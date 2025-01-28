#include "quant.hpp"

#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>

#define decl_kernel_pair(impl) \
    extern auto f32_q8_##impl( \
      const float* __restrict__ x, \
        std::uint8_t* __restrict__ o, \
        std::size_t n, \
        float inv_scale, \
        std::int32_t zero_point, \
        const bool sto_rnd, \
        quant::prng_state& prng \
    ) noexcept -> void; \
    extern auto f32_q4_##impl( \
        const float* __restrict__ x, \
        std::uint8_t* __restrict__ o, \
        std::size_t n, \
        float inv_scale, \
        std::int32_t zero_point, \
        const bool sto_rnd, \
        quant::prng_state& prng \
    ) noexcept -> void

decl_kernel_pair(generic);

#ifdef __x86_64__
    #include <cpuid.h>

    [[nodiscard]] auto check_sse42_support() noexcept -> bool {
        int info[4] = {-1};
        __cpuid(0, info[0], info[1], info[2], info[3]);
        if (info[0] < 1) return false;
        __cpuid(1, info[0], info[1], info[2], info[3]);
        return (info[2] & (1<<20)) != 0;
    }

    [[nodiscard]] auto check_avx2_support() noexcept -> bool {
        int info[4] = {-1};
        __cpuid(0, info[0], info[1], info[2], info[3]);
        if (info[0] < 7) return false;
        __cpuid(1, info[0], info[1], info[2], info[3]);
        if ((info[2] & 0x38081001) != 0x38081001) return false;
        __cpuid_count(7, 0, info[0], info[1], info[2], info[3]);
        if ((info[1] & 0x20) != 0x20) return false;
        std::uint32_t lo, hi;
        asm volatile("xgetbv\n\t" : "=a" (lo), "=d" (hi) : "c" (0));
        return ((static_cast<uint64_t>(lo)|(static_cast<uint64_t>(hi) << 32)) & 6) == 6;
    }

    [[nodiscard]] auto check_avx512f_support() noexcept -> bool {
        int info[4] = {-1};
       __cpuid(0, info[0], info[1], info[2], info[3]);
        if (info[0] < 7) return false;
        __cpuid(1, info[0], info[1], info[2], info[3]);
        if ((info[2] & 0x8000000) == 0 || (info[2] & 0x10000000) == 0) return false;
        __cpuid_count(7, 0, info[0], info[1], info[2], info[3]);
        if ((info[1] & 0x10000) == 0) return false;
        std::uint32_t lo, hi;
        asm volatile("xgetbv\n\t" : "=a" (lo), "=d" (hi) : "c" (0));
        return ((static_cast<uint64_t>(lo)|(static_cast<uint64_t>(hi) << 32)) & 0xe0) == 0xe0;
    }

    decl_kernel_pair(amd64_sse42);
    decl_kernel_pair(amd64_avx2);
    decl_kernel_pair(amd64_avx512f);
#endif

#undef decl_kernel_pair

namespace quant {
    context::context(std::size_t num_threads) {
        num_threads = std::max<std::size_t>(1, num_threads);
        m_workers.reserve(num_threads);
        for (std::int64_t ti {0}; ti < num_threads; ++ti) { // Initialize workers (main thread is worker 0)
            m_workers.emplace_back(*this, ti, static_cast<std::int64_t>(num_threads));
        }
        #ifdef __x86_64__
            m_sse42_supported = check_sse42_support();
            m_avx2_supported = check_avx2_support();
            m_avx512f_supported = check_avx512f_support();
        #endif
        while (m_workers_online.load(std::memory_order_seq_cst) != num_threads-1)
            std::this_thread::yield();
    }

    context::worker::worker(context& ctx, const std::int64_t ti, const std::int64_t tc)
        : ctx{&ctx}, payload{static_cast<std::uint32_t>(ti ^ tc)} {
        payload.ti = ti;
        payload.tc = tc;
        payload.phase = 0;
        if (ti != 0) { // ti != 0 are extra worker thread, ti == 0 is main thread
            thread.emplace(&worker::entry, this);
        }
    }

    auto context::worker::await_work() -> bool {
        std::unique_lock lock {ctx->m_mtx};
        ctx->m_cv.wait(lock, [this]() noexcept -> bool { return ctx->m_interrupt || ctx->m_phase > payload.phase; });
        if (ctx->m_interrupt) [[unlikely]] return false;
        payload.phase = ctx->m_phase;
        return true;
    }

    auto context::worker::entry() -> void {
        ctx->m_workers_online.fetch_add(1, std::memory_order_seq_cst);
        while (await_work()) [[likely]]
            exec_and_broadcast();
    }

    auto context::worker::exec_and_broadcast() -> void {
        const auto* const bx {op.in};
        auto* const br {op.out};
        const std::int64_t tc {payload.tc};
        const std::int64_t ti {payload.ti};
        const std::int64_t numel {op.numel};
        const std::int64_t chunk {(numel + tc - 1)/tc};
        const std::int64_t ra {chunk*ti};
        const std::int64_t rb {std::min(ra + chunk, numel)};
        if (const std::int64_t vmel {rb - ra}; vmel > 0) [[likely]] {
            const auto* const px {bx + ra};
            auto* const pr {br + ra};
            const float scale {op.scale};
            const std::int32_t zp {op.zero_point};
            const bool sto_rnd {op.rnd_mode == round_mode::stochastic};
            #ifdef __x86_64__
                if (ctx.m_avx512f_supported) f32_q8_amd64_avx512f(px, pr, vmel, scale, zp, sto_rnd, payload.prng);
                else if (ctx.m_avx2_supported) f32_q8_amd64_avx2(px, pr, vmel, scale, zp, sto_rnd, payload.prng);
                else if (ctx.m_sse42_supported) f32_q8_amd64_sse42(px, pr, vmel, scale, zp, sto_rnd, payload.prng);
                else f32_q8_generic(px, pr, vmel, scale, zp, sto_rnd, payload.prng);
            #else
                f32_q8_generic(px, pr, vmel, scale, zp, sto_rnd, payload.prng);
            #endif
        }
        if (1 + ctx->m_num_completed.fetch_add(1, std::memory_order::relaxed) == ctx->m_workers.size()) {
            std::unique_lock lock {ctx->m_mtx};
            ctx->m_cv.notify_all();
        }
    }

    auto context::kickoff_workers(
        const std::span<const float> in,
        const std::span<std::uint8_t> out,
        const float scale,
        const std::int32_t zero_point,
        const round_mode mode
    ) -> void {
        std::unique_lock lock {m_mtx};
        for (auto& worker : m_workers) {
            worker.op = op_info {
                .in = in.data(),
                .out = out.data(),
                .numel = static_cast<std::int64_t>(in.size()),
                .scale = scale,
                .zero_point = zero_point,
                .rnd_mode = mode
            };
        }
        ++m_phase;
        m_num_completed.store(0, std::memory_order::relaxed);
        m_cv.notify_all();
    }

    auto context::barrier() -> void {
        std::unique_lock lock {m_mtx};
        m_cv.wait(lock, [&]() noexcept -> bool { return m_num_completed.load(std::memory_order_relaxed) == m_workers.size(); });
    }

    context::~context() {
        std::unique_lock lock {m_mtx};
        m_interrupt = true;
        ++m_phase;
        lock.unlock();
        m_cv.notify_all();
        for (auto& worker : m_workers)
            if (worker.thread && worker.thread->joinable())
                worker.thread->join();
        m_workers.clear();
    }

    auto context::quantize_int8(
        const std::span<const float> in,
        const std::span<std::uint8_t> out,
        const float scale,
        const std::int32_t zero_point,
        const round_mode mode
    ) -> void {
        kickoff_workers(in, out, scale, zero_point, mode);
        worker& w0 {m_workers[0]}; // Main thread does work too
        w0.exec_and_broadcast();
        barrier();
    }
}
