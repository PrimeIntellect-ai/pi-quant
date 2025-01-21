#include "quant.hpp"

#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>

extern auto f32_q8_generic(
    const float* __restrict__ x,
    std::uint8_t* __restrict__ o,
    std::size_t n,
    float inv_scale,
    std::int32_t zero_point
) noexcept -> void;

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

    extern auto f32_q8_amd64_sse42(
        const float* __restrict__ x,
        std::uint8_t* __restrict__ o,
        std::size_t n,
        float inv_scale,
        std::int32_t zero_point
    ) noexcept -> void;
    extern auto f32_q4_amd64_sse42(
        const float* __restrict__ x,
        std::uint8_t* __restrict__ o,
        std::size_t n,
        float inv_scale,
        std::int32_t zero_point
    ) noexcept -> void;

    extern auto f32_q8_amd64_avx2(
        const float* __restrict__ x,
        std::uint8_t* __restrict__ o,
        std::size_t n,
        float inv_scale,
        std::int32_t zero_point
    ) noexcept -> void;
    extern auto f32_q4_amd64_avx2(
        const float* __restrict__ x,
        std::uint8_t* __restrict__ o,
        std::size_t n,
        float inv_scale,
        std::int32_t zero_point
    ) noexcept -> void;

    extern auto f32_q8_amd64_avx512f(
        const float* __restrict__ x,
        std::uint8_t* __restrict__ o,
        std::size_t n,
        float inv_scale,
        std::int32_t zero_point
    ) noexcept -> void;
    extern auto f32_q4_amd64_avx512f(
        const float* __restrict__ x,
        std::uint8_t* __restrict__ o,
        std::size_t n,
        float inv_scale,
        std::int32_t zero_point
    ) noexcept -> void;
#endif

namespace quant {
    context::context(std::size_t num_threads) {
        num_threads = std::max<std::size_t>(1, num_threads);
        m_workers.resize(num_threads);
        for (std::int64_t ti {0}; ti < m_workers.size(); ++ti) { // Initialize workers (main thread is worker 0)
            auto& worker {m_workers[ti]};
            worker.payload.ti = ti;
            worker.payload.tc = static_cast<std::int64_t>(num_threads);
            worker.payload.phase = 0;
            if (ti != 0)
                worker.thread = std::thread{&worker::entry, &worker, std::ref(*this)};
        }
        #ifdef __x86_64__
            m_sse42_supported = check_sse42_support();
            m_avx2_supported = check_avx2_support();
            m_avx512f_supported = check_avx512f_support();
        #endif
        while (m_workers_online.load(std::memory_order_seq_cst) != num_threads-1)
            std::this_thread::yield();
    }

    auto context::worker::await_work(context& ctx) -> bool {
        std::unique_lock lock {ctx.m_mtx};
        ctx.m_cv.wait(lock, [&]() noexcept -> bool { return ctx.m_interrupt || ctx.m_phase > payload.phase; });
        if (ctx.m_interrupt) [[unlikely]] return false;
        payload.phase = ctx.m_phase;
        return true;
    }

    auto context::worker::entry(context& ctx) -> void {
        ctx.m_workers_online.fetch_add(1, std::memory_order_seq_cst);
        while (await_work(ctx)) [[likely]]
            exec_and_broadcast(ctx);
    }

    auto context::worker::exec_and_broadcast(context& ctx) -> void {
        const auto* const bx {payload.in};
        auto* const br {payload.out};
        const std::int64_t tc {payload.tc};
        const std::int64_t ti {payload.ti};
        const std::int64_t numel {payload.numel};
        const std::int64_t chunk {(numel + tc - 1)/tc};
        const std::int64_t ra {chunk*ti};
        const std::int64_t rb {std::min(ra + chunk, numel)};
        if (rb > ra) [[likely]] {
            const std::int64_t vmel {rb - ra};
            const auto* const px {bx + ra};
            auto* const pr {br + ra};
            const float scale {payload.scale};
            const std::int32_t zp {payload.zero_point};
            #ifdef __x86_64__
                if (ctx.m_avx512f_supported) f32_q8_amd64_avx512f(px, pr, vmel, scale, zp);
                else if (ctx.m_avx2_supported) f32_q8_amd64_avx2(px, pr, vmel, scale, zp);
                else if (ctx.m_sse42_supported) f32_q8_amd64_sse42(px, pr, vmel, scale, zp);
                else f32_q8_generic(px, pr, vmel, scale, zp);
            #else
                f32_q8_generic(px, pr, vmel, scale, zp);
            #endif
        }
        if (1 + ctx.m_num_completed.fetch_add(1, std::memory_order::relaxed) == ctx.m_workers.size()) {
            std::unique_lock lock {ctx.m_mtx};
            ctx.m_cv.notify_all();
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
            worker.payload.in = in.data();
            worker.payload.out = out.data();
            worker.payload.numel = static_cast<std::int64_t>(in.size());
            worker.payload.scale = scale;
            worker.payload.zero_point = zero_point;
            worker.payload.mode = mode;
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
        w0.exec_and_broadcast(*this);
        barrier();
    }
}
