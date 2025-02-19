#include "quant.hpp"

#include <cassert>
#include <cstdarg>
#include <thread>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace quant {
    #define decl_kernel_fn(impl) \
        extern auto impl( \
        const float* __restrict__ x, \
        std::uint8_t* __restrict__ o, \
        std::int64_t numel, \
        float inv_scale, \
        std::int32_t zero_point, \
        const bool sto_rnd, \
        quant::prng_state& prng \
    ) noexcept -> void;

    decl_kernel_fn(f32_q8_generic);
    decl_kernel_fn(f32_q4_generic);

    using kernel_fn = auto (
        const float* __restrict__ x,
        std::uint8_t* __restrict__ o,
        std::int64_t n,
        float inv_scale,
        std::int32_t zero_point,
        bool sto_rnd,
        quant::prng_state& prng
    ) -> void;

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

        decl_kernel_fn(f32_q8_amd64_sse42);
        decl_kernel_fn(f32_q4_amd64_sse42);
        decl_kernel_fn(f32_q8_amd64_avx2);
        decl_kernel_fn(f32_q4_amd64_avx2);
        decl_kernel_fn(f32_q8_amd64_avx512f);
        decl_kernel_fn(f32_q4_amd64_avx512f);

        static constexpr std::array<kernel_fn*, static_cast<std::size_t>(amd64_cpu_caps::num_)> k_dispatch_i8 = {
            &f32_q8_generic,
            &f32_q8_amd64_sse42,
            &f32_q8_amd64_avx2,
            &f32_q8_amd64_avx512f
        };
        static constexpr std::array<kernel_fn*, static_cast<std::size_t>(amd64_cpu_caps::num_)> k_dispatch_i4 = {
            &f32_q4_generic,
            &f32_q4_amd64_sse42,
            &f32_q4_amd64_avx2,
            &f32_q4_amd64_avx512f
        };
    #endif

    #undef decl_kernel_pair

    auto compute_quant_config_from_data(const std::span<const float> x) -> std::pair<float, std::int32_t> {
        if (x.empty()) [[unlikely]] return {0.0f, 0.0f};
        float mean {std::accumulate(x.begin(), x.end(), 0.0f) / static_cast<float>(x.size())};
        float sq_delta {std::transform_reduce(
            x.begin(), x.end(),
            0.0f,
            std::plus{},
            [mean](const float value) noexcept -> float {
                const float delta {value - mean};
                return delta * delta;
            }
        )};
        const float std {std::sqrt(sq_delta / static_cast<float>((x.size()-1)))};
        const float scale {12.0f*std/255.0f};
        const std::int32_t zp {127 - static_cast<std::int32_t>(std::round(mean/scale))};
        return {scale, zp};
    }

    auto panic(const char* msg, ...) -> void {
        std::va_list args;
        va_start(args, msg);
        char tmp[8192];
        int delta{std::snprintf(tmp, sizeof(tmp), "%s", "\x1b[31m")};
        delta += std::vsnprintf(tmp+delta, sizeof(tmp)-delta, msg, args);
        std::snprintf(tmp+delta, sizeof(tmp)-delta, "%s", "\x1b[0m");
        std::cerr << tmp << std::endl;
        va_end(args);
        std::abort();
    }

    context::context(std::size_t num_threads) {
        num_threads = std::max<std::size_t>(1, num_threads);
        m_workers.reserve(num_threads);
        for (std::int64_t ti {}; ti < num_threads; ++ti) { // Initialize workers (main thread is worker 0)
            m_workers.emplace_back(*this, ti, static_cast<std::int64_t>(num_threads));
        }
        #ifdef __x86_64__
            if (check_avx512f_support()) cpu_caps = amd64_cpu_caps::avx512;
            else if (check_avx2_support()) cpu_caps = amd64_cpu_caps::avx2;
            else if (check_sse42_support()) cpu_caps = amd64_cpu_caps::sse_4_2;
            else cpu_caps = amd64_cpu_caps::none;
        #endif
        while (m_workers_online.load(std::memory_order_seq_cst) != num_threads-1)
            std::this_thread::yield();
    }

    context::worker::worker(context& ctx, const std::int64_t ti, const std::int64_t tc)
        : ctx{&ctx}, pl{static_cast<std::uint32_t>(ti ^ tc)} {
        pl.ti = ti;
        pl.tc = tc;
        pl.phase = 0;
        if (ti != 0) { // ti != 0 are extra worker thread, ti == 0 is main thread
            thread.emplace(&worker::entry, this);
        }
    }

    auto context::worker::await_work() -> bool {
        std::unique_lock lock {ctx->m_mtx};
        ctx->m_cv.wait(lock, [this]() noexcept -> bool { return ctx->m_interrupt || ctx->m_phase > pl.phase; });
        if (ctx->m_interrupt) [[unlikely]] return false;
        pl.phase = ctx->m_phase;
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
        const std::int64_t tc {pl.tc};
        const std::int64_t ti {pl.ti};
        const std::int64_t numel {op.numel};
        const bool is_i8 {op.format == op_info::q_i8};
        const auto dispatch {[=](const float* px, std::uint8_t* pr, std::int64_t vmel) noexcept -> void {
            #ifdef __x86_64__
                const auto cap_idx {static_cast<std::size_t>(ctx->cpu_caps)};
                auto* const kernel {is_i8 ? k_dispatch_i8[cap_idx] : k_dispatch_i4[cap_idx]};
                (*kernel)(px, pr, vmel, op.scale, op.zero_point, op.rnd_mode == round_mode::stochastic, pl.prng);
            #else
                auto* const kernel {op.format == op_info::q_i8 ? &f32_q8_generic : &f32_q4_generic};
                (*kernel)(px, pr, vmel, op.scale, op.zero_point, op.rnd_mode == round_mode::stochastic, pl.prng);
            #endif
        }};
        if (is_i8) {
            const std::int64_t chunk = (numel + tc - 1)/tc;
            const std::int64_t ra = chunk*ti;
            const std::int64_t rb = std::min(ra + chunk, numel);
            if (rb > ra) [[likely]] {
                dispatch(bx + ra, br + ra, rb - ra);
            }
        } else {
            const std::int64_t pairs = (numel + 1)>>1;
            const std::int64_t pair_chunk = (pairs + tc - 1) / tc;
            const std::int64_t pra = pair_chunk * ti;
            const std::int64_t prb = std::min(pra + pair_chunk, pairs);
            if (prb > pra) [[likely]] {
                const std::int64_t ra = pra<<1;
                const std::int64_t rb = prb<<1 > numel ? numel : prb<<1; /* When numel is odd, the last pair is incomplete */
                dispatch(bx + ra, br + pra, rb - ra);
            }
        }
        if (1+ctx->m_num_completed.fetch_add(1, std::memory_order::relaxed) == ctx->m_workers.size()) {
            std::unique_lock lock {ctx->m_mtx};
            ctx->m_cv.notify_all();
        }
    }

    auto context::kickoff_workers(const op_info& info) -> void {
        std::unique_lock lock {m_mtx};
        for (auto& worker : m_workers)
            worker.op = info;
        ++m_phase;
        m_num_completed.store(0, std::memory_order::relaxed);
        m_cv.notify_all();
    }

    auto context::barrier() -> void {
        std::unique_lock lock {m_mtx};
        m_cv.wait(lock, [&]() noexcept -> bool { return m_num_completed.load(std::memory_order_relaxed) == m_workers.size(); });
    }

    auto context::operator()(const op_info& info) -> void {
        kickoff_workers(info);
        worker& w0 {m_workers[0]}; // Main thread does work too
        w0.exec_and_broadcast();
        barrier();
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

    auto context::quantize_uint8(
        const std::span<const float> in,
        const std::span<std::uint8_t> out,
        const float scale,
        const std::int32_t zero_point,
        const round_mode mode
    ) -> void {
        quant_assert(in.size() == out.size(), "input and output spans must have the same length, but %zu != %zu", in.size(), out.size());
        const op_info info {
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()),
            .scale = scale,
            .zero_point = zero_point,
            .rnd_mode = mode,
            .format = op_info::q_i8
        };
        (*this)(info);
    }

    auto context::dequantize_uint8(
        const std::span<std::uint8_t> in,
        const std::span<float> out,
        const float scale,
        const std::int32_t zero_point
    ) -> void {
        quant_assert(in.size() == out.size(), "input and output spans must have the same length, but %zu != %zu", in.size(), out.size());
    }

    auto context::quantize_uint4(
        const std::span<const float> in,
        const std::span<std::uint8_t> out,
        const float scale,
        const std::int32_t zero_point,
        const round_mode mode
    ) -> void {
        std::size_t output_len {(in.size() + 1)>>1};
        quant_assert(in.size() == output_len, "input and output spans must have the same length, but %zu != %zu", in.size(), output_len);
        const op_info info {
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()),
            .scale = scale,
            .zero_point = zero_point,
            .rnd_mode = mode,
            .format = op_info::q_i4
        };
        (*this)(info);
    }
}
