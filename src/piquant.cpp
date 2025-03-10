#include "piquant.hpp"

#include <cassert>
#include <cstdarg>
#include <thread>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace piquant {
    #define decl_quant_kernel_fn(impl) \
        extern auto impl( \
        const float* __restrict__ x, \
        std::uint8_t* __restrict__ o, \
        std::int64_t numel, \
        float scale, \
        std::int32_t zero_point, \
        const bool sto_rnd, \
        prng_state& prng \
    ) noexcept -> void

    #define decl_dequant_kernel_fn(impl) \
        extern auto impl( \
            const std::uint8_t* __restrict__ x, \
            float* __restrict__ o, \
            std::int64_t numel, \
            float scale, \
            std::int32_t zero_point, \
            reduce_op op \
        ) noexcept -> void

    decl_quant_kernel_fn(f32_quant8_generic);
    decl_dequant_kernel_fn(f32_dequant8_generic);
    decl_quant_kernel_fn(f32_quant4_generic);
    decl_dequant_kernel_fn(f32_dequant4_generic);

    using quant_kernel = auto (
        const float* __restrict__ x,
        std::uint8_t* __restrict__ o,
        std::int64_t numel,
        float scale,
        std::int32_t zero_point,
        bool sto_rnd,
        prng_state& prng
    ) -> void;

    using dequant_kernel = auto (
        const std::uint8_t* __restrict__ x,
        float* __restrict__ o,
        std::int64_t numel,
        float scale,
        std::int32_t zero_point,
        reduce_op op
    ) -> void;

    #ifdef __x86_64__
        #include <cpuid.h>

        [[nodiscard]] static auto check_sse42_support() noexcept -> bool {
            int info[4] = {-1};
            __cpuid(0, info[0], info[1], info[2], info[3]);
            if (info[0] < 1) return false;
            __cpuid(1, info[0], info[1], info[2], info[3]);
            return (info[2] & (1<<20)) != 0;
        }

        [[nodiscard]] static auto check_avx2_support() noexcept -> bool {
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

        [[nodiscard]] static auto check_avx512f_support() noexcept -> bool {
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

        decl_quant_kernel_fn(f32_quant8_amd64_sse42);
        decl_dequant_kernel_fn(f32_dequant8_amd64_sse42);
        decl_quant_kernel_fn(f32_quant4_amd64_sse42);
        decl_dequant_kernel_fn(f32_dequant4_amd64_sse42);
        decl_quant_kernel_fn(f32_quant8_amd64_avx2);
        decl_dequant_kernel_fn(f32_dequant8_amd64_avx2);
        decl_quant_kernel_fn(f32_quant4_amd64_avx2);
        decl_dequant_kernel_fn(f32_dequant4_amd64_avx2);
        decl_quant_kernel_fn(f32_quant8_amd64_avx512f);
        decl_dequant_kernel_fn(f32_dequant8_amd64_avx512f);
        decl_quant_kernel_fn(f32_quant4_amd64_avx512f);
        decl_dequant_kernel_fn(f32_dequant4_amd64_avx512f);

        static constexpr std::array<quant_kernel*, static_cast<std::size_t>(amd64_cpu_caps::num_)> quant8_routines = {
            &f32_quant8_generic,
            &f32_quant8_amd64_sse42,
            &f32_quant8_amd64_avx2,
            &f32_quant8_amd64_avx512f
        };
        static constexpr std::array<dequant_kernel*, static_cast<std::size_t>(amd64_cpu_caps::num_)> dequant8_routines = {
            &f32_dequant8_generic,
            &f32_dequant8_amd64_sse42,
            &f32_dequant8_amd64_avx2,
            &f32_dequant8_amd64_avx512f
        };
        static constexpr std::array<quant_kernel*, static_cast<std::size_t>(amd64_cpu_caps::num_)> quant4_routines = {
            &f32_quant4_generic,
            &f32_quant4_amd64_sse42,
            &f32_quant4_amd64_avx2,
            &f32_quant4_amd64_avx512f
        };
        static constexpr std::array<dequant_kernel*, static_cast<std::size_t>(amd64_cpu_caps::num_)> dequant4_routines = {
            &f32_dequant4_generic,
            &f32_dequant4_amd64_sse42,
            &f32_dequant4_amd64_avx2,
            &f32_dequant4_amd64_avx512f
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
        std::int64_t numel {};
        bool is_i8 {};
        bool is_dequant {};
        if (const auto* quant_desc {std::get_if<quant_descriptor>(&cmd)}; quant_desc) {
            numel = quant_desc->numel;
            is_i8 = quant_desc->format == dtype::uint8;
            is_dequant = false;
        } else if (const auto* dequant_desc {std::get_if<dequant_descriptor>(&cmd)}; dequant_desc) {
            numel = dequant_desc->numel;
            is_i8 = dequant_desc->format == dtype::uint8;
            is_dequant = true;
        } else {
            panic("Invalid command type");
        }

        const std::int64_t tc {pl.tc};
        const std::int64_t ti {pl.ti};

        const auto partition_row {[=] (const bool is_uint8) noexcept -> std::optional<std::array<std::int64_t, 3>> {
            if (is_uint8) {
                const std::int64_t chunk {(numel + tc - 1)/tc};
                const std::int64_t ra {chunk*ti};
                const std::int64_t rb {std::min(ra + chunk, numel)};
                if (ra >= rb) [[unlikely]] return {};
                return {{ra, ra, rb-ra}};
            }
            const std::int64_t pairs {(numel + 1)>>1};
            const std::int64_t pair_chunk {(pairs + tc - 1)/tc};
            const std::int64_t pra {pair_chunk*ti};
            const std::int64_t prb {std::min(pra + pair_chunk, pairs)};
            if (pra >= prb) [[unlikely]] return {};
            const std::int64_t ra {pra<<1};
            const std::int64_t rb {prb<<1 > numel ? numel : prb<<1}; /* When numel is odd, the last pair is incomplete */
            return {{ra, pra, rb-ra}};
        }};

        const auto dispatch_quant {[=, this](const std::int64_t oa, const std::int64_t ob, const std::int64_t n, const quant_descriptor& cmd) noexcept -> void {
            #ifdef __x86_64__
                const auto level {static_cast<std::size_t>(ctx->cpu_caps)};
                auto* const kernel {is_i8 ? quant8_routines[level] : quant4_routines[level]};
                (*kernel)(cmd.in+oa, cmd.out+ob, n, cmd.scale, cmd.zero_point, cmd.rnd_mode == round_mode::stochastic, pl.prng);
            #else
                auto* const kernel {is_i8 ? &f32_quant8_generic : &f32_quant4_generic};
                (*kernel)(cmd.in+oa, cmd.out+ob, n, cmd.scale, cmd.zero_point, cmd.rnd_mode == round_mode::stochastic, pl.prng);
            #endif
        }};

        const auto dispatch_dequant {[=, this](const std::int64_t oa, const std::int64_t ob, const std::int64_t n, const dequant_descriptor& cmd) noexcept -> void {
            #ifdef __x86_64__
                const auto level {static_cast<std::size_t>(ctx->cpu_caps)};
                auto* const kernel {is_i8 ? dequant8_routines[level] : dequant4_routines[level]};
                (*kernel)(cmd.in+oa, cmd.out+ob, n, cmd.scale, cmd.zero_point, cmd.op);
            #else
                auto* const kernel {is_i8 ? &f32_dequant8_generic : &f32_dequant4_generic};
                (*kernel)(cmd.in+oa, cmd.out+ob, n, cmd.scale, cmd.zero_point, cmd.op);
            #endif
        }};

        if (const auto partition {partition_row(is_i8)}; partition) [[likely]] {
            const auto [oa, ob, n] {*partition};
            if (is_dequant) dispatch_dequant(oa, ob, n, std::get<dequant_descriptor>(cmd));
            else dispatch_quant(oa, ob, n, std::get<quant_descriptor>(cmd));
        }

        if (1+ctx->m_num_completed.fetch_add(1, std::memory_order::relaxed) == ctx->m_workers.size()) { // Last worker
            std::unique_lock lock {ctx->m_mtx};
            ctx->m_cv.notify_all();
        }
    }

    auto context::kickoff_workers(quant_command&& cmd) -> void {
        std::unique_lock lock {m_mtx};
        for (auto& worker : m_workers)
            worker.cmd = cmd;
        ++m_phase;
        m_num_completed.store(0, std::memory_order::relaxed);
        m_cv.notify_all();
    }

    auto context::barrier() -> void {
        std::unique_lock lock {m_mtx};
        m_cv.wait(lock, [&]() noexcept -> bool { return m_num_completed.load(std::memory_order_relaxed) == m_workers.size(); });
    }

    auto context::operator()(quant_command&& cmd) -> void {
        kickoff_workers(std::forward<decltype(cmd)>(cmd));
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
        quant_descriptor info {
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()),
            .scale = scale,
            .zero_point = zero_point,
            .rnd_mode = mode,
            .format = dtype::uint8,
        };
        (*this)(info);
    }

    auto context::dequantize_uint8(
        const std::span<const std::uint8_t> in,
        const std::span<float> out,
        const float scale,
        const std::int32_t zero_point,
        const reduce_op op
    ) -> void {
        quant_assert(in.size() == out.size(), "input and output spans must have the same length, but %zu != %zu", in.size(), out.size());
        dequant_descriptor info {
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()),
            .scale = scale,
            .zero_point = zero_point,
            .format = dtype::uint4,
            .op = op
        };
        (*this)(info);
    }

    auto context::quantize_uint4(
        const std::span<const float> in,
        const std::span<std::uint8_t> out,
        const float scale,
        const std::int32_t zero_point,
        const round_mode mode
    ) -> void {
        std::size_t output_len {(in.size() + 1)>>1};
        quant_assert(out.size() == output_len, "input and output spans must have the same length, but %zu != %zu", out.size(), output_len);
        const quant_descriptor info {
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()),
            .scale = scale,
            .zero_point = zero_point,
            .rnd_mode = mode,
            .format = dtype::uint8
        };
        (*this)(info);
    }

    auto context::dequantize_uint4(
        const std::span<const std::uint8_t> in,
        const std::span<float> out,
        const float scale,
        const std::int32_t zero_point,
        const reduce_op op
    ) -> void {
        quant_assert(in.size() == out.size(), "input and output spans must have the same length, but %zu != %zu", in.size(), out.size());
        dequant_descriptor info {
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()),
            .scale = scale,
            .zero_point = zero_point,
            .format = dtype::uint4,
            .op = op
        };
        (*this)(info);
    }

    auto quantize_dequantize_redundant(
        std::span<const float> in,
        std::span<std::uint8_t> out,
        dtype format
    ) -> void {

    }
}
