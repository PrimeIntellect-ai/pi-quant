#include "piquant.hpp"

#include <cassert>
#include <cstdarg>
#include <thread>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <variant>
#include <condition_variable>

namespace piquant {
#define decl_quant_kernel_fn(impl) \
extern __attribute__((hot)) auto impl( \
const void* x, \
void* o, \
std::int64_t range, \
const context::quant_descriptor& desc, \
prng_state& prng \
) noexcept -> void

    decl_quant_kernel_fn(quant_generic);

    using quant_kernel = auto (
        const void* x,
        void* o,
        std::int64_t range,
        const context::quant_descriptor& desc,
        prng_state& prng
    ) noexcept -> void;

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

    decl_quant_kernel_fn(quant_amd64_sse42);
    decl_quant_kernel_fn(quant_amd64_avx2);
    decl_quant_kernel_fn(quant_amd64_avx512f);

    static constexpr std::array<quant_kernel*, static_cast<std::size_t>(amd64_cpu_caps::num_)> quant_routines = {
        &quant_generic,
        &quant_amd64_sse42,
        &quant_amd64_avx2,
        &quant_amd64_avx512f
    };

#endif

#undef decl_kernel_pair

    template <class... T>
    struct overloads final : T... { using T::operator()...; };

    template <typename T> requires std::is_floating_point_v<T>
    [[nodiscard]] static auto compute_quant_config_from_data(const std::span<const T> x, std::int64_t tmax) -> std::pair<T, std::int64_t> {
        if (x.empty()) [[unlikely]] return {0.0, 0.0};
        //tmax &= (1ull<<52)-1; // Remove superfluous precision bit for float64
        auto mean {static_cast<T>(std::accumulate(x.begin(), x.end(), 0.0) / static_cast<T>(x.size()))};
        auto sq_delta {static_cast<T>(std::transform_reduce(
            x.begin(), x.end(),
            0.0,
            std::plus{},
            [mean](const T value) noexcept -> T {
                const T delta {value - mean};
                return delta * delta;
            }
        ))};
        const auto std {static_cast<T>(std::sqrt(sq_delta / static_cast<T>(x.size()-1)))};
        const auto scale {static_cast<T>(12.0*std/static_cast<T>(tmax))};
        const std::int64_t zp {(tmax>>1) - static_cast<std::int64_t>(std::round(mean/scale))};
        return {scale, zp};
    }

    auto compute_quant_config_from_data(const std::span<const float> x, const std::int64_t tmax) -> std::pair<float, std::int64_t> {
        return compute_quant_config_from_data<float>(x, tmax);
    }

    auto compute_quant_config_from_data(const std::span<const double> x, const std::int64_t tmax) -> std::pair<double, std::int64_t> {
        return compute_quant_config_from_data<double>(x, tmax);
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

    static constexpr std::size_t cache_line {
        #ifdef __cpp_lib_hardware_interference_size
            std::hardware_destructive_interference_size
        #else
            64
        #endif
    };

    struct payload {
        prng_state prng;
        std::int64_t ti {}; // thread index
        std::int64_t tc {}; // thread count
        std::uint64_t phase {};

        explicit constexpr payload(const std::uint32_t seed) noexcept : prng{seed} {}
    };

    struct worker final {
        context::pimpl* pimpl;
        alignas(cache_line) payload pl;
        alignas(cache_line) context::quant_descriptor cmd {};
        std::optional<std::thread> thread {};

        explicit worker(context::pimpl& ctx, const std::int64_t ti, const std::int64_t tc) : pimpl{&ctx}, pl{static_cast<std::uint32_t>(ti ^ tc)} {
            pl.ti = ti;
            pl.tc = tc;
            pl.phase = 0;
            if (ti != 0) { // ti != 0 are extra worker thread, ti == 0 is main thread
                thread.emplace(&worker::entry, this);
            }
        }

        worker(const worker&) = delete;
        worker(worker&&) = default;
        auto operator = (const worker&) -> worker& = delete;
        auto operator = (worker&&) -> worker& = default;

        ~worker() {
            if (thread && thread->joinable())
                thread->join();
        }

        [[nodiscard]] auto await_work() -> bool;
        auto entry() -> void;
        auto exec_and_broadcast() -> void;
    };

    class context::pimpl final {
    public:
        explicit pimpl(std::size_t num_threads);
        pimpl(const pimpl&) = delete;
        pimpl(pimpl&&) = delete;
        auto operator = (const pimpl&) -> pimpl& = delete;
        auto operator = (pimpl&&) -> pimpl& = delete;
        ~pimpl();

        alignas(cache_line) volatile bool m_interrupt {};
        alignas(cache_line) std::uint64_t m_phase {};
        alignas(cache_line) std::atomic_int64_t m_num_completed {};
        std::vector<worker> m_workers {};
        std::condition_variable m_cv {};
        std::mutex m_mtx {};
        std::atomic_size_t m_workers_online {};
#ifdef __x86_64__
        amd64_cpu_caps cpu_caps {};
#endif

        auto kickoff_workers(quant_descriptor&& cmd) -> void;
        auto barrier() -> void;
        auto operator()(quant_descriptor&& cmd) -> void;
    };

    auto worker::await_work() -> bool {
        std::unique_lock lock {pimpl->m_mtx};
        pimpl->m_cv.wait(lock, [this]() noexcept -> bool { return pimpl->m_interrupt || pimpl->m_phase > pl.phase; });
        if (pimpl->m_interrupt) [[unlikely]] return false;
        pl.phase = pimpl->m_phase;
        return true;
    }

    auto worker::entry() -> void {
        pimpl->m_workers_online.fetch_add(1, std::memory_order_seq_cst);
        while (await_work()) [[likely]]
            exec_and_broadcast();
    }

    auto worker::exec_and_broadcast() -> void {
        const std::int64_t tc {pl.tc};
        const std::int64_t ti {pl.ti};
        const auto partition_row {[=, this] () noexcept -> std::optional<std::array<std::int64_t, 3>> {
            if (dtype_info_of(cmd.dt_out).bit_size < 8) {       // Subbyte granularity requires special handling to not split packed bit pairs
                const std::int64_t pairs {(cmd.numel + 1)>>1};
                const std::int64_t pair_chunk {(pairs + tc - 1)/tc};
                const std::int64_t pra {pair_chunk*ti};
                const std::int64_t prb {std::min(pra + pair_chunk, pairs)};
                if (pra >= prb) [[unlikely]] return {};
                const std::int64_t ra {pra<<1};
                const std::int64_t rb {prb<<1 > cmd.numel ? cmd.numel : prb<<1}; // When numel is odd, the last pair is incomplete
                return {{ra, pra, rb-ra}};
            }
            const std::int64_t chunk {(cmd.numel + tc - 1)/tc};
            const std::int64_t ra {chunk*ti};
            const std::int64_t rb {std::min(ra + chunk, cmd.numel)};
            if (ra >= rb) [[unlikely]] return {};
            return {{ra, ra, rb-ra}};
        }};
        const auto dispatch_quant {[=, this](const std::int64_t oa, const std::int64_t ob, const std::int64_t range, const context::quant_descriptor& cmd) noexcept -> void {
            #ifdef __x86_64__
                const auto level {static_cast<std::size_t>(pimpl->cpu_caps)};
                piquant_assert2(level < quant_routines.size());
                auto* const kernel {quant_routines[level]};
                piquant_assert2(kernel != nullptr);
                const auto si {dtype_info_of(cmd.dt_in).stride};
                const auto so {cmd.type == context::command_type::quant_dequant ? si : dtype_info_of(cmd.dt_out).stride};
                (*kernel)(
                    cmd.in + si*oa,
                    cmd.out + so*ob,
                    range,
                    cmd,
                    pl.prng
                );
            #else
                auto* const kernel {&quant_generic};
                piquant_assert2(kernel != nullptr);
                const auto si {dtype_info_of(cmd.dt_in).stride};
                const auto so {cmd.type == context::command_type::quant_dequant ? si : dtype_info_of(cmd.dt_out).stride};
                (*kernel)(
                    cmd.in + si*oa,
                    cmd.out + so*ob,
                    range,
                    cmd,
                    pl.prng
                );
            #endif
        }};

        if (const auto partition {partition_row()}; partition) [[likely]] {
            const auto [oa, ob, n] {*partition};
            dispatch_quant(oa, ob, n, cmd);
        }

        if (1+pimpl->m_num_completed.fetch_add(1, std::memory_order::relaxed) == pimpl->m_workers.size()) { // Last worker
            std::unique_lock lock {pimpl->m_mtx};
            pimpl->m_cv.notify_all();
        }
    }

    context::pimpl::pimpl(std::size_t num_threads) {
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

    context::pimpl::~pimpl() {
        std::unique_lock lock {m_mtx};
        m_interrupt = true;
        ++m_phase;
        lock.unlock();
        m_cv.notify_all();
        m_workers.clear();
    }

    auto context::pimpl::kickoff_workers(quant_descriptor&& cmd) -> void {
        std::unique_lock lock {m_mtx};
        for (auto& worker : m_workers)
            worker.cmd = cmd;
        ++m_phase;
        m_num_completed.store(0, std::memory_order::relaxed);
        m_cv.notify_all();
    }

    auto context::pimpl::barrier() -> void {
        std::unique_lock lock {m_mtx};
        m_cv.wait(lock, [&]() noexcept -> bool { return m_num_completed.load(std::memory_order_relaxed) == m_workers.size(); });
    }

    auto context::pimpl::operator()(quant_descriptor&& cmd) -> void {
        kickoff_workers(std::forward<decltype(cmd)>(cmd));
        worker& w0 {m_workers[0]}; // Main thread does work too
        w0.exec_and_broadcast();
        barrier();
    }

    context::context(std::size_t num_threads) {
        m_pimpl = std::make_shared<pimpl>(num_threads);
    }

    context::~context() = default;

    auto context::quantize(
        const std::span<const std::byte> in,
        const dtype dtype_in,
        const std::span<std::byte> out,
        const dtype dtype_out,
        const float scale,
        const std::int32_t zero_point,
        const round_mode mode
    ) const -> void {
        piquant_assert(!dtype_info_of(dtype_in).is_quant, "input dtype must be a dequantized type");
        piquant_assert(dtype_info_of(dtype_out).is_quant, "output dtype must be a quantized type");
        if (dtype_info_of(dtype_out).bit_size < 8) { // Packed (sub 1 byte) types require a splitted numel of all pairs
            piquant_assert(out.size() == (in.size()+1)>>1, "output span requires (in.size() + 1) / 2 elements, as it is a packed datatype with sub-byte granularity, numel in: %zu, numel out: %zu", in.size(), out.size());
        } else {
            piquant_assert(in.size() == out.size(), "input and output spans must have the same length, but %zu != %zu", in.size(), out.size());
        }
        quant_descriptor info {
            .type = command_type::quant,
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()),
            .scale = scale,
            .zero_point = zero_point,
            .dt_in = dtype_in,
            .dt_out = dtype_out,
            .rnd_mode = mode
        };
        (*this->m_pimpl)(std::move(info));
    }

    auto context::dequantize(
        const std::span<const std::byte> in,
        const dtype dtype_in,
        const std::span<std::byte> out,
        const dtype dtype_out,
        const float scale,
        const std::int32_t zero_point,
        const reduce_op op
    ) const -> void {
        piquant_assert(dtype_info_of(dtype_in).is_quant, "input dtype must be a quantized type");
        piquant_assert(!dtype_info_of(dtype_out).is_quant, "output dtype must be a dequantized type");
        if (dtype_info_of(dtype_in).bit_size < 8) { // Packed (sub 1 byte) types require a splitted numel of all pairs
            piquant_assert(in.size() == (out.size()+1)>>1, "output span requires (out.size() + 1) / 2 elements, as it is a packed datatype with sub-byte granularity, numel in: %zu, numel out: %zu", in.size(), out.size());
        } else {
            piquant_assert(in.size() == out.size(), "input and output spans must have the same length, but %zu != %zu", in.size(), out.size());
        }
        quant_descriptor info {
            .type = command_type::dequant,
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()),
            .scale = scale,
            .zero_point = zero_point,
            .dt_in = dtype_in,
            .dt_out = dtype_out,
            .reduce = op
        };
        (*this->m_pimpl)(std::move(info));
    }

    auto context::quantize_dequantize_fused(
        const std::span<const std::byte> in,
        const dtype dtype_in_out,
        const std::span<std::byte> out,
        const dtype quant_type,
        const float scale,
        const std::int32_t zero_point,
        const round_mode mode,
        const reduce_op op
    ) const -> void {
        piquant_assert(!dtype_info_of(dtype_in_out).is_quant, "input dtype must be a dequantized type");
        piquant_assert(dtype_info_of(quant_type).is_quant, "quant dtype must be a quantized type");
        piquant_assert(in.size() == out.size(), "input and output spans must have the same length, but %zu != %zu", in.size(), out.size());
        quant_descriptor info {
            .type = command_type::quant_dequant,
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()),
            .scale = scale,
            .zero_point = zero_point,
            .dt_in = dtype_in_out,
            .dt_out = quant_type,
            .rnd_mode = mode,
            .reduce = op
        };
        (*this->m_pimpl)(std::move(info));
    }

    auto context::reseed_thread_local_rng(const std::uint32_t seed) const -> void {
        for (auto& worker : m_pimpl->m_workers)
            worker.pl.prng = prng_state{seed};
    }
}
