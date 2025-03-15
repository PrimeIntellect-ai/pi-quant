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
        extern auto impl( \
        const void* __restrict__ x, \
        void* __restrict__ o, \
        std::int64_t numel, \
        float scale, \
        std::int32_t zero_point, \
        const bool sto_rnd, \
        prng_state& prng \
    ) noexcept -> void

    #define decl_dequant_kernel_fn(impl) \
        extern auto impl( \
            const void* __restrict__ x, \
            void* __restrict__ o, \
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
        const void* __restrict__ x,
        void* __restrict__ o,
        std::int64_t numel,
        float scale,
        std::int32_t zero_point,
        bool sto_rnd,
        prng_state& prng
    ) -> void;

    using dequant_kernel = auto (
        const void* __restrict__ x,
        void* __restrict__ o,
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

        static constexpr auto k_num_cpu_levels {static_cast<std::size_t>(amd64_cpu_caps::num_)};
        static constexpr auto k_num_dtypes {static_cast<std::size_t>(dtype::num_)};

        static constexpr std::array quant_routines = {
            std::array<quant_kernel*, k_num_dtypes> {
                nullptr,                    // f32
                &f32_quant8_generic,        // quint8
                &f32_quant4_generic         // quint4
            },
            std::array<quant_kernel*, k_num_dtypes> {
                nullptr,                    // f32
                &f32_quant8_amd64_sse42,    // quint8
                &f32_quant4_amd64_sse42     // quint4
            },
            std::array<quant_kernel*, k_num_dtypes> {
                nullptr,                    // f32
                &f32_quant8_amd64_avx2,     // quint8
                &f32_quant4_amd64_avx2      // quint4
            },
            std::array<quant_kernel*, k_num_dtypes> {
                nullptr,                    // f32
                &f32_quant8_amd64_avx512f,  // quint8
                &f32_quant4_amd64_avx512f   // quint4
            },
        };

        static constexpr std::array dequant_routines = {
            std::array<dequant_kernel*, k_num_dtypes> {
                nullptr,                        // f32
                &f32_dequant8_generic,          // quint8
                &f32_dequant4_generic           // quint4
            },
            std::array<dequant_kernel*, k_num_dtypes> {
                nullptr,                        // f32
                &f32_dequant8_amd64_sse42,      // quint8
                &f32_dequant4_amd64_sse42       // quint4
            },
            std::array<dequant_kernel*, k_num_dtypes> {
                nullptr,                        // f32
                &f32_dequant8_amd64_avx2,       // quint8
                &f32_dequant4_amd64_avx2        // quint4
            },
            std::array<dequant_kernel*, k_num_dtypes> {
                nullptr,                        // f32
                &f32_dequant8_amd64_avx512f,    // quint8
                &f32_dequant4_amd64_avx512f     // quint4
            },
        };

    #endif

    #undef decl_kernel_pair

    template <class... T>
    struct overloads final : T... { using T::operator()...; };

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

    static constexpr std::size_t cache_line {
        #ifdef __cpp_lib_hardware_interference_size
            std::hardware_destructive_interference_size
        #else
            64
        #endif
    };

    struct quant_descriptor final {
        const std::byte* in {};
        std::byte* out {};
        std::int64_t numel {};
        float scale {};
        std::int32_t zero_point {};
        dtype dt_in {};
        dtype dt_out {};
        round_mode rnd_mode {};
    };

    struct dequant_descriptor final {
        const std::byte* in {};
        std::byte* out {};
        std::int64_t numel {};
        float scale {};
        std::int32_t zero_point {};
        dtype dt_in {};
        dtype dt_out {};
        reduce_op op {};
    };

    struct quant_depiquant_descriptor final {
        quant_descriptor quant {};
        dequant_descriptor dequant {};
    };

    using quant_command = std::variant<std::monostate, quant_descriptor, dequant_descriptor, quant_depiquant_descriptor>;

    struct payload {
        prng_state prng;
        std::int64_t ti {}; // thread index
        std::int64_t tc {}; // thread count
        std::uint64_t phase {};

        explicit constexpr payload(const std::uint32_t seed) noexcept : prng{seed} {}
    };

    struct worker;

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

        auto kickoff_workers(quant_command&& cmd) -> void;
        auto barrier() -> void;
        auto operator()(quant_command&& cmd) -> void;
    };

    struct worker final {
        context::pimpl* pimpl;
        alignas(cache_line) payload pl;
        alignas(cache_line) quant_command cmd {};
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

        [[nodiscard]] auto await_work() -> bool {
            std::unique_lock lock {pimpl->m_mtx};
            pimpl->m_cv.wait(lock, [this]() noexcept -> bool { return pimpl->m_interrupt || pimpl->m_phase > pl.phase; });
            if (pimpl->m_interrupt) [[unlikely]] return false;
            pl.phase = pimpl->m_phase;
            return true;
        }

        auto entry() -> void {
            pimpl->m_workers_online.fetch_add(1, std::memory_order_seq_cst);
            while (await_work()) [[likely]]
                exec_and_broadcast();
        }

        auto exec_and_broadcast() -> void {
            std::int64_t numel {};
            dtype dt_in {}, dt_out {};
            bool is_dequant {};
            const auto visitor = overloads {
                [](std::monostate) -> void {},
                [&](const quant_descriptor& desc) -> void {
                    numel = desc.numel;
                    dt_in = desc.dt_in;
                    dt_out = desc.dt_out;
                    is_dequant = false;
                },
                [&](const dequant_descriptor& desc) -> void {
                    numel = desc.numel;
                    dt_in = desc.dt_in;
                    dt_out = desc.dt_out;
                    is_dequant = true;
                },
                [&](const quant_depiquant_descriptor& desc) -> void {

                }
            };
            std::visit(visitor, cmd);

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
                    const auto level {static_cast<std::size_t>(pimpl->cpu_caps)};
                    const auto fmt {static_cast<std::size_t>(dt_out)};
                    const std::size_t stride_in {dtype_info_of(dt_in).stride};
                    const std::size_t stride_out {dtype_info_of(dt_out).stride};
                    piquant_assert2(level < quant_routines.size());
                    piquant_assert2(fmt < quant_routines[0].size());
                    auto* const kernel {quant_routines[level][fmt]};
                    piquant_assert2(kernel != nullptr);
                    (*kernel)(
                        reinterpret_cast<const std::byte*>(cmd.in) + oa*stride_in,
                        reinterpret_cast<std::byte*>(cmd.out)+ ob*stride_out,
                        n,
                        cmd.scale,
                        cmd.zero_point,
                        cmd.rnd_mode == round_mode::stochastic,
                        pl.prng
                    );
                #else
                    auto* const kernel {is_i8 ? &f32_quant8_generic : &f32_quant4_generic};
                    piquant_assert2(kernel != nullptr);
                    (*kernel)(cmd.in+oa, cmd.out+ob, n, cmd.scale, cmd.zero_point, cmd.rnd_mode == round_mode::stochastic, pl.prng);
                #endif
            }};

            const auto dispatch_dequant {[=, this](const std::int64_t oa, const std::int64_t ob, const std::int64_t n, const dequant_descriptor& cmd) noexcept -> void {
                #ifdef __x86_64__
                    const auto level {static_cast<std::size_t>(pimpl->cpu_caps)};
                    const auto fmt {static_cast<std::size_t>(dt_in)};
                    const std::size_t stride_in {dtype_info_of(dt_in).stride};
                    const std::size_t stride_out {dtype_info_of(dt_out).stride};
                    piquant_assert2(level < dequant_routines.size());
                    piquant_assert2(fmt < dequant_routines[0].size());
                    auto* const kernel {dequant_routines[level][fmt]};
                    piquant_assert2(kernel != nullptr);
                    (*kernel)(
                        reinterpret_cast<const std::byte*>(cmd.in) + oa*stride_in,
                        reinterpret_cast<std::byte*>(cmd.out)+ ob*stride_out,
                        n,
                        cmd.scale,
                        cmd.zero_point,
                        cmd.op
                    );
                #else
                    auto* const kernel {is_i8 ? &f32_dequant8_generic : &f32_dequant4_generic};
                    piquant_assert2(kernel != nullptr);
                    (*kernel)(cmd.in+oa, cmd.out+ob, n, cmd.scale, cmd.zero_point, cmd.op);
                #endif
            }};

            if (const auto partition {partition_row(dt_out == dtype::uint8)}; partition) [[likely]] {
                const auto [oa, ob, n] {*partition};
                if (is_dequant) dispatch_dequant(oa, ob, n, std::get<dequant_descriptor>(cmd));
                else dispatch_quant(oa, ob, n, std::get<quant_descriptor>(cmd));
            }

            if (1+pimpl->m_num_completed.fetch_add(1, std::memory_order::relaxed) == pimpl->m_workers.size()) { // Last worker
                std::unique_lock lock {pimpl->m_mtx};
                pimpl->m_cv.notify_all();
            }
        }
    };

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

    auto context::pimpl::kickoff_workers(quant_command&& cmd) -> void {
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

    auto context::pimpl::operator()(quant_command&& cmd) -> void {
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
        piquant_assert(dtype_info_of(dtype_in).is_dequant, "input dtype must be a dequantized type");
        piquant_assert(dtype_info_of(dtype_out).is_quant, "output dtype must be a quantized type");
        if (dtype_info_of(dtype_out).bit_size < 8) { // Packed (sub 1 byte) types require a splitted numel of all pairs
            piquant_assert(out.size() == in.size()+1>>1, "output span requires (in.size() + 1) / 2 elements, as it is a packed datatype with sub-byte granularity, numel in: %zu, numel out: %zu", in.size(), out.size());
        } else {
            piquant_assert(in.size() == out.size(), "input and output spans must have the same length, but %zu != %zu", in.size(), out.size());
        }
        quant_descriptor info {
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()),
            .scale = scale,
            .zero_point = zero_point,
            .dt_in = dtype_in,
            .dt_out = dtype_out,
            .rnd_mode = mode,
        };
        (*this->m_pimpl)(info);
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
        piquant_assert(dtype_info_of(dtype_out).is_dequant, "output dtype must be a dequantized type");
        piquant_assert(in.size() == out.size(), "input and output spans must have the same length, but %zu != %zu", in.size(), out.size());
        dequant_descriptor info {
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()),
            .scale = scale,
            .zero_point = zero_point,
            .dt_in = dtype_in,
            .dt_out = dtype_out,
            .op = op
        };
        (*this->m_pimpl)(info);
    }
}
