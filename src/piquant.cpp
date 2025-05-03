#include "piquant.hpp"
#include "piquant_internal.hpp"

#include <cassert>
#include <cstdarg>
#include <thread>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <condition_variable>

#include <pithreadpool/threadpool.hpp>

namespace piquant {
    #define decl_quant_kernel_installer_fn(impl) \
        [[nodiscard]] extern auto impl() noexcept -> kernel_registry

    decl_quant_kernel_installer_fn(install_quant_generic);

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

        decl_quant_kernel_installer_fn(install_quant_amd64_sse42);
        decl_quant_kernel_installer_fn(install_quant_amd64_avx2);
        decl_quant_kernel_installer_fn(install_quant_amd64_avx512f);

    #endif

    #undef decl_kernel_pair

    template <class... T>
    struct overloads final : T... { using T::operator()...; };

    auto panic(const char* msg, ...) -> void {
        std::va_list args;
        va_start(args, msg);
        std::array<char, 8192> tmp {};
        int delta{std::snprintf(tmp.data(), sizeof(tmp), "%s", "\x1b[31m")};
        delta += std::vsnprintf(tmp.data()+delta, sizeof(tmp)-delta, msg, args);
        std::snprintf(tmp.data()+delta, sizeof(tmp)-delta, "%s", "\x1b[0m");
        std::cerr << tmp.data() << std::endl;
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

    struct partition {
        std::int64_t ti {}; // thread index
        std::int64_t tc {}; // thread count
    };

    class context::pimpl final {
    public:
        explicit pimpl(std::size_t num_threads);
        pimpl(const pimpl&) = delete;
        pimpl(pimpl&&) = delete;
        auto operator = (const pimpl&) -> pimpl& = delete;
        auto operator = (pimpl&&) -> pimpl& = delete;
        ~pimpl();

        kernel_registry registry {};
        std::size_t num_threads;
        pi::threadpool::ThreadPool m_pool;

        auto operator ()(const quant_descriptor& desc) -> void;
        auto operator ()(std::span<const float> x, std::int64_t type_max) -> std::pair<float, std::int64_t>;
        auto operator ()(std::span<const double> x, std::int64_t type_max) -> std::pair<float, std::int64_t>;
        auto job_entry(partition& pl, const quant_descriptor& cmd) const -> void;
    };

    auto context::pimpl::job_entry(partition& pl, const quant_descriptor& cmd) const -> void {
        const std::int64_t tc {std::max(std::int64_t{1}, pl.tc)};
        const std::int64_t ti {pl.ti};
        const auto partition_row {[&] () noexcept -> std::optional<std::array<std::int64_t, 3>> {
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
        const auto dispatch_quant {[&](const std::int64_t oa, const std::int64_t ob, const std::int64_t range, const context::quant_descriptor& cmd) noexcept -> void {
            auto* const kernel {&registry.quant_kernel};
            piquant_assert2(kernel != nullptr);
            const auto si {dtype_info_of(cmd.dt_in).stride};
            const auto so {cmd.type == context::command_type::quant_dequant ? si : dtype_info_of(cmd.dt_out).stride};
            (*kernel)(
                cmd.in + si*oa,
                cmd.out + so*ob,
                range,
                cmd
            );
        }};

        if (const auto partition {partition_row()}; partition) [[likely]] {
            const auto [oa, ob, n] {*partition};
            dispatch_quant(oa, ob, n, cmd);
        }
    }

    context::pimpl::pimpl(const std::size_t num_threads) : num_threads(num_threads),
        m_pool{1, 64} { // TODO: use num_threads
        registry = install_quant_generic();
        m_pool.startup();
        #ifdef __x86_64__
            if (check_avx512f_support()) registry = install_quant_amd64_avx512f();
            else if (check_avx2_support()) registry = install_quant_amd64_avx2();
            else if (check_sse42_support())  registry = install_quant_amd64_sse42();

        #endif
    }

    context::pimpl::~pimpl() {
        m_pool.shutdown();
    }

    auto context::pimpl::operator()(const quant_descriptor& desc) -> void {
        const size_t num_threads {this->num_threads};
        pi::threadpool::MultiTaskResult jobs_future = m_pool.scheduleSequence<void>(0u, num_threads, [this, &desc, num_threads](const std::size_t ti) {
            partition pl {
                .ti = static_cast<std::int64_t>(ti),
                .tc = static_cast<std::int64_t>(num_threads)
            };
            job_entry(pl, desc);
        });
        jobs_future.join();
    }

    template <typename T, typename F> requires std::is_floating_point_v<T>
    static auto compute_quant_config(
        pi::threadpool::ThreadPool& pool,
        F&& kernel,
        std::span<const T> x,
        std::int64_t type_max
    ) -> std::pair<float, std::int64_t> {
        const auto* base {x.data()};
        pi::threadpool::MultiTaskResult jobs_future = pool.scheduleBlocks<std::array<T, 2>>(0u, x.size(), [base, &kernel](std::size_t start, std::size_t end) -> std::array<T, 2> {
            std::size_t numel {end - start};
            if (numel <= 0) return {0.0, 0.0};
            std::span<const T> x {base + start, numel};
            return std::invoke(kernel, x);
        });
        jobs_future.join();
        double sum {};
        double sum_sq {};
        for (size_t i = 0; i < jobs_future.size(); ++i) {
            auto [s, ss] {jobs_future.get(i)};
            sum += static_cast<double>(s);
            sum_sq += static_cast<double>(ss);
        }
        double fnumel {static_cast<double>(x.size())};
        double mean {sum / fnumel};
        double variance {(sum_sq - sum*sum / fnumel) / (fnumel-1.0)};
        double stddev {std::sqrt(variance)};
        double scale {static_cast<float>(stddev_scale*stddev / static_cast<float>(type_max))};
        if (scale == 0.0) [[unlikely]] {
            return {1.0f, (type_max+1)>>1};
        }
        std::int64_t zp {((type_max+1)>>1) - static_cast<std::int64_t>(std::round(mean / scale))};
        return {scale, zp};
    }

    auto context::pimpl::operator()(std::span<const float> x, std::int64_t type_max) -> std::pair<float, std::int64_t> {
        auto& kernel {(*registry.quant_config_kernel_f32)};
        return compute_quant_config(m_pool, kernel, x, type_max);
    }

    auto context::pimpl::operator()(std::span<const double> x, std::int64_t type_max) -> std::pair<float, std::int64_t> {
        auto& kernel {(*registry.quant_config_kernel_f64)};
        return compute_quant_config(m_pool, kernel, x, type_max);
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
        const std::int64_t zero_point,
        const round_mode mode
    ) const -> void {
        const auto& dti {dtype_info_of(dtype_in)};
        const auto& dto {dtype_info_of(dtype_out)};
        piquant_assert(!(dti.flags & dtype_flags::is_quant), "input dtype must be a dequantized type");
        piquant_assert(dto.flags & dtype_flags::is_quant, "output dtype must be a quantized type");
        if (dto.bit_size < 8) { // Packed (sub 1 byte) types require a splitted numel of all pairs
            piquant_assert(out.size()/(dto.stride) == (in.size()/(dti.stride)+1)>>1, "output span requires (in.size() + 1) / 2 elements, as it is a packed datatype with sub-byte granularity, numel in: %zu, numel out: %zu", in.size(), out.size());
        } else {
            piquant_assert(in.size()/dti.stride == out.size()/dto.stride, "input and output spans must have the same length, but %zu != %zu", in.size()/(dti.bit_size>>3), out.size()/(dto.bit_size>>3));
        }
        quant_descriptor info {
            .type = command_type::quant,
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()/dti.stride),
            .scale = scale,
            .zero_point = zero_point,
            .dt_in = dtype_in,
            .dt_out = dtype_out,
            .rnd_mode = mode
        };
        (*this->m_pimpl)(info);
    }

    auto context::dequantize(
        const std::span<const std::byte> in,
        const dtype dtype_in,
        const std::span<std::byte> out,
        const dtype dtype_out,
        const float scale,
        const std::int64_t zero_point,
        const reduce_op op
    ) const -> void {
        const auto& dti {dtype_info_of(dtype_in)};
        const auto& dto {dtype_info_of(dtype_out)};
        piquant_assert(dti.flags & dtype_flags::is_quant, "input dtype must be a quantized type");
        piquant_assert(!(dto.flags & dtype_flags::is_quant), "output dtype must be a dequantized type");
        if (dti.bit_size < 8) { // Packed (sub 1 byte) types require a splitted numel of all pairs
            piquant_assert(in.size()/dti.stride == (out.size()/(dto.stride)+1)>>1, "output span requires (out.size() + 1) / 2 elements, as it is a packed datatype with sub-byte granularity, numel in: %zu, numel out: %zu", in.size(), out.size());
        } else {
            piquant_assert(in.size()/dti.stride == out.size()/dto.stride, "input and output spans must have the same length, but %zu != %zu", in.size()/(dti.bit_size>>3), out.size()/(dto.bit_size>>3));
        }
        quant_descriptor info {
            .type = command_type::dequant,
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()/(dti.stride)),
            .scale = scale,
            .zero_point = zero_point,
            .dt_in = dtype_in,
            .dt_out = dtype_out,
            .reduce = op
        };
        (*this->m_pimpl)(info);
    }

    auto context::quantize_dequantize_fused(
        const std::span<const std::byte> in,
        const dtype dtype_in_out,
        const std::span<std::byte> out,
        const dtype quant_type,
        const float scale,
        const std::int64_t zero_point,
        const round_mode mode,
        const reduce_op op
    ) const -> void {
        const auto& dti{dtype_info_of(dtype_in_out)};
        piquant_assert(!(dti.flags & dtype_flags::is_quant), "input dtype must be a dequantized type");
        piquant_assert(dtype_info_of(quant_type).flags & dtype_flags::is_quant, "quant dtype must be a quantized type");
        piquant_assert(in.size() == out.size(), "input and output spans must have the same length, but %zu != %zu", in.size(), out.size());
        quant_descriptor info {
            .type = command_type::quant_dequant,
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(in.size()/dti.stride),
            .scale = scale,
            .zero_point = zero_point,
            .dt_in = dtype_in_out,
            .dt_out = quant_type,
            .rnd_mode = mode,
            .reduce = op
        };
        (*this->m_pimpl)(info);
    }

    [[nodiscard]] static auto compute_type_max(dtype dt) noexcept -> std::int64_t {
        std::size_t width {dtype_info_of(dt).bit_size};
        piquant_assert(width > 0 && width <= 64, "invalid width %zu for type %s", width, dtype_info_of(dt).name.data());
        std::uint64_t max {width == 64 ? std::numeric_limits<std::uint64_t>::max() : std::uint64_t{1} << width};
        --max;
        if (dt == dtype::uint64 || dtype_info_of(dt).flags & dtype_flags::is_signed) max >>= 1;
        return static_cast<std::int64_t>(max);
    }

    auto context::compute_quant_config_from_data(std::span<const float> x, dtype quant_dst_dtype) const -> std::pair<float, std::int64_t> {
        auto result {(*this->m_pimpl)(x, compute_type_max(quant_dst_dtype))};
        piquant_assert(!std::isnan(result.first) && result.first >= 0.0f, "scale must be positive");
        return result;
    }

    auto context::compute_quant_config_from_data(std::span<const double> x, dtype quant_dst_dtype) const -> std::pair<float, std::int64_t> {
        auto result {(*this->m_pimpl)(x, compute_type_max(quant_dst_dtype))};
        piquant_assert(result.first >= 0.0f, "scale must be positive");
        return result;
    }
}
