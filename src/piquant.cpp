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
#include <numbers>
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

        auto operator ()(const quant_descriptor& desc) const -> void;
        auto operator ()(std::span<const float32_t> x, dtype quant_dst_type) -> std::pair<float32_t, std::int64_t>;
        auto operator ()(std::span<const float64_t> x, dtype quant_dst_type) -> std::pair<float32_t, std::int64_t>;
        auto job_entry(partition& pl, const quant_descriptor& cmd) const -> void;
    };

    auto context::pimpl::job_entry(partition& pl, const quant_descriptor& cmd) const -> void {
        const std::int64_t tc {std::max(std::int64_t{1}, pl.tc)};
        const std::int64_t ti {pl.ti};
        const auto partition_row {[&] () noexcept -> std::optional<std::array<std::int64_t, 3>> {
            std::int64_t chunk_size {(cmd.numel + tc - 1)/tc};
            // If we have nibble input or output, we really don't want the chunk size to be and odd number of elements
            // because it would trigger trailing element handling in every thread. We want to avoid that to eliminate
            // the need for synchronization, as now two threads want to modify the same output byte to place their respective nibble-parts.
            // Hence, we round up to the next even number if we have a packed type.
            {
                bool packed_input {dtype_info_of(cmd.dt_in ).bit_size < 8};
                bool packed_output {dtype_info_of(cmd.dt_out).bit_size < 8};
                bool split_by_pairs {
                    (cmd.type == command_type::quant && packed_output) ||
                    (cmd.type == command_type::dequant && packed_input ) ||
                    (cmd.type == command_type::quant_dequant && packed_output)
                };
                if (split_by_pairs && chunk_size & 1) ++chunk_size;
            }
            std::int64_t ra {chunk_size*ti};
            std::int64_t rb {std::min(ra + chunk_size, cmd.numel)};
            if (ra >= rb) [[unlikely]] return {};
            return {{ra, ra, rb-ra}};
        }};
        const auto dispatch_quant {[&](const std::int64_t oa, const std::int64_t ob, const std::int64_t range, const quant_descriptor& cmd) noexcept -> void {
            auto* const kernel {&registry.quant_kernel};
            piquant_assert2(kernel != nullptr);
            const auto si {dtype_info_of(cmd.dt_in).bit_size};
            const auto so {cmd.type == command_type::quant_dequant ? si : dtype_info_of(cmd.dt_out).bit_size};
            (*kernel)(
                cmd.in + (si*oa) / 8,
                cmd.out + (so*ob) / 8,
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
        m_pool{static_cast<int>(num_threads), 64} {
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

    auto context::pimpl::operator()(const quant_descriptor& desc) const -> void {
        const size_t num_threads {this->num_threads};
        const pi::threadpool::MultiTaskResult jobs_future = m_pool.scheduleSequence<void>(0u, num_threads, [this, &desc, num_threads](const std::size_t ti) {
            partition pl {
                .ti = static_cast<std::int64_t>(ti),
                .tc = static_cast<std::int64_t>(num_threads)
            };
            job_entry(pl, desc);
        });
        jobs_future.join();
    }

    [[nodiscard]] static auto compute_type_max(dtype dt) noexcept -> std::uint64_t {
        dtype_info info {dtype_info_of(dt)};
        piquant_assert(info.flags & dtype_flags::is_quant && info.flags & dtype_flags::is_int, "type %s is not a quantization type", info.name.data());
        std::size_t width {dtype_info_of(dt).bit_size};
        piquant_assert(width > 0 && width <= 64, "invalid width %zu for type %s", width, info.name.data());
        if (info.flags & dtype_flags::is_signed) --width;
        if (width == 64) return std::numeric_limits<std::uint64_t>::max();
        return (1ull<<width) - 1;
    }

    template <typename T, typename F> requires std::is_floating_point_v<T>
    static auto compute_quant_config(
        pi::threadpool::ThreadPool& pool,
        F&& kernel,
        std::span<const T> x,
        dtype quant_dst_type
    ) -> std::pair<float32_t, std::int64_t> {
        const auto* base {x.data()};
        pi::threadpool::MultiTaskResult jobs_future = pool.scheduleBlocks<std::array<T, 2>>(0u, x.size(), [base, &kernel](std::size_t start, std::size_t end) -> std::array<T, 2> {
            std::size_t numel {end - start};
            if (numel <= 0) return {0.0, 0.0};
            std::span<const T> x {base + start, numel};
            return std::invoke(kernel, x);
        });
        jobs_future.join();
        float64_t r_min {std::numeric_limits<float64_t>::max()};
        float64_t r_max {std::numeric_limits<float64_t>::lowest()};
        for (std::size_t i {}; i < jobs_future.size(); ++i) {
            auto [min, max] {jobs_future.get(i)};
            r_min = std::min(r_min, static_cast<float64_t>(min));
            r_max = std::max(r_max, static_cast<float64_t>(max));
        }
        std::uint64_t type_max {compute_type_max(quant_dst_type)};
        std::int64_t type_min {0};
        if (dtype_info_of(quant_dst_type).flags & dtype_flags::is_signed)
            type_min = -static_cast<std::int64_t>(type_max) - 1;
        if (r_max == r_min) [[unlikely]] {
            const std::int64_t mid = (type_max + type_min) >> 1;
            return {1.0f, mid};
        }
        float64_t q_min {static_cast<float64_t>(type_min)};
        float64_t q_max {static_cast<float64_t>(type_max)};
        float64_t scale = (r_max - r_min)/(q_max - q_min);
        float64_t zero_point = q_min - r_min/scale;
        zero_point = std::max(std::min(static_cast<float64_t>(static_cast<std::int64_t>(std::round(zero_point))), q_max), q_min);
        return {scale, zero_point};
    }

    auto context::pimpl::operator()(std::span<const float32_t> x, dtype quant_dst_type) -> std::pair<float32_t, std::int64_t> {
        auto& kernel {(*registry.find_min_max_f32)};
        return compute_quant_config(m_pool, kernel, x, quant_dst_type);
    }

    auto context::pimpl::operator()(std::span<const float64_t> x, dtype quant_dst_type) -> std::pair<float32_t, std::int64_t> {
        auto& kernel {(*registry.find_min_max_f64)};
        return compute_quant_config(m_pool, kernel, x, quant_dst_type);
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
        const float32_t scale,
        const std::int64_t zero_point,
        const round_mode mode
    ) const -> void {
        const auto& dti {dtype_info_of(dtype_in)};
        const auto& dto {dtype_info_of(dtype_out)};
        piquant_assert(!(dti.flags & dtype_flags::is_quant), "input dtype must be a dequantized type");
        piquant_assert(dto.flags & dtype_flags::is_quant, "output dtype must be a quantized type");
        switch (dto.bit_size) {
            case 4: piquant_assert(out.size()/(dto.stride) == (in.size()/(dti.stride)+1)>>1, "output span requires (in.size() + 1) / 2 elements, as it is a packed datatype with sub-byte granularity, numel in: %zu, numel out: %zu", in.size(), out.size()); break;
            case 2: piquant_assert(out.size()/(dto.stride) == (in.size()/(dti.stride)+3)>>2, "output span requires (in.size() + 1) / 4 elements, as it is a packed datatype with sub-byte granularity, numel in: %zu, numel out: %zu", in.size(), out.size()); break;
            default:  piquant_assert(in.size()/dti.stride == out.size()/dto.stride, "input and output spans must have the same length, but %zu != %zu", in.size()/(dti.bit_size>>3), out.size()/(dto.bit_size>>3));
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
            .rounding = mode
        };
        (*this->m_pimpl)(info);
    }

    auto context::dequantize(
        const std::span<const std::byte> in,
        const dtype dtype_in,
        const std::span<std::byte> out,
        const dtype dtype_out,
        const float32_t scale,
        const std::int64_t zero_point,
        const reduce_op op
    ) const -> void {
        const auto& dti {dtype_info_of(dtype_in)};
        const auto& dto {dtype_info_of(dtype_out)};
        piquant_assert(dti.flags & dtype_flags::is_quant, "input dtype must be a quantized type");
        piquant_assert(!(dto.flags & dtype_flags::is_quant), "output dtype must be a dequantized type");
        switch (dti.bit_size) {
            case 4: piquant_assert(in.size()/dti.stride == (out.size()/(dto.stride)+1)>>1, "output span requires (out.size() + 1) / 2 elements, as it is a packed datatype with sub-byte granularity, numel in: %zu, numel out: %zu", in.size(), out.size()); break;
            case 2: piquant_assert(in.size()/dti.stride == (out.size()/(dto.stride)+3)>>2, "output span requires (out.size() + 3) / 4 elements, as it is a packed datatype with sub-byte granularity, numel in: %zu, numel out: %zu", in.size(), out.size()); break;
            default: piquant_assert(in.size()/dti.stride == out.size()/dto.stride, "input and output spans must have the same length, but %zu != %zu", in.size()/(dti.bit_size>>3), out.size()/(dto.bit_size>>3)); break;
        }
        quant_descriptor info {
            .type = command_type::dequant,
            .in = in.data(),
            .out = out.data(),
            .numel = static_cast<std::int64_t>(out.size()/(dto.stride)),
            .scale = scale,
            .zero_point = zero_point,
            .dt_in = dtype_in,
            .dt_out = dtype_out,
            .reducing = op
        };
        (*this->m_pimpl)(info);
    }

    auto context::quantize_dequantize_fused(
        const std::span<const std::byte> in,
        const dtype dtype_in_out,
        const std::span<std::byte> out,
        const dtype quant_type,
        const float32_t scale,
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
            .rounding = mode,
            .reducing = op
        };
        (*this->m_pimpl)(info);
    }

    auto context::compute_quant_config_from_data(std::span<const float32_t> x, dtype quant_dst_dtype) const -> std::pair<float32_t, std::int64_t> {
        auto result {(*this->m_pimpl)(x, quant_dst_dtype)};
        piquant_assert(!std::isnan(result.first) && result.first >= 0.0f, "scale must be positive");
        return result;
    }

    auto context::compute_quant_config_from_data(std::span<const float64_t> x, dtype quant_dst_dtype) const -> std::pair<float32_t, std::int64_t> {
        auto result {(*this->m_pimpl)(x, quant_dst_dtype)};
        piquant_assert(result.first >= 0.0f, "scale must be positive");
        return result;
    }
}
