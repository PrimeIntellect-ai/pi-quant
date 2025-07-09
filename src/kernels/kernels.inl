// This inline file is directly included into the kernels.inl file, which is cloned (recompiled) in multiple compilation units for different CPU architectures.
// ! Make sure all functions are static, to make them local to the compilation unit.

#ifndef QUANT_KERNEL_IMPL
#error "Kernel impl is not defined"
#endif

#include <piquant.hpp>
#include "../piquant_internal.hpp"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <numeric>
#include <sstream>

namespace piquant {
    struct kernel_registry;
}

#define concat(a, b) a ## b
#define impl_namespace(a, b) piquant::concat(a, _impl)

namespace impl_namespace(QUANT_KERNEL_IMPL, _) {

    // Include order matters, implementations are cloned per specialized compilation unit
    #include "prng.inl"
    #include "kernels_specialized.inl"
    #include "quantize.inl"
    #include "dequantize.inl"

    template <typename T> requires std::is_floating_point_v<T>
    [[nodiscard]] auto compute_quant_config_from_data(std::span<const T> in) -> std::array<T, 2> {
        if (in.empty()) [[unlikely]] return {0.0, 0.0};
        T sum {};
        T sum_sq {};
        for (T v : in) {
            sum += v;
            sum_sq += v*v;
        }
        return {sum, sum_sq};
    }

    template <>
    [[nodiscard]] auto compute_quant_config_from_data(std::span<const float> in) -> std::array<float, 2> {
        if (in.empty()) [[unlikely]] return {0.0f, 0.0};
        const auto* p {in.data()};
        float sum {};
        float sum_sq {};
        std::int64_t i {};
        #if defined(__AVX512F__) && defined(__AVX512BW__) && 0

        #elif defined(__AVX2__)
            static constexpr auto hsum {[](__m256 x) noexcept -> float {
                __m128 hiq {_mm256_extractf128_ps(x, 1)};
                __m128 loq {_mm256_castps256_ps128(x)};
                __m128 suq {_mm_add_ps(loq, hiq)};
                __m128 hid {_mm_movehl_ps(suq, suq)};
                __m128 sud {_mm_add_ps(suq, hid)};
                __m128 hi {_mm_shuffle_ps(sud, sud, 0x1)};git
                return _mm_cvtss_f32(_mm_add_ss(sud, hi));
            }};
            __m256 vsum1 {_mm256_setzero_ps()};
            __m256 vsum2 {_mm256_setzero_ps()};
            __m256 vsum3 {_mm256_setzero_ps()};
            __m256 vsum4 {_mm256_setzero_ps()};
            __m256 vsum5 {_mm256_setzero_ps()};
            __m256 vsum6 {_mm256_setzero_ps()};
            __m256 vsum7 {_mm256_setzero_ps()};
            __m256 vsum8 {_mm256_setzero_ps()};
            __m256 vsum_sq1 {_mm256_setzero_ps()};
            __m256 vsum_sq2 {_mm256_setzero_ps()};
            __m256 vsum_sq3 {_mm256_setzero_ps()};
            __m256 vsum_sq4 {_mm256_setzero_ps()};
            __m256 vsum_sq5 {_mm256_setzero_ps()};
            __m256 vsum_sq6 {_mm256_setzero_ps()};
            __m256 vsum_sq7 {_mm256_setzero_ps()};
            __m256 vsum_sq8 {_mm256_setzero_ps()};
            for (; i+63 < x.size(); i += 64) {
                __m256 v1 {_mm256_loadu_ps(p+i+8*0)};
                __m256 v2 {_mm256_loadu_ps(p+i+8*1)};
                __m256 v3 {_mm256_loadu_ps(p+i+8*2)};
                __m256 v4 {_mm256_loadu_ps(p+i+8*3)};
                __m256 v5 {_mm256_loadu_ps(p+i+8*4)};
                __m256 v6 {_mm256_loadu_ps(p+i+8*5)};
                __m256 v7 {_mm256_loadu_ps(p+i+8*6)};
                __m256 v8 {_mm256_loadu_ps(p+i+8*7)};
                vsum1 = _mm256_add_ps(vsum1, v1);
                vsum2 = _mm256_add_ps(vsum2, v2);
                vsum3 = _mm256_add_ps(vsum3, v3);
                vsum4 = _mm256_add_ps(vsum4, v4);
                vsum5 = _mm256_add_ps(vsum5, v5);
                vsum6 = _mm256_add_ps(vsum6, v6);
                vsum7 = _mm256_add_ps(vsum7, v7);
                vsum8 = _mm256_add_ps(vsum8, v8);
                vsum_sq1 = _mm256_fmadd_ps(v1, v1, vsum_sq1);
                vsum_sq2 = _mm256_fmadd_ps(v2, v2, vsum_sq2);
                vsum_sq3 = _mm256_fmadd_ps(v3, v3, vsum_sq3);
                vsum_sq4 = _mm256_fmadd_ps(v4, v4, vsum_sq4);
                vsum_sq5 = _mm256_fmadd_ps(v5, v5, vsum_sq5);
                vsum_sq6 = _mm256_fmadd_ps(v6, v6, vsum_sq6);
                vsum_sq7 = _mm256_fmadd_ps(v7, v7, vsum_sq7);
                vsum_sq8 = _mm256_fmadd_ps(v8, v8, vsum_sq8);
            }
            __m256 vsum_total {_mm256_add_ps(vsum1, _mm256_add_ps(vsum2, _mm256_add_ps(vsum3, _mm256_add_ps(vsum4, _mm256_add_ps(vsum5, _mm256_add_ps(vsum6, _mm256_add_ps(vsum7, vsum8)))))))};
            __m256 vsum_sq_total {_mm256_add_ps(vsum_sq1, _mm256_add_ps(vsum_sq2, _mm256_add_ps(vsum_sq3, _mm256_add_ps(vsum_sq4, _mm256_add_ps(vsum_sq5, _mm256_add_ps(vsum_sq6, _mm256_add_ps(vsum_sq7, vsum_sq8)))))))};
            sum = hsum(vsum_total);
            sum_sq = hsum(vsum_sq_total);
        #elif defined(__SSE4_2__)
            __m128 vsum1 {_mm_setzero_ps()};
            __m128 vsum2 {_mm_setzero_ps()};
            __m128 vsum3 {_mm_setzero_ps()};
            __m128 vsum4 {_mm_setzero_ps()};
            __m128 vsum5 {_mm_setzero_ps()};
            __m128 vsum6 {_mm_setzero_ps()};
            __m128 vsum7 {_mm_setzero_ps()};
            __m128 vsum8 {_mm_setzero_ps()};
            __m128 vsum_sq1 {_mm_setzero_ps()};
            __m128 vsum_sq2 {_mm_setzero_ps()};
            __m128 vsum_sq3 {_mm_setzero_ps()};
            __m128 vsum_sq4 {_mm_setzero_ps()};
            __m128 vsum_sq5 {_mm_setzero_ps()};
            __m128 vsum_sq6 {_mm_setzero_ps()};
            __m128 vsum_sq7 {_mm_setzero_ps()};
            __m128 vsum_sq8 {_mm_setzero_ps()};
            for (; i+31 < x.size(); i += 32) {
                __m128 v1 {_mm_loadu_ps(p+i+4*0)};
                __m128 v2 {_mm_loadu_ps(p+i+4*1)};
                __m128 v3 {_mm_loadu_ps(p+i+4*2)};
                __m128 v4 {_mm_loadu_ps(p+i+4*3)};
                __m128 v5 {_mm_loadu_ps(p+i+4*4)};
                __m128 v6 {_mm_loadu_ps(p+i+4*5)};
                __m128 v7 {_mm_loadu_ps(p+i+4*6)};
                __m128 v8 {_mm_loadu_ps(p+i+4*7)};
                vsum1 = _mm_add_ps(vsum1, v1);
                vsum2 = _mm_add_ps(vsum2, v2);
                vsum3 = _mm_add_ps(vsum3, v3);
                vsum4 = _mm_add_ps(vsum4, v4);
                vsum5 = _mm_add_ps(vsum5, v5);
                vsum6 = _mm_add_ps(vsum6, v6);
                vsum7 = _mm_add_ps(vsum7, v7);
                vsum8 = _mm_add_ps(vsum8, v8);
                vsum_sq1 = _mm_add_ps(vsum_sq1, _mm_mul_ps(v1, v1));
                vsum_sq2 = _mm_add_ps(vsum_sq2, _mm_mul_ps(v2, v2));
                vsum_sq3 = _mm_add_ps(vsum_sq3, _mm_mul_ps(v3, v3));
                vsum_sq4 = _mm_add_ps(vsum_sq4, _mm_mul_ps(v4, v4));
                vsum_sq5 = _mm_add_ps(vsum_sq5, _mm_mul_ps(v5, v5));
                vsum_sq6 = _mm_add_ps(vsum_sq6, _mm_mul_ps(v6, v6));
                vsum_sq7 = _mm_add_ps(vsum_sq7, _mm_mul_ps(v7, v7));
                vsum_sq8 = _mm_add_ps(vsum_sq8, _mm_mul_ps(v8, v8));
            }
            __m128 vsum_total {_mm_add_ps(vsum1, _mm_add_ps(vsum2, _mm_add_ps(vsum3, _mm_add_ps(vsum4, _mm_add_ps(vsum5, _mm_add_ps(vsum6, _mm_add_ps(vsum7, vsum8)))))))};
            __m128 vsum_sq_total {_mm_add_ps(vsum_sq1, _mm_add_ps(vsum_sq2, _mm_add_ps(vsum_sq3, _mm_add_ps(vsum_sq4, _mm_add_ps(vsum_sq5, _mm_add_ps(vsum_sq6, _mm_add_ps(vsum_sq7, vsum_sq8)))))))};
            sum = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(vsum_total, vsum_total), vsum_total));
            sum_sq = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(vsum_sq_total, vsum_sq_total), vsum_sq_total));
        #elif defined(__aarch64__) && defined(__ARM_NEON__)
            float32x4_t vsum1 {vdupq_n_f32(0.0f)};
            float32x4_t vsum2 {vdupq_n_f32(0.0f)};
            float32x4_t vsum3 {vdupq_n_f32(0.0f)};
            float32x4_t vsum4 {vdupq_n_f32(0.0f)};
            float32x4_t vsum5 {vdupq_n_f32(0.0f)};
            float32x4_t vsum6 {vdupq_n_f32(0.0f)};
            float32x4_t vsum7 {vdupq_n_f32(0.0f)};
            float32x4_t vsum8 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq1 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq2 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq3 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq4 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq5 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq6 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq7 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq8 {vdupq_n_f32(0.0f)};
            for (; i+31 < in.size(); i += 32) {
                float32x4_t v1 {vld1q_f32(p+i+(0<<2))};
                float32x4_t v2 {vld1q_f32(p+i+(1<<2))};
                float32x4_t v3 {vld1q_f32(p+i+(2<<2))};
                float32x4_t v4 {vld1q_f32(p+i+(3<<2))};
                float32x4_t v5 {vld1q_f32(p+i+(4<<2))};
                float32x4_t v6 {vld1q_f32(p+i+(5<<2))};
                float32x4_t v7 {vld1q_f32(p+i+(6<<2))};
                float32x4_t v8 {vld1q_f32(p+i+(7<<2))};
                vsum1 = vaddq_f32(vsum1, v1);
                vsum2 = vaddq_f32(vsum2, v2);
                vsum3 = vaddq_f32(vsum3, v3);
                vsum4 = vaddq_f32(vsum4, v4);
                vsum5 = vaddq_f32(vsum5, v5);
                vsum6 = vaddq_f32(vsum6, v6);
                vsum7 = vaddq_f32(vsum7, v7);
                vsum8 = vaddq_f32(vsum8, v8);
                vsum_sq1 = vmlaq_f32(vsum_sq1, v1, v1);
                vsum_sq2 = vmlaq_f32(vsum_sq2, v2, v2);
                vsum_sq3 = vmlaq_f32(vsum_sq3, v3, v3);
                vsum_sq4 = vmlaq_f32(vsum_sq4, v4, v4);
                vsum_sq5 = vmlaq_f32(vsum_sq5, v5, v5);
                vsum_sq6 = vmlaq_f32(vsum_sq6, v6, v6);
                vsum_sq7 = vmlaq_f32(vsum_sq7, v7, v7);
                vsum_sq8 = vmlaq_f32(vsum_sq8, v8, v8);
            }
            float32x4_t vsum_total {vaddq_f32(vsum1, vaddq_f32(vsum2, vaddq_f32(vsum3, vaddq_f32(vsum4, vaddq_f32(vsum5, vaddq_f32(vsum6, vaddq_f32(vsum7, vsum8)))))))};
            float32x4_t vsum_sq_total {vaddq_f32(vsum_sq1, vaddq_f32(vsum_sq2, vaddq_f32(vsum_sq3, vaddq_f32(vsum_sq4, vaddq_f32(vsum_sq5, vaddq_f32(vsum_sq6, vaddq_f32(vsum_sq7, vsum_sq8)))))))};
            sum = vaddvq_f32(vsum_total);
            sum_sq = vaddvq_f32(vsum_sq_total);
        #endif
        for (; i < in.size(); ++i) {
            float v {p[i]};
            sum += v;
            sum_sq += v*v;
        }
        return {sum, sum_sq};
    }

    static auto PIQUANT_HOT quant_config_kernel_f32(std::span<const float> in) noexcept -> std::array<float, 2> {
        return compute_quant_config_from_data(in);
    }

    static auto PIQUANT_HOT quant_config_kernel_f64(std::span<const double> in) noexcept -> std::array<double, 2> {
        return compute_quant_config_from_data(in);
    }

    template <typename In, typename Out, const round_mode RoundMode, const reduce_op ReduceOp>
    static auto PIQUANT_HOT requant_generic(
      const void* in,
      void* out,
      std::int64_t numel,
      float scale,
      std::int64_t zp
    ) noexcept -> void {
        const auto* PIQUANT_RESTRICT x {static_cast<const In*>(in)};
        auto* PIQUANT_RESTRICT o {static_cast<In*>(out)};
        double inv_scale {1.0 / scale};
        if constexpr (ReduceOp == reduce_op::set) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] = dequant_step<Out, In>(scale, zp, quant_step_scalar<In, Out, RoundMode>(x[i], inv_scale, zp));
            return;
        }
        if constexpr (ReduceOp == reduce_op::add) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] += dequant_step<Out, In>(scale, zp, quant_step_scalar<In, Out, RoundMode>(x[i], inv_scale, zp));
            return;
        }
    }
};

namespace piquant {
    [[nodiscard]] constexpr auto make_pair_perm(dtype from,dtype to) noexcept -> std::uint16_t {
        auto ito {static_cast<std::underlying_type_t<decltype(to)>>(to)};
        auto ifrom {static_cast<std::underlying_type_t<decltype(from)>>(from)};
        return ((255&ifrom)<<8)+(255&ito);
    }

    using quant_fn = auto (*)(const void*, void*, std::int64_t, float, std::int64_t) noexcept -> void;

    template <typename Src, typename Dst, round_mode M>
    [[nodiscard]] consteval auto quant_entry() noexcept -> quant_fn { return &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<Src, Dst, M>; }

    template <typename Src, typename Dst, reduce_op R>
    [[nodiscard]] consteval auto dequant_entry() noexcept -> quant_fn { return &impl_namespace(QUANT_KERNEL_IMPL, _)::dequant_generic<Dst, Src, R>; }

    template <typename Src, typename Dst, round_mode M, reduce_op R>
    [[nodiscard]] consteval auto requant_entry() noexcept -> quant_fn { return &impl_namespace(QUANT_KERNEL_IMPL, _)::requant_generic<Src, Dst, M, R>; }

    template <typename...> struct type_set {};

    template <typename TL> struct type_set_size;
    template <typename... Ts> struct type_set_size<type_set<Ts...>>  : std::integral_constant<std::size_t, sizeof...(Ts)> {};

    using quant_types = type_set<
        uint2_t, int2_t, uint4_t, int4_t,
        uint8_t, int8_t, uint16_t, int16_t,
        uint32_t, int32_t, uint64_t, int64_t
    >;

    using fp_types = type_set<float, double>;

    template <typename Src, round_mode M, typename TL> struct make_quant_row;
    template <typename Src, round_mode M, typename... Dst>
    struct make_quant_row<Src, M, type_set<Dst...>> {
        static constexpr std::array<quant_fn, 2+sizeof...(Dst)> value = {nullptr, nullptr, quant_entry<Src, Dst, M>()...};
    };

    template <typename Src, reduce_op R, typename TL> struct make_dequant_row;
    template <typename Src, reduce_op R, typename... Dst>
    struct make_dequant_row<Src, R, type_set<Dst...>> {
        static constexpr std::array<quant_fn, 2+sizeof...(Dst)> value = {nullptr, nullptr, dequant_entry<Src, Dst, R>()...};
    };

    template <typename Src, round_mode M, reduce_op R, typename TL> struct make_requant_row;
    template <typename Src, round_mode M, reduce_op R, typename... Dst>
    struct make_requant_row<Src, M, R, type_set<Dst...>> {
        static constexpr std::array<quant_fn, 2+sizeof...(Dst)> value = {nullptr, nullptr, requant_entry<Src, Dst, M, R>()...};
    };

    template <round_mode M, reduce_op R, typename FPSrcSet> struct make_requant_block;

    template <round_mode M, reduce_op R, typename... Src>
    struct make_requant_block<M, R, type_set<Src...>> {
        static constexpr std::array<std::array<quant_fn, 2+type_set_size<quant_types>::value>, sizeof...(Src)> value {
            make_requant_row<Src, M, R, quant_types>::value...
        };
    };

    // 3D Dispatch table for quantization kernels. Order matters.
    static constexpr std::array quant_functions {
        std::array {
            std::array {
                make_quant_row<float, round_mode::nearest, quant_types>::value,
                make_quant_row<double, round_mode::nearest, quant_types>::value
            },
        },
        std::array {
            std::array {
                make_quant_row<float, round_mode::stochastic, quant_types>::value,
                make_quant_row<double, round_mode::stochastic, quant_types>::value
            },
        }
    };

    // 3D Dispatch table for dequantization kernels. Order matters.
    static constexpr std::array dequant_functions {
        std::array {
            std::array {
                make_dequant_row<float, reduce_op::set, quant_types>::value,
                make_dequant_row<double, reduce_op::set, quant_types>::value
            },
        },
        std::array {
            std::array {
                make_dequant_row<float, reduce_op::add, quant_types>::value,
                make_dequant_row<double, reduce_op::add, quant_types>::value
            },
        }
    };

    // 4D Dispatch table for requantization kernels. Order matters.
    static constexpr std::array requant_functions {
        std::array{
            make_requant_block<round_mode::nearest, reduce_op::set, fp_types>::value,
            make_requant_block<round_mode::nearest, reduce_op::add, fp_types>::value
        },
        std::array{
            make_requant_block<round_mode::stochastic, reduce_op::set, fp_types>::value,
            make_requant_block<round_mode::stochastic, reduce_op::add, fp_types>::value
        }
    };

    static void dispatch_quantize(const void* in, void* out, std::int64_t range, const context::quant_descriptor& desc) {
        const dtype_info& dt_in {dtype_info_of(desc.dt_in)};
        const dtype_info& dt_out {dtype_info_of(desc.dt_out)};
        piquant_assert2(!(dt_in.flags & dtype_flags::is_quant));
        piquant_assert2(dt_out.flags & dtype_flags::is_quant);
        const auto& stubs_round_mode {quant_functions[static_cast<std::size_t>(desc.rounding)]};
        const auto& stubs_dtype_fp {stubs_round_mode[static_cast<std::size_t>(desc.dt_in)]};
        auto* kernel {stubs_dtype_fp[static_cast<std::size_t>(desc.dt_out)]};
        piquant_assert(kernel != nullptr, "invalid quantization types: %s -> %s", dtype_info_of(desc.dt_in).name, dtype_info_of(desc.dt_out).name);
        (*kernel)(in, out, range, desc.scale, desc.zero_point);
    }

    static void dispatch_dequantize(const void* in, void* out, std::int64_t range, const context::quant_descriptor& desc) {
        const dtype_info& dt_in {dtype_info_of(desc.dt_in)};
        const dtype_info& dt_out {dtype_info_of(desc.dt_out)};
        piquant_assert2(dt_in.flags & dtype_flags::is_quant);
        piquant_assert2(!(dt_out.flags & dtype_flags::is_quant));
        const auto& stubs_reduce_mode {dequant_functions[static_cast<std::size_t>(desc.reducing)]};
        const auto& stubs_dtype_fp {stubs_reduce_mode[static_cast<std::size_t>(desc.dt_out)]};
        auto* kernel {stubs_dtype_fp[static_cast<std::size_t>(desc.dt_in)]};
        piquant_assert(kernel != nullptr, "invalid dequantization types: %s -> %s", dtype_info_of(desc.dt_in).name, dtype_info_of(desc.dt_out).name);
        (*kernel)(in, out, range, desc.scale, desc.zero_point);
    }

    static void dispatch_requantize(const void* in, void* out, std::int64_t range, const context::quant_descriptor& desc) {
        using enum dtype;
        const dtype_info& dt_in {dtype_info_of(desc.dt_in)};
        const dtype_info& dt_out {dtype_info_of(desc.dt_out)};
        piquant_assert2(!(dt_in.flags & dtype_flags::is_quant));
        piquant_assert2(dt_out.flags & dtype_flags::is_quant);
        const auto& stubs_round_mode {requant_functions[static_cast<std::size_t>(desc.rounding)]};
        const auto& stubs_reduce_op {stubs_round_mode[static_cast<std::size_t>(desc.reducing)]};
        const auto& stubs_fp {stubs_reduce_op[static_cast<std::size_t>(desc.dt_in)]};
        auto* kernel {stubs_fp[static_cast<std::size_t>(desc.dt_out)]};
        piquant_assert(kernel != nullptr, "invalid requantization types: %s -> %s", dtype_info_of(desc.dt_in).name, dtype_info_of(desc.dt_out).name);
        (*kernel)(in, out, range, desc.scale, desc.zero_point);
    }

    static auto PIQUANT_HOT quantize_dispatch(const void* in, void* out, std::int64_t range, const context::quant_descriptor& desc) noexcept -> void {
        switch (desc.type) {
            case context::command_type::quant: dispatch_quantize(in, out, range, desc); return;
            case context::command_type::dequant: dispatch_dequantize(in, out, range, desc); return;
            case context::command_type::quant_dequant: dispatch_requantize(in, out, range, desc); return;
            default: panic("invalid quantization command type: %d", static_cast<int>(desc.type));
        }
    }

    auto QUANT_KERNEL_IMPL() noexcept -> kernel_registry {
        return kernel_registry {
            .quant_kernel = &quantize_dispatch,
            .quant_config_kernel_f32 = &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_config_kernel_f32,
            .quant_config_kernel_f64 = &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_config_kernel_f64,
        };
    }
}
