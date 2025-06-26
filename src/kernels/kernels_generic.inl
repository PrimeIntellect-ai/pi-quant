// This inline file is directly included into the kernels.inl file, which is cloned (recompiled) in multiple compilation units for different CPU architectures.
// ! Make sure all functions are static, to make them local to the compilation unit.

#ifndef QUANT_KERNEL_IMPL
#error "Kernel impl is not defined"
#endif

#include <piquant.hpp>
#include "../piquant_internal.hpp"

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
    [[nodiscard]] auto compute_quant_config_from_data(std::span<const T> x) -> std::array<T, 2> {
        if (x.empty()) [[unlikely]] return {0.0, 0.0};
        T sum {};
        T sum_sq {};
        for (T v : x) {
            sum += v;
            sum_sq += v*v;
        }
        return {sum, sum_sq};
    }

    #ifdef __AVX2__
        [[nodiscard]] static auto avx2_hsum256(__m256 x) noexcept -> float {
            __m128 hiQuad = _mm256_extractf128_ps(x, 1);
            __m128 loQuad = _mm256_castps256_ps128(x);
            __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
            __m128 loDual = sumQuad;
            __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
            __m128 sumDual = _mm_add_ps(loDual, hiDual);
            __m128 lo = sumDual;
            __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
            __m128 sum = _mm_add_ss(lo, hi);
            return _mm_cvtss_f32(sum);
        }
    #endif

    template <>
    [[nodiscard]] auto compute_quant_config_from_data(std::span<const float> x) -> std::array<float, 2> {
        if (x.empty()) [[unlikely]] return {0.0f, 0.0};
        const auto* p {x.data()};
        float sum {};
        float sum_sq {};
        std::int64_t i {};
        #if defined(__AVX512F__) && defined(__AVX512BW__) && 0

        #elif defined(__AVX2__)
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
            sum = avx2_hsum256(vsum_total);
            sum_sq = avx2_hsum256(vsum_sq_total);
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
            for (; i+31 < x.size(); i += 32) {
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
        for (; i < x.size(); ++i) {
            float v {p[i]};
            sum += v;
            sum_sq += v*v;
        }
        return {sum, sum_sq};
    }

    static auto PIQUANT_HOT quant_config_kernel_f32(std::span<const float> x) noexcept -> std::array<float, 2> {
        return compute_quant_config_from_data(x);
    }

    static auto PIQUANT_HOT quant_config_kernel_f64(std::span<const double> x) noexcept -> std::array<double, 2> {
        return compute_quant_config_from_data(x);
    }

    template <typename In, typename Out, const round_mode RoundMode, const reduce_op ReduceOp>
    static auto PIQUANT_HOT quant_dequant_generic(
      const void* in,
      void* out,
      std::int64_t numel,
      double scale,
      std::int64_t zp
    ) noexcept -> void {
        /* TODO
        const auto* PIQUANT_RESTRICT x {static_cast<const In*>(in)};
        auto* PIQUANT_RESTRICT o {static_cast<Out*>(out)};
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
        */
    }
};

namespace piquant {
    [[nodiscard]] constexpr auto make_pair_perm(dtype from,dtype to) noexcept -> std::uint16_t {
        auto ito {static_cast<std::underlying_type_t<decltype(to)>>(to)};
        auto ifrom {static_cast<std::underlying_type_t<decltype(from)>>(from)};
        return ((255&ifrom)<<8)+(255&ito);
    }

    using dispatch_fn = auto (*)(const void*, void*, std::int64_t, float, std::int64_t) noexcept -> void;

    // Dispatch table for quantization kernels.  aOrder matters.
    static constexpr std::array<std::array<std::array<dispatch_fn, static_cast<std::size_t>(dtype::count_)>, float_dtype_count>, static_cast<std::size_t>(round_mode::count_)> stubs_quant = {
        std::array<std::array<dispatch_fn, static_cast<std::size_t>(dtype::count_)>, float_dtype_count> {
            std::array<dispatch_fn, static_cast<std::size_t>(dtype::count_)> { // float, nearest
                nullptr, // same type <-> not supported
                nullptr, // same type <-> not supported
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint2_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int2_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint4_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int4_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint8_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int8_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint16_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int16_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint32_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int32_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint64_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int64_t, round_mode::nearest>
            },
            std::array<dispatch_fn, static_cast<std::size_t>(dtype::count_)> { // double, nearest
                nullptr, // same type <-> not supported
                nullptr, // same type <-> not supported
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint2_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int2_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint4_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int4_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint8_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int8_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint16_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int16_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint32_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int32_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint64_t, round_mode::nearest>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int64_t, round_mode::nearest>
            }
        },
        std::array<std::array<dispatch_fn, static_cast<std::size_t>(dtype::count_)>, float_dtype_count> {
            std::array<dispatch_fn, static_cast<std::size_t>(dtype::count_)> { // float, stochastic
                nullptr, // same type <-> not supported
                nullptr, // same type <-> not supported
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint2_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int2_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint4_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int4_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint8_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int8_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint16_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int16_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint32_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int32_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, uint64_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<float, int64_t, round_mode::stochastic>
                },
            std::array<dispatch_fn, static_cast<std::size_t>(dtype::count_)> { // double, stochastic
                nullptr, // same type <-> not supported
                nullptr, // same type <-> not supported
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint2_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int2_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint4_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int4_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint8_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int8_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint16_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int16_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint32_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int32_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, uint64_t, round_mode::stochastic>,
                &impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<double, int64_t, round_mode::stochastic>
            }
        }
    };

    static auto PIQUANT_HOT quantize_dispatch(
        const void* x,
        void* o,
        std::int64_t range,
        const context::quant_descriptor& desc
    ) noexcept -> void {
        using enum dtype;
        const dtype_info& dt_in {dtype_info_of(desc.dt_in)};
        const dtype_info& dt_out {dtype_info_of(desc.dt_out)};
        switch (desc.type) {
            case context::command_type::quant: {
                piquant_assert2(!(dt_in.flags & dtype_flags::is_quant));
                piquant_assert2(dt_out.flags & dtype_flags::is_quant);
                const auto& stubs_round_mode {stubs_quant[static_cast<std::size_t>(desc.rnd_mode)]};
                const auto& stubs_dtype_fp {stubs_round_mode[static_cast<std::size_t>(desc.dt_in)]};
                auto* kernel {stubs_dtype_fp[static_cast<std::size_t>(desc.dt_out)]};
                piquant_assert(kernel != nullptr, "invalid quantization types: %s -> %s", dtype_info_of(desc.dt_in).name, dtype_info_of(desc.dt_out).name);
                (*kernel)(x, o, range, desc.scale, desc.zero_point);
            } return;
            case context::command_type::dequant:    // out[i] = dequantize(in[i])
                piquant_assert2(dt_in.flags & dtype_flags::is_quant);
                piquant_assert2(!(dt_out.flags & dtype_flags::is_quant));
                #define impl_dequant_perm(dti, dto, ti, to) \
                    case make_pair_perm(dti, dto): \
                        switch (desc.reduce) { \
                            case reduce_op::set: impl_namespace(QUANT_KERNEL_IMPL, _)::dequant_generic<ti, to, reduce_op::set>(x, o, range, desc.scale, desc.zero_point); return; \
                            case reduce_op::add: impl_namespace(QUANT_KERNEL_IMPL, _)::dequant_generic<ti, to, reduce_op::add>(x, o, range, desc.scale, desc.zero_point); return; \
                            default: panic("Invalid reduce operation"); \
                        }  \
                    return
                switch (make_pair_perm(desc.dt_in, desc.dt_out)) {
                    impl_dequant_perm(uint4, f32, uint4_t, float);
                    impl_dequant_perm(int4, f32, int4_t, float);
                    impl_dequant_perm(uint2, f32, uint2_t, float);
                    impl_dequant_perm(int2, f32, int2_t, float);
                    impl_dequant_perm(uint8, f32, uint8_t, float);
                    impl_dequant_perm(int8, f32, int8_t, float);
                    impl_dequant_perm(uint16, f32, uint16_t, float);
                    impl_dequant_perm(int16, f32, int16_t, float);
                    impl_dequant_perm(uint32, f32, uint32_t, float);
                    impl_dequant_perm(int32, f32, int32_t, float);
                    impl_dequant_perm(uint64, f32, uint64_t, float);
                    impl_dequant_perm(int64, f32, int64_t, float);
                    impl_dequant_perm(uint4, f64, uint4_t, double);
                    impl_dequant_perm(int4, f64, int4_t, double);
                    impl_dequant_perm(uint2, f64, uint2_t, double);
                    impl_dequant_perm(int2, f64, int2_t, double);
                    impl_dequant_perm(uint8, f64, uint8_t, double);
                    impl_dequant_perm(int8, f64, int8_t, double);
                    impl_dequant_perm(uint16, f64, uint16_t, double);
                    impl_dequant_perm(int16, f64, int16_t, double);
                    impl_dequant_perm(uint32, f64, uint32_t, double);
                    impl_dequant_perm(int32, f64, int32_t, double);
                    impl_dequant_perm(uint64, f64, uint64_t, double);
                    impl_dequant_perm(int64, f64, int64_t, double);
                    default: panic("Invalid dequantization pair");
                }
            return;
            case context::command_type::quant_dequant:  // out[i] = dequantize(quantize(in[i])))
                piquant_assert2(!(dt_in.flags & dtype_flags::is_quant));
                piquant_assert2(dt_out.flags & dtype_flags::is_quant); // dt_out acts as the quantized type, but dtype in == dtype out
               #define impl_quant_perm(dti, dto, ti, to) \
                    case make_pair_perm(dti, dto): \
                        if (desc.reduce == reduce_op::set) \
                            if (desc.rnd_mode == round_mode::stochastic) \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::stochastic, reduce_op::set>(x, o, range, desc.scale, desc.zero_point); \
                            else \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::nearest, reduce_op::set>(x, o, range, desc.scale, desc.zero_point); \
                        else \
                            if (desc.rnd_mode == round_mode::stochastic) \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::stochastic, reduce_op::add>(x, o, range, desc.scale, desc.zero_point); \
                            else \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::nearest, reduce_op::add>(x, o, range, desc.scale, desc.zero_point); \
                    return
                switch (make_pair_perm(desc.dt_in, desc.dt_out)) {
                    impl_quant_perm(f32, uint4, float, uint4_t);
                    impl_quant_perm(f32, int4, float, int4_t);
                    impl_quant_perm(f32, uint2, float, uint2_t);
                    impl_quant_perm(f32, int2, float, int2_t);
                    impl_quant_perm(f32, uint8, float, uint8_t);
                    impl_quant_perm(f32, int8, float, int8_t);
                    impl_quant_perm(f32, uint16, float, uint16_t);
                    impl_quant_perm(f32, int16, float, int16_t);
                    impl_quant_perm(f32, uint32, float, uint32_t);
                    impl_quant_perm(f32, int32, float, int32_t);
                    impl_quant_perm(f32, uint64, float, uint64_t);
                    impl_quant_perm(f32, int64, float, int64_t);
                    impl_quant_perm(f64, uint4, double, uint4_t);
                    impl_quant_perm(f64, int4, double, int4_t);
                    impl_quant_perm(f64, uint2, double, uint2_t);
                    impl_quant_perm(f64, int2, double, int2_t);
                    impl_quant_perm(f64, uint8, double, uint8_t);
                    impl_quant_perm(f64, int8, double, int8_t);
                    impl_quant_perm(f64, uint16, double, uint16_t);
                    impl_quant_perm(f64, int16, double, int16_t);
                    impl_quant_perm(f64, uint32, double, uint32_t);
                    impl_quant_perm(f64, int32, double, int32_t);
                    impl_quant_perm(f64, uint64, double, uint64_t);
                    impl_quant_perm(f64, int64, double, int64_t);
                    default: panic("Invalid quantization pair");
                }
            return;
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
