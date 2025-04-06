#ifndef QUANT_KERNEL_IMPL
#error "Kernel impl is not defined"
#endif

#include <piquant.hpp>
#include "piquant_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#ifdef _MSC_VER
#define PIQUANT_HOT
#define PIQUANT_AINLINE __forceinline
#define PIQUANT_RESTRICT __restrict
#else
#define PIQUANT_HOT __attribute__((hot))
#define PIQUANT_AINLINE __attribute__((always_inline))
#define PIQUANT_RESTRICT __restrict__
#endif

namespace piquant {
    static constexpr double std_scale {12.0};

    struct kernel_registry;

    [[nodiscard]] static constexpr auto PIQUANT_AINLINE prng_canonical(prng_state& p) -> float { // returns ξ ∈ [0, 1)
        auto& remaining {p.remaining};
        auto& next {p.next};
        auto& state {p.state};
        if (--remaining <= 0) {
            remaining = 624;
            next = 0;
            uint32_t y, i;
            for (i = 0; i < 624-397; ++i) {
                y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                state[i] = state[i+397] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
            }
            for (; i < 624-1; ++i) {
                y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                state[i] = state[i + (397-624)] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
            }
            y = (state[624-1] & 0x80000000u) | (state[0] & 0x7fffffffu);
            state[624-1] = state[397-1] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
        }
        uint32_t y = state[next++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return (1.f/static_cast<float>(1<<23)*(static_cast<float>(y>>9) + 0.5f));
    }
}

#define concat(a, b) a ## b
#define impl_namespace(a, b) piquant::concat(a, _impl)

namespace impl_namespace(QUANT_KERNEL_IMPL, _) {
    static auto PIQUANT_HOT quant_f32_to_uint8_nearest(
        const float* const PIQUANT_RESTRICT x,
        std::uint8_t* const PIQUANT_RESTRICT o,
        const std::int64_t numel,
        float scale,
        const std::int32_t zp
   ) noexcept -> void {
        scale = 1.0f / scale; /* We multiply by reciprocal */
        std::int64_t i {};
        #if defined(__AVX512F__) && defined(__AVX512BW__) && 0
            const __m512 vinv_scale {_mm512_set1_ps(inv_scale)};
            const __m512i vzero_point {_mm512_set1_epi32(zp)};
            const __m512i vmin {_mm512_setzero_si512()};
            const __m512i vmax {_mm512_set1_epi32(0xff)};
            constexpr int k_round_mode {_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC};
            constexpr std::size_t step {64};
            for (; i+step <= numel; i += step) {
                __m512 xf0 {_mm512_loadu_ps(x+i+(0<<4))};
                __m512 xf1 {_mm512_loadu_ps(x+i+(1<<4))};
                __m512 xf2 {_mm512_loadu_ps(x+i+(2<<4))};
                __m512 xf3 {_mm512_loadu_ps(x+i+(3<<4))};
                __m512i xi0 {_mm512_max_epi32(vmin, _mm512_min_epi32(vmax, _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(xf0, vinv_scale), k_round_mode)), vzero_point)))};
                __m512i xi1 {_mm512_max_epi32(vmin, _mm512_min_epi32(vmax, _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(xf1, vinv_scale), k_round_mode)), vzero_point)))};
                __m512i xi2 {_mm512_max_epi32(vmin, _mm512_min_epi32(vmax, _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(xf2, vinv_scale), k_round_mode)), vzero_point)))};
                __m512i xi3 {_mm512_max_epi32(vmin, _mm512_min_epi32(vmax, _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(xf3, vinv_scale), k_round_mode)),vzero_point)))};
                __m512i pack16_0 {_mm512_packus_epi32(xi0, xi1)};
                __m512i pack16_1 {_mm512_packus_epi32(xi2, xi3)};
                __m512i result {_mm512_packus_epi16(pack16_0, pack16_1)};
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(o+i), result);
            }
        #elif defined(__AVX2__)
            const __m256 vinv_scale {_mm256_set1_ps(scale)};
            const __m256 vhalf {_mm256_set1_ps(0.5f)};
            const __m256 vneg_half {_mm256_set1_ps(-0.5f)};
            const __m256 vzero {_mm256_setzero_ps()};
            const __m256i vzero_point {_mm256_set1_epi32(zp)};
            const __m256i vmin {_mm256_setzero_si256()};
            const __m256i vmax {_mm256_set1_epi32(0xff)};
            constexpr std::size_t step {32};
            static const __m256i shuffle_matrix {_mm256_setr_epi8(
                0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15,
                0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15
            )};
            for (; i + step <= numel; i += step) {
                __m256 xf0 {_mm256_loadu_ps(x+i+(0<<3))};
                __m256 xf1 {_mm256_loadu_ps(x+i+(1<<3))};
                __m256 xf2 {_mm256_loadu_ps(x+i+(2<<3))};
                __m256 xf3 {_mm256_loadu_ps(x+i+(3<<3))};
                __m256 prod0 {_mm256_mul_ps(xf0, vinv_scale)};
                __m256 prod1 {_mm256_mul_ps(xf1, vinv_scale)};
                __m256 prod2 {_mm256_mul_ps(xf2, vinv_scale)};
                __m256 prod3 {_mm256_mul_ps(xf3, vinv_scale)};
                __m256 mask0   {_mm256_cmp_ps(prod0, vzero, _CMP_GE_OQ)};
                __m256 offset0 {_mm256_blendv_ps(vneg_half, vhalf, mask0)};
                __m256 adjusted0 {_mm256_add_ps(prod0, offset0)};
                __m256 mask1   {_mm256_cmp_ps(prod1, vzero, _CMP_GE_OQ)};
                __m256 offset1 {_mm256_blendv_ps(vneg_half, vhalf, mask1)};
                __m256 adjusted1 {_mm256_add_ps(prod1, offset1)};
                __m256 mask2   {_mm256_cmp_ps(prod2, vzero, _CMP_GE_OQ)};
                __m256 offset2 {_mm256_blendv_ps(vneg_half, vhalf, mask2)};
                __m256 adjusted2 {_mm256_add_ps(prod2, offset2)};
                __m256 mask3   {_mm256_cmp_ps(prod3, vzero, _CMP_GE_OQ)};
                __m256 offset3 {_mm256_blendv_ps(vneg_half, vhalf, mask3)};
                __m256 adjusted3 {_mm256_add_ps(prod3, offset3)};
                __m256i xi0 {_mm256_cvttps_epi32(adjusted0)};
                __m256i xi1 {_mm256_cvttps_epi32(adjusted1)};
                __m256i xi2 {_mm256_cvttps_epi32(adjusted2)};
                __m256i xi3 {_mm256_cvttps_epi32(adjusted3)};
                xi0 = _mm256_add_epi32(xi0, vzero_point);
                xi1 = _mm256_add_epi32(xi1, vzero_point);
                xi2 = _mm256_add_epi32(xi2, vzero_point);
                xi3 = _mm256_add_epi32(xi3, vzero_point);
                xi0 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, xi0));
                xi1 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, xi1));
                xi2 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, xi2));
                xi3 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, xi3));
                __m256i pack16_0 {_mm256_packus_epi32(xi0, xi1)};
                __m256i pack16_1 {_mm256_packus_epi32(xi2, xi3)};
                __m256i result {_mm256_packus_epi16(pack16_0, pack16_1)};
                result = _mm256_permute4x64_epi64(result, 0xD8);
                __m256i final {_mm256_shuffle_epi8(result, shuffle_matrix)};
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(o+i), final);
            }
        #elif defined(__SSE4_2__)
            const __m128 vinv_scale {_mm_set1_ps(scale)};
            const __m128 vhalf {_mm_set1_ps(0.5f)};
            const __m128 vneg_half {_mm_set1_ps(-0.5f)};
            const __m128 vzero {_mm_setzero_ps()};
            const __m128i vzero_point {_mm_set1_epi32(zp)};
            const __m128i vmin {_mm_setzero_si128()};
            const __m128i vmax {_mm_set1_epi32(0xff)};
            constexpr std::size_t step = 16;
            for (; i + step <= numel; i += step) {
                __m128 xf0 = _mm_loadu_ps(x+i+(0<<2));
                __m128 xf1 = _mm_loadu_ps(x+i+(1<<2));
                __m128 xf2 = _mm_loadu_ps(x+i+(2<<2));
                __m128 xf3 = _mm_loadu_ps(x+i+(3<<2));
                xf0 = _mm_mul_ps(xf0, vinv_scale);
                xf1 = _mm_mul_ps(xf1, vinv_scale);
                xf2 = _mm_mul_ps(xf2, vinv_scale);
                xf3 = _mm_mul_ps(xf3, vinv_scale);
                __m128 mask0   {_mm_cmpge_ps(xf0, vzero)};
                __m128 offs0 {_mm_blendv_ps(vneg_half, vhalf, mask0)};
                __m128 adj0 {_mm_add_ps(xf0, offs0)};
                __m128 mask1   {_mm_cmpge_ps(xf1, vzero)};
                __m128 offs1 {_mm_blendv_ps(vneg_half, vhalf, mask1)};
                __m128 adj1 {_mm_add_ps(xf1, offs1)};
                __m128 mask2   {_mm_cmpge_ps(xf2, vzero)};
                __m128 offs2 {_mm_blendv_ps(vneg_half, vhalf, mask2)};
                __m128 adj2 {_mm_add_ps(xf2, offs2)};
                __m128 mask3   {_mm_cmpge_ps(xf3, vzero)};
                __m128 offs3 {_mm_blendv_ps(vneg_half, vhalf, mask3)};
                __m128 adj3 {_mm_add_ps(xf3, offs3)};
                __m128i xi0 {_mm_cvttps_epi32(adj0)};
                __m128i xi1 {_mm_cvttps_epi32(adj1)};
                __m128i xi2 {_mm_cvttps_epi32(adj2)};
                __m128i xi3 {_mm_cvttps_epi32(adj3)};
                xi0 = _mm_add_epi32(xi0, vzero_point);
                xi1 = _mm_add_epi32(xi1, vzero_point);
                xi2 = _mm_add_epi32(xi2, vzero_point);
                xi3 = _mm_add_epi32(xi3, vzero_point);
                xi0 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, xi0));
                xi1 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, xi1));
                xi2 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, xi2));
                xi3 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, xi3));
                __m128i pack16_0 {_mm_packus_epi32(xi0, xi1)};
                __m128i pack16_1 {_mm_packus_epi32(xi2, xi3)};
                __m128i result {_mm_packus_epi16(pack16_0, pack16_1)};
                _mm_storeu_si128(reinterpret_cast<__m128i*>(o+i), result);
            }
        #elif defined(__aarch64__) && defined(__ARM_NEON__)
            const float32x4_t vinv_scale = vdupq_n_f32(scale);
            const int32x4_t vzero_point = vdupq_n_s32(zp);
            const int32x4_t vmin = vdupq_n_s32(0);
            const int32x4_t vmax = vdupq_n_s32(0xff);
            constexpr std::size_t step = 16;
            for (; i+step <= numel; i += step) {
                float32x4_t xf0 = vld1q_f32(x+i+(0<<2));
                float32x4_t xf1 = vld1q_f32(x+i+(1<<2));
                float32x4_t xf2 = vld1q_f32(x+i+(2<<2));
                float32x4_t xf3 = vld1q_f32(x+i+(3<<2));
                xf0 = vmulq_f32(xf0, vinv_scale);
                xf1 = vmulq_f32(xf1, vinv_scale);
                xf2 = vmulq_f32(xf2, vinv_scale);
                xf3 = vmulq_f32(xf3, vinv_scale);
                int32x4_t xi0 = vcvtaq_s32_f32(xf0);
                int32x4_t xi1 = vcvtaq_s32_f32(xf1);
                int32x4_t xi2 = vcvtaq_s32_f32(xf2);
                int32x4_t xi3 = vcvtaq_s32_f32(xf3);
                xi0 = vaddq_s32(xi0, vzero_point);
                xi1 = vaddq_s32(xi1, vzero_point);
                xi2 = vaddq_s32(xi2, vzero_point);
                xi3 = vaddq_s32(xi3, vzero_point);
                xi0 = vmaxq_s32(xi0, vmin);
                xi1 = vmaxq_s32(xi1, vmin);
                xi2 = vmaxq_s32(xi2, vmin);
                xi3 = vmaxq_s32(xi3, vmin);
                xi0 = vminq_s32(xi0, vmax);
                xi1 = vminq_s32(xi1, vmax);
                xi2 = vminq_s32(xi2, vmax);
                xi3 = vminq_s32(xi3, vmax);
                int16x4_t p16_0_low = vqmovn_s32(xi0);
                int16x4_t p16_0_high = vqmovn_s32(xi1);
                int16x8_t pack16_0 = vcombine_s16(p16_0_low, p16_0_high);
                int16x4_t p16_1_low = vqmovn_s32(xi2);
                int16x4_t p16_1_high = vqmovn_s32(xi3);
                int16x8_t pack16_1 = vcombine_s16(p16_1_low, p16_1_high);
                uint8x8_t result_low = vqmovun_s16(pack16_0);
                uint8x8_t result_high = vqmovun_s16(pack16_1);
                uint8x16_t result = vcombine_u8(result_low, result_high);
                vst1q_u8(o+i, result);
            }
        #endif
        for (; i < numel; ++i) {
            const float rnd {std::round(x[i] * scale)};
            const std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
    }

    template <typename T>
    struct dtype_limits final {
        static constexpr T min {std::numeric_limits<T>::min()};
        static constexpr T max {std::numeric_limits<T>::max()};
    };

    template<>
    struct dtype_limits<uint4_t> final {
        static constexpr std::uint8_t min {0};
        static constexpr std::uint8_t max {0xf};
    };

    template<>
    struct dtype_limits<int4_t> final {
        static constexpr std::int8_t min {-0x8};
        static constexpr std::int8_t max {0x7};
    };

    template <typename T>
    concept is_int4 = std::is_same_v<T, uint4_t> || std::is_same_v<T, int4_t>;

    template <typename OUT> requires is_int4<OUT>
    [[nodiscard]] static constexpr auto PIQUANT_AINLINE pack_nibbles(const OUT x, const OUT y) -> OUT {
        const auto xi {static_cast<std::uint8_t>(x)};
        const auto yi {static_cast<std::uint8_t>(y)};
        constexpr auto m {dtype_limits<uint4_t>::max};
        const auto pa {static_cast<std::uint8_t>((m&xi)<<4|m&yi)};
        return static_cast<OUT>(pa);
    }

    template <const round_mode RND, typename IN, typename OUT, typename... Args>
         requires (std::is_floating_point_v<IN> && (std::is_integral_v<OUT> || is_int4<OUT>) && std::is_same_v<std::common_type_t<Args...>, IN> && sizeof...(Args) != 0)
    static inline auto PIQUANT_AINLINE quant_step(const double inv_scale, const std::int32_t zp, prng_state& prng, const Args... args) noexcept -> OUT {
        if constexpr (RND == round_mode::stochastic) {
            const auto Q{[&](const IN x) noexcept -> OUT {
                double rnd {x * inv_scale};
                const double dec {std::abs(rnd - std::trunc(rnd))};
                const double xi {prng_canonical(prng)};
                double adj {xi < dec ? 1.0f : 0.0f};
                if (rnd < 0.0f) adj = -1.0f * adj;
                rnd = std::trunc(rnd) + adj;
                const auto integral {static_cast<std::int64_t>(rnd) + zp};
                return static_cast<OUT>(std::clamp<decltype(integral)>(integral, dtype_limits<OUT>::min, dtype_limits<OUT>::max));
            }};
            if constexpr (sizeof...(Args) == 1) return Q(args...);
            else return pack_nibbles(Q(args)...);
        } else {
            const auto Q {[=](const IN x) noexcept -> OUT {
                const double rnd {std::round(static_cast<double>(x) * inv_scale)};
                const auto integral {static_cast<std::int64_t>(rnd) + zp};
                return static_cast<OUT>(std::clamp<decltype(integral)>(integral, dtype_limits<OUT>::min, dtype_limits<OUT>::max));
            }};
            if constexpr (sizeof...(Args) == 1) return Q(args...);
            else return pack_nibbles(Q(args)...);
        }
    }

    template <typename IN, typename OUT, const round_mode RND>
        requires (std::is_floating_point_v<IN> && (std::is_integral_v<OUT> || is_int4<OUT>))
    static auto PIQUANT_HOT quant_generic(
        const void* const in,
        void* const out,
        std::int64_t numel,
        float scale,
        const std::int64_t zp,
        prng_state& prng
    ) noexcept -> void {
        if constexpr (std::is_same_v<IN, float> && std::is_same_v<OUT, std::uint8_t> && RND == round_mode::nearest) { // Use SIMD optimized kernels for some dtype permutations
            quant_f32_to_uint8_nearest(static_cast<const float*>(in), static_cast<std::uint8_t*>(out), numel, scale, zp);
            return;
        }
        const auto* PIQUANT_RESTRICT const x {static_cast<const IN*>(in)};
        auto* PIQUANT_RESTRICT const o {static_cast<OUT*>(out)};
        const double inv_scale {1.0 / static_cast<double>(scale)}; // We multiply by reciprocal
        if constexpr (is_int4<OUT>) numel = (numel+1)>>1;
        for (std::int64_t i {}; i < numel; ++i)
            if constexpr (is_int4<OUT>)
                o[i] = quant_step<RND, IN, OUT>(inv_scale, zp, prng, x[i], x[i+1]);
            else
                o[i] = quant_step<RND, IN, OUT>(inv_scale, zp, prng, x[i]);
    }

    template <typename IN, typename OUT>
          requires (std::is_floating_point_v<OUT> && (std::is_integral_v<IN> || is_int4<IN>))
    static inline auto PIQUANT_AINLINE dequant_step(const double scale, const std::int32_t zp, const IN x) noexcept -> OUT {
        return static_cast<OUT>(static_cast<std::make_signed_t<IN>>(x) - zp)*scale;
    }

    template <typename IN, typename OUT, const reduce_op RDO>
            requires (std::is_floating_point_v<OUT> && (std::is_integral_v<IN> || is_int4<IN>))
    static auto PIQUANT_HOT dequant_generic(
        const void* const in,
        void* const out,
        const std::int64_t numel,
        double scale,
        const std::int64_t zp
    ) noexcept -> void {
        const auto* PIQUANT_RESTRICT const x {static_cast<const IN*>(in)};
        auto* PIQUANT_RESTRICT const o {static_cast<OUT*>(out)};
        if constexpr (RDO == reduce_op::set) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] = dequant_step<IN, OUT>(scale, zp, x[i]);
        } else if constexpr (RDO == reduce_op::add) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] += dequant_step<IN, OUT>(scale, zp, x[i]);
        } else
            panic("Invalid reduce operation");
    }

    template <typename IN, typename QUANT, const round_mode RND, const reduce_op RDO>
      requires (std::is_floating_point_v<IN> && (std::is_integral_v<QUANT> || is_int4<QUANT>))
    static auto PIQUANT_HOT quant_dequant_generic(
      const void* const in,
      void* const out,
      const std::int64_t numel,
      double scale,
      const std::int64_t zp,
      prng_state& prng
    ) noexcept -> void {
        const auto* PIQUANT_RESTRICT const x {static_cast<const IN*>(in)};
        auto* PIQUANT_RESTRICT const o {static_cast<IN*>(out)};
        const double inv_scale {1.0 / scale};
        if constexpr (RDO == reduce_op::set) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] = dequant_step<QUANT, IN>(scale, zp, quant_step<RND, IN, QUANT>(inv_scale, zp, prng, x[i]));
        } else if constexpr (RDO == reduce_op::add) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] += dequant_step<QUANT, IN>(scale, zp, quant_step<RND, IN, QUANT>(inv_scale, zp, prng, x[i]));
        } else
            panic("Invalid reduce operation");
    }

    template <typename T> requires std::is_floating_point_v<T>
    [[nodiscard]] auto compute_quant_config_from_data(const T* p, std::int64_t numel, std::int64_t tmax) -> std::pair<T, std::int64_t> {
        if (!numel) [[unlikely]] return {0.0, 0.0};
        auto mean {static_cast<T>(std::accumulate(p, p+numel, 0.0) / static_cast<T>(numel))};
        auto sq_delta {static_cast<T>(std::transform_reduce(
            p, p+numel,
            0.0,
            std::plus{},
            [mean](const T value) noexcept -> T {
                T delta {value - mean};
                return delta*delta;
            }
        ))};
        const auto std {static_cast<T>(std::sqrt(sq_delta / static_cast<T>(numel-1)))};
        const auto scale {static_cast<T>(std_scale*std/static_cast<T>(tmax))};
        const std::int64_t zp {(tmax>>1) - static_cast<std::int64_t>(std::round(mean/scale))};
        return {scale, zp};
    }

    template <>
    [[nodiscard]] auto compute_quant_config_from_data(const float* p, std::int64_t numel, std::int64_t tmax) -> std::pair<float, std::int64_t> {
        if (!numel) [[unlikely]] return {0.0, 0.0};
        float sum {};
        {
            std::int64_t i {};
            #ifdef __ARM_NEON
                float32x4_t vsum1 {vdupq_n_f32(0.0f)};
                float32x4_t vsum2 {vdupq_n_f32(0.0f)};
                float32x4_t vsum3 {vdupq_n_f32(0.0f)};
                float32x4_t vsum4 {vdupq_n_f32(0.0f)};
                for (; i+16 <= numel; i += 16) {
                    vsum1 = vaddq_f32(vsum1, vld1q_f32(p+i));
                    vsum2 = vaddq_f32(vsum2, vld1q_f32(p+i+4));
                    vsum3 = vaddq_f32(vsum3, vld1q_f32(p+i+8));
                    vsum4 = vaddq_f32(vsum4, vld1q_f32(p+i+12));
                }
                sum = vaddvq_f32(vsum1) + vaddvq_f32(vsum2) +
                      vaddvq_f32(vsum3) + vaddvq_f32(vsum4);
            #endif
            for (; i < numel; ++i) {
                sum += p[i];
            }
        }
        float mean {sum / static_cast<float>(numel)};
        float sq_delta {};
        {
            std::int64_t i {};
            #ifdef __ARM_NEON
                float32x4_t vsq_delta1 {vdupq_n_f32(0.0f)};
                float32x4_t vsq_delta2 {vdupq_n_f32(0.0f)};
                float32x4_t vsq_delta3 {vdupq_n_f32(0.0f)};
                float32x4_t vsq_delta4 {vdupq_n_f32(0.0f)};
                float32x4_t vmean {vdupq_n_f32(mean)};
                for (; i+16 <= numel; i += 16) {
                    float32x4_t vdelta1 {vsubq_f32(vld1q_f32(p+i), vmean)};
                    float32x4_t vdelta2 {vsubq_f32(vld1q_f32(p+i+4), vmean)};
                    float32x4_t vdelta3 {vsubq_f32(vld1q_f32(p+i+8), vmean)};
                    float32x4_t vdelta4 {vsubq_f32(vld1q_f32(p+i+12), vmean)};
                    vsq_delta1 = vmlaq_f32(vsq_delta1, vdelta1, vdelta1);
                    vsq_delta2 = vmlaq_f32(vsq_delta2, vdelta2, vdelta2);
                    vsq_delta3 = vmlaq_f32(vsq_delta3, vdelta3, vdelta3);
                    vsq_delta4 = vmlaq_f32(vsq_delta4, vdelta4, vdelta4);
                }
                sq_delta = vaddvq_f32(vsq_delta1) + vaddvq_f32(vsq_delta2) +
                            vaddvq_f32(vsq_delta3) + vaddvq_f32(vsq_delta4);
            #endif
            for (; i < numel; ++i) {
                float delta {p[i] - mean};
                sq_delta += delta*delta;
            }
        }
        auto stddev {(std::sqrt(sq_delta / static_cast<float>(numel-1)))};
        auto scale {static_cast<float>(std_scale * stddev / static_cast<float>(tmax))};
        std::int64_t zp {(tmax>>1) - static_cast<std::int64_t>(std::round(mean / scale))};
        return {scale, zp};
    }

    static auto PIQUANT_HOT quant_config_kernel_f32(std::span<const float> x, std::int64_t tmax) noexcept -> std::pair<float, std::int32_t> {
        return compute_quant_config_from_data(x.data(), x.size(), tmax>>1);
    }

    static auto PIQUANT_HOT quant_config_kernel_f64(std::span<const double> x, std::int64_t tmax) noexcept -> std::pair<float, std::int32_t> {
        return compute_quant_config_from_data(x.data(), x.size(), tmax>>1);
    }
};

namespace piquant {
    [[nodiscard]] constexpr auto make_pair_perm(const dtype from, const dtype to) noexcept -> std::uint16_t {
        auto ito {static_cast<std::underlying_type_t<decltype(to)>>(to)};
        auto ifrom {static_cast<std::underlying_type_t<decltype(from)>>(from)};
        return ((255&ifrom)<<8)+(255&ito);
    }

    static auto PIQUANT_HOT quantize_dispatch(
        const void* x,
        void* o,
        std::int64_t range,
        const context::quant_descriptor& desc,
        prng_state& prng
    ) noexcept -> void {
        using enum dtype;
        const dtype_info& dt_in {dtype_info_of(desc.dt_in)};
        const dtype_info& dt_out {dtype_info_of(desc.dt_out)};
        switch (desc.type) {
            case context::command_type::quant:  // out[i] = quantize(in[i])
                piquant_assert2(!dt_in.is_quant);
                piquant_assert2(dt_out.is_quant);
                #define impl_quant_perm(dti, dto, ti, to) \
                    case make_pair_perm(dti, dto): \
                        if (desc.rnd_mode == round_mode::stochastic) \
                            impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<ti, to, round_mode::stochastic>(x, o, range, desc.scale, desc.zero_point, prng); \
                        else \
                            impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<ti, to, round_mode::nearest>(x, o, range, desc.scale, desc.zero_point, prng); \
                    return
                switch (make_pair_perm(desc.dt_in, desc.dt_out)) {
                    impl_quant_perm(f32, uint4, float, uint4_t);
                    impl_quant_perm(f32, int4, float, int4_t);
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
            #undef impl_quant_perm
            return;
            case context::command_type::dequant:    // out[i] = dequantize(in[i])
                piquant_assert2(dt_in.is_quant);
                piquant_assert2(!dt_out.is_quant);
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
                piquant_assert2(!dt_in.is_quant);
                piquant_assert2(dt_out.is_quant); // dt_out acts as the quantized type, but dtype in == dtype out
               #define impl_quant_perm(dti, dto, ti, to) \
                    case make_pair_perm(dti, dto): \
                        if (desc.reduce == reduce_op::set) \
                            if (desc.rnd_mode == round_mode::stochastic) \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::stochastic, reduce_op::set>(x, o, range, desc.scale, desc.zero_point, prng); \
                            else \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::nearest, reduce_op::set>(x, o, range, desc.scale, desc.zero_point, prng); \
                        else \
                            if (desc.rnd_mode == round_mode::stochastic) \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::stochastic, reduce_op::add>(x, o, range, desc.scale, desc.zero_point, prng); \
                            else \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::nearest, reduce_op::add>(x, o, range, desc.scale, desc.zero_point, prng); \
                    return
                switch (make_pair_perm(desc.dt_in, desc.dt_out)) {
                    impl_quant_perm(f32, uint4, float, uint4_t);
                    impl_quant_perm(f32, int4, float, int4_t);
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
