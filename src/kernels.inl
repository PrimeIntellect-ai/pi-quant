#ifndef QUANT8_KERNEL_IMPL
#error "Q8 impl is not defined"
#endif
#ifndef QUANT4_KERNEL_IMPL
#error "Q4 impl is not defined"
#endif

#include <piquant.hpp>


#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <iostream>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace piquant {
    [[nodiscard]] static constexpr auto prng_canonical(prng_state& p) -> float { // returns ξ ∈ [0, 1)
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

namespace impl_namespace(QUANT8_KERNEL_IMPL, _) {
    static auto __attribute__((hot)) nearest(
        const float* const __restrict__ x,
        std::uint8_t* const __restrict__ o,
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

    static auto __attribute__((hot)) stochastic(
        const float* const __restrict__ x,
        std::uint8_t* const __restrict__ o,
        const std::int64_t numel,
        float scale,
        const std::int32_t zp,
        piquant::prng_state& prng
    ) noexcept -> void {
        scale = 1.0f / scale; /* We multiply by reciprocal */
        std::int64_t i {};
        for (; i < numel; ++i) {
            float rnd {x[i] * scale};
            const float dec {std::abs(rnd - std::trunc(rnd))};
            const float xi {prng_canonical(prng)};
            float adj {xi < dec ? 1.0f : 0.0f};
            if (rnd < 0.0f) adj = -1.0f * adj;
            rnd = std::trunc(rnd) + adj;
            const std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
    }

    template <const piquant::reduce_op op>
    static auto __attribute__((hot)) dequant(
       const std::uint8_t* const __restrict__ x,
       float* const __restrict__ o,
       const std::int64_t numel,
       const float scale,
       const std::int32_t zp
    ) noexcept -> void {
        std::int64_t i {};
        #if defined(__AVX512F__) && defined(__AVX512BW__) && 0

        #elif defined(__AVX2__) && 0

        #elif defined(__SSE4_2__)

        #elif defined(__aarch64__) && defined(__ARM_NEON__)

        #endif
        if constexpr (op == piquant::reduce_op::set) {
            for (; i < numel; ++i) {
                o[i] = static_cast<float>(x[i] - zp) * scale;
            }
        } else if constexpr (op == piquant::reduce_op::add) {
            for (; i < numel; ++i) {
                o[i] += static_cast<float>(x[i] - zp) * scale;
            }
        } else {
            piquant::panic("Invalid reduce operation");
        }
    }
};

namespace impl_namespace(QUANT4_KERNEL_IMPL, _) {
    static auto __attribute__((hot)) nearest(
        const float* const __restrict__ x,
        std::uint8_t* const __restrict__ o,
        std::int64_t numel,
        float scale,
        const std::int32_t zp
    ) noexcept -> void {
        scale = 1.0f / scale; /* We multiply by reciprocal */
        numel = (numel + 1) / 2;
        const auto f = [=](float x) noexcept -> std::uint8_t {
            return std::clamp<int>(std::round(x * scale) + zp, 0, 0xf);
        };
        for (std::size_t i{0}; i < numel; ++i) {
            auto hi {f(x[(i<<1)])     & 0b0000'1111};
            auto lo {f(x[(i<<1) + 1]) & 0b0000'1111};
            o[i] = (hi << 4) | lo;
        }
    }

    static auto __attribute__((hot)) stochastic(
        const float* const __restrict__ x,
        std::uint8_t* const __restrict__ o,
        const std::int64_t numel,
        float scale,
        const std::int32_t zp,
        piquant::prng_state& prng
    ) noexcept -> void {
        scale = 1.0f / scale; /* We multiply by reciprocal */
        std::int64_t i {};
        for (; i < numel; ++i) {
            float rnd {x[i] * scale};
            const float dec {std::abs(rnd - std::trunc(rnd))};
            const float xi {prng_canonical(prng)};
            float adj {xi < dec ? 1.0f : 0.0f};
            if (rnd < 0.0f) adj = -1.0f * adj;
            rnd = std::trunc(rnd) + adj;
            const std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
    }

    template <const piquant::reduce_op op>
    static auto __attribute__((hot)) dequant(
       const std::uint8_t* const __restrict__ x,
       float* const __restrict__ o,
       const std::int64_t numel,
       const float scale,
       const std::int32_t zp
    ) noexcept -> void {
        std::int64_t i {};
        #if defined(__AVX512F__) && defined(__AVX512BW__) && 0

        #elif defined(__AVX2__) && 0
            const __m256 vscale = _mm256_set1_ps(scale);
            const __m256i vzp16 = _mm256_set1_epi16(static_cast<short>(zp));
            constexpr std::size_t step = 32;
            for (; i+step <= numel; i += step) {
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+i));
                __m128i v_low = _mm256_castsi256_si128(v);
                __m128i v_high = _mm256_extracti128_si256(v, 1);
                __m256i v_low_16 = _mm256_cvtepu8_epi16(v_low);
                __m256i v_high_16 = _mm256_cvtepu8_epi16(v_high);
                v_low_16 = _mm256_sub_epi16(v_low_16, vzp16);
                v_high_16 = _mm256_sub_epi16(v_high_16, vzp16);
                __m256i v_low_32_0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_low_16));
                __m256i v_low_32_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_low_16, 1));
                __m256i v_high_32_0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_high_16));
                __m256i v_high_32_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_high_16, 1));
                __m256 f_low_0 = _mm256_cvtepi32_ps(v_low_32_0);
                __m256 f_low_1 = _mm256_cvtepi32_ps(v_low_32_1);
                __m256 f_high_0 = _mm256_cvtepi32_ps(v_high_32_0);
                __m256 f_high_1 = _mm256_cvtepi32_ps(v_high_32_1);
                f_low_0 = _mm256_mul_ps(f_low_0, vscale);
                f_low_1 = _mm256_mul_ps(f_low_1, vscale);
                f_high_0 = _mm256_mul_ps(f_high_0, vscale);
                f_high_1 = _mm256_mul_ps(f_high_1, vscale);
                _mm256_storeu_ps(o+i, f_low_0);
                _mm256_storeu_ps(o+i+8, f_low_1);
                _mm256_storeu_ps(o+i+16, f_high_0);
                _mm256_storeu_ps(o+i+24, f_high_1);
            }
        #elif defined(__SSE4_2__)

        #elif defined(__aarch64__) && defined(__ARM_NEON__)

        #endif
        if constexpr (op == piquant::reduce_op::set) {
            for (; i < numel; ++i) {
                o[i] = static_cast<float>(x[i] - zp) * scale;
            }
        } else if constexpr (op == piquant::reduce_op::add) {
            for (; i < numel; ++i) {
                o[i] += static_cast<float>(x[i] - zp) * scale;
            }
        } else {
            piquant::panic("Invalid reduce operation");
        }
    }
};

namespace piquant {
    auto __attribute__((hot)) QUANT8_KERNEL_IMPL(
      const float* const __restrict__ x,
      std::uint8_t* const __restrict__ o,
      const std::int64_t numel,
      const float scale,
      const std::int32_t zp,
      const bool sto_rnd,
      prng_state& prng
    ) noexcept -> void {
        if (sto_rnd) {
            impl_namespace(QUANT8_KERNEL_IMPL, _)::stochastic(x, o, numel, scale, zp, prng);
        } else {
            impl_namespace(QUANT8_KERNEL_IMPL, _)::nearest(x, o, numel, scale, zp);
        }
    }

    auto __attribute__((hot)) DEQUANT8_KERNEL_IMPL(
      const std::uint8_t* const __restrict__ x,
      float* const __restrict__ o,
      const std::int64_t numel,
      const float scale,
      const std::int32_t zp,
      const reduce_op op
    ) noexcept -> void {
        switch (op) {
            case reduce_op::set: impl_namespace(QUANT8_KERNEL_IMPL, _)::dequant<reduce_op::set>(x, o, numel, scale, zp); break;
            case reduce_op::add: impl_namespace(QUANT8_KERNEL_IMPL, _)::dequant<reduce_op::add>(x, o, numel, scale, zp); break;
            default: panic("Invalid reduce_op");
        }
    }

    auto __attribute__((hot)) QUANT4_KERNEL_IMPL(
      const float* const __restrict__ x,
      std::uint8_t* const __restrict__ o,
      const std::int64_t numel,
      const float scale,
      const std::int32_t zp,
      const bool sto_rnd,
      prng_state& prng
    ) noexcept -> void {
        if (sto_rnd) {
            impl_namespace(QUANT4_KERNEL_IMPL, _)::stochastic(x, o, numel, scale, zp, prng);
        } else {
            impl_namespace(QUANT4_KERNEL_IMPL, _)::nearest(x, o, numel, scale, zp);
        }
    }

    auto __attribute__((hot)) DEQUANT4_KERNEL_IMPL(
        const std::uint8_t* const __restrict__ x,
        float* const __restrict__ o,
        const std::int64_t numel,
        const float scale,
        const std::int32_t zp,
        const reduce_op op
    ) noexcept -> void {
        switch (op) {
            case reduce_op::set: impl_namespace(QUANT4_KERNEL_IMPL, _)::dequant<reduce_op::set>(x, o, numel, scale, zp); break;
            case reduce_op::add: impl_namespace(QUANT4_KERNEL_IMPL, _)::dequant<reduce_op::add>(x, o, numel, scale, zp); break;
            default: panic("Invalid reduce_op");
        }
    }
}
