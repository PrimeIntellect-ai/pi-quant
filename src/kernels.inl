#ifndef Q8_KERNEL_IMPL
#error "Q8 impl is not defined"
#endif
#ifndef Q4_KERNEL_IMPL
#error "Q4 impl is not defined"
#endif

#include <quant.hpp>


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

#define concat(a, b) a ## b
#define impl_namespace(a, b) concat(a, _impl)

namespace impl_namespace(Q8_KERNEL_IMPL, _) {
    static auto __attribute__((hot)) nearest(
        const float* const __restrict__ x,
        std::uint8_t* const __restrict__ o,
        const std::int64_t numel,
        const float inv_scale,
        const std::int32_t zp
   ) noexcept -> void {
        std::int64_t i {};
        #if defined(__AVX512F__) && defined(__AVX512BW__) && 0
            const __m512 vinv_scale = _mm512_set1_ps(inv_scale);
            const __m512i vzero_point = _mm512_set1_epi32(zp);
            const __m512i vmin = _mm512_setzero_si512();
            const __m512i vmax = _mm512_set1_epi32(0xff);
            constexpr int k_round_mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
            constexpr std::size_t step = 64;
            for (; i+step <= numel; i += step) {
                __m512 xf0 = _mm512_loadu_ps(x+i+(0<<4));
                __m512 xf1 = _mm512_loadu_ps(x+i+(1<<4));
                __m512 xf2 = _mm512_loadu_ps(x+i+(2<<4));
                __m512 xf3 = _mm512_loadu_ps(x+i+(3<<4));
                __m512i xi0 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(xf0, vinv_scale), k_round_mode)), vzero_point)));
                __m512i xi1 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(xf1, vinv_scale), k_round_mode)), vzero_point)));
                __m512i xi2 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(xf2, vinv_scale), k_round_mode)), vzero_point)));
                __m512i xi3 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(xf3, vinv_scale), k_round_mode)),vzero_point)));
                __m512i pack16_0 = _mm512_packus_epi32(xi0, xi1);
                __m512i pack16_1 = _mm512_packus_epi32(xi2, xi3);
                __m512i result = _mm512_packus_epi16(pack16_0, pack16_1);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(o+i), result);
            }
        #elif defined(__AVX2__)
            const __m256 vinv_scale = _mm256_set1_ps(inv_scale);
            const __m256i vzero_point = _mm256_set1_epi32(zp);
            const __m256i vmin = _mm256_setzero_si256();
            const __m256i vmax = _mm256_set1_epi32(0xff);
            constexpr int k_round_mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
            constexpr std::size_t step = 32;
            for (; i+step <= numel; i += step) {
                __m256 xf0 = _mm256_loadu_ps(x+i+(0<<3));
                __m256 xf1 = _mm256_loadu_ps(x+i+(1<<3));
                __m256 xf2 = _mm256_loadu_ps(x+i+(2<<3));
                __m256 xf3 = _mm256_loadu_ps(x+i+(3<<3));
                __m256i xi0 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(xf0, vinv_scale), k_round_mode)), vzero_point)));
                __m256i xi1 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(xf1, vinv_scale), k_round_mode)), vzero_point)));
                __m256i xi2 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(xf2, vinv_scale), k_round_mode)), vzero_point)));
                __m256i xi3 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(xf3, vinv_scale), k_round_mode)), vzero_point)));
                __m256i pack16_0 = _mm256_packus_epi32(xi0, xi1);
                __m256i pack16_1 = _mm256_packus_epi32(xi2, xi3);
                __m256i result = _mm256_packus_epi16(pack16_0, pack16_1);
                result = _mm256_permute4x64_epi64(result, 0xD8);
                static const __m256i shuffle_mask = _mm256_setr_epi8(
                    0,  1,  2,  3,   8,  9, 10, 11,   4,  5,  6,  7,  12, 13, 14, 15,
                    0,  1,  2,  3,   8,  9, 10, 11,   4,  5,  6,  7,  12, 13, 14, 15
                );
                __m256i final = _mm256_shuffle_epi8(result, shuffle_mask);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(o+i), final);
            }
        #elif defined(__SSE4_2__)
            const __m128 vinv_scale = _mm_set1_ps(inv_scale);
            const __m128i vzero_point = _mm_set1_epi32(zp);
            const __m128i vmin = _mm_setzero_si128();
            const __m128i vmax = _mm_set1_epi32(0xff);
            constexpr int k_round_mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
            constexpr std::size_t step = 16;
            for (; i+step <= numel; i += step) {
                __m128 xf0 = _mm_loadu_ps(x+i+(0<<2));
                __m128 xf1 = _mm_loadu_ps(x+i+(1<<2));
                __m128 xf2 = _mm_loadu_ps(x+i+(2<<2));
                __m128 xf3 = _mm_loadu_ps(x+i+(3<<2));
                __m128i xi0 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(_mm_cvtps_epi32(_mm_round_ps(_mm_mul_ps(xf0, vinv_scale), k_round_mode)), vzero_point)));
                __m128i xi1 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(_mm_cvtps_epi32(_mm_round_ps(_mm_mul_ps(xf1, vinv_scale), k_round_mode)), vzero_point)));
                __m128i xi2 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(_mm_cvtps_epi32(_mm_round_ps(_mm_mul_ps(xf2, vinv_scale), k_round_mode)), vzero_point)));
                __m128i xi3 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(_mm_cvtps_epi32(_mm_round_ps(_mm_mul_ps(xf3, vinv_scale), k_round_mode)), vzero_point)));
                __m128i pack16_0 = _mm_packus_epi32(xi0, xi1);
                __m128i pack16_1 = _mm_packus_epi32(xi2, xi3);
                __m128i result = _mm_packus_epi16(pack16_0, pack16_1);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(o+i), result);
            }
        #elif defined(__aarch64__) && defined(__ARM_NEON__)
            const float32x4_t vinv_scale = vdupq_n_f32(inv_scale);
            const int32x4_t vzero_point = vdupq_n_s32(zp);
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
                int32x4_t xi0 = vcvtnq_s32_f32(xf0);
                int32x4_t xi1 = vcvtnq_s32_f32(xf1);
                int32x4_t xi2 = vcvtnq_s32_f32(xf2);
                int32x4_t xi3 = vcvtnq_s32_f32(xf3);
                xi0 = vaddq_s32(xi0, vzero_point);
                xi1 = vaddq_s32(xi1, vzero_point);
                xi2 = vaddq_s32(xi2, vzero_point);
                xi3 = vaddq_s32(xi3, vzero_point);
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
            const float rnd {std::round(x[i] * inv_scale)};
            const std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
    }

    static auto __attribute__((hot)) stochastic(
        const float* const __restrict__ x,
        std::uint8_t* const __restrict__ o,
        const std::int64_t numel,
        const float inv_scale,
        const std::int32_t zp,
        [[maybe_unused]] quant::prng_state& prng
    ) noexcept -> void {
        std::int64_t i {};
        for (; i < numel; ++i) {
            float rnd {x[i] * inv_scale};
            const float dec {std::abs(rnd - std::trunc(rnd))};
            const float xi {prng.gen_canonical()};
            float adj {xi < dec ? 1.0f : 0.0f};
            if (rnd < 0.0f) adj = -1.0f * adj;
            rnd = std::trunc(rnd) + adj;
            const std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
    }
};

namespace impl_namespace(Q4_KERNEL_IMPL, _) {
    static auto __attribute__((hot)) nearest(
        const float* const __restrict__ x,
        std::uint8_t* const __restrict__ o,
        std::int64_t numel,
        const float inv_scale,
        const std::int32_t zp
    ) noexcept -> void {
        numel = (numel + 1) / 2;
        const auto f = [=](float x) noexcept -> std::uint8_t {
            return std::clamp<int>(std::round(x * inv_scale) + zp, 0, 0xf);
        };
        for (std::size_t i{0}; i < numel; ++i) {
            std::uint8_t hi = f(x[2 * i])     & 15;
            std::uint8_t lo = f(x[2 * i + 1]) & 15;
            o[i] = (hi << 4) | lo;
        }
    }

    static auto __attribute__((hot)) stochastic(
        const float* const __restrict__ x,
        std::uint8_t* const __restrict__ o,
        const std::int64_t numel,
        const float inv_scale,
        const std::int32_t zp,
        [[maybe_unused]] quant::prng_state& prng
    ) noexcept -> void {
        std::int64_t i {};
        for (; i < numel; ++i) {
            float rnd {x[i] * inv_scale};
            const float dec {std::abs(rnd - std::trunc(rnd))};
            const float xi {prng.gen_canonical()};
            float adj {xi < dec ? 1.0f : 0.0f};
            if (rnd < 0.0f) adj = -1.0f * adj;
            rnd = std::trunc(rnd) + adj;
            const std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
    };
};

auto __attribute__((hot)) Q8_KERNEL_IMPL(
  const float* const __restrict__ x,
  std::uint8_t* const __restrict__ o,
  const std::int64_t numel,
  const float inv_scale,
  const std::int32_t zp,
  const bool sto_rnd,
  quant::prng_state& prng
) noexcept -> void {
    if (sto_rnd) impl_namespace(Q8_KERNEL_IMPL, _)::stochastic(x, o, numel, inv_scale, zp, prng);
    else impl_namespace(Q8_KERNEL_IMPL, _)::nearest(x, o, numel, inv_scale, zp);
}

auto __attribute__((hot)) Q4_KERNEL_IMPL(
  const float* const __restrict__ x,
  std::uint8_t* const __restrict__ o,
  const std::int64_t numel,
  const float inv_scale,
  const std::int32_t zp,
  const bool sto_rnd,
  quant::prng_state& prng
) noexcept -> void {
    if (sto_rnd) impl_namespace(Q4_KERNEL_IMPL, _)::stochastic(x, o, numel, inv_scale, zp, prng);
    else impl_namespace(Q4_KERNEL_IMPL, _)::nearest(x, o, numel, inv_scale, zp);
}
