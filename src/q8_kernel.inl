#ifndef Q8_KERNEL_IMPL
#error "impl is not defined"
#endif

#include <algorithm>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

static auto __attribute__((hot)) Q8_KERNEL_IMPL(
  const float* const __restrict__ x,
  std::uint8_t* const __restrict__ o,
  const std::size_t n,
  const float inv_scale,
  const std::int32_t zero_point
) noexcept -> void {
    std::size_t i {};
    #ifdef __SSE4_1__
        const __m128 vinv_scale = _mm_set1_ps(inv_scale);
        const __m128i vzero_point = _mm_set1_epi32(zero_point);
        constexpr int k_round_mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
        constexpr std::size_t step = 16;
        for (; i + step <= n; i += step) {
            __m128 xf0 = _mm_loadu_ps(x+i+(0<<2));
            __m128 xf1 = _mm_loadu_ps(x+i+(1<<2));
            __m128 xf2 = _mm_loadu_ps(x+i+(2<<2));
            __m128 xf3 = _mm_loadu_ps(x+i+(3<<2));
            __m128i xi0 = _mm_add_epi32(_mm_cvtps_epi32(_mm_round_ps(_mm_mul_ps(xf0, vinv_scale), k_round_mode)), vzero_point);
            __m128i xi1 = _mm_add_epi32(_mm_cvtps_epi32(_mm_round_ps(_mm_mul_ps(xf1, vinv_scale), k_round_mode)), vzero_point);
            __m128i xi2 = _mm_add_epi32(_mm_cvtps_epi32(_mm_round_ps(_mm_mul_ps(xf2, vinv_scale), k_round_mode)), vzero_point);
            __m128i xi3 = _mm_add_epi32(_mm_cvtps_epi32(_mm_round_ps(_mm_mul_ps(xf3, vinv_scale), k_round_mode)),vzero_point);
            __m128i pack16_0 = _mm_packus_epi32(xi0, xi1);
            __m128i pack16_1 = _mm_packus_epi32(xi2, xi3);
            __m128i result = _mm_packus_epi16(pack16_0, pack16_1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o+i), result);
        }
    #endif
    for (; i < n; ++i) {
        o[i] = static_cast<std::uint8_t>(std::clamp(static_cast<std::int32_t>(std::round(x[i] * inv_scale)) + zero_point, 0, 0xff));
    }
}