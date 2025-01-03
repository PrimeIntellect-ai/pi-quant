#ifndef impl
#error "impl is not defined"
#endif
#define Q8_KERNEL_IMPL(I) f32_q8_kernel_##I

#include <cassert>
#include <algorithm>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

static auto __attribute__((flatten)) Q8_KERNEL_IMPL(impl)(
  const float* const x,
  std::uint8_t* const o,
  const std::size_t n,
  const float inv_scale,
  const std::int32_t zero_point
) noexcept -> void {
    std::size_t i {};
    #ifdef __SSE2__
        const __m128 xmm0 {_mm_set1_ps(inv_scale)};
        const __m128i xmm1 {_mm_set1_epi32(zero_point)};
        const __m128i xmm2 {_mm_set1_epi32(0xff)};
        constexpr int k_round_mode {_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC}; // Round to nearest, and suppress exceptions
        for (; i+3 < n; i += 4) {
          const __m128 xi = _mm_loadu_ps(x+i);
          const __m128i yi = _mm_cvtps_epi32(_mm_round_ps(_mm_mul_ps(xi, xmm0), k_round_mode));
          const __m128i zi = _mm_add_epi32(yi, xmm1);
          const __m128i wi = _mm_max_epi32(_mm_min_epi32(zi, xmm2), _mm_setzero_si128());
          _mm_storeu_si128(reinterpret_cast<__m128i*>(o+i), wi);
        }
    #endif
    for (; i < n; ++i) {
        o[i] = static_cast<std::uint8_t>(std::clamp(static_cast<std::int32_t>(std::round(x[i] * inv_scale)) + zero_point, 0, 0xff));
    }
}