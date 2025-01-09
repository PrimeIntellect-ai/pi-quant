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
  const float* const x,
  std::uint8_t* const o,
  const std::size_t n,
  const float inv_scale,
  const std::int32_t zero_point
) noexcept -> void {
    std::size_t i {};
    #if 0
        __m128 vinv_scale {_mm_set1_ps(inv_scale)};
        __m128i vzero_point {_mm_set1_epi32(zero_point)};
        __m128i vmax {_mm_set1_epi8(0xff)};
        constexpr int k_round_mode = _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC; // Round to nearest, and suppress exceptions
        for (; i+3 < n; i += 4) {
            __m128 xi = _mm_loadu_ps(x+i);
            __m128i yi = _mm_cvtps_epi32(_mm_round_ps(_mm_mul_ps(xi, vinv_scale), k_round_mode));
            __m128i zi = _mm_add_epi32(yi, vzero_point);
            __m128i i8 = _mm_packus_epi32(zi, _mm_setzero_si128());
            __m128i wi = _mm_max_epi8(_mm_min_epi8(zi, vmax), _mm_setzero_si128());
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o+i), wi);
        }
    #endif
    for (; i < n; ++i) {
        o[i] = static_cast<std::uint8_t>(std::clamp(static_cast<std::int32_t>(std::round(x[i] * inv_scale)) + zero_point, 0, 0xff));
    }
}