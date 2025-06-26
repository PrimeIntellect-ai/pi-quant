// This inline file is directly included into the kernels.inl file, which is cloned (recompiled) in multiple compilation units for different CPU architectures.
// ! Make sure all functions are static, to make them local to the compilation unit.

#include "../piquant_internal.hpp"

using namespace piquant;

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

static auto PIQUANT_HOT quant_f32_to_uint8_nearest(
    const float* PIQUANT_RESTRICT x,
    std::uint8_t* PIQUANT_RESTRICT o,
    std::int64_t numel,
    float scale,
    std::int32_t zp
) noexcept -> void {
    scale = 1.0f / scale; /* We multiply by reciprocal */
    std::int64_t i {};
    #if defined(__AVX512F__) && defined(__AVX512BW__) && 0

    #elif defined(__AVX2__)
        __m256 vinv_scale {_mm256_set1_ps(scale)};
        __m256 vhalf {_mm256_set1_ps(0.5f)};
        __m256 vneg_half {_mm256_set1_ps(-0.5f)};
        __m256 vzero {_mm256_setzero_ps()};
        __m256i vzero_point {_mm256_set1_epi32(zp)};
        __m256i vmin {_mm256_setzero_si256()};
        __m256i vmax {_mm256_set1_epi32(0xff)};
        static const __m256i shuffle_matrix {_mm256_setr_epi8(
            0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15,
            0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15
        )};
        for (; i < numel && std::bit_cast<std::uintptr_t>(x+i)&31; ++i) {
            float rnd {std::round(x[i]*scale)};
            std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
        for (; i+31 < numel; i += 32) {
            __m256 xf0 {_mm256_castsi256_ps(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(x+i+(0<<3))))};
            __m256 xf1 {_mm256_castsi256_ps(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(x+i+(1<<3))))};
            __m256 xf2 {_mm256_castsi256_ps(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(x+i+(2<<3))))};
            __m256 xf3 {_mm256_castsi256_ps(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(x+i+(3<<3))))};
            __m256 prod0 {_mm256_mul_ps(xf0, vinv_scale)};
            __m256 prod1 {_mm256_mul_ps(xf1, vinv_scale)};
            __m256 prod2 {_mm256_mul_ps(xf2, vinv_scale)};
            __m256 prod3 {_mm256_mul_ps(xf3, vinv_scale)};
            __m256 adj0 {_mm256_add_ps(prod0, _mm256_blendv_ps(vneg_half, vhalf, _mm256_cmp_ps(prod0, vzero, _CMP_GE_OQ)))};
            __m256 adj1 {_mm256_add_ps(prod1, _mm256_blendv_ps(vneg_half, vhalf, _mm256_cmp_ps(prod1, vzero, _CMP_GE_OQ)))};
            __m256 adj2 {_mm256_add_ps(prod2, _mm256_blendv_ps(vneg_half, vhalf, _mm256_cmp_ps(prod2, vzero, _CMP_GE_OQ)))};
            __m256 adj3 {_mm256_add_ps(prod3, _mm256_blendv_ps(vneg_half, vhalf, _mm256_cmp_ps(prod3, vzero, _CMP_GE_OQ)))};
            __m256i xi0 {_mm256_add_epi32(_mm256_cvttps_epi32(adj0), vzero_point)};
            __m256i xi1 {_mm256_add_epi32(_mm256_cvttps_epi32(adj1), vzero_point)};
            __m256i xi2 {_mm256_add_epi32(_mm256_cvttps_epi32(adj2), vzero_point)};
            __m256i xi3 {_mm256_add_epi32(_mm256_cvttps_epi32(adj3), vzero_point)};
            xi0 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, xi0));
            xi1 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, xi1));
            xi2 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, xi2));
            xi3 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, xi3));
            __m256i packed {_mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_packus_epi32(xi0, xi1), _mm256_packus_epi32(xi2, xi3)), 0xd8)};
            __m256i shuffled {_mm256_shuffle_epi8(packed, shuffle_matrix)};
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(o+i), shuffled);
        }
    #elif defined(__SSE4_2__)
        __m128 vinv_scale {_mm_set1_ps(scale)};
        __m128 vhalf {_mm_set1_ps(0.5f)};
        __m128 vneg_half {_mm_set1_ps(-0.5f)};
        __m128 vzero {_mm_setzero_ps()};
        __m128i vzero_point {_mm_set1_epi32(zp)};
        __m128i vmin {_mm_setzero_si128()};
        __m128i vmax {_mm_set1_epi32(0xff)};
        for (; i < numel && std::bit_cast<std::uintptr_t>(x+i)&15; ++i) {
            float rnd {std::round(x[i]*scale)};
            std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
        for (; i+15 < numel; i += 16) {
            __m128 xf0 {_mm_castsi128_ps(_mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<float*>(x+i+(0<<2)))))};
            __m128 xf1 {_mm_castsi128_ps(_mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<float*>(x+i+(1<<2)))))};
            __m128 xf2 {_mm_castsi128_ps(_mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<float*>(x+i+(2<<2)))))};
            __m128 xf3 {_mm_castsi128_ps(_mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<float*>(x+i+(3<<2)))))};
            xf0 = _mm_mul_ps(xf0, vinv_scale);
            xf1 = _mm_mul_ps(xf1, vinv_scale);
            xf2 = _mm_mul_ps(xf2, vinv_scale);
            xf3 = _mm_mul_ps(xf3, vinv_scale);
            __m128 mask0 {_mm_cmpge_ps(xf0, vzero)};
            __m128 offs0 {_mm_blendv_ps(vneg_half, vhalf, mask0)};
            __m128 adj0 {_mm_add_ps(xf0, offs0)};
            __m128 mask1 {_mm_cmpge_ps(xf1, vzero)};
            __m128 offs1 {_mm_blendv_ps(vneg_half, vhalf, mask1)};
            __m128 adj1 {_mm_add_ps(xf1, offs1)};
            __m128 mask2 {_mm_cmpge_ps(xf2, vzero)};
            __m128 offs2 {_mm_blendv_ps(vneg_half, vhalf, mask2)};
            __m128 adj2 {_mm_add_ps(xf2, offs2)};
            __m128 mask3 {_mm_cmpge_ps(xf3, vzero)};
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
        float32x4_t vinv_scale {vdupq_n_f32(scale)};
        int32x4_t vzero_point {vdupq_n_s32(zp)};
        int32x4_t vmin {vdupq_n_s32(0)};
        int32x4_t vmax {vdupq_n_s32(0xff)};
        for (; i+15 < numel; i += 16) {
            float32x4_t xf0 {vld1q_f32(x+i+(0<<2))};
            float32x4_t xf1 {vld1q_f32(x+i+(1<<2))};
            float32x4_t xf2 {vld1q_f32(x+i+(2<<2))};
            float32x4_t xf3 {vld1q_f32(x+i+(3<<2))};
            xf0 = vmulq_f32(xf0, vinv_scale);
            xf1 = vmulq_f32(xf1, vinv_scale);
            xf2 = vmulq_f32(xf2, vinv_scale);
            xf3 = vmulq_f32(xf3, vinv_scale);
            int32x4_t xi0 {vcvtaq_s32_f32(xf0)};
            int32x4_t xi1 {vcvtaq_s32_f32(xf1)};
            int32x4_t xi2 {vcvtaq_s32_f32(xf2)};
            int32x4_t xi3 {vcvtaq_s32_f32(xf3)};
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
            int16x8_t pack16_0 {vcombine_s16(vqmovn_s32(xi0), vqmovn_s32(xi1))};
            int16x8_t pack16_1 {vcombine_s16(vqmovn_s32(xi2), vqmovn_s32(xi3))};
            uint8x16_t result {vcombine_u8(vqmovun_s16(pack16_0), vqmovun_s16(pack16_1))};
            vst1q_u8(o+i, result);
        }
    #endif
    for (; i < numel; ++i) {
        float rnd {std::round(x[i] * scale)};
        std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
        o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
    }
}

template <const bool SUM>
static auto PIQUANT_HOT dequant_uint8_to_f32(
    const std::uint8_t* PIQUANT_RESTRICT x,
    float* PIQUANT_RESTRICT o,
    std::int64_t numel,
    float scale,
    std::int32_t zp
) noexcept -> void {
    std::int64_t i {};
    #if defined(__AVX512F__) && defined(__AVX512BW__) && 0

    #elif defined(__AVX2__)
        __m256i vzp256    {_mm256_set1_epi32(zp)};
        __m256  vscale256 {_mm256_set1_ps(scale)};
        static constexpr auto expand_u8_to_s32_avx2_fast = [](const __m256i &v) {
            __m256i zero {_mm256_setzero_si256()};
            __m256i w_lo {_mm256_unpacklo_epi8(v, zero)};
            __m256i w_hi {_mm256_unpackhi_epi8(v, zero)};
            __m256i d0 {_mm256_unpacklo_epi16(w_lo, zero)};
            __m256i d1 {_mm256_unpackhi_epi16(w_lo, zero)};
            __m256i d2 {_mm256_unpacklo_epi16(w_hi, zero)};
            __m256i d3 {_mm256_unpackhi_epi16(w_hi, zero)};
            return std::array{d0,d1,d2,d3};
        };
        static constexpr auto expand_u8_to_s32_avx2 = [](const __m256i &v) {
            __m128i lo128 = _mm256_castsi256_si128(v);
            __m128i hi128 = _mm256_extracti128_si256(v, 1);
            __m256i a0 = _mm256_cvtepu8_epi32(lo128);
            __m256i a1 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo128, 8));
            __m256i b0 = _mm256_cvtepu8_epi32(hi128);
            __m256i b1 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi128, 8));
            return std::array{a0, a1, b0, b1};
        };
        for (; i+63 < numel; i += 64) {
            __m256i in0 {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+i+(0<<5)))};
            __m256i in1 {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+i+(1<<5)))};
            auto [vs00, vs10, vs20, vs30] {expand_u8_to_s32_avx2(in0)};
            auto [vs01, vs11, vs21, vs31] {expand_u8_to_s32_avx2(in1)};
            vs00 = _mm256_sub_epi32(vs00, vzp256);
            vs10 = _mm256_sub_epi32(vs10, vzp256);
            vs20 = _mm256_sub_epi32(vs20, vzp256);
            vs30 = _mm256_sub_epi32(vs30, vzp256);
            vs01 = _mm256_sub_epi32(vs01, vzp256);
            vs11 = _mm256_sub_epi32(vs11, vzp256);
            vs21 = _mm256_sub_epi32(vs21, vzp256);
            vs31 = _mm256_sub_epi32(vs31, vzp256);
            __m256 vf00 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs00), vscale256)};
            __m256 vf10 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs10), vscale256)};
            __m256 vf20 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs20), vscale256)};
            __m256 vf30 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs30), vscale256)};
            __m256 vf01 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs01), vscale256)};
            __m256 vf11 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs11), vscale256)};
            __m256 vf21 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs21), vscale256)};
            __m256 vf31 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs31), vscale256)};
            if constexpr (SUM) {
                vf00 = _mm256_add_ps(vf00, _mm256_loadu_ps(o+i+(0<<3)));
                vf10 = _mm256_add_ps(vf10, _mm256_loadu_ps(o+i+(1<<3)));
                vf20 = _mm256_add_ps(vf20, _mm256_loadu_ps(o+i+(2<<3)));
                vf30 = _mm256_add_ps(vf30, _mm256_loadu_ps(o+i+(3<<3)));
                vf01 = _mm256_add_ps(vf01, _mm256_loadu_ps(o+i+(4<<3)));
                vf11 = _mm256_add_ps(vf11, _mm256_loadu_ps(o+i+(5<<3)));
                vf21 = _mm256_add_ps(vf21, _mm256_loadu_ps(o+i+(6<<3)));
                vf31 = _mm256_add_ps(vf31, _mm256_loadu_ps(o+i+(7<<3)));
            }
            _mm256_storeu_ps(o+i+(0<<3), vf00);
            _mm256_storeu_ps(o+i+(1<<3), vf10);
            _mm256_storeu_ps(o+i+(2<<3), vf20);
            _mm256_storeu_ps(o+i+(3<<3), vf30);
            _mm256_storeu_ps(o+i+(4<<3), vf01);
            _mm256_storeu_ps(o+i+(5<<3), vf11);
            _mm256_storeu_ps(o+i+(6<<3), vf21);
            _mm256_storeu_ps(o+i+(7<<3), vf31);
        }
    #elif defined(__SSE4_2__)
        __m128i vzp {_mm_set1_epi32(zp)};
        __m128 vscale {_mm_set1_ps(scale)};
        for (; i+63 < numel; i += 64) {
            __m128i in0 {_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+i+(0<<4)))};
            __m128i in1 {_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+i+(1<<4)))};
            __m128i in2 {_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+i+(2<<4)))};
            __m128i in3 {_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+i+(3<<4)))};
            auto expand16 = [&](const __m128i &v) {
                __m128i zero {_mm_setzero_si128()};
                __m128i w_lo {_mm_unpacklo_epi8(v, zero)};
                __m128i w_hi {_mm_unpackhi_epi8(v, zero)};
                __m128i d0 {_mm_unpacklo_epi16(w_lo, zero)};
                __m128i d1 {_mm_unpackhi_epi16(w_lo, zero)};
                __m128i d2 {_mm_unpacklo_epi16(w_hi, zero)};
                __m128i d3 {_mm_unpackhi_epi16(w_hi, zero)};
                return std::array{d0,d1,d2,d3};
            };
            auto [vs00, vs10, vs20, vs30] = expand16(in0);
            auto [vs01, vs11, vs21, vs31] = expand16(in1);
            auto [vs02, vs12, vs22, vs32] = expand16(in2);
            auto [vs03, vs13, vs23, vs33] = expand16(in3);
            vs00 = _mm_sub_epi32(vs00, vzp);
            vs10 = _mm_sub_epi32(vs10, vzp);
            vs20 = _mm_sub_epi32(vs20, vzp);
            vs30 = _mm_sub_epi32(vs30, vzp);
            vs01 = _mm_sub_epi32(vs01, vzp);
            vs11 = _mm_sub_epi32(vs11, vzp);
            vs21 = _mm_sub_epi32(vs21, vzp);
            vs31 = _mm_sub_epi32(vs31, vzp);
            vs02 = _mm_sub_epi32(vs02, vzp);
            vs12 = _mm_sub_epi32(vs12, vzp);
            vs22 = _mm_sub_epi32(vs22, vzp);
            vs32 = _mm_sub_epi32(vs32, vzp);
            vs03 = _mm_sub_epi32(vs03, vzp);
            vs13 = _mm_sub_epi32(vs13, vzp);
            vs23 = _mm_sub_epi32(vs23, vzp);
            vs33 = _mm_sub_epi32(vs33, vzp);
            __m128 vf00 {_mm_mul_ps(_mm_cvtepi32_ps(vs00), vscale)};
            __m128 vf01 {_mm_mul_ps(_mm_cvtepi32_ps(vs10), vscale)};
            __m128 vf02 {_mm_mul_ps(_mm_cvtepi32_ps(vs20), vscale)};
            __m128 vf03 {_mm_mul_ps(_mm_cvtepi32_ps(vs30), vscale)};
            __m128 vf10 {_mm_mul_ps(_mm_cvtepi32_ps(vs01), vscale)};
            __m128 vf11 {_mm_mul_ps(_mm_cvtepi32_ps(vs11), vscale)};
            __m128 vf12 {_mm_mul_ps(_mm_cvtepi32_ps(vs21), vscale)};
            __m128 vf13 {_mm_mul_ps(_mm_cvtepi32_ps(vs31), vscale)};
            __m128 vf20 {_mm_mul_ps(_mm_cvtepi32_ps(vs02), vscale)};
            __m128 vf21 {_mm_mul_ps(_mm_cvtepi32_ps(vs12), vscale)};
            __m128 vf22 {_mm_mul_ps(_mm_cvtepi32_ps(vs22), vscale)};
            __m128 vf23 {_mm_mul_ps(_mm_cvtepi32_ps(vs32), vscale)};
            __m128 vf30 {_mm_mul_ps(_mm_cvtepi32_ps(vs03), vscale)};
            __m128 vf31 {_mm_mul_ps(_mm_cvtepi32_ps(vs13), vscale)};
            __m128 vf32 {_mm_mul_ps(_mm_cvtepi32_ps(vs23), vscale)};
            __m128 vf33 {_mm_mul_ps(_mm_cvtepi32_ps(vs33), vscale)};
            if constexpr (SUM) {
                vf00 = _mm_add_ps(vf00, _mm_loadu_ps(o+i+ ( 0<<2)));
                vf01 = _mm_add_ps(vf01, _mm_loadu_ps(o+i+ ( 1<<2)));
                vf02 = _mm_add_ps(vf02, _mm_loadu_ps(o+i+ ( 2<<2)));
                vf03 = _mm_add_ps(vf03, _mm_loadu_ps(o+i+ ( 3<<2)));
                vf10 = _mm_add_ps(vf10, _mm_loadu_ps(o+i+ ( 4<<2)));
                vf11 = _mm_add_ps(vf11, _mm_loadu_ps(o+i+ ( 5<<2)));
                vf12 = _mm_add_ps(vf12, _mm_loadu_ps(o+i+ ( 6<<2)));
                vf13 = _mm_add_ps(vf13, _mm_loadu_ps(o+i+ ( 7<<2)));
                vf20 = _mm_add_ps(vf20, _mm_loadu_ps(o+i+ ( 8<<2)));
                vf21 = _mm_add_ps(vf21, _mm_loadu_ps(o+i+ ( 9<<2)));
                vf22 = _mm_add_ps(vf22, _mm_loadu_ps(o+i+ (10<<2)));
                vf23 = _mm_add_ps(vf23, _mm_loadu_ps(o+i+ (11<<2)));
                vf30 = _mm_add_ps(vf30, _mm_loadu_ps(o+i+ (12<<2)));
                vf31 = _mm_add_ps(vf31, _mm_loadu_ps(o+i+ (13<<2)));
                vf32 = _mm_add_ps(vf32, _mm_loadu_ps(o+i+ (14<<2)));
                vf33 = _mm_add_ps(vf33, _mm_loadu_ps(o+i+ (15<<2)));
            }
            _mm_storeu_ps(o+i+( 0<<2), vf00);
            _mm_storeu_ps(o+i+( 1<<2), vf01);
            _mm_storeu_ps(o+i+( 2<<2), vf02);
            _mm_storeu_ps(o+i+( 3<<2), vf03);
            _mm_storeu_ps(o+i+( 4<<2), vf10);
            _mm_storeu_ps(o+i+( 5<<2), vf11);
            _mm_storeu_ps(o+i+( 6<<2), vf12);
            _mm_storeu_ps(o+i+( 7<<2), vf13);
            _mm_storeu_ps(o+i+( 8<<2), vf20);
            _mm_storeu_ps(o+i+( 9<<2), vf21);
            _mm_storeu_ps(o+i+(10<<2), vf22);
            _mm_storeu_ps(o+i+(11<<2), vf23);
            _mm_storeu_ps(o+i+(12<<2), vf30);
            _mm_storeu_ps(o+i+(13<<2), vf31);
            _mm_storeu_ps(o+i+(14<<2), vf32);
            _mm_storeu_ps(o+i+(15<<2), vf33);
        }
    #elif defined(__aarch64__) && defined(__ARM_NEON__)
        static constexpr auto expand_u8_to_s32{[](uint8x16_t u8) noexcept-> std::array<int32x4_t, 4> {
            uint16x8_t u16l {vmovl_u8(vget_low_u8(u8))};
            uint16x8_t u16h {vmovl_u8(vget_high_u8(u8))};
            uint16x4_t a0 {vget_low_u16(u16l)};
            uint16x4_t a1 {vget_high_u16(u16l)};
            uint16x4_t a2 {vget_low_u16(u16h)};
            uint16x4_t a3 {vget_high_u16(u16h)};
            int32x4_t vs0 {vmovl_s16(vreinterpret_s16_u16(a0))};
            int32x4_t vs1 {vmovl_s16(vreinterpret_s16_u16(a1))};
            int32x4_t vs2 {vmovl_s16(vreinterpret_s16_u16(a2))};
            int32x4_t vs3 {vmovl_s16(vreinterpret_s16_u16(a3))};
            return {vs0, vs1, vs2, vs3};
        }};
        int32x4_t vzp {vdupq_n_s32(zp)};
        float32x4_t vscale {vdupq_n_f32(scale)};
        for (; i+63 < numel; i += 64) {
            uint8x16_t u80 {vld1q_u8(x+i+(0<<4))};
            uint8x16_t u81 {vld1q_u8(x+i+(1<<4))};
            uint8x16_t u82 {vld1q_u8(x+i+(2<<4))};
            uint8x16_t u83 {vld1q_u8(x+i+(3<<4))};
            auto [vs00, vs10, vs20, vs30] {expand_u8_to_s32(u80)};
            auto [vs01, vs11, vs21, vs31] {expand_u8_to_s32(u81)};
            auto [vs02, vs12, vs22, vs32] {expand_u8_to_s32(u82)};
            auto [vs03, vs13, vs23, vs33] {expand_u8_to_s32(u83)};
            vs00 = vsubq_s32(vs00, vzp);
            vs10 = vsubq_s32(vs10, vzp);
            vs20 = vsubq_s32(vs20, vzp);
            vs30 = vsubq_s32(vs30, vzp);
            vs01 = vsubq_s32(vs01, vzp);
            vs11 = vsubq_s32(vs11, vzp);
            vs21 = vsubq_s32(vs21, vzp);
            vs31 = vsubq_s32(vs31, vzp);
            vs02 = vsubq_s32(vs02, vzp);
            vs12 = vsubq_s32(vs12, vzp);
            vs22 = vsubq_s32(vs22, vzp);
            vs32 = vsubq_s32(vs32, vzp);
            vs03 = vsubq_s32(vs03, vzp);
            vs13 = vsubq_s32(vs13, vzp);
            vs23 = vsubq_s32(vs23, vzp);
            vs33 = vsubq_s32(vs33, vzp);
            float32x4_t vf00 {vmulq_f32(vcvtq_f32_s32(vs00), vscale)};
            float32x4_t vf01 {vmulq_f32(vcvtq_f32_s32(vs10), vscale)};
            float32x4_t vf02 {vmulq_f32(vcvtq_f32_s32(vs20), vscale)};
            float32x4_t vf03 {vmulq_f32(vcvtq_f32_s32(vs30), vscale)};
            float32x4_t vf10 {vmulq_f32(vcvtq_f32_s32(vs01), vscale)};
            float32x4_t vf11 {vmulq_f32(vcvtq_f32_s32(vs11), vscale)};
            float32x4_t vf12 {vmulq_f32(vcvtq_f32_s32(vs21), vscale)};
            float32x4_t vf13 {vmulq_f32(vcvtq_f32_s32(vs31), vscale)};
            float32x4_t vf20 {vmulq_f32(vcvtq_f32_s32(vs02), vscale)};
            float32x4_t vf21 {vmulq_f32(vcvtq_f32_s32(vs12), vscale)};
            float32x4_t vf22 {vmulq_f32(vcvtq_f32_s32(vs22), vscale)};
            float32x4_t vf23 {vmulq_f32(vcvtq_f32_s32(vs32), vscale)};
            float32x4_t vf30 {vmulq_f32(vcvtq_f32_s32(vs03), vscale)};
            float32x4_t vf31 {vmulq_f32(vcvtq_f32_s32(vs13), vscale)};
            float32x4_t vf32 {vmulq_f32(vcvtq_f32_s32(vs23), vscale)};
            float32x4_t vf33 {vmulq_f32(vcvtq_f32_s32(vs33), vscale)};
            if constexpr (SUM) {
                vf00 = vaddq_f32(vf00, vld1q_f32(o+i+(0<<2)));
                vf01 = vaddq_f32(vf01, vld1q_f32(o+i+(1<<2)));
                vf02 = vaddq_f32(vf02, vld1q_f32(o+i+(2<<2)));
                vf03 = vaddq_f32(vf03, vld1q_f32(o+i+(3<<2)));
                vf10 = vaddq_f32(vf10, vld1q_f32(o+i+(4<<2)));
                vf11 = vaddq_f32(vf11, vld1q_f32(o+i+(5<<2)));
                vf12 = vaddq_f32(vf12, vld1q_f32(o+i+(6<<2)));
                vf13 = vaddq_f32(vf13, vld1q_f32(o+i+(7<<2)));
                vf20 = vaddq_f32(vf20, vld1q_f32(o+i+(8<<2)));
                vf21 = vaddq_f32(vf21, vld1q_f32(o+i+(9<<2)));
                vf22 = vaddq_f32(vf22, vld1q_f32(o+i+(10<<2)));
                vf23 = vaddq_f32(vf23, vld1q_f32(o+i+(11<<2)));
                vf30 = vaddq_f32(vf30, vld1q_f32(o+i+(12<<2)));
                vf31 = vaddq_f32(vf31, vld1q_f32(o+i+(13<<2)));
                vf32 = vaddq_f32(vf32, vld1q_f32(o+i+(14<<2)));
                vf33 = vaddq_f32(vf33, vld1q_f32(o+i+(15<<2)));
            }
            vst1q_f32((o+i+(0<<2)), vf00);
            vst1q_f32((o+i+(1<<2)), vf01);
            vst1q_f32((o+i+(2<<2)), vf02);
            vst1q_f32((o+i+(3<<2)), vf03);
            vst1q_f32((o+i+(4<<2)), vf10);
            vst1q_f32((o+i+(5<<2)), vf11);
            vst1q_f32((o+i+(6<<2)), vf12);
            vst1q_f32((o+i+(7<<2)), vf13);
            vst1q_f32((o+i+(8<<2)), vf20);
            vst1q_f32((o+i+(9<<2)), vf21);
            vst1q_f32((o+i+(10<<2)), vf22);
            vst1q_f32((o+i+(11<<2)), vf23);
            vst1q_f32((o+i+(12<<2)), vf30);
            vst1q_f32((o+i+(13<<2)), vf31);
            vst1q_f32((o+i+(14<<2)), vf32);
            vst1q_f32((o+i+(15<<2)), vf33);
        }
    #endif
    for (; i < numel; ++i) {
        float dq {static_cast<float>(static_cast<std::int32_t>(x[i]) - zp)*scale};
        if constexpr (SUM) o[i] += dq;
        else o[i] = dq;
    }
}
