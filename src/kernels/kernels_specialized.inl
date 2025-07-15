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
    const fp32_t* PIQUANT_RESTRICT x,
    std::uint8_t* PIQUANT_RESTRICT o,
    std::int64_t numel,
    fp32_t scale,
    std::int32_t zp
) noexcept -> void {
    scale = 1.0f / scale;
    std::int64_t i {};
    #if defined(__AVX512F__) && defined(__AVX512BW__)
        __m512 vinv_scale {_mm512_set1_ps(scale)};
        __m512 vhalf {_mm512_set1_ps(0.5f)};
        __m512 vneg_half {_mm512_set1_ps(-0.5f)};
        __m512 vzero_ps {_mm512_setzero_ps()};
        __m512i vzero_point {_mm512_set1_epi32(zp)};
        for (; i < numel && ((std::bit_cast<std::uintptr_t>(o+i) & 15) != 0); ++i) {
            fp32_t r {std::round(x[i]*scale)};
            std::int32_t q32 {static_cast<std::int32_t>(r) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(q32, 0, 0xff));
        }
        for (; i+63 < numel; i += 64) {
            __m512 xf0 {_mm512_loadu_ps(x+i+0)};
            __m512 xf1 {_mm512_loadu_ps(x+i+16)};
            __m512 xf2 {_mm512_loadu_ps(x+i+32)};
            __m512 xf3 {_mm512_loadu_ps(x+i+48)};
            __m512 prod0 {_mm512_mul_ps(xf0, vinv_scale)};
            __m512 prod1 {_mm512_mul_ps(xf1, vinv_scale)};
            __m512 prod2 {_mm512_mul_ps(xf2, vinv_scale)};
            __m512 prod3 {_mm512_mul_ps(xf3, vinv_scale)};
            __m512 adj0 {_mm512_add_ps(prod0, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(prod0, vzero_ps, _CMP_GE_OQ), vneg_half, vhalf))};
            __m512 adj1 {_mm512_add_ps(prod1, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(prod1, vzero_ps, _CMP_GE_OQ), vneg_half, vhalf))};
            __m512 adj2 {_mm512_add_ps(prod2, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(prod2, vzero_ps, _CMP_GE_OQ), vneg_half, vhalf))};
            __m512 adj3 {_mm512_add_ps(prod3, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(prod3, vzero_ps, _CMP_GE_OQ), vneg_half, vhalf))};
            __m512i xi0 {_mm512_add_epi32(_mm512_cvttps_epi32(adj0), vzero_point)};
            __m512i xi1 {_mm512_add_epi32(_mm512_cvttps_epi32(adj1), vzero_point)};
            __m512i xi2 {_mm512_add_epi32(_mm512_cvttps_epi32(adj2), vzero_point)};
            __m512i xi3 {_mm512_add_epi32(_mm512_cvttps_epi32(adj3), vzero_point)};
            _mm_stream_si128(reinterpret_cast<__m128i*>(o+i+0), _mm512_cvtusepi32_epi8(xi0));
            _mm_stream_si128(reinterpret_cast<__m128i*>(o+i+16), _mm512_cvtusepi32_epi8(xi1));
            _mm_stream_si128(reinterpret_cast<__m128i*>(o+i+32), _mm512_cvtusepi32_epi8(xi2));
            _mm_stream_si128(reinterpret_cast<__m128i*>(o+i+48), _mm512_cvtusepi32_epi8(xi3));
        }
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
        for (; i < numel && std::bit_cast<std::uintptr_t>(o+i) & 31; ++i) {
            fp32_t rnd {std::round(x[i]*scale)};
            std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
        for (; i+31 < numel; i += 32) {
          __m256 xf0   {_mm256_loadu_ps(x+i+(0<<3))};
            __m256 xf1 {_mm256_loadu_ps(x+i+(1<<3))};
            __m256 xf2 {_mm256_loadu_ps(x+i+(2<<3))};
            __m256 xf3 {_mm256_loadu_ps(x+i+(3<<3))};
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
            _mm256_stream_si256(reinterpret_cast<__m256i*>(o+i), shuffled);
        }
    #elif defined(__SSE4_2__)
        __m128 vinv_scale {_mm_set1_ps(scale)};
        __m128 vhalf {_mm_set1_ps(0.5f)};
        __m128 vneg_half {_mm_set1_ps(-0.5f)};
        __m128 vzero {_mm_setzero_ps()};
        __m128i vzero_point {_mm_set1_epi32(zp)};
        __m128i vmin {_mm_setzero_si128()};
        __m128i vmax {_mm_set1_epi32(0xff)};
        for (; i < numel && std::bit_cast<std::uintptr_t>(o+i) & 15; ++i) {
            fp32_t rnd {std::round(x[i]*scale)};
            std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
        for (; i+15 < numel; i += 16) {
            __m128 xf0 {_mm_loadu_ps(x+i+(0<<2))};
            __m128 xf1 {_mm_loadu_ps(x+i+(1<<2))};
            __m128 xf2 {_mm_loadu_ps(x+i+(2<<2))};
            __m128 xf3 {_mm_loadu_ps(x+i+(3<<2))};
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
            _mm_stream_si128(reinterpret_cast<__m128i*>(o+i), result);
        }
    #elif defined(__aarch64__) && defined(__ARM_NEON__)
        float32x4_t vinv_scale {vdupq_n_f32(scale)};
        int32x4_t vzero_point {vdupq_n_s32(zp)};
        for (; i+15 < numel; i += 16) {
            int32x4_t xi00 {vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x+i+(0<<2)), vinv_scale))};
            int32x4_t xi01 {vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x+i+(1<<2)), vinv_scale))};
            int32x4_t xi02 {vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x+i+(2<<2)), vinv_scale))};
            int32x4_t xi03 {vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x+i+(3<<2)), vinv_scale))};
            xi00 = vaddq_s32(xi00, vzero_point);
            xi01 = vaddq_s32(xi01, vzero_point);
            xi02 = vaddq_s32(xi02, vzero_point);
            xi03 = vaddq_s32(xi03, vzero_point);
            int16x8_t pack16_00 {vcombine_s16(vqmovn_s32(xi00), vqmovn_s32(xi01))};
            int16x8_t pack16_01 {vcombine_s16(vqmovn_s32(xi02), vqmovn_s32(xi03))};
            uint8x16_t r0 {vcombine_u8(vqmovun_s16(pack16_00), vqmovun_s16(pack16_01))};
            vst1q_u8(o+i, r0);
        }
    #endif
    for (; i < numel; ++i) {
        fp32_t rnd {std::round(x[i] * scale)};
        std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
        o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
    }
}

template <const bool SUM>
static auto PIQUANT_HOT dequant_uint8_to_f32(
    const std::uint8_t* PIQUANT_RESTRICT x,
    fp32_t* PIQUANT_RESTRICT o,
    std::int64_t numel,
    fp32_t scale,
    std::int32_t zp
) noexcept -> void {
    std::int64_t i {};
    #if defined(__AVX512F__) && defined(__AVX512BW__)
        __m512i vzp {_mm512_set1_epi32(zp)};
        __m512  vscale {_mm512_set1_ps(scale)};
        for (; i+63 < numel; i += 64) {
            __m128i in0 {_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+i+ 0))};
            __m128i in1 {_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+i+16))};
            __m128i in2 {_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+i+32))};
            __m128i in3 {_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+i+48))};
            __m512i s0 {_mm512_sub_epi32(_mm512_cvtepu8_epi32(in0), vzp)};
            __m512i s1 {_mm512_sub_epi32(_mm512_cvtepu8_epi32(in1), vzp)};
            __m512i s2 {_mm512_sub_epi32(_mm512_cvtepu8_epi32(in2), vzp)};
            __m512i s3 {_mm512_sub_epi32(_mm512_cvtepu8_epi32(in3), vzp)};
            __m512 f0 {_mm512_mul_ps(_mm512_cvtepi32_ps(s0), vscale)};
            __m512 f1 {_mm512_mul_ps(_mm512_cvtepi32_ps(s1), vscale)};
            __m512 f2 {_mm512_mul_ps(_mm512_cvtepi32_ps(s2), vscale)};
            __m512 f3 {_mm512_mul_ps(_mm512_cvtepi32_ps(s3), vscale)};
            if constexpr (SUM) {
                f0 = _mm512_add_ps(f0, _mm512_loadu_ps(o+i+ 0));
                f1 = _mm512_add_ps(f1, _mm512_loadu_ps(o+i+16));
                f2 = _mm512_add_ps(f2, _mm512_loadu_ps(o+i+32));
                f3 = _mm512_add_ps(f3, _mm512_loadu_ps(o+i+48));
            }
            _mm512_storeu_ps(o+i+ 0, f0);
            _mm512_storeu_ps(o+i+16, f1);
            _mm512_storeu_ps(o+i+32, f2);
            _mm512_storeu_ps(o+i+48, f3);
        }
    #elif defined(__AVX2__)
        __m256i vzp256 {_mm256_set1_epi32(zp)};
        __m256 vscale256 {_mm256_set1_ps(scale)};
        static constexpr auto expand_u8_to_s32_avx2 = [](__m256i v) noexcept {
            __m128i lo128 {_mm256_castsi256_si128(v)};
            __m128i hi128 {_mm256_extracti128_si256(v, 1)};
            __m256i a0 {_mm256_cvtepu8_epi32(lo128)};
            __m256i a1 {_mm256_cvtepu8_epi32(_mm_srli_si128(lo128, 8))};
            __m256i b0 {_mm256_cvtepu8_epi32(hi128)};
            __m256i b1 {_mm256_cvtepu8_epi32(_mm_srli_si128(hi128, 8))};
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
            int32x4_t vs0 {vmovl_s16(vreinterpret_s16_u16(vget_low_u16(u16l)))};
            int32x4_t vs1 {vmovl_s16(vreinterpret_s16_u16(vget_high_u16(u16l)))};
            int32x4_t vs2 {vmovl_s16(vreinterpret_s16_u16(vget_low_u16(u16h)))};
            int32x4_t vs3 {vmovl_s16(vreinterpret_s16_u16(vget_high_u16(u16h)))};
            return {vs0, vs1, vs2, vs3};
        }};
        int32x4_t vzp {vdupq_n_s32(zp)};
        float32x4_t vscale {vdupq_n_f32(scale)};
        for (; i+15 < numel; i += 16) {
            uint8x16_t u80 {vld1q_u8(x+i+(0<<4))};
            auto [vs00, vs10, vs20, vs30] {expand_u8_to_s32(u80)};
            float32x4_t vf00 {vmulq_f32(vcvtq_f32_s32(vsubq_s32(vs00, vzp)), vscale)};
            float32x4_t vf01 {vmulq_f32(vcvtq_f32_s32(vsubq_s32(vs10, vzp)), vscale)};
            float32x4_t vf02 {vmulq_f32(vcvtq_f32_s32(vsubq_s32(vs20, vzp)), vscale)};
            float32x4_t vf03 {vmulq_f32(vcvtq_f32_s32(vsubq_s32(vs30, vzp)), vscale)};
            if constexpr (SUM) {
                vf00 = vaddq_f32(vf00, vld1q_f32(o+i+(0<<2)));
                vf01 = vaddq_f32(vf01, vld1q_f32(o+i+(1<<2)));
                vf02 = vaddq_f32(vf02, vld1q_f32(o+i+(2<<2)));
                vf03 = vaddq_f32(vf03, vld1q_f32(o+i+(3<<2)));
            }
            vst1q_f32((o+i+(0<<2)), vf00);
            vst1q_f32((o+i+(1<<2)), vf01);
            vst1q_f32((o+i+(2<<2)), vf02);
            vst1q_f32((o+i+(3<<2)), vf03);
        }
    #endif
    for (; i < numel; ++i) {
        fp32_t dq {static_cast<fp32_t>(static_cast<std::int32_t>(x[i]) - zp)*scale};
        if constexpr (SUM) o[i] += dq;
        else o[i] = dq;
    }
}

static auto PIQUANT_HOT find_min_max_f32(std::span<const fp32_t> in) noexcept -> std::array<fp32_t, 2> {
    const fp32_t* PIQUANT_RESTRICT x {in.data()};
    auto numel {static_cast<std::int64_t>(in.size())};
    std::int64_t i {};
    fp32_t min {std::numeric_limits<fp32_t>::max()};
    fp32_t max {std::numeric_limits<fp32_t>::lowest()};
    #if defined(__AVX512F__) && defined(__AVX512BW__)
        __m512 vmin {_mm512_set1_ps(min)};
        __m512 vmax {_mm512_set1_ps(max)};
        for (; i+63 < numel; i += 63) {
            __m512 v0 {_mm512_loadu_ps(x+i)};
            __m512 v1 {_mm512_loadu_ps(x+i+16)};
            __m512 v2 {_mm512_loadu_ps(x+i+32)};
            __m512 v3 {_mm512_loadu_ps(x+i+48)};
            vmin = _mm512_min_ps(vmin, v0);
            vmax = _mm512_max_ps(vmax, v0);
            vmin = _mm512_min_ps(vmin, v1);
            vmax = _mm512_max_ps(vmax, v1);
            vmin = _mm512_min_ps(vmin, v2);
            vmax = _mm512_max_ps(vmax, v2);
            vmin = _mm512_min_ps(vmin, v3);
            vmax = _mm512_max_ps(vmax, v3);
        }
        min = _mm512_reduce_min_ps(vmin);
        max = _mm512_reduce_max_ps(vmax);
    #elif defined(__AVX2__)
        __m256 vmin {_mm256_set1_ps(min)};
        __m256 vmax {_mm256_set1_ps(max)};
        for (; i+31 < numel; i += 32) {
            __m256 v0 {_mm256_loadu_ps(x+i)};
            __m256 v1 {_mm256_loadu_ps(x+i+8)};
            __m256 v2 {_mm256_loadu_ps(x+i+16)};
            __m256 v3 {_mm256_loadu_ps(x+i+24)};
            vmin = _mm256_min_ps(vmin, v0);
            vmax = _mm256_max_ps(vmax, v0);
            vmin = _mm256_min_ps(vmin, v1);
            vmax = _mm256_max_ps(vmax, v1);
            vmin = _mm256_min_ps(vmin, v2);
            vmax = _mm256_max_ps(vmax, v2);
            vmin = _mm256_min_ps(vmin, v3);
            vmax = _mm256_max_ps(vmax, v3);
        }
        __m128 lo {_mm256_castps256_ps128(vmin)};
        __m128 hi {_mm256_extractf128_ps(vmin, 1)};
        __m128 minima {_mm_min_ps(lo, hi)};
        minima = _mm_min_ps(minima, _mm_movehl_ps(minima, minima));
        minima = _mm_min_ps(minima, _mm_shuffle_ps(minima, minima, 0b01));
        min = _mm_cvtss_f32(minima);
        lo = _mm256_castps256_ps128(vmax);
        hi = _mm256_extractf128_ps(vmax, 1);
        __m128 maxima {_mm_max_ps(lo, hi)};
        maxima = _mm_max_ps(maxima, _mm_movehl_ps(maxima, maxima));
        maxima = _mm_max_ps(maxima, _mm_shuffle_ps(maxima, maxima, 0b01));
        max = _mm_cvtss_f32(maxima);
    #elif defined(__aarch64__) && defined(__ARM_NEON__)
        float32x4_t vmin {vdupq_n_f32(min)};
        float32x4_t vmax {vdupq_n_f32(max)};
        for (; i+15 < numel; i += 16) {
            float32x4_t v0 {vld1q_f32(x+i+(0<<2))};
            float32x4_t v1 {vld1q_f32(x+i+(1<<2))};
            float32x4_t v2 {vld1q_f32(x+i+(2<<2))};
            float32x4_t v3 {vld1q_f32(x+i+(3<<2))};
            vmin = vminq_f32(vmin, v0);
            vmax = vmaxq_f32(vmax, v0);
            vmin = vminq_f32(vmin, v1);
            vmax = vmaxq_f32(vmax, v1);
            vmin = vminq_f32(vmin, v2);
            vmax = vmaxq_f32(vmax, v2);
            vmin = vminq_f32(vmin, v3);
            vmax = vmaxq_f32(vmax, v3);
        }
        min = std::min({vgetq_lane_f32(vmin, 0), vgetq_lane_f32(vmin, 1), vgetq_lane_f32(vmin, 2), vgetq_lane_f32(vmin, 3)});
        max = std::max({vgetq_lane_f32(vmax, 0), vgetq_lane_f32(vmax, 1), vgetq_lane_f32(vmax, 2), vgetq_lane_f32(vmax, 3)});
    #endif
    for (; i < numel; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
    }
    return {min, max};
}

static auto PIQUANT_HOT find_min_max_bf16(std::span<const bfp16_t> in) noexcept -> std::array<fp32_t, 2> {
    const bfp16_t* PIQUANT_RESTRICT x {in.data()};
    auto numel {static_cast<std::int64_t>(in.size())};
    std::int64_t i {};
    fp32_t min {dtype_limits<fp32_t>::max};
    fp32_t max {dtype_limits<fp32_t>::min};
    #if defined(__AVX512F__) && defined(__AVX512BW__)

    #elif defined(__AVX2__)
    #elif defined(__aarch64__) && defined(__ARM_NEON__)

    #endif
    for (; i < numel; ++i) {
        auto xi {static_cast<fp32_t>(x[i])};
        if (xi < min) min = xi;
        if (xi > max) max = xi;
    }
    return {min, max};
}

static auto PIQUANT_HOT quant_f32_to_uint4_nearest(
    const fp32_t* PIQUANT_RESTRICT x,
    uint4_t* PIQUANT_RESTRICT o,
    std::int64_t numel,
    fp32_t scale,
    std::int32_t zp
) noexcept -> void {
    scale = 1.0f / scale;

    std::int64_t i {};
    #if defined(__AVX512F__) && defined(__AVX512BW__)
        __m512 vinv_scale {_mm512_set1_ps(scale)};
        __m512 vhalf {_mm512_set1_ps(0.5f)};
        __m512 vneg_half {_mm512_set1_ps(-0.5f)};
        __m512 vzero_ps {_mm512_setzero_ps()};
        __m512i vzp {_mm512_set1_epi32(zp)};
        __m512i vmin {_mm512_setzero_si512()};
        __m512i vmax {_mm512_set1_epi32(15)};
        __m128i shuf_even {_mm_setr_epi8(0,2,4,6,8,10,12,14,-1,-1,-1,-1,-1,-1,-1,-1)};
        __m128i shuf_odd {_mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-1,-1,-1,-1,-1,-1,-1)};
        for (; i+15 < numel; i += 16) {
            __m512 xf {_mm512_loadu_ps(x+i)};
            __m512 scaled {_mm512_mul_ps(xf, vinv_scale)};
            __m512 rnd {_mm512_mask_blend_ps(
                _mm512_cmp_ps_mask(scaled, vzero_ps, _CMP_GE_OQ),
                vneg_half,
                vhalf
            )};
            __m512i qi { _mm512_min_epi32(
                _mm512_max_epi32(
                    _mm512_add_epi32(
                        _mm512_cvttps_epi32(
                            _mm512_add_ps(scaled, rnd)),
                            vzp
                        ),
                    vmin
                ), vmax
            )};
            __m128i u8 {_mm512_cvtusepi32_epi8(qi)};
            _mm_storel_epi64(reinterpret_cast<__m128i*>(o+(i>>1)), _mm_or_si128(_mm_shuffle_epi8(u8, shuf_even), _mm_slli_epi16(_mm_shuffle_epi8(u8, shuf_odd), 4)));
        }
    #endif

    const auto quant_step_packed {[=](fp32_t a, fp32_t b) noexcept -> std::uint8_t {
        auto qa {std::clamp(static_cast<std::int32_t>(std::round(a * scale)) + zp, 0, 15)};
        auto qb {std::clamp(static_cast<std::int32_t>(std::round(b * scale)) + zp, 0, 15)};
        return qa & 15 | (qb & 15)<<4;
    }};

    for (; i+1 < numel; i += 2) {
        fp32_t a {x[i]};
        fp32_t b {x[i+1]};
        o[i>>1].bits = quant_step_packed(a, b);
    }
    if (numel & 1) {
        o[i>>1].bits = quant_step_packed(x[numel-1], 0);
        o[i>>1].bits &= 15;
    }
}