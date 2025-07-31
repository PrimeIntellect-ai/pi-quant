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
        __m512i vmin {_mm512_setzero_si512()};
        __m512i vmax {_mm512_set1_epi32(0xff)};
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
            xi0 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, xi0));
            xi1 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, xi1));
            xi2 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, xi2));
            xi3 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, xi3));
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
            __m256 xf0 {_mm256_loadu_ps(x+i+(0<<3))};
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
            __m128i xi0 {_mm_cvttps_epi32(_mm_add_ps(xf0, _mm_blendv_ps(vneg_half, vhalf, _mm_cmpge_ps(xf0, vzero))))};
            __m128i xi1 {_mm_cvttps_epi32(_mm_add_ps(xf1, _mm_blendv_ps(vneg_half, vhalf, _mm_cmpge_ps(xf1, vzero))))};
            __m128i xi2 {_mm_cvttps_epi32(_mm_add_ps(xf2, _mm_blendv_ps(vneg_half, vhalf, _mm_cmpge_ps(xf2, vzero))))};
            __m128i xi3 {_mm_cvttps_epi32(_mm_add_ps(xf3, _mm_blendv_ps(vneg_half, vhalf, _mm_cmpge_ps(xf3, vzero))))};
            xi0 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(xi0, vzero_point)));
            xi1 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(xi1, vzero_point)));
            xi2 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(xi2, vzero_point)));
            xi3 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(xi3, vzero_point)));
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

static auto PIQUANT_HOT quant_bf16_to_uint8_nearest(
    const bfp16_t* PIQUANT_RESTRICT x,
    std::uint8_t* PIQUANT_RESTRICT o,
    std::int64_t numel,
    fp32_t scale,
    std::int32_t zp
) noexcept -> void {
    scale = 1.0f / scale;
    std::int64_t i {};
    #if defined(__AVX512F__) && defined(__AVX512BW__)

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
            fp32_t rnd {std::round(static_cast<fp32_t>(x[i])*scale)};
            std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
        for (; i+31 < numel; i += 32) {
            __m256 xf0 {_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i))), 16))};
            __m256 xf1 {_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+8))), 16))};
            __m256 xf2 {_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+16))), 16))};
            __m256 xf3 {_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+24))), 16))};
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
            fp32_t rnd {std::round(static_cast<fp32_t>(x[i])*scale)};
            std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
            o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
        }
        for (; i+31 < numel; i += 32) {
            __m128 xf0 {_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i))), 16))};
            __m128 xf1 {_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+8))), 16))};
            __m128 xf2 {_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+16))), 16))};
            __m128 xf3 {_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+24))), 16))};
            xf0 = _mm_mul_ps(xf0, vinv_scale);
            xf1 = _mm_mul_ps(xf1, vinv_scale);
            xf2 = _mm_mul_ps(xf2, vinv_scale);
            xf3 = _mm_mul_ps(xf3, vinv_scale);
            __m128i xi0 {_mm_cvttps_epi32(_mm_add_ps(xf0, _mm_blendv_ps(vneg_half, vhalf, _mm_cmpge_ps(xf0, vzero))))};
            __m128i xi1 {_mm_cvttps_epi32(_mm_add_ps(xf1, _mm_blendv_ps(vneg_half, vhalf, _mm_cmpge_ps(xf1, vzero))))};
            __m128i xi2 {_mm_cvttps_epi32(_mm_add_ps(xf2, _mm_blendv_ps(vneg_half, vhalf, _mm_cmpge_ps(xf2, vzero))))};
            __m128i xi3 {_mm_cvttps_epi32(_mm_add_ps(xf3, _mm_blendv_ps(vneg_half, vhalf, _mm_cmpge_ps(xf3, vzero))))};
            xi0 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(xi0, vzero_point)));
            xi1 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(xi1, vzero_point)));
            xi2 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(xi2, vzero_point)));
            xi3 = _mm_max_epi32(vmin, _mm_min_epi32(vmax, _mm_add_epi32(xi3, vzero_point)));
            __m128i pack16_0 {_mm_packus_epi32(xi0, xi1)};
            __m128i pack16_1 {_mm_packus_epi32(xi2, xi3)};
            __m128i result {_mm_packus_epi16(pack16_0, pack16_1)};
            _mm_stream_si128(reinterpret_cast<__m128i*>(o+i), result);
        }
    #elif defined(__aarch64__) && defined(__ARM_NEON__)

    #endif
    for (; i < numel; ++i) {
        fp32_t rnd {std::round(static_cast<fp32_t>(x[i]) * scale)};
        std::int32_t i32 {static_cast<std::int32_t>(rnd) + zp};
        o[i] = static_cast<std::uint8_t>(std::clamp(i32, 0, 0xff));
    }
}

template <const reduce_op ReduceOp>
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
            if constexpr (ReduceOp == reduce_op::add) {
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
        __m256i vzp {_mm256_set1_epi32(zp)};
        __m256 vscale {_mm256_set1_ps(scale)};
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
            vs00 = _mm256_sub_epi32(vs00, vzp);
            vs10 = _mm256_sub_epi32(vs10, vzp);
            vs20 = _mm256_sub_epi32(vs20, vzp);
            vs30 = _mm256_sub_epi32(vs30, vzp);
            vs01 = _mm256_sub_epi32(vs01, vzp);
            vs11 = _mm256_sub_epi32(vs11, vzp);
            vs21 = _mm256_sub_epi32(vs21, vzp);
            vs31 = _mm256_sub_epi32(vs31, vzp);
            __m256 vf00 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs00), vscale)};
            __m256 vf10 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs10), vscale)};
            __m256 vf20 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs20), vscale)};
            __m256 vf30 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs30), vscale)};
            __m256 vf01 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs01), vscale)};
            __m256 vf11 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs11), vscale)};
            __m256 vf21 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs21), vscale)};
            __m256 vf31 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs31), vscale)};
            if constexpr (ReduceOp == reduce_op::add) {
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
            static constexpr auto expand16 {[&](const __m128i &v) {
                __m128i zero {_mm_setzero_si128()};
                __m128i w_lo {_mm_unpacklo_epi8(v, zero)};
                __m128i w_hi {_mm_unpackhi_epi8(v, zero)};
                __m128i d0 {_mm_unpacklo_epi16(w_lo, zero)};
                __m128i d1 {_mm_unpackhi_epi16(w_lo, zero)};
                __m128i d2 {_mm_unpacklo_epi16(w_hi, zero)};
                __m128i d3 {_mm_unpackhi_epi16(w_hi, zero)};
                return std::array{d0,d1,d2,d3};
            }};
            auto [vs00, vs10, vs20, vs30] = expand16(in0);
            auto [vs01, vs11, vs21, vs31] = expand16(in1);
            auto [vs02, vs12, vs22, vs32] = expand16(in2);
            auto [vs03, vs13, vs23, vs33] = expand16(in3);
            __m128 vf00 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs00, vzp)), vscale)};
            __m128 vf01 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs10, vzp)), vscale)};
            __m128 vf02 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs20, vzp)), vscale)};
            __m128 vf03 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs30, vzp)), vscale)};
            __m128 vf10 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs01, vzp)), vscale)};
            __m128 vf11 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs11, vzp)), vscale)};
            __m128 vf12 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs21, vzp)), vscale)};
            __m128 vf13 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs31, vzp)), vscale)};
            __m128 vf20 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs02, vzp)), vscale)};
            __m128 vf21 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs12, vzp)), vscale)};
            __m128 vf22 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs22, vzp)), vscale)};
            __m128 vf23 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs32, vzp)), vscale)};
            __m128 vf30 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs03, vzp)), vscale)};
            __m128 vf31 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs13, vzp)), vscale)};
            __m128 vf32 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs23, vzp)), vscale)};
            __m128 vf33 {_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(vs33, vzp)), vscale)};
            if constexpr (ReduceOp == reduce_op::add) {
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
            _mm_storeu_ps(o+i+(0<<2), vf00);
            _mm_storeu_ps(o+i+(1<<2), vf01);
            _mm_storeu_ps(o+i+(2<<2), vf02);
            _mm_storeu_ps(o+i+(3<<2), vf03);
            _mm_storeu_ps(o+i+(4<<2), vf10);
            _mm_storeu_ps(o+i+(5<<2), vf11);
            _mm_storeu_ps(o+i+(6<<2), vf12);
            _mm_storeu_ps(o+i+(7<<2), vf13);
            _mm_storeu_ps(o+i+(8<<2), vf20);
            _mm_storeu_ps(o+i+(9<<2), vf21);
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
            if constexpr (ReduceOp == reduce_op::add) {
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
        if constexpr (ReduceOp == reduce_op::add) o[i] += dq;
        else o[i] = dq;
    }
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
    #elif defined(__AVX2__)
        __m256 vinv_scale {_mm256_set1_ps(scale)};
        __m256 vhalf {_mm256_set1_ps(0.5f)};
        __m256 vneg_half {_mm256_set1_ps(-0.5f)};
        __m256 vzero_ps {_mm256_setzero_ps()};
        __m256i vzp {_mm256_set1_epi32(zp)};
        __m256i vmin {_mm256_setzero_si256()};
        __m256i vmax {_mm256_set1_epi32(15)};
        __m128i shuf_even {_mm_setr_epi8(
            0,2,4,6,8,10,12,14,
            0x80,0x80,0x80,0x80,
            0x80,0x80,0x80,0x80
        )};
        __m128i shuf_odd {_mm_setr_epi8(
            1,3,5,7,9,11,13,15,
            0x80,0x80,0x80,0x80,
            0x80,0x80,0x80,0x80
        )};
        for (; i+15 < numel; i += 16) {
            __m256 xf0 {_mm256_loadu_ps(x+i)};
            __m256 xf1 {_mm256_loadu_ps(x+i+8)};
            __m256 scaled0 {_mm256_mul_ps(xf0, vinv_scale)};
            __m256 scaled1 {_mm256_mul_ps(xf1, vinv_scale)};
            __m256 mask0 {_mm256_cmp_ps(scaled0, vzero_ps, _CMP_GE_OQ)};
            __m256 mask1 {_mm256_cmp_ps(scaled1, vzero_ps, _CMP_GE_OQ)};
            __m256 rnd0 {_mm256_blendv_ps(vneg_half, vhalf, mask0)};
            __m256 rnd1 {_mm256_blendv_ps(vneg_half, vhalf, mask1)};
            __m256i qi0 {_mm256_cvttps_epi32(_mm256_add_ps(scaled0, rnd0))};
            __m256i qi1 {_mm256_cvttps_epi32(_mm256_add_ps(scaled1, rnd1))};
            qi0 = _mm256_add_epi32(qi0, vzp);
            qi1 = _mm256_add_epi32(qi1, vzp);
            qi0 = _mm256_max_epi32(qi0, vmin);
            qi1 = _mm256_max_epi32(qi1, vmin);
            qi0 = _mm256_min_epi32(qi0, vmax);
            qi1 = _mm256_min_epi32(qi1, vmax);
            __m128i qi0_lo {_mm256_castsi256_si128(qi0)};
            __m128i qi0_hi {_mm256_extracti128_si256(qi0, 1)};
            __m128i q16_0 {_mm_packs_epi32(qi0_lo, qi0_hi)};
            __m128i qi1_lo {_mm256_castsi256_si128(qi1)};
            __m128i qi1_hi {_mm256_extracti128_si256(qi1, 1)};
            __m128i q16_1 {_mm_packs_epi32(qi1_lo, qi1_hi)};
            __m128i q8 {_mm_packus_epi16(q16_0, q16_1)};
            __m128i lo_nib {_mm_shuffle_epi8(q8, shuf_even)};
            __m128i hi_nib {_mm_shuffle_epi8(q8, shuf_odd)};
            __m128i packed {_mm_or_si128(lo_nib, _mm_slli_epi16(hi_nib, 4))};
            _mm_storel_epi64(reinterpret_cast<__m128i*>(o+(i>>1)), packed);
        }
    #elif defined(__SSE4_2__)
        __m128 vinv_scale {_mm_set1_ps(scale)};
        __m128 vhalf {_mm_set1_ps(0.5f)};
        __m128 vneg_half {_mm_set1_ps(-0.5f)};
        __m128 vzero_ps {_mm_setzero_ps()};
        __m128i vzp {_mm_set1_epi32(zp)};
        __m128i vmin {_mm_setzero_si128()};
        __m128i vmax {_mm_set1_epi32(15)};
        __m128i shuf_even {_mm_setr_epi8(
            0,2,4,6,8,10,12,14,
            0x80,0x80,0x80,0x80,
            0x80,0x80,0x80,0x80
        )};
        __m128i shuf_odd  {_mm_setr_epi8(
            1,3,5,7,9,11,13,15,
            0x80,0x80,0x80,0x80,
            0x80,0x80,0x80,0x80
        )};

        for (; i+15 < numel; i += 16) {
            __m128 xf0 {_mm_loadu_ps(x+i)};
            __m128 xf1 {_mm_loadu_ps(x+i+4)};
            __m128 xf2 {_mm_loadu_ps(x+i+8)};
            __m128 xf3 {_mm_loadu_ps(x+i+12)};
            __m128 scaled0 {_mm_mul_ps(xf0, vinv_scale)};
            __m128 scaled1 {_mm_mul_ps(xf1, vinv_scale)};
            __m128 scaled2 {_mm_mul_ps(xf2, vinv_scale)};
            __m128 scaled3 {_mm_mul_ps(xf3, vinv_scale)};
            __m128 mask0 {_mm_cmpge_ps(scaled0, vzero_ps)};
            __m128 mask1 {_mm_cmpge_ps(scaled1, vzero_ps)};
            __m128 mask2 {_mm_cmpge_ps(scaled2, vzero_ps)};
            __m128 mask3 {_mm_cmpge_ps(scaled3, vzero_ps)};
            __m128 rnd0 {_mm_blendv_ps(vneg_half, vhalf, mask0)};
            __m128 rnd1 {_mm_blendv_ps(vneg_half, vhalf, mask1)};
            __m128 rnd2 {_mm_blendv_ps(vneg_half, vhalf, mask2)};
            __m128 rnd3 {_mm_blendv_ps(vneg_half, vhalf, mask3)};
            __m128i qi0 {_mm_cvttps_epi32(_mm_add_ps(scaled0, rnd0))};
            __m128i qi1 {_mm_cvttps_epi32(_mm_add_ps(scaled1, rnd1))};
            __m128i qi2 {_mm_cvttps_epi32(_mm_add_ps(scaled2, rnd2))};
            __m128i qi3 {_mm_cvttps_epi32(_mm_add_ps(scaled3, rnd3))};
            qi0 = _mm_add_epi32(qi0, vzp);
            qi1 = _mm_add_epi32(qi1, vzp);
            qi2 = _mm_add_epi32(qi2, vzp);
            qi3 = _mm_add_epi32(qi3, vzp);
            qi0 = _mm_max_epi32(_mm_min_epi32(qi0, vmax), vmin);
            qi1 = _mm_max_epi32(_mm_min_epi32(qi1, vmax), vmin);
            qi2 = _mm_max_epi32(_mm_min_epi32(qi2, vmax), vmin);
            qi3 = _mm_max_epi32(_mm_min_epi32(qi3, vmax), vmin);
            __m128i q16_a {_mm_packs_epi32(qi0, qi1)};
            __m128i q16_b {_mm_packs_epi32(qi2, qi3)};
            __m128i q8 {_mm_packus_epi16(q16_a, q16_b)};
            __m128i lo_nib {_mm_shuffle_epi8(q8, shuf_even)};
            __m128i hi_nib {_mm_shuffle_epi8(q8, shuf_odd)};
            __m128i packed {_mm_or_si128(lo_nib, _mm_slli_epi16(hi_nib, 4))};
            _mm_storel_epi64(reinterpret_cast<__m128i*>(o+(i>>1)), packed);
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

template <const reduce_op ReduceOp>
static auto PIQUANT_HOT dequant_uint4_to_f32(
    const uint4_t* PIQUANT_RESTRICT x,
    fp32_t* PIQUANT_RESTRICT o,
    std::int64_t numel,
    fp32_t scale,
    std::int32_t zp
) noexcept -> void {
    std::int64_t i {};
    #if defined(__AVX512F__) && defined(__AVX512BW__)
        __m512i vzp {_mm512_set1_epi32(zp)};
        __m512 vscale {_mm512_set1_ps(scale)};
        __m512i vmaskLo {_mm512_set1_epi8(0x0f)};
        __m512i idx {_mm512_set_epi8(
            63,62,61,60,59,58,57,56, 47,46,45,44,43,42,41,40,
            31,30,29,28,27,26,25,24, 15,14,13,12,11,10, 9, 8,
            55,54,53,52,51,50,49,48, 39,38,37,36,35,34,33,32,
            23,22,21,20,19,18,17,16,  7, 6, 5, 4, 3, 2, 1, 0
        )};
        static constexpr auto expand_u8_to_s32 {[](__m512i v) noexcept -> std::array<__m512i, 4> {
            __m128i l0 {_mm512_extracti32x4_epi32(v, 0)};
            __m128i l1 {_mm512_extracti32x4_epi32(v, 1)};
            __m128i l2 {_mm512_extracti32x4_epi32(v, 2)};
            __m128i l3 {_mm512_extracti32x4_epi32(v, 3)};
            __m512i a0 {_mm512_cvtepu8_epi32(l0)};
            __m512i a1 {_mm512_cvtepu8_epi32(l1)};
            __m512i a2 {_mm512_cvtepu8_epi32(l2)};
            __m512i a3 {_mm512_cvtepu8_epi32(l3)};
            return {a0, a1, a2, a3};
        }};
        const auto unpack_nibbles {[&vmaskLo](__m512i v) noexcept -> std::array<__m512i, 2> {
            __m512i lo {_mm512_and_si512(v, vmaskLo)};
            __m512i hi {_mm512_and_si512(_mm512_srli_epi16(v, 4), vmaskLo)};
            __m512i t0 {_mm512_unpacklo_epi8(lo, hi)};
            __m512i t1 {_mm512_unpackhi_epi8(lo, hi)};
            __m512i v0 {_mm512_castsi128_si512(_mm512_extracti32x4_epi32(t0, 0))};
            v0 = _mm512_inserti32x4(v0, _mm512_extracti32x4_epi32(t1, 0), 1);
            v0 = _mm512_inserti32x4(v0, _mm512_extracti32x4_epi32(t0, 1), 2);
            v0 = _mm512_inserti32x4(v0, _mm512_extracti32x4_epi32(t1, 1), 3);
            __m512i v1 {_mm512_castsi128_si512(_mm512_extracti32x4_epi32(t0, 2))};
            v1 = _mm512_inserti32x4(v1, _mm512_extracti32x4_epi32(t1, 2), 1);
            v1 = _mm512_inserti32x4(v1, _mm512_extracti32x4_epi32(t0, 3), 2);
            v1 = _mm512_inserti32x4(v1, _mm512_extracti32x4_epi32(t1, 3), 3);
            return {v0, v1};
        }};
        for (; i+127 < numel; i += 128) {
            __m512i packed {_mm512_loadu_si512(reinterpret_cast<const void*>(reinterpret_cast<const std::uint8_t*>(x) + (i >> 1)))};
            auto [nib0, nib1] = unpack_nibbles(packed);
            auto [vs00, vs10, vs20, vs30] = expand_u8_to_s32(nib0);
            auto [vs01, vs11, vs21, vs31] = expand_u8_to_s32(nib1);
            vs00 = _mm512_sub_epi32(vs00, vzp);
            vs10 = _mm512_sub_epi32(vs10, vzp);
            vs20 = _mm512_sub_epi32(vs20, vzp);
            vs30 = _mm512_sub_epi32(vs30, vzp);
            vs01 = _mm512_sub_epi32(vs01, vzp);
            vs11 = _mm512_sub_epi32(vs11, vzp);
            vs21 = _mm512_sub_epi32(vs21, vzp);
            vs31 = _mm512_sub_epi32(vs31, vzp);
            __m512 vf00 {_mm512_mul_ps(_mm512_cvtepi32_ps(vs00), vscale)};
            __m512 vf10 {_mm512_mul_ps(_mm512_cvtepi32_ps(vs10), vscale)};
            __m512 vf20 {_mm512_mul_ps(_mm512_cvtepi32_ps(vs20), vscale)};
            __m512 vf30 {_mm512_mul_ps(_mm512_cvtepi32_ps(vs30), vscale)};
            __m512 vf01 {_mm512_mul_ps(_mm512_cvtepi32_ps(vs01), vscale)};
            __m512 vf11 {_mm512_mul_ps(_mm512_cvtepi32_ps(vs11), vscale)};
            __m512 vf21 {_mm512_mul_ps(_mm512_cvtepi32_ps(vs21), vscale)};
            __m512 vf31 {_mm512_mul_ps(_mm512_cvtepi32_ps(vs31), vscale)};
            if constexpr (ReduceOp == reduce_op::add) {
                vf00 = _mm512_add_ps(vf00, _mm512_loadu_ps(o+i+0));
                vf10 = _mm512_add_ps(vf10, _mm512_loadu_ps(o+i+16));
                vf20 = _mm512_add_ps(vf20, _mm512_loadu_ps(o+i+32));
                vf30 = _mm512_add_ps(vf30, _mm512_loadu_ps(o+i+48));
                vf01 = _mm512_add_ps(vf01, _mm512_loadu_ps(o+i+64));
                vf11 = _mm512_add_ps(vf11, _mm512_loadu_ps(o+i+80));
                vf21 = _mm512_add_ps(vf21, _mm512_loadu_ps(o+i+96));
                vf31 = _mm512_add_ps(vf31, _mm512_loadu_ps(o+i+112));
            }
            _mm512_storeu_ps(o+i+0, vf00);
            _mm512_storeu_ps(o+i+16, vf10);
            _mm512_storeu_ps(o+i+32, vf20);
            _mm512_storeu_ps(o+i+48, vf30);
            _mm512_storeu_ps(o+i+64, vf01);
            _mm512_storeu_ps(o+i+80, vf11);
            _mm512_storeu_ps(o+i+96, vf21);
            _mm512_storeu_ps(o+i+112, vf31);
        }
    #elif defined(__AVX2__)
        __m256i vzp {_mm256_set1_epi32(zp)};
        __m256 vscale {_mm256_set1_ps(scale)};
        __m256i vmask_lo {_mm256_set1_epi8(0x0f)};
        static constexpr auto expand_u8_to_s32 {[](__m256i v) noexcept -> std::array<__m256i, 4> {
            __m128i l {_mm256_castsi256_si128(v)};
            __m128i h {_mm256_extracti128_si256(v, 1)};
            __m256i a0 {_mm256_cvtepu8_epi32(l)};
            __m256i a1 {_mm256_cvtepu8_epi32(_mm_srli_si128(l, 8))};
            __m256i b0 {_mm256_cvtepu8_epi32(h)};
            __m256i b1 {_mm256_cvtepu8_epi32(_mm_srli_si128(h, 8))};
            return std::array{a0, a1, b0, b1};
        }};
        const auto unpack_nibbles {[&vmask_lo](__m256i v) noexcept -> std::array<__m256i, 2> {
            __m256i lo {_mm256_and_si256(v, vmask_lo)};
            __m256i hi {_mm256_and_si256(_mm256_srli_epi16(v,4), vmask_lo)};
            __m256i t0 {_mm256_unpacklo_epi8(lo, hi)};
            __m256i t1 {_mm256_unpackhi_epi8(lo, hi)};
            __m256i v0 {_mm256_permute2x128_si256(t0, t1, 0x20)};
            __m256i v1 {_mm256_permute2x128_si256(t0, t1, 0x31)};
            return std::array{v0, v1};
        }};
        for (; i+63 < numel; i += 64) {
            __m256i packed {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(reinterpret_cast<const std::uint8_t*>(x) + (i>>1)))};
            auto [nib0, nib1] {unpack_nibbles(packed)};
            auto [vs00, vs10, vs20, vs30] {expand_u8_to_s32(nib0)};
            auto [vs01, vs11, vs21, vs31] {expand_u8_to_s32(nib1)};
            vs00 = _mm256_sub_epi32(vs00, vzp);
            vs10 = _mm256_sub_epi32(vs10, vzp);
            vs20 = _mm256_sub_epi32(vs20, vzp);
            vs30 = _mm256_sub_epi32(vs30, vzp);
            vs01 = _mm256_sub_epi32(vs01, vzp);
            vs11 = _mm256_sub_epi32(vs11, vzp);
            vs21 = _mm256_sub_epi32(vs21, vzp);
            vs31 = _mm256_sub_epi32(vs31, vzp);
            __m256 vf00 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs00), vscale)};
            __m256 vf10 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs10), vscale)};
            __m256 vf20 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs20), vscale)};
            __m256 vf30 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs30), vscale)};
            __m256 vf01 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs01), vscale)};
            __m256 vf11 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs11), vscale)};
            __m256 vf21 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs21), vscale)};
            __m256 vf31 {_mm256_mul_ps(_mm256_cvtepi32_ps(vs31), vscale)};
            if constexpr (ReduceOp == reduce_op::add) {
                vf00 = _mm256_add_ps(vf00, _mm256_loadu_ps(o+i+ 0));
                vf10 = _mm256_add_ps(vf10, _mm256_loadu_ps(o+i+ 8));
                vf20 = _mm256_add_ps(vf20, _mm256_loadu_ps(o+i+16));
                vf30 = _mm256_add_ps(vf30, _mm256_loadu_ps(o+i+24));
                vf01 = _mm256_add_ps(vf01, _mm256_loadu_ps(o+i+32));
                vf11 = _mm256_add_ps(vf11, _mm256_loadu_ps(o+i+40));
                vf21 = _mm256_add_ps(vf21, _mm256_loadu_ps(o+i+48));
                vf31 = _mm256_add_ps(vf31, _mm256_loadu_ps(o+i+56));
            }
            _mm256_storeu_ps(o+i+0, vf00);
            _mm256_storeu_ps(o+i+8, vf10);
            _mm256_storeu_ps(o+i+16, vf20);
            _mm256_storeu_ps(o+i+24, vf30);
            _mm256_storeu_ps(o+i+32, vf01);
            _mm256_storeu_ps(o+i+40, vf11);
            _mm256_storeu_ps(o+i+48, vf21);
            _mm256_storeu_ps(o+i+56, vf31);
        }
    #elif defined(__SSE4_2__)
        __m128i vzp32 {_mm_set1_epi32(zp)};
        __m128  vscale {_mm_set1_ps(scale)};
        __m128i vmaskLo {_mm_set1_epi8(0x0f)};
        const auto process16 {[&](__m128i b, std::int64_t v) noexcept -> void {
            __m128i w0 {_mm_cvtepu8_epi16(b)};
            __m128i d00 {_mm_cvtepi16_epi32(w0)};
            __m128i d01 {_mm_cvtepi16_epi32(_mm_srli_si128(w0, 8))};
            d00 = _mm_sub_epi32(d00, vzp32);
            d01 = _mm_sub_epi32(d01, vzp32);
            __m128 f00 {_mm_mul_ps(_mm_cvtepi32_ps(d00), vscale)};
            __m128 f01 {_mm_mul_ps(_mm_cvtepi32_ps(d01), vscale)};
            __m128i w1 {_mm_cvtepu8_epi16(_mm_srli_si128(b, 8))};
            __m128i d10 {_mm_cvtepi16_epi32(w1)};
            __m128i d11 {_mm_cvtepi16_epi32(_mm_srli_si128(w1, 8))};
            d10 = _mm_sub_epi32(d10, vzp32);
            d11 = _mm_sub_epi32(d11, vzp32);
            __m128 f10 {_mm_mul_ps(_mm_cvtepi32_ps(d10), vscale)};
            __m128 f11 {_mm_mul_ps(_mm_cvtepi32_ps(d11), vscale)};
            if constexpr (ReduceOp == reduce_op::add) {
                f00 = _mm_add_ps(f00, _mm_loadu_ps(o+v+0));
                f01 = _mm_add_ps(f01, _mm_loadu_ps(o+v+4));
                f10 = _mm_add_ps(f10, _mm_loadu_ps(o+v+8));
                f11 = _mm_add_ps(f11, _mm_loadu_ps(o+v+12));
            }
            _mm_storeu_ps(o+v+0, f00);
            _mm_storeu_ps(o+v+4, f01);
            _mm_storeu_ps(o+v+8, f10);
            _mm_storeu_ps(o+v+12, f11);
        }};
        for (; i+31 < numel; i += 32) {
            __m128i packed {_mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const std::uint8_t*>(x) + (i>>1)))};
            __m128i lo {_mm_and_si128(packed, vmaskLo)};
            __m128i hi {_mm_and_si128(_mm_srli_epi16(packed, 4), vmaskLo)};
            __m128i t0 {_mm_unpacklo_epi8(lo, hi)};
            __m128i t1 {_mm_unpackhi_epi8(lo, hi)};
            process16(t0, i+0);
            process16(t1, i+16);
        }
    #elif defined(__aarch64__) && defined(__ARM_NEON__)

    #endif

    const auto dequant_step {[=](std::int32_t x) noexcept -> fp32_t {
        return static_cast<fp32_t>(x - zp)*scale;
    }};

    for (; i+1 < numel; i += 2) {
        auto p {x[i>>1].bits};
        auto qa {p & 15};
        auto qb {p >> 4};
        if constexpr (ReduceOp == reduce_op::set) {
            o[i] = dequant_step(qa);
            o[i+1] = dequant_step(qb);
        } else if constexpr (ReduceOp == reduce_op::add) {
            o[i] += dequant_step(qa);
            o[i+1] += dequant_step(qb);
        }
    }
    if (numel & 1) {
        auto r {dequant_step(x[i>>1].bits & 15)};
        if constexpr (ReduceOp == reduce_op::set) o[numel-1] = r;
        else if constexpr (ReduceOp == reduce_op::add) o[numel-1] += r;
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
            __m256 v0 {_mm256_loadu_ps(reinterpret_cast<const float*>(x+i))};
            __m256 v1 {_mm256_loadu_ps(reinterpret_cast<const float*>(x+i+8))};
            __m256 v2 {_mm256_loadu_ps(reinterpret_cast<const float*>(x+i+16))};
            __m256 v3 {_mm256_loadu_ps(reinterpret_cast<const float*>(x+i+24))};
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
        __m128 m {_mm_min_ps(lo, hi)};
        m = _mm_min_ps(m, _mm_movehl_ps(m, m));
        m = _mm_min_ps(m, _mm_shuffle_ps(m, m, 0b01));
        min = _mm_cvtss_f32(m);
        lo = _mm256_castps256_ps128(vmax);
        hi = _mm256_extractf128_ps(vmax, 1);
        m = _mm_max_ps(lo, hi);
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0b01));
        max = _mm_cvtss_f32(m);
    #elif defined(__SSE4_2__)
        __m128 vmin {_mm_set1_ps(min)};
        __m128 vmax {_mm_set1_ps(max)};
        for (; i+15 < numel; i += 16) {
            __m128 v0 {_mm_loadu_ps(x+i+(0<<2))};
            __m128 v1 {_mm_loadu_ps(x+i+(1<<2))};
            __m128 v2 {_mm_loadu_ps(x+i+(2<<2))};
            __m128 v3 {_mm_loadu_ps(x+i+(3<<2))};
            vmin = _mm_min_ps(vmin, v0);
            vmax = _mm_max_ps(vmax, v0);
            vmin = _mm_min_ps(vmin, v1);
            vmax = _mm_max_ps(vmax, v1);
            vmin = _mm_min_ps(vmin, v2);
            vmax = _mm_max_ps(vmax, v2);
            vmin = _mm_min_ps(vmin, v3);
            vmax = _mm_max_ps(vmax, v3);
        }
        min = _mm_cvtss_f32(_mm_min_ps(_mm_min_ps(vmin, _mm_movehl_ps(vmin, vmin)), _mm_shuffle_ps(vmin, vmin, 0b01)));
        max = _mm_cvtss_f32(_mm_max_ps(_mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax)), _mm_shuffle_ps(vmax, vmax, 0b01)));
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
        __m256 vmin {_mm256_set1_ps(min)};
        __m256 vmax {_mm256_set1_ps(max)};
        for (; i+31 < numel; i += 32) {
            __m256 v0 {_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i))), 16))};
            __m256 v1 {_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+8))), 16))};
            __m256 v2 {_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+16))), 16))};
            __m256 v3 {_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+24))), 16))};
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
        __m128 m {_mm_min_ps(lo, hi)};
        m = _mm_min_ps(m, _mm_movehl_ps(m, m));
        m = _mm_min_ps(m, _mm_shuffle_ps(m, m, 0b01));
        min = _mm_cvtss_f32(m);
        lo = _mm256_castps256_ps128(vmax);
        hi = _mm256_extractf128_ps(vmax, 1);
        m = _mm_max_ps(lo, hi);
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0b01));
        max = _mm_cvtss_f32(m);
    #elif defined(__SSE4_2__)
        __m128 vmin {_mm_set1_ps(min)};
        __m128 vmax {_mm_set1_ps(max)};
        for (; i+31 < numel; i += 32) {
            __m128 v0 {_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i))), 16))};
            __m128 v1 {_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+8))), 16))};
            __m128 v2 {_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+16))), 16))};
            __m128 v3 {_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x+i+24))), 16))};
            vmin = _mm_min_ps(vmin, v0);
            vmax = _mm_max_ps(vmax, v0);
            vmin = _mm_min_ps(vmin, v1);
            vmax = _mm_max_ps(vmax, v1);
            vmin = _mm_min_ps(vmin, v2);
            vmax = _mm_max_ps(vmax, v2);
            vmin = _mm_min_ps(vmin, v3);
            vmax = _mm_max_ps(vmax, v3);
        }
        min = _mm_cvtss_f32(_mm_min_ps(_mm_min_ps(vmin, _mm_movehl_ps(vmin, vmin)), _mm_shuffle_ps(vmin, vmin, 0b01)));
        max = _mm_cvtss_f32(_mm_max_ps(_mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax)), _mm_shuffle_ps(vmax, vmax, 0b01)));
    #elif defined(__aarch64__) && defined(__ARM_NEON__)

    #endif
    for (; i < numel; ++i) {
        auto xi {static_cast<fp32_t>(x[i])};
        if (xi < min) min = xi;
        if (xi > max) max = xi;
    }
    return {min, max};
}
