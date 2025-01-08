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
    #if defined(__aarch64__) && defined(__ARM_NEON__) && 0
        float32x4_t vinv_scale = vdupq_n_f32(inv_scale);
        int32x4_t vzero_point  = vdupq_n_s32(zero_point);
        int32x4_t vmin_val = vdupq_n_s32(0);
        int32x4_t vmax_val = vdupq_n_s32(255);
        // Process 8 floats per loop iteration
    for (; i + 7 < n; i += 8) {
        // -------------------------------------------------------------
        //  Load 8 floats
        // -------------------------------------------------------------
        float32x4_t vf0 = vld1q_f32(x + i);
        float32x4_t vf1 = vld1q_f32(x + i + 4);

        // -------------------------------------------------------------
        //  Multiply by inv_scale
        // -------------------------------------------------------------
        vf0 = vmulq_f32(vf0, vinv_scale);
        vf1 = vmulq_f32(vf1, vinv_scale);

        // -------------------------------------------------------------
        //  Round half away from zero
        //
        //  Option A: If Armv8.2+:
        //    int32x4_t vi0 = vcvtaq_s32_f32(vf0); // round half away from zero
        //    int32x4_t vi1 = vcvtaq_s32_f32(vf1);
        //
        //  Option B: Manual std::round() emulation if vcvtaq_s32_f32 not available:
        //    (Add 0.5 if positive, subtract 0.5 if negative, then truncate)
        // -------------------------------------------------------------
        int32x4_t vi0 = vcvtaq_s32_f32(vf0);
        int32x4_t vi1 = vcvtaq_s32_f32(vf1);

        // -------------------------------------------------------------
        //  Add zero_point and clamp to [0..255]
        // -------------------------------------------------------------
        vi0 = vaddq_s32(vi0, vzero_point);
        vi1 = vaddq_s32(vi1, vzero_point);

        vi0 = vmaxq_s32(vi0, vmin_val);
        vi1 = vmaxq_s32(vi1, vmin_val);
        vi0 = vminq_s32(vi0, vmax_val);
        vi1 = vminq_s32(vi1, vmax_val);

        // -------------------------------------------------------------
        //  Narrow from s32 -> s16 -> u8 (all 8 bytes at once)
        // -------------------------------------------------------------
        int16x4_t vi0_16s = vqmovn_s32(vi0); // saturate to 16 bits
        int16x4_t vi1_16s = vqmovn_s32(vi1);
        // reinterpret as unsigned 16
        uint16x4_t vu0_16u = vreinterpret_u16_s16(vi0_16s);
        uint16x4_t vu1_16u = vreinterpret_u16_s16(vi1_16s);

        // combine into 8x 16-bit
        uint16x8_t v16u = vcombine_u16(vu0_16u, vu1_16u);

        // narrow 8x 16-bit -> 8x 8-bit
        uint8x8_t vu8 = vqmovn_u16(v16u);

        // -------------------------------------------------------------
        //  Store 8 bytes in one shot
        // -------------------------------------------------------------
        vst1_u8(o + i, vu8);
    }
    #endif
    for (; i < n; ++i) {
        o[i] = static_cast<std::uint8_t>(std::clamp(static_cast<std::int32_t>(std::round(x[i] * inv_scale)) + zero_point, 0, 0xff));
    }
}