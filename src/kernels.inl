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
    struct kernel_registry;
}

#define concat(a, b) a ## b
#define impl_namespace(a, b) piquant::concat(a, _impl)

namespace impl_namespace(QUANT_KERNEL_IMPL, _) {
    // Xorshift 128 plus PRNG (scalar) used for stochastic rounding.
    // Generates a canonical float ∈ [0, 1) using a 64-bit state.
    struct xs128p_state final {
        std::uint64_t p1 {};
        std::uint64_t p2 {};

        constexpr xs128p_state(std::uint64_t p1, std::uint64_t p2) noexcept : p1{p1}, p2{p2} {}

        [[nodiscard]] inline auto PIQUANT_AINLINE operator ()() noexcept -> std::uint64_t {
            std::uint64_t s1 {p1};
            std::uint64_t s0 {p2};
            p1 = s0;
            s1 ^= s1<<23;
            p2 = s1^s0^(s1>>18)^(s0>>5);
            return p2 + s0;
        }

        [[nodiscard]] inline auto PIQUANT_AINLINE canonical() noexcept -> float {
            static constexpr auto bias_scale {1.0f/static_cast<float>(0x800000)};
            std::uint64_t y {~0u & (*this)()};
            return (bias_scale*(static_cast<float>(y>>9) + 0.5f));
        }

        [[nodiscard]] inline auto PIQUANT_AINLINE bounded(std::array<std::uint32_t, 2> bounds) noexcept -> std::array<std::uint32_t, 2> {
            auto [b1, b2] {bounds}; // [0, bound1), [0, bound2)
            std::uint64_t y {(*this)()};
            return {
                static_cast<std::uint32_t>(((y&~0u)*b1)>>32),
                static_cast<std::uint32_t>(((y>>32)*b2)>>32)
            };
        }
    };

    // Xorshift 128 plus PRNG (SIMD) used for stochastic rounding.
    // Generates N canonical floats ∈ [0, 1), where N is vector_width / sizeof(float).
    struct xs128pv_state final {

        static constexpr std::array<std::uint64_t, 2> jump_tab { 0x8a5cd789635d2dff, 0x121fd2155c472f96 };

        static constexpr auto jump(std::uint64_t i1, std::uint64_t i2, std::uint64_t& s0, std::uint64_t& s1) noexcept -> void {
            constexpr auto jump_to_keys {
                [](std::uint64_t& ps0, std::uint64_t& ps1) noexcept -> void {
                    uint64_t s1 {ps0};
                    uint64_t s0 {ps1};
                    ps0 = s0;
                    s1 ^= s1<<23;
                    ps1 = s1^s0^(s1>>18)^(s0>>5);
                }
            };
            for (std::size_t i {}; i < jump_tab.size(); ++i)
                for (std::uint32_t j {}; j < 64; ++j) {
                    if (1&jump_tab[i]<<j) {
                        s0 ^= i1;
                        s1 ^= i2;
                    }
                    jump_to_keys(i1, i2);
                }
        }

        #if defined(__AVX512F__) && defined(__AVX512BW__) && 0
            __m128i p1;
            __m128i p2;
        #elif defined(__AVX2__)
            __m256i p1;
            __m256i p2;

            xs128pv_state(std::uint64_t p1, std::uint64_t p2) noexcept {
                std::array<std::uint64_t, 4> S0 {};
                std::array<std::uint64_t, 4> S1 {};
                S0[0] = p1;
                S1[0] = p2;
                jump(S0[0], S1[0], S0[1], S1[1]);
                jump(S0[1], S1[1], S0[2], S1[2]);
                jump(S0[2], S1[2], S0[3], S1[3]);
                this->p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(S0.data()));
                this->p2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(S1.data()));
            }

            [[nodiscard]] inline auto PIQUANT_AINLINE operator ()() noexcept -> __m256i {
                __m256i s1 {p1};
                __m256i s0 {p2};
                p1 = p2;
                s1 = _mm256_xor_si256(p2, _mm256_slli_epi64(p2, 23));
                p2 = _mm256_xor_si256(
                        _mm256_xor_si256(
                            _mm256_xor_si256(s1, s0),
                                _mm256_srli_epi64(s1, 18)),
                                _mm256_srli_epi64(s0, 5));
                return _mm256_add_epi64(p2, s0);
            }

        [[nodiscard]] inline auto PIQUANT_AINLINE canonical() noexcept -> __m256 {
            static constexpr auto bias_scale {1.0f/static_cast<float>(0x800000)};
            __m256i y {(*this)()};
            y = _mm256_and_si256(y, _mm256_set1_epi32(~0));
            y = _mm256_srli_epi32(y, 9);
            __m256 r {_mm256_add_ps(_mm256_castsi256_ps(y), _mm256_set1_ps(0.5f))};
            return _mm256_mul_ps(r, _mm256_set1_ps(bias_scale));
        }
        #else
            xs128p_state scalar;
            constexpr xs128pv_state(std::uint64_t p1, std::uint64_t p2) noexcept : scalar{p1, p2} {}
        #endif
    };

    static thread_local xs128p_state s_sprng {0x123456789abcdef0, 0x0fedcba987654321};

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
    [[nodiscard]] static constexpr auto PIQUANT_AINLINE pack_nibbles(OUT x, OUT y) -> OUT {
        return static_cast<OUT>(static_cast<std::uint8_t>((0xF & static_cast<std::uint8_t>(x)) | ((0xF & static_cast<std::uint8_t>(y)) << 4)));
    }

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
                __m256 xf0 {_mm256_cvtepi32_ps(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(x+i+(0<<3))))};
                __m256 xf1 {_mm256_cvtepi32_ps(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(x+i+(1<<3))))};
                __m256 xf2 {_mm256_cvtepi32_ps(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(x+i+(2<<3))))};
                __m256 xf3 {_mm256_cvtepi32_ps(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(x+i+(3<<3))))};
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
                __m128 xf0 {_mm_cvtepi32_ps(_mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<float*>(x+i+(0<<2)))))};
                __m128 xf1 {_mm_cvtepi32_ps(_mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<float*>(x+i+(1<<2)))))};
                __m128 xf2 {_mm_cvtepi32_ps(_mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<float*>(x+i+(2<<2)))))};
                __m128 xf3 {_mm_cvtepi32_ps(_mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<float*>(x+i+(3<<2)))))};
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
            static constexpr auto expand_u8_to_s32_avx2 = [](const __m256i &v) {
                __m256i zero {_mm256_setzero_si256()};
                __m256i w_lo {_mm256_unpacklo_epi8(v, zero)};
                __m256i w_hi {_mm256_unpackhi_epi8(v, zero)};
                __m256i d0 {_mm256_unpacklo_epi16(w_lo, zero)};
                __m256i d1 {_mm256_unpackhi_epi16(w_lo, zero)};
                __m256i d2 {_mm256_unpacklo_epi16(w_hi, zero)};
                __m256i d3 {_mm256_unpackhi_epi16(w_hi, zero)};
                return std::array{d0,d1,d2,d3};
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
                __m128i in0 {_mm_loadu_si128((__m128i*)(x+i+0*16))};
                __m128i in1 {_mm_loadu_si128((__m128i*)(x+i+1*16))};
                __m128i in2 {_mm_loadu_si128((__m128i*)(x+i+2*16))};
                __m128i in3 {_mm_loadu_si128((__m128i*)(x+i+3*16))};
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
                    vf00 = _mm_add_ps(vf00, _mm_loadu_ps(o + i +  0*4));
                    vf01 = _mm_add_ps(vf01, _mm_loadu_ps(o + i +  1*4));
                    vf02 = _mm_add_ps(vf02, _mm_loadu_ps(o + i +  2*4));
                    vf03 = _mm_add_ps(vf03, _mm_loadu_ps(o + i +  3*4));
                    vf10 = _mm_add_ps(vf10, _mm_loadu_ps(o + i +  4*4));
                    vf11 = _mm_add_ps(vf11, _mm_loadu_ps(o + i +  5*4));
                    vf12 = _mm_add_ps(vf12, _mm_loadu_ps(o + i +  6*4));
                    vf13 = _mm_add_ps(vf13, _mm_loadu_ps(o + i +  7*4));
                    vf20 = _mm_add_ps(vf20, _mm_loadu_ps(o + i +  8*4));
                    vf21 = _mm_add_ps(vf21, _mm_loadu_ps(o + i +  9*4));
                    vf22 = _mm_add_ps(vf22, _mm_loadu_ps(o + i + 10*4));
                    vf23 = _mm_add_ps(vf23, _mm_loadu_ps(o + i + 11*4));
                    vf30 = _mm_add_ps(vf30, _mm_loadu_ps(o + i + 12*4));
                    vf31 = _mm_add_ps(vf31, _mm_loadu_ps(o + i + 13*4));
                    vf32 = _mm_add_ps(vf32, _mm_loadu_ps(o + i + 14*4));
                    vf33 = _mm_add_ps(vf33, _mm_loadu_ps(o + i + 15*4));
                }
                _mm_storeu_ps(o+i+ 0*4, vf00);
                _mm_storeu_ps(o+i+ 1*4, vf01);
                _mm_storeu_ps(o+i+ 2*4, vf02);
                _mm_storeu_ps(o+i+ 3*4, vf03);
                _mm_storeu_ps(o+i+ 4*4, vf10);
                _mm_storeu_ps(o+i+ 5*4, vf11);
                _mm_storeu_ps(o+i+ 6*4, vf12);
                _mm_storeu_ps(o+i+ 7*4, vf13);
                _mm_storeu_ps(o+i+ 8*4, vf20);
                _mm_storeu_ps(o+i+ 9*4, vf21);
                _mm_storeu_ps(o+i+10*4, vf22);
                _mm_storeu_ps(o+i+11*4, vf23);
                _mm_storeu_ps(o+i+12*4, vf30);
                _mm_storeu_ps(o+i+13*4, vf31);
                _mm_storeu_ps(o+i+14*4, vf32);
                _mm_storeu_ps(o+i+15*4, vf33);
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

    template <const round_mode RND, typename IN, typename OUT, typename... Args>
         requires (std::is_floating_point_v<IN> && (std::is_integral_v<OUT> || is_int4<OUT>)
             && std::is_same_v<std::common_type_t<Args...>, IN> && sizeof...(Args) != 0)
    static inline auto PIQUANT_AINLINE quant_step(double inv_scale, std::int64_t zp, Args... args) noexcept -> OUT {
        if constexpr (RND == round_mode::stochastic) {
            const auto Q{[&](const IN x) noexcept -> OUT {
                double rnd {x * inv_scale};
                double dec {std::abs(rnd - std::trunc(rnd))};
                double xi {(s_sprng.canonical())};
                double adj {xi < dec ? 1.0f : 0.0f};
                if (rnd < 0.0f) adj = -1.0f * adj;
                rnd = std::trunc(rnd) + adj;
                auto integral {static_cast<std::int64_t>(rnd) + zp};
                return static_cast<OUT>(std::clamp<decltype(integral)>(integral, dtype_limits<OUT>::min, dtype_limits<OUT>::max));
            }};
            if constexpr (sizeof...(Args) == 1) return Q(args...);
            else return pack_nibbles(Q(args)...);
        } else {
            const auto Q {[=](const IN x) noexcept -> OUT {
                double rnd {std::round(static_cast<double>(x) * inv_scale)};
                auto integral {static_cast<std::int64_t>(rnd) + zp};
                return static_cast<OUT>(std::clamp<decltype(integral)>(integral, dtype_limits<OUT>::min, dtype_limits<OUT>::max));
            }};
            if constexpr (sizeof...(Args) == 1) return Q(args...);
            else return pack_nibbles(Q(args)...);
        }
    }

    template <typename IN, typename OUT, const round_mode RND>
        requires (std::is_floating_point_v<IN> && (std::is_integral_v<OUT> || is_int4<OUT>))
    static auto PIQUANT_HOT quant_generic(
        const void* in,
        void* out,
        std::int64_t numel,
        float scale,
        std::int64_t zp
    ) noexcept -> void {
        // Use SIMD optimized kernels for some dtype permutations
        if constexpr (std::is_same_v<IN, float> && std::is_same_v<OUT, std::uint8_t> && RND == round_mode::nearest) {
            quant_f32_to_uint8_nearest(static_cast<const float*>(in), static_cast<std::uint8_t*>(out), numel, scale, zp);
            return;
        }
        const auto* PIQUANT_RESTRICT x {static_cast<const IN*>(in)};
        auto* PIQUANT_RESTRICT o {static_cast<OUT*>(out)};
        double inv_scale {1.0 / static_cast<double>(scale)}; // We multiply by reciprocal
        if constexpr (is_int4<OUT>) {
            std::int64_t numel_out {(numel+1)>>1};
            for (std::int64_t i{}, j{}; j < numel_out; ++j, i += 2) {
                IN a {x[i]};
                IN b {i+1 < numel ? x[i+1] : x[i]};
                o[j] = quant_step<RND, IN, OUT>(inv_scale, zp, a, b);
            }
        } else {
            for (std::int64_t i = 0; i < numel; ++i)
                o[i] = quant_step<RND, IN, OUT>(inv_scale, zp, x[i]);
        }
    }

    template <typename IN, typename OUT>
          requires (std::is_floating_point_v<OUT> && (std::is_integral_v<IN> || is_int4<IN>))
    static inline auto PIQUANT_AINLINE dequant_step(double scale, std::int64_t zp, const IN x) noexcept -> OUT {
        return static_cast<OUT>(static_cast<std::int64_t>(x) - zp)*scale;
    }

    template <typename IN, typename OUT, const reduce_op RDO>
            requires (std::is_floating_point_v<OUT> && (std::is_integral_v<IN> || is_int4<IN>))
    static auto PIQUANT_HOT dequant_generic(
        const void* in,
        void* out,
        std::int64_t numel,
        double scale,
        std::int64_t zp
    ) noexcept -> void {
        const auto* PIQUANT_RESTRICT x {static_cast<const IN*>(in)};
        auto* PIQUANT_RESTRICT o {static_cast<OUT*>(out)};
        // Use SIMD optimized kernels for some dtype permutations
        if constexpr (std::is_same_v<IN, std::uint8_t> && std::is_same_v<OUT, float>) {
            if constexpr (RDO == reduce_op::set) {
                dequant_uint8_to_f32<false>(static_cast<const std::uint8_t*>(in), static_cast<float*>(out), numel, static_cast<float>(scale), static_cast<std::int32_t>(zp));
                return;
            } else if constexpr (RDO == reduce_op::add) {
                dequant_uint8_to_f32<true>(static_cast<const std::uint8_t*>(in), static_cast<float*>(out), numel, static_cast<float>(scale), static_cast<std::int32_t>(zp));
                return;
            }
        } else if constexpr (is_int4<IN>) {
            std::int64_t numel_packed {(numel+1)>>1};
            constexpr auto unpack {[](std::uint8_t nib) noexcept -> std::int8_t {
                if constexpr (std::is_same_v<IN, int4_t>) { // 4-bit sign extension
                    return nib & 0x8 ? static_cast<std::int8_t>(nib | 0xF0) : static_cast<std::int8_t>(nib);
                } else {
                    return static_cast<std::int8_t>(nib);
                }
            }};
            for (std::int64_t j{}, i{}; j < numel_packed; ++j) {
                std::uint8_t byte {static_cast<std::uint8_t>(x[j])};
                std::int8_t qa {unpack(byte&0xF)};
                std::int8_t qb {unpack(byte>>4)};
                if constexpr (RDO == reduce_op::set) {
                    o[i++] = dequant_step<std::int8_t, OUT>(scale, zp, qa);
                    if (i < numel)
                        o[i++] = dequant_step<std::int8_t, OUT>(scale, zp, qb);
                } else if constexpr (RDO == reduce_op::add) {
                    o[i++] += dequant_step<std::int8_t, OUT>(scale, zp, qa);
                    if (i < numel)
                        o[i++] += dequant_step<std::int8_t, OUT>(scale, zp, qb);
                } else {
                    static_assert(RDO == reduce_op::set || RDO == reduce_op::add, "Invalid reduce operation");
                }
            }
            return;
        }
        if constexpr (RDO == reduce_op::set) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] = dequant_step<IN, OUT>(scale, zp, x[i]);
        } else if constexpr (RDO == reduce_op::add) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] += dequant_step<IN, OUT>(scale, zp, x[i]);
        } else {
            static_assert(RDO == reduce_op::set || RDO == reduce_op::add, "Invalid reduce operation");
        }
    }

    template <typename IN, typename QUANT, const round_mode RND, const reduce_op RDO>
      requires (std::is_floating_point_v<IN> && (std::is_integral_v<QUANT> || is_int4<QUANT>))
    static auto PIQUANT_HOT quant_dequant_generic(
      const void* in,
      void* out,
      std::int64_t numel,
      double scale,
      std::int64_t zp
    ) noexcept -> void {
        const auto* PIQUANT_RESTRICT x {static_cast<const IN*>(in)};
        auto* PIQUANT_RESTRICT o {static_cast<IN*>(out)};
        double inv_scale {1.0 / scale};
        if constexpr (RDO == reduce_op::set) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] = dequant_step<QUANT, IN>(scale, zp, quant_step<RND, IN, QUANT>(inv_scale, zp, x[i]));
        } else if constexpr (RDO == reduce_op::add) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] += dequant_step<QUANT, IN>(scale, zp, quant_step<RND, IN, QUANT>(inv_scale, zp, x[i]));
        } else
            panic("Invalid reduce operation");
    }

    template <typename T> requires std::is_floating_point_v<T>
    [[nodiscard]] auto compute_quant_config_from_data(const T* p, std::int64_t numel) -> std::array<T, 2> {
        if (!numel) [[unlikely]] return {0.0, 0.0};
        T sum {};
        T sum_sq {};
        for (std::int64_t i {}; i < numel; ++i) {
            T x {p[i]};
            sum += x;
            sum_sq += x*x;
        }
        return {sum, sum_sq};
    }

    #ifdef __AVX2__
        [[nodiscard]] static auto avx2_hsum256(__m256 x) noexcept -> float {
            __m128 hiQuad = _mm256_extractf128_ps(x, 1);
            __m128 loQuad = _mm256_castps256_ps128(x);
            __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
            __m128 loDual = sumQuad;
            __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
            __m128 sumDual = _mm_add_ps(loDual, hiDual);
            __m128 lo = sumDual;
            __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
            __m128 sum = _mm_add_ss(lo, hi);
            return _mm_cvtss_f32(sum);
        }
    #endif

    template <>
    [[nodiscard]] auto compute_quant_config_from_data(const float* p, std::int64_t numel) -> std::array<float, 2> {
        if (!numel) [[unlikely]] return {0.0f, 0.0};
        float sum {};
        float sum_sq {};
        std::int64_t i {};
        #if defined(__AVX512F__) && defined(__AVX512BW__) && 0

        #elif defined(__AVX2__)
            __m256 vsum1 {_mm256_setzero_ps()};
            __m256 vsum2 {_mm256_setzero_ps()};
            __m256 vsum3 {_mm256_setzero_ps()};
            __m256 vsum4 {_mm256_setzero_ps()};
            __m256 vsum5 {_mm256_setzero_ps()};
            __m256 vsum6 {_mm256_setzero_ps()};
            __m256 vsum7 {_mm256_setzero_ps()};
            __m256 vsum8 {_mm256_setzero_ps()};
            __m256 vsum_sq1 {_mm256_setzero_ps()};
            __m256 vsum_sq2 {_mm256_setzero_ps()};
            __m256 vsum_sq3 {_mm256_setzero_ps()};
            __m256 vsum_sq4 {_mm256_setzero_ps()};
            __m256 vsum_sq5 {_mm256_setzero_ps()};
            __m256 vsum_sq6 {_mm256_setzero_ps()};
            __m256 vsum_sq7 {_mm256_setzero_ps()};
            __m256 vsum_sq8 {_mm256_setzero_ps()};
            for (; i+63 < numel; i += 64) {
                __m256 v1 {_mm256_loadu_ps(p+i+8*0)};
                __m256 v2 {_mm256_loadu_ps(p+i+8*1)};
                __m256 v3 {_mm256_loadu_ps(p+i+8*2)};
                __m256 v4 {_mm256_loadu_ps(p+i+8*3)};
                __m256 v5 {_mm256_loadu_ps(p+i+8*4)};
                __m256 v6 {_mm256_loadu_ps(p+i+8*5)};
                __m256 v7 {_mm256_loadu_ps(p+i+8*6)};
                __m256 v8 {_mm256_loadu_ps(p+i+8*7)};
                vsum1 = _mm256_add_ps(vsum1, v1);
                vsum2 = _mm256_add_ps(vsum2, v2);
                vsum3 = _mm256_add_ps(vsum3, v3);
                vsum4 = _mm256_add_ps(vsum4, v4);
                vsum5 = _mm256_add_ps(vsum5, v5);
                vsum6 = _mm256_add_ps(vsum6, v6);
                vsum7 = _mm256_add_ps(vsum7, v7);
                vsum8 = _mm256_add_ps(vsum8, v8);
                vsum_sq1 = _mm256_fmadd_ps(v1, v1, vsum_sq1);
                vsum_sq2 = _mm256_fmadd_ps(v2, v2, vsum_sq2);
                vsum_sq3 = _mm256_fmadd_ps(v3, v3, vsum_sq3);
                vsum_sq4 = _mm256_fmadd_ps(v4, v4, vsum_sq4);
                vsum_sq5 = _mm256_fmadd_ps(v5, v5, vsum_sq5);
                vsum_sq6 = _mm256_fmadd_ps(v6, v6, vsum_sq6);
                vsum_sq7 = _mm256_fmadd_ps(v7, v7, vsum_sq7);
                vsum_sq8 = _mm256_fmadd_ps(v8, v8, vsum_sq8);
            }
            __m256 vsum_total {_mm256_add_ps(vsum1, _mm256_add_ps(vsum2, _mm256_add_ps(vsum3, _mm256_add_ps(vsum4, _mm256_add_ps(vsum5, _mm256_add_ps(vsum6, _mm256_add_ps(vsum7, vsum8)))))))};
            __m256 vsum_sq_total {_mm256_add_ps(vsum_sq1, _mm256_add_ps(vsum_sq2, _mm256_add_ps(vsum_sq3, _mm256_add_ps(vsum_sq4, _mm256_add_ps(vsum_sq5, _mm256_add_ps(vsum_sq6, _mm256_add_ps(vsum_sq7, vsum_sq8)))))))};
            sum = avx2_hsum256(vsum_total);
            sum_sq = avx2_hsum256(vsum_sq_total);
        #elif defined(__SSE4_2__)
            __m128 vsum1 {_mm_setzero_ps()};
            __m128 vsum2 {_mm_setzero_ps()};
            __m128 vsum3 {_mm_setzero_ps()};
            __m128 vsum4 {_mm_setzero_ps()};
            __m128 vsum5 {_mm_setzero_ps()};
            __m128 vsum6 {_mm_setzero_ps()};
            __m128 vsum7 {_mm_setzero_ps()};
            __m128 vsum8 {_mm_setzero_ps()};
            __m128 vsum_sq1 {_mm_setzero_ps()};
            __m128 vsum_sq2 {_mm_setzero_ps()};
            __m128 vsum_sq3 {_mm_setzero_ps()};
            __m128 vsum_sq4 {_mm_setzero_ps()};
            __m128 vsum_sq5 {_mm_setzero_ps()};
            __m128 vsum_sq6 {_mm_setzero_ps()};
            __m128 vsum_sq7 {_mm_setzero_ps()};
            __m128 vsum_sq8 {_mm_setzero_ps()};
            for (; i+31 < numel; i += 32) {
                __m128 v1 {_mm_loadu_ps(p+i+4*0)};
                __m128 v2 {_mm_loadu_ps(p+i+4*1)};
                __m128 v3 {_mm_loadu_ps(p+i+4*2)};
                __m128 v4 {_mm_loadu_ps(p+i+4*3)};
                __m128 v5 {_mm_loadu_ps(p+i+4*4)};
                __m128 v6 {_mm_loadu_ps(p+i+4*5)};
                __m128 v7 {_mm_loadu_ps(p+i+4*6)};
                __m128 v8 {_mm_loadu_ps(p+i+4*7)};
                vsum1 = _mm_add_ps(vsum1, v1);
                vsum2 = _mm_add_ps(vsum2, v2);
                vsum3 = _mm_add_ps(vsum3, v3);
                vsum4 = _mm_add_ps(vsum4, v4);
                vsum5 = _mm_add_ps(vsum5, v5);
                vsum6 = _mm_add_ps(vsum6, v6);
                vsum7 = _mm_add_ps(vsum7, v7);
                vsum8 = _mm_add_ps(vsum8, v8);
                vsum_sq1 = _mm_add_ps(vsum_sq1, _mm_mul_ps(v1, v1));
                vsum_sq2 = _mm_add_ps(vsum_sq2, _mm_mul_ps(v2, v2));
                vsum_sq3 = _mm_add_ps(vsum_sq3, _mm_mul_ps(v3, v3));
                vsum_sq4 = _mm_add_ps(vsum_sq4, _mm_mul_ps(v4, v4));
                vsum_sq5 = _mm_add_ps(vsum_sq5, _mm_mul_ps(v5, v5));
                vsum_sq6 = _mm_add_ps(vsum_sq6, _mm_mul_ps(v6, v6));
                vsum_sq7 = _mm_add_ps(vsum_sq7, _mm_mul_ps(v7, v7));
                vsum_sq8 = _mm_add_ps(vsum_sq8, _mm_mul_ps(v8, v8));
            }
            __m128 vsum_total {_mm_add_ps(vsum1, _mm_add_ps(vsum2, _mm_add_ps(vsum3, _mm_add_ps(vsum4, _mm_add_ps(vsum5, _mm_add_ps(vsum6, _mm_add_ps(vsum7, vsum8)))))))};
            __m128 vsum_sq_total {_mm_add_ps(vsum_sq1, _mm_add_ps(vsum_sq2, _mm_add_ps(vsum_sq3, _mm_add_ps(vsum_sq4, _mm_add_ps(vsum_sq5, _mm_add_ps(vsum_sq6, _mm_add_ps(vsum_sq7, vsum_sq8)))))))};
            sum = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(vsum_total, vsum_total), vsum_total));
            sum_sq = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(vsum_sq_total, vsum_sq_total), vsum_sq_total));
        #elif defined(__aarch64__) && defined(__ARM_NEON__)
            float32x4_t vsum1 {vdupq_n_f32(0.0f)};
            float32x4_t vsum2 {vdupq_n_f32(0.0f)};
            float32x4_t vsum3 {vdupq_n_f32(0.0f)};
            float32x4_t vsum4 {vdupq_n_f32(0.0f)};
            float32x4_t vsum5 {vdupq_n_f32(0.0f)};
            float32x4_t vsum6 {vdupq_n_f32(0.0f)};
            float32x4_t vsum7 {vdupq_n_f32(0.0f)};
            float32x4_t vsum8 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq1 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq2 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq3 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq4 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq5 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq6 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq7 {vdupq_n_f32(0.0f)};
            float32x4_t vsum_sq8 {vdupq_n_f32(0.0f)};
            for (; i+31 < numel; i += 32) {
                float32x4_t v1 {vld1q_f32(p+i+(0<<2))};
                float32x4_t v2 {vld1q_f32(p+i+(1<<2))};
                float32x4_t v3 {vld1q_f32(p+i+(2<<2))};
                float32x4_t v4 {vld1q_f32(p+i+(3<<2))};
                float32x4_t v5 {vld1q_f32(p+i+(4<<2))};
                float32x4_t v6 {vld1q_f32(p+i+(5<<2))};
                float32x4_t v7 {vld1q_f32(p+i+(6<<2))};
                float32x4_t v8 {vld1q_f32(p+i+(7<<2))};
                vsum1 = vaddq_f32(vsum1, v1);
                vsum2 = vaddq_f32(vsum2, v2);
                vsum3 = vaddq_f32(vsum3, v3);
                vsum4 = vaddq_f32(vsum4, v4);
                vsum5 = vaddq_f32(vsum5, v5);
                vsum6 = vaddq_f32(vsum6, v6);
                vsum7 = vaddq_f32(vsum7, v7);
                vsum8 = vaddq_f32(vsum8, v8);
                vsum_sq1 = vmlaq_f32(vsum_sq1, v1, v1);
                vsum_sq2 = vmlaq_f32(vsum_sq2, v2, v2);
                vsum_sq3 = vmlaq_f32(vsum_sq3, v3, v3);
                vsum_sq4 = vmlaq_f32(vsum_sq4, v4, v4);
                vsum_sq5 = vmlaq_f32(vsum_sq5, v5, v5);
                vsum_sq6 = vmlaq_f32(vsum_sq6, v6, v6);
                vsum_sq7 = vmlaq_f32(vsum_sq7, v7, v7);
                vsum_sq8 = vmlaq_f32(vsum_sq8, v8, v8);
            }
            float32x4_t vsum_total {vaddq_f32(vsum1, vaddq_f32(vsum2, vaddq_f32(vsum3, vaddq_f32(vsum4, vaddq_f32(vsum5, vaddq_f32(vsum6, vaddq_f32(vsum7, vsum8)))))))};
            float32x4_t vsum_sq_total {vaddq_f32(vsum_sq1, vaddq_f32(vsum_sq2, vaddq_f32(vsum_sq3, vaddq_f32(vsum_sq4, vaddq_f32(vsum_sq5, vaddq_f32(vsum_sq6, vaddq_f32(vsum_sq7, vsum_sq8)))))))};
            sum = vaddvq_f32(vsum_total);
            sum_sq = vaddvq_f32(vsum_sq_total);
        #endif
        for (; i < numel; ++i) {
            float x {p[i]};
            sum += x;
            sum_sq += x*x;
        }
        return {sum, sum_sq};
    }

    static auto PIQUANT_HOT quant_config_kernel_f32(std::span<const float> x) noexcept -> std::array<float, 2> {
        return compute_quant_config_from_data(x.data(), static_cast<std::int64_t>(x.size()));
    }

    static auto PIQUANT_HOT quant_config_kernel_f64(std::span<const double> x) noexcept -> std::array<double, 2> {
        return compute_quant_config_from_data(x.data(), static_cast<std::int64_t>(x.size()));
    }
};

namespace piquant {
    [[nodiscard]] constexpr auto make_pair_perm(dtype from,dtype to) noexcept -> std::uint16_t {
        auto ito {static_cast<std::underlying_type_t<decltype(to)>>(to)};
        auto ifrom {static_cast<std::underlying_type_t<decltype(from)>>(from)};
        return ((255&ifrom)<<8)+(255&ito);
    }

    static auto PIQUANT_HOT quantize_dispatch(
        const void* x,
        void* o,
        std::int64_t range,
        const context::quant_descriptor& desc
    ) noexcept -> void {
        using enum dtype;
        const dtype_info& dt_in {dtype_info_of(desc.dt_in)};
        const dtype_info& dt_out {dtype_info_of(desc.dt_out)};
        switch (desc.type) {
            case context::command_type::quant:  // out[i] = quantize(in[i])
                piquant_assert2(!(dt_in.flags & dtype_flags::is_quant));
                piquant_assert2(dt_out.flags & dtype_flags::is_quant);
                #define impl_quant_perm(dti, dto, ti, to) \
                    case make_pair_perm(dti, dto): \
                        if (desc.rnd_mode == round_mode::stochastic) \
                            impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<ti, to, round_mode::stochastic>(x, o, range, desc.scale, desc.zero_point); \
                        else \
                            impl_namespace(QUANT_KERNEL_IMPL, _)::quant_generic<ti, to, round_mode::nearest>(x, o, range, desc.scale, desc.zero_point); \
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
                piquant_assert2(dt_in.flags & dtype_flags::is_quant);
                piquant_assert2(!(dt_out.flags & dtype_flags::is_quant));
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
                piquant_assert2(!(dt_in.flags & dtype_flags::is_quant));
                piquant_assert2(dt_out.flags & dtype_flags::is_quant); // dt_out acts as the quantized type, but dtype in == dtype out
               #define impl_quant_perm(dti, dto, ti, to) \
                    case make_pair_perm(dti, dto): \
                        if (desc.reduce == reduce_op::set) \
                            if (desc.rnd_mode == round_mode::stochastic) \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::stochastic, reduce_op::set>(x, o, range, desc.scale, desc.zero_point); \
                            else \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::nearest, reduce_op::set>(x, o, range, desc.scale, desc.zero_point); \
                        else \
                            if (desc.rnd_mode == round_mode::stochastic) \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::stochastic, reduce_op::add>(x, o, range, desc.scale, desc.zero_point); \
                            else \
                                impl_namespace(QUANT_KERNEL_IMPL, _)::quant_dequant_generic<ti, to, round_mode::nearest, reduce_op::add>(x, o, range, desc.scale, desc.zero_point); \
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
