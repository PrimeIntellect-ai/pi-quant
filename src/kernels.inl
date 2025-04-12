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

        [[nodiscard]] auto PIQUANT_AINLINE operator ()() noexcept -> std::uint64_t {
            std::uint64_t s1 {p1};
            std::uint64_t s0 {p2};
            p1 = s0;
            s1 ^= s1<<23;
            p2 = s1^s0^(s1>>18)^(s0>>5);
            return p2 + s0;
        }

        [[nodiscard]] auto PIQUANT_AINLINE canonical() noexcept -> float {
            static constexpr auto bias_scale {1.0f/static_cast<float>(0x800000)};
            std::uint64_t y {~0u & (*this)()};
            return (bias_scale*(static_cast<float>(y>>9) + 0.5f));
        }

        [[nodiscard]] auto PIQUANT_AINLINE bounded(std::array<std::uint32_t, 2> bounds) noexcept -> std::array<std::uint32_t, 2> {
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

            [[nodiscard]] auto operator ()() noexcept -> __m256i {
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

        [[nodiscard]] auto canonical() noexcept -> __m256 {
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

    static thread_local constinit xs128p_state s_sprng {0x123456789abcdef0, 0x0fedcba987654321};

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
        auto xi {static_cast<std::uint8_t>(x)};
        auto yi {static_cast<std::uint8_t>(y)};
        auto pa {static_cast<std::uint8_t>((0xf&xi)<<4|0xf&yi)};
        return static_cast<OUT>(pa);
    }

    static auto PIQUANT_HOT quant_f32_to_uint4_nearest(
        const float* PIQUANT_RESTRICT x,
        uint4_t* PIQUANT_RESTRICT o,
        std::int64_t numel,
        float scale,
        std::int32_t zp
   ) noexcept -> void {
        scale = 1.0f / scale; /* We multiply by reciprocal */
        std::int64_t i {};
        #if defined(__AVX512F__) && defined(__AVX512BW__) && 0

        #elif defined(__AVX2__)

        #elif defined(__SSE4_2__)

        #elif defined(__aarch64__) && defined(__ARM_NEON__)

        #endif
        numel = (numel+1)>>1;
        for (; i < numel; ++i) {
            auto x1 {static_cast<uint4_t>(std::clamp(static_cast<std::int32_t>(std::round(x[i]*scale)) + zp, 0, 0xf))};
            auto x2 {static_cast<uint4_t>(std::clamp(static_cast<std::int32_t>(std::round(x[i+1]*scale)) + zp, 0, 0xf))};
            o[i] = pack_nibbles(x1, x2);
        }
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
            __m512 vinv_scale {_mm512_set1_ps(inv_scale)};
            __m512i vzero_point {_mm512_set1_epi32(zp)};
            __m512i vmin {_mm512_setzero_si512()};
            __m512i vmax {_mm512_set1_epi32(0xff)};
            constexpr auto k_round_mode {_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC};
            for (; i+63 < numel; i += 64) {
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
            float32x4_t vinv_scale {vdupq_n_f32(scale)};
            int32x4_t vzero_point {vdupq_n_s32(zp)};
            int32x4_t vmin {vdupq_n_s32(0)};
            int32x4_t vmax {vdupq_n_s32(0xff)};
            for (; i+15 <= numel; i += 16) {
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

    template <const round_mode RND, typename IN, typename OUT, typename... Args>
         requires (std::is_floating_point_v<IN> && (std::is_integral_v<OUT> || is_int4<OUT>)
             && std::is_same_v<std::common_type_t<Args...>, IN> && sizeof...(Args) != 0)
    static auto PIQUANT_AINLINE quant_step(double inv_scale, std::int64_t zp, Args... args) noexcept -> OUT {
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
        } else if constexpr (std::is_same_v<IN, float> && std::is_same_v<OUT, uint4_t> && RND == round_mode::nearest) {
            quant_f32_to_uint4_nearest(static_cast<const float*>(in), static_cast<uint4_t*>(out), numel, scale, zp);
            return;
        }
        const auto* PIQUANT_RESTRICT x {static_cast<const IN*>(in)};
        auto* PIQUANT_RESTRICT o {static_cast<OUT*>(out)};
        double inv_scale {1.0 / static_cast<double>(scale)}; // We multiply by reciprocal
        if constexpr (is_int4<OUT>) numel = (numel+1)>>1;
        for (std::int64_t i {}; i < numel; ++i)
            if constexpr (is_int4<OUT>)
                o[i] = quant_step<RND, IN, OUT>(inv_scale, zp, x[i], x[i+1]);
            else
                o[i] = quant_step<RND, IN, OUT>(inv_scale, zp, x[i]);
    }

    template <typename IN, typename OUT>
          requires (std::is_floating_point_v<OUT> && (std::is_integral_v<IN> || is_int4<IN>))
    static auto PIQUANT_AINLINE dequant_step(double scale, std::int64_t zp, const IN x) noexcept -> OUT {
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
        if constexpr (RDO == reduce_op::set) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] = dequant_step<IN, OUT>(scale, zp, x[i]);
        } else if constexpr (RDO == reduce_op::add) {
            for (std::int64_t i {}; i < numel; ++i)
                o[i] += dequant_step<IN, OUT>(scale, zp, x[i]);
        } else
            panic("Invalid reduce operation");
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
    [[nodiscard]] auto compute_quant_config_from_data(const T* p, std::int64_t numel, std::int64_t tmax) -> std::pair<T, std::int64_t> {
        if (!numel) [[unlikely]] return {0.0, 0.0};
        T mean {static_cast<T>(std::accumulate(p, p+numel, 0.0) / static_cast<T>(numel))};
        const auto sq_delta {static_cast<T>(std::transform_reduce(
            p, p+numel,
            0.0,
            std::plus{},
            [mean](const T value) noexcept -> T {
                T delta {value - mean};
                return delta*delta;
            }
        ))};
        T std {static_cast<T>(std::sqrt(sq_delta / static_cast<T>(numel-1)))};
        T scale {static_cast<T>(stddev_scale*std/static_cast<T>(tmax))};
        if (scale == 0.0) [[unlikely]] {
            return {1.0f, (tmax+1)>>1};
        }
        std::int64_t zp {(tmax>>1) - static_cast<std::int64_t>(std::round(mean/scale))};
        return {scale, zp};
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
    [[nodiscard]] auto compute_quant_config_from_data(const float* p, std::int64_t numel, std::int64_t tmax) -> std::pair<float, std::int64_t> {
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
                float32x4_t v1 {vld1q_f32(p+i+4*0)};
                float32x4_t v2 {vld1q_f32(p+i+4*1)};
                float32x4_t v3 {vld1q_f32(p+i+4*2)};
                float32x4_t v4 {vld1q_f32(p+i+4*3)};
                float32x4_t v5 {vld1q_f32(p+i+4*4)};
                float32x4_t v6 {vld1q_f32(p+i+4*5)};
                float32x4_t v7 {vld1q_f32(p+i+4*6)};
                float32x4_t v8 {vld1q_f32(p+i+4*7)};
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
        double mean {};
        double m2k {};
        for (std::int64_t i=0; i < numel; ++i) {
            double x {p[i]};
            double delta {x - mean};
            mean += delta/static_cast<double>(i + 1);
            m2k += delta*(x - mean);
        }
        double variance {(numel > 1) ? (m2k/static_cast<double>(numel - 1)) : 0.0};
        double stddev {std::sqrt(variance)};
        double scale {(stddev_scale*stddev / static_cast<double>(tmax))};
        if (scale == 0.0) [[unlikely]] {
           return {1.0f, (tmax+1)>>1};
        }
        std::int64_t zp {((tmax+1)>>1) - static_cast<std::int64_t>(std::round(mean / scale))};
        return {static_cast<float>(scale), zp};
    }

    static auto PIQUANT_HOT quant_config_kernel_f32(std::span<const float> x, std::int64_t tmax) noexcept -> std::pair<float, std::int64_t> {
        return compute_quant_config_from_data(x.data(), x.size(), tmax);
    }

    static auto PIQUANT_HOT quant_config_kernel_f64(std::span<const double> x, std::int64_t tmax) noexcept -> std::pair<float, std::int64_t> {
        return compute_quant_config_from_data(x.data(), x.size(), tmax);
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
