// This inline file is directly included into the kernels.inl file, which is cloned (recompiled) in multiple compilation units for different CPU architectures.
// ! Make sure all functions are static, to make them local to the compilation unit.

#include "../piquant_internal.hpp"

using namespace piquant;

static constinit thread_local xs128p_state s_sprng {0x123456789abcdef0, 0x0fedcba987654321};

template <typename In, typename Out> requires is_float_type<In> && is_quant_type<Out>
[[nodiscard]] static auto PIQUANT_AINLINE quant_step_scalar_stochastic(In x, float64_t inv_scale, std::int64_t zp) noexcept -> Out {
    float64_t rnd {x * inv_scale};
    float64_t dec {std::abs(rnd - std::trunc(rnd))};
    float64_t xi {(s_sprng.canonical())};
    float64_t adj {xi < dec ? 1.0f : 0.0f};
    if (rnd < 0.0f) adj = -1.0f * adj;
    rnd = std::trunc(rnd) + adj;
    auto integral {static_cast<std::int64_t>(rnd) + zp};
    const auto min = dtype_limits<Out>::min;
    const auto max = dtype_limits<Out>::max;
    return static_cast<Out>(std::clamp<decltype(integral)>(integral, min, max));
}

template <typename In, typename Out> requires is_float_type<In> && is_quant_type<Out>
[[nodiscard]] static auto PIQUANT_AINLINE quant_step_scalar_nearest(In x, float64_t inv_scale, std::int64_t zp) noexcept -> Out {
    float64_t rnd {std::round(static_cast<float64_t>(x) * inv_scale)};
    auto integral {static_cast<std::int64_t>(rnd) + zp};
    return static_cast<Out>(std::clamp<decltype(integral)>(integral, dtype_limits<Out>::min, dtype_limits<Out>::max));
}

template <typename In, typename Out, const round_mode RoundMode> requires is_float_type<In> && is_quant_type<Out>
[[nodiscard]] static auto PIQUANT_AINLINE quant_step_scalar(In x, float64_t inv_scale, std::int64_t zp) noexcept -> Out {
    if constexpr (RoundMode == round_mode::stochastic)
        return quant_step_scalar_stochastic<In, Out>(x, inv_scale, zp);
    else
        return quant_step_scalar_nearest<In, Out>(x, inv_scale, zp);
}

template <typename In, typename Out, const round_mode RoundMode> requires is_float_type<In> && is_quant_type<Out>
[[nodiscard]] static auto PIQUANT_AINLINE quant_step_packed(In a, In b, float64_t inv_scale, std::int64_t zp) noexcept -> Out {
    auto qa {quant_step_scalar<In, Out, RoundMode>(a, inv_scale, zp).bits};
    auto qb {quant_step_scalar<In, Out, RoundMode>(b, inv_scale, zp).bits};
    return qa & 15 | (qb & 15)<<4;
}

template <typename In, typename Out, const round_mode RoundMode> requires is_float_type<In> && is_quant_type<Out>
[[nodiscard]] static auto PIQUANT_AINLINE quant_step_packed(In a, In b, In c, In d, float64_t inv_scale, std::int64_t zp) noexcept -> Out {
    auto qa {quant_step_scalar<In, Out, RoundMode>(a, inv_scale, zp).bits};
    auto qb {quant_step_scalar<In, Out, RoundMode>(b, inv_scale, zp).bits};
    auto qc {quant_step_scalar<In, Out, RoundMode>(c, inv_scale, zp).bits};
    auto qd {quant_step_scalar<In, Out, RoundMode>(d, inv_scale, zp).bits};
    return qa & 3 | (qb & 3)<<2 | (qc & 3)<<4 | (qd & 3)<<6;
}

template <typename In, typename Out, const round_mode RoundMode> requires is_float_type<In> && is_int4<Out>
static auto PIQUANT_HOT quant_int4(
    const In* PIQUANT_RESTRICT x,
    Out* PIQUANT_RESTRICT o,
    std::int64_t numel,
    float64_t inv_scale,
    std::int64_t zp
) noexcept -> void {
    std::int64_t i {};
    for (; i+1 < numel; i += 2) {
        In a {x[i]};
        In b {x[i+1]};
        o[i>>1] = quant_step_packed<In, Out, RoundMode>(a, b, inv_scale, zp);
    }
    if (numel & 1) {
        o[i>>1] = quant_step_packed<In, Out, RoundMode>(x[numel-1], 0, inv_scale, zp);
        o[i>>1].bits &= 15;
    }
}

template <typename In, typename Out, const round_mode RoundMode> requires is_float_type<In> && is_int2<Out>
static auto PIQUANT_HOT quant_int2(
    const In* PIQUANT_RESTRICT x,
    Out* PIQUANT_RESTRICT o,
    std::int64_t numel,
    float64_t inv_scale,
    std::int64_t zp
) noexcept -> void {
    std::int64_t i {};
    for (; i+3 < numel; i += 4) {
        In a {x[i]};
        In b {x[i+1]};
        In c {x[i+2]};
        In d {x[i+3]};
        o[i>>2] = quant_step_packed<In, Out, RoundMode>(a, b, c, d, inv_scale, zp);
    }
    if (numel & 3) { /* Handle 1-, 2- or 3-value tail */
        typename Out::packed_storage p {};
        switch (numel & 3) {
            case 3: p |= (quant_step_scalar<In, Out, RoundMode>(x[i+2], inv_scale, zp).bits&3) << 4;
            case 2: p |= (quant_step_scalar<In, Out, RoundMode>(x[i+1], inv_scale, zp).bits&3) << 2;
            case 1: p |= (quant_step_scalar<In, Out, RoundMode>(x[i], inv_scale, zp).bits&3);
        }
        o[i>>2] = p;
    }
}

template <typename In, typename Out, const round_mode RoundMode> requires is_float_type<In> && is_quant_type<Out>
static auto PIQUANT_HOT quant_generic(
    const void* in,
    void* out,
    std::int64_t numel,
    float32_t scale,
    std::int64_t zp
) noexcept -> void {
    // Use SIMD optimized kernels for some dtype permutations
    if constexpr (std::is_same_v<In, float32_t> && std::is_same_v<Out, std::uint8_t> && RoundMode == round_mode::nearest) {
        quant_f32_to_uint8_nearest(static_cast<const float32_t*>(in), static_cast<std::uint8_t*>(out), numel, scale, zp);
        return;
    }
    if constexpr (std::is_same_v<In, float32_t> && std::is_same_v<Out, uint4_t> && RoundMode == round_mode::nearest) {
        quant_f32_to_uint4_nearest(static_cast<const float32_t*>(in), static_cast<uint4_t*>(out), numel, scale, zp);
        return;
    }

    const auto* PIQUANT_RESTRICT x {static_cast<const In*>(in)};
    auto* PIQUANT_RESTRICT o {static_cast<Out*>(out)};
    float64_t inv_scale {1.0 / static_cast<float64_t>(scale)}; // We multiply by reciprocal

    if constexpr (is_int4<Out>) { // Special case for int4
        quant_int4<In, Out, RoundMode>(x, o, numel, inv_scale, zp);
        return;
    }

    if constexpr (is_int2<Out>) { // Special case for int2
        quant_int2<In, Out, RoundMode>(x, o, numel, inv_scale, zp);
        return;
    }

    // Generic quantization for other dtypes
    for (std::int64_t i=0; i < numel; ++i)
        o[i] = quant_step_scalar<In, Out, RoundMode>(x[i], inv_scale, zp);
}