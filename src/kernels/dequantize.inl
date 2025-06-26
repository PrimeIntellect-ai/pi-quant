// This inline file is directly included into the kernels.inl file, which is cloned (recompiled) in multiple compilation units for different CPU architectures.
// ! Make sure all functions are static, to make them local to the compilation unit.

#include "../piquant_internal.hpp"

using namespace piquant;

template <typename In, typename Out> requires is_quant_type<In> && is_float_type<Out>
[[nodiscard]] static auto dequant_step(double scale, std::int64_t zp, const In x) noexcept -> Out {
    if constexpr (is_packed_int<In>)
        return static_cast<Out>(static_cast<std::int64_t>(x.u8) - zp)*scale;
    else
        return static_cast<Out>(static_cast<std::int64_t>(x) - zp)*scale;
}

template <typename In, typename Out, const reduce_op ReduceOp> requires is_int4<In> && is_float_type<Out>
static auto PIQUANT_HOT dequant_int4(
    const In* x,
    Out* o,
    std::int64_t numel,
    double scale,
    std::int64_t zp
) noexcept -> void {
    std::int64_t i{};
    for (std::int64_t j{}; i+1 < numel; i += 2, j++) {
        auto [qa, qb]  {x[j].unpack()};
        if constexpr (ReduceOp == reduce_op::set) {
            o[i] = dequant_step<In, Out>(scale, zp, qa);
            o[i+1] = dequant_step<In, Out>(scale, zp, qb);
        } else if constexpr (ReduceOp == reduce_op::add) {
            o[i] += dequant_step<In, Out>(scale, zp, qa);
            o[i+1] += dequant_step<In, Out>(scale, zp, qb);
        } else
            static_assert(ReduceOp == reduce_op::set || ReduceOp == reduce_op::add, "Invalid reduce operation");
    }
    if (numel & 1) {
        auto [qa, qb] {x[i>>1].unpack()};
        Out r = dequant_step<In, Out>(scale, zp, qa);
        if constexpr (ReduceOp == reduce_op::set)
            o[numel-1] = r;
        else if constexpr (ReduceOp == reduce_op::add)
            o[numel-1] += r;
        else
            static_assert(ReduceOp == reduce_op::set || ReduceOp == reduce_op::add, "Invalid reduce operation");
    }
}

template <typename In, typename Out, const reduce_op ReduceOp> requires is_int2<In> && is_float_type<Out>
static auto PIQUANT_HOT dequant_int2(
    const In* x,
    Out* o,
    std::int64_t numel,
    double scale,
    std::int64_t zp
) noexcept -> void {
    std::int64_t i {};
    std::int64_t j {};
    for (; i+3 < numel; i += 4, ++j) {
        auto [qa, qb, qc, qd] = x[j].unpack();
        if constexpr (ReduceOp == reduce_op::set) {
            o[i] = dequant_step<In, Out>(scale, zp, qa);
            o[i+1] = dequant_step<In, Out>(scale, zp, qb);
            o[i+2] = dequant_step<In, Out>(scale, zp, qc);
            o[i+3] = dequant_step<In, Out>(scale, zp, qd);
        } else if constexpr (ReduceOp == reduce_op::add) {
            o[i] += dequant_step<In, Out>(scale, zp, qa);
            o[i+1] += dequant_step<In, Out>(scale, zp, qb);
            o[i+2] += dequant_step<In, Out>(scale, zp, qc);
            o[i+3] += dequant_step<In, Out>(scale, zp, qd);
        } else {
            static_assert(ReduceOp == reduce_op::set || ReduceOp == reduce_op::add, "Invalid reduce operation");
        }
    }
    if (numel & 3) { /* Handle 1-, 2- or 3-value tail */
        auto p {x[i>>2].u8};
        if (numel & 1) o[i] = dequant_step<In, Out>(scale, zp, In{p & 3});
        if (numel & 2) o[i+1] = dequant_step<In, Out>(scale, zp, In{p>>2 & 3});
        if (numel & 3) o[i+((numel & 3) == 3 ? 2 : 0)] = dequant_step<In, Out>(scale, zp, In{p>>4 & 3});
    }
}

template <typename In, typename Out, const reduce_op ReduceOp> requires is_quant_type<In> && is_float_type<Out>
static auto PIQUANT_HOT dequant_generic(
    const void* in,
    void* out,
    std::int64_t numel,
    double scale,
    std::int64_t zp
) noexcept -> void {
    const auto* PIQUANT_RESTRICT x {static_cast<const In*>(in)};
    auto* PIQUANT_RESTRICT o {static_cast<Out*>(out)};

    // Use SIMD optimized kernels for some dtype permutations
    if constexpr (std::is_same_v<In, std::uint8_t> && std::is_same_v<Out, float>) {
        if constexpr (ReduceOp == reduce_op::set) {
            dequant_uint8_to_f32<false>(static_cast<const std::uint8_t*>(in), static_cast<float*>(out), numel, static_cast<float>(scale), static_cast<std::int32_t>(zp));
            return;
        } else if constexpr (ReduceOp == reduce_op::add) {
            dequant_uint8_to_f32<true>(static_cast<const std::uint8_t*>(in), static_cast<float*>(out), numel, static_cast<float>(scale), static_cast<std::int32_t>(zp));
            return;
        }
    }
    if constexpr (is_int4<In>) { // Special case for int4
        dequant_int4<In, Out, ReduceOp>(x, o, numel, scale, zp);
        return;
    }

    if constexpr (is_int2<In>) { // Special case for int2
        dequant_int2<In, Out, ReduceOp>(x, o, numel, scale, zp);
        return;
    }

    // Generic case for other quantized types
    if constexpr (ReduceOp == reduce_op::set) {
        for (std::int64_t i {}; i < numel; ++i)
            o[i] = dequant_step<In, Out>(scale, zp, x[i]);
    } else if constexpr (ReduceOp == reduce_op::add) {
        for (std::int64_t i {}; i < numel; ++i)
            o[i] += dequant_step<In, Out>(scale, zp, x[i]);
    }
}