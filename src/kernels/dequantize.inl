// This inline file is directly included into the kernels.inl file, which is cloned (recompiled) in multiple compilation units for different CPU architectures.
// ! Make sure all functions are static, to make them local to the compilation unit.

#include "../piquant_internal.hpp"

using namespace piquant;

template <typename In, typename Out> requires is_quant_type<In> && is_float_type<Out>
[[nodiscard]] static auto dequant_step(fp32_t scale, std::int64_t zp, const In x) noexcept -> Out {
    return static_cast<Out>(static_cast<std::int64_t>(x) - zp)*scale;
}

template <typename In, typename Out, const reduce_op ReduceOp> requires std::is_same_v<uint4_t, In> && is_float_type<Out>
static auto PIQUANT_HOT dequant_uint4(
    const In* x,
    Out* o,
    std::int64_t numel,
    fp32_t scale,
    std::int64_t zp
) noexcept -> void {
    std::int64_t i{};
    for (std::int64_t j {}; i+1 < numel; i += 2, ++j) {
        auto p {x[j].bits};
        auto qa {p & 15};
        auto qb {p >> 4};
        if constexpr (ReduceOp == reduce_op::set) {
            o[i] = dequant_step<In, Out>(scale, zp, qa);
            o[i+1] = dequant_step<In, Out>(scale, zp, qb);
        } else if constexpr (ReduceOp == reduce_op::add) {
            o[i] += dequant_step<In, Out>(scale, zp, qa);
            o[i+1] += dequant_step<In, Out>(scale, zp, qb);
        }
    }
    if (numel & 1) {
        auto qa {x[i>>1].bits & 15};
        Out r = dequant_step<In, Out>(scale, zp, qa);
        if constexpr (ReduceOp == reduce_op::set) o[numel-1] = r;
        else if constexpr (ReduceOp == reduce_op::add) o[numel-1] += r;
    }
}

template <typename In, typename Out, const reduce_op ReduceOp> requires std::is_same_v<uint2_t, In> && is_float_type<Out>
static auto PIQUANT_HOT dequant_uint2(
    const In* x,
    Out* o,
    std::int64_t numel,
    fp32_t scale,
    std::int64_t zp
) noexcept -> void {
    std::int64_t i {};
    std::int64_t j {};
    for (; i+3 < numel; i += 4, ++j) {
        auto p {x[j].bits};
        auto qa {p & 3};
        auto qb {p>>2 & 3};
        auto qc {p>>4 & 3};
        auto qd {p>>6 & 3};
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
    auto p {x[i>>2].bits};
    switch (numel&3) {
        case 1:
            o[i] = dequant_step<In, Out>(scale, zp, p&3);
            break;
        case 2:
            o[i] = dequant_step<In, Out>(scale, zp, p&3);
            o[i+1] = dequant_step<In, Out>(scale, zp, (p>>2)&3);
            break;
        case 3:
            o[i] = dequant_step<In, Out>(scale, zp, p&3);
            o[i+1] = dequant_step<In, Out>(scale, zp, (p>>2)&3);
            o[i+2] = dequant_step<In, Out>(scale, zp, (p>>4)&3);
            break;
    }
}

template <typename In, typename Out, const reduce_op ReduceOp> requires is_quant_type<In> && is_float_type<Out>
static auto PIQUANT_HOT dequant_generic(
    const void* in,
    void* out,
    std::int64_t numel,
    fp32_t scale,
    std::int64_t zp
) noexcept -> void {
    const auto* PIQUANT_RESTRICT x {static_cast<const In*>(in)};
    auto* PIQUANT_RESTRICT o {static_cast<Out*>(out)};

    // Use SIMD optimized kernels for some dtype permutations
    if constexpr (std::is_same_v<In, std::uint8_t> && std::is_same_v<Out, fp32_t>) {
        dequant_uint8_to_f32<ReduceOp>(static_cast<const std::uint8_t*>(in), static_cast<fp32_t*>(out), numel, scale, static_cast<std::int32_t>(zp));
        return;
    }
    if constexpr (std::is_same_v<In, uint4_t> && std::is_same_v<Out, fp32_t>) {
        dequant_uint4_to_f32<ReduceOp>(static_cast<const uint4_t*>(in), static_cast<fp32_t*>(out), numel, scale, static_cast<std::int32_t>(zp));
        return;
    }
    if constexpr (std::is_same_v<In, std::uint8_t> && std::is_same_v<Out, bfp16_t>) {
        dequant_uint8_to_bf16<ReduceOp>(static_cast<const std::uint8_t*>(in), static_cast<bfp16_t*>(out), numel, scale, static_cast<std::int32_t>(zp));
        return;
    }
    if constexpr (std::is_same_v<In, uint4_t> && std::is_same_v<Out, bfp16_t>) {
        dequant_uint4_to_bf16<ReduceOp>(static_cast<const uint4_t*>(in), static_cast<bfp16_t*>(out), numel, scale, static_cast<std::int32_t>(zp));
        return;
    }
    if constexpr (std::is_same_v<In, uint2_t> && std::is_same_v<Out, bfp16_t>) {
        dequant_uint2_to_bf16<ReduceOp>(static_cast<const uint2_t*>(in), static_cast<bfp16_t*>(out), numel, scale, static_cast<std::int32_t>(zp));
        return;
    }

    if constexpr (std::is_same_v<uint4_t, In>) { // Special case for int4
        dequant_uint4<In, Out, ReduceOp>(x, o, numel, scale, zp);
        return;
    }

    if constexpr (std::is_same_v<uint2_t, In>) { // Special case for int2
        dequant_uint2<In, Out, ReduceOp>(x, o, numel, scale, zp);
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