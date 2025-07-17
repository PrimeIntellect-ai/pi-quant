#include <piquant.h>
#include <piquant.hpp>
#include "piquant_internal.hpp"

#include <bit>

using namespace piquant;

static_assert(static_cast<int>(dtype::f32) == PIQUANT_DTYPE_F32);
static_assert(static_cast<int>(dtype::bf16) == PIQUANT_DTYPE_BF16);
static_assert(static_cast<int>(dtype::uint2) == PIQUANT_DTYPE_UINT2);
static_assert(static_cast<int>(dtype::uint4) == PIQUANT_DTYPE_UINT4);
static_assert(static_cast<int>(dtype::uint8) == PIQUANT_DTYPE_UINT8);

struct piquant_context_t final {
    context* ctx {};
};

extern "C" auto piquant_context_create(const std::size_t num_threads) -> piquant_context_t* {
    auto* ctx {new context{num_threads}}; // todo
    return std::bit_cast<piquant_context_t*>(ctx);
}

extern "C" auto piquant_context_destroy(piquant_context_t* ctx) -> void {
    delete std::bit_cast<context*>(ctx);
}

extern "C" auto piquant_quantize(
    piquant_context_t* ctx,
    const void* in,
    piquant_dtype_t dtype_in,
    void* out,
    piquant_dtype_t dtype_out,
    size_t numel,
    fp32_t scale,
    int64_t zero_point,
    piquant_round_mode_t mode
) -> void {
    const auto& dti {dtype_info_of(static_cast<dtype>(dtype_in))};
    const auto& dto {dtype_info_of(static_cast<dtype>(dtype_out))};
    std::size_t in_bytes {numel*dti.stride};
    std::size_t out_bytes {dto.bit_size == 8 ? numel*dto.stride : packed_numel(numel, dto)*dto.stride};
    std::span in_span {static_cast<const std::byte*>(in), in_bytes};
    std::span out_span {static_cast<std::byte*>(out), out_bytes};
    std::bit_cast<context*>(ctx)->quantize(
        in_span,
        static_cast<dtype>(dtype_in),
        out_span,
        static_cast<dtype>(dtype_out),
        scale,
        zero_point,
        static_cast<round_mode>(mode)
    );
}

extern "C" auto piquant_dequantize(
    piquant_context_t* ctx,
    const void* in,
    piquant_dtype_t dtype_in,
    void* out,
    piquant_dtype_t dtype_out,
    size_t numel,
    fp32_t scale,
    int64_t zero_point,
    piquant_reduce_op_t op
) -> void {
    const auto& dti {dtype_info_of(static_cast<dtype>(dtype_in))};
    const auto& dto {dtype_info_of(static_cast<dtype>(dtype_out))};
    std::size_t in_bytes  {dti.bit_size == 8 ? numel*dti.stride : packed_numel(numel, dti)*dti.stride};
    std::size_t out_bytes {numel*dto.stride};
    std::span in_span {static_cast<const std::byte*>(in), in_bytes};
    std::span out_span {static_cast<std::byte*>(out), out_bytes};
    std::bit_cast<context*>(ctx)->dequantize(
        in_span,
        static_cast<dtype>(dtype_in),
        out_span,
        static_cast<dtype>(dtype_out),
        scale,
        zero_point,
        static_cast<reduce_op>(op)
    );
}

extern "C" auto piquant_compute_quant_params_float32(piquant_context_t* ctx, const fp32_t* const x, const std::size_t n, const piquant_dtype_t target_quant_dtype, fp32_t* const out_scale, int64_t* const out_zero_point) -> void {
    const auto [scale, zero_point] {
        std::bit_cast<context*>(ctx)->compute_quant_config_from_data(
            std::span{x, n},
            static_cast<dtype>(target_quant_dtype)
        )
    };
    *out_scale = scale;
    *out_zero_point = zero_point;
}

extern "C" auto piquant_compute_quant_params_bfloat16(piquant_context_t* ctx, const uint16_t* const x, const std::size_t n, const piquant_dtype_t target_quant_dtype, fp32_t* const out_scale, int64_t* const out_zero_point) -> void {
    const auto [scale, zero_point] {
        std::bit_cast<context*>(ctx)->compute_quant_config_from_data(
            std::span{reinterpret_cast<const bfp16_t*>(x), n},
            static_cast<dtype>(target_quant_dtype)
        )
    };
    *out_scale = scale;
    *out_zero_point = zero_point;
}
