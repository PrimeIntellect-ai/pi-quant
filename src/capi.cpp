#include <piquant.h>
#include <piquant.hpp>
#include <bit>

using namespace piquant;

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
    float scale,
    int32_t zero_point,
    piquant_round_mode_t mode
) -> void {
    std::span in_span {static_cast<const std::byte*>(in), numel*(dtype_info_of(static_cast<dtype>(dtype_in)).bit_size>>3)};
    std::span out_span {static_cast<std::byte*>(out), numel*(dtype_info_of(static_cast<dtype>(dtype_out)).bit_size>>3)};
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
    float scale,
    int32_t zero_point,
    piquant_reduce_op_t op
) -> void {
    std::span in_span {static_cast<const std::byte*>(in), numel*(dtype_info_of(static_cast<dtype>(dtype_in)).bit_size>>3)};
    std::span out_span {static_cast<std::byte*>(out), numel*(dtype_info_of(static_cast<dtype>(dtype_out)).bit_size>>3)};
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

extern "C" auto piquant_compute_quant_config_from_data(piquant_context_t* ctx, const float* const x, const std::size_t n, const piquant_dtype_t target_quant_dtype, float* const out_scale, int64_t* const out_zero_point) -> void {
    const auto [scale, zero_point] {
        std::bit_cast<context*>(ctx)->compute_quant_config_from_data(std::span{x, n}, static_cast<dtype>(target_quant_dtype))
    };
    *out_scale = scale;
    *out_zero_point = zero_point;
}
