#include <quant.h>
#include <quant.hpp>
#include <bit>

using namespace quant;

struct quant_context_t final {
    context* ctx {};
};

extern "C" auto compute_quant_config_from_data(const float* const x, const std::size_t n, float* const out_scale, int32_t* const out_zero_point) -> void {
    const auto [scale, zero_point] {compute_quant_config_from_data(std::span{x, n})};
    *out_scale = scale;
    *out_zero_point = zero_point;
}

extern "C" auto quant_context_create(const std::size_t num_threads) -> quant_context_t* {
    auto* ctx {new context{num_threads}};
    return std::bit_cast<quant_context_t*>(ctx);
}

extern "C" auto quant_context_destroy(quant_context_t* ctx) -> void {
    delete std::bit_cast<context*>(ctx);
}

extern "C" auto quant_uint8(
    quant_context_t* const ctx,
    const float* const in,
    std::uint8_t* const out,
    const std::size_t numel,
    const float scale,
    const std::int32_t zero_point,
    const quant_round_mode_t mode
) -> void {
    auto* const ct {std::bit_cast<context*>(ctx)};
    std::span<const float> span_in {in, in+numel};
    std::span<std::uint8_t> span_out {out, out+numel};
    ct->quantize_uint8(
        span_in,
        span_out,
        scale,
        zero_point,
        mode == QUANT_NEAREST ? round_mode::nearest : round_mode::stochastic
    );
}

extern "C" auto dequant_uint8(
    quant_context_t* const ctx,
    const std::uint8_t* const in,
    float* const out,
    const std::size_t numel,
    const float scale,
    const std::int32_t zero_point
) -> void {
    auto* const ct {std::bit_cast<context*>(ctx)};
    std::span<const std::uint8_t> span_in {in, in+numel};
    std::span<float> span_out {out, out+numel};
    ct->dequantize_uint8(
        span_in,
        span_out,
        scale,
        zero_point
    );
}

extern "C" auto quant_uint4(
    quant_context_t* const ctx,
    const float* const in,
    std::uint8_t* const out,
    const std::size_t numel,
    const float scale,
    const std::int32_t zero_point,
    const quant_round_mode_t mode
) -> void {
    auto* const ct {std::bit_cast<context*>(ctx)};
    std::span<const float> span_in {in, in+numel};
    std::span<std::uint8_t> span_out {out, out+numel};
    ct->quantize_uint4(
        span_in,
        span_out,
        scale,
        zero_point,
        mode == QUANT_NEAREST ? round_mode::nearest : round_mode::stochastic
    );
}
