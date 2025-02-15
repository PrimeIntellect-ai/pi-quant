#include <quant.h>
#include <quant.hpp>
#include <bit>

using namespace quant;

struct quant_context_t {
    context* ctx {};
};

extern "C" auto quant_context_create(size_t num_threads) -> quant_context_t* {
    context* ctx {new context{num_threads}};
    return std::bit_cast<quant_context_t*>(ctx);
}

extern "C" auto quant_context_destroy(quant_context_t* ctx) -> void {
    delete std::bit_cast<context*>(ctx);
}

extern "C" auto quant_uint8(
    quant_context_t* ctx,
    const float* in,
    uint8_t* out,
    size_t numel,
    float scale,
    int32_t zero_point,
    quant_round_mode_t mode
) -> void {
    auto* ct {std::bit_cast<context*>(ctx)};
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

extern "C" auto quant_uint4(
    quant_context_t* ctx,
    const float* in,
    uint8_t* out,
    size_t numel,
    float scale,
    int32_t zero_point,
    quant_round_mode_t mode
) -> void {
    auto* ct {std::bit_cast<context*>(ctx)};
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
