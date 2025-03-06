/* Minimal C99 API */

#ifndef CAPI_H
#define CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#define QUANT_EXPORT __declspec(dllexport)
#else
#define QUANT_EXPORT __attribute__((visibility("default")))
#endif


/* computes and returns {scale, zero_point} derived from the data's mean and stddev. */
extern QUANT_EXPORT void compute_quant_config_from_data(const float* x, size_t n, float* out_scale, int32_t* out_zero_point);

typedef struct quant_context_t quant_context_t; /* Opaque context ptr */

typedef enum quant_round_mode_t {
    QUANT_NEAREST,
    QUANT_STOCHASTIC
} quant_round_mode_t;

typedef enum quant_reduce_op_t {
    QUANT_REDUCE_OP_SET, /* output[i] = quantize(input[i]) */
    QUANT_REDUCE_OP_ADD, /* output[i] += quantize(input[i]) */
} quant_reduce_op_t;

extern QUANT_EXPORT quant_context_t* quant_context_create(size_t num_threads);
extern QUANT_EXPORT void quant_context_destroy(quant_context_t* ctx);

extern QUANT_EXPORT void quant_uint8(
    quant_context_t* ctx,
    const float* in,
    uint8_t* out,
    size_t numel,
    float scale,
    int32_t zero_point,
    quant_round_mode_t mode,
    quant_reduce_op_t op
);

extern QUANT_EXPORT void dequant_uint8(
    quant_context_t* ctx,
    const uint8_t* in,
    float* out,
    size_t numel,
    float scale,
    int32_t zero_point
);

extern QUANT_EXPORT void quant_uint4(
    quant_context_t* ctx,
    const float* in,
    uint8_t* out,
    size_t numel,
    float scale,
    int32_t zero_point,
    quant_round_mode_t mode,
    quant_reduce_op_t op
);

#ifdef __cplusplus
}
#endif
#endif
