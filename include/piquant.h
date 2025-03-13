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
extern QUANT_EXPORT void piquant_compute_quant_config_from_data(const float* x, size_t n, float* out_scale, int32_t* out_zero_point);

typedef struct piquant_context_t piquant_context_t; /* Opaque context ptr */

typedef enum piquant_round_mode_t {
    QUANT_NEAREST,
    QUANT_STOCHASTIC
} piquant_round_mode_t;

typedef enum piquant_reduce_op_t {
    QUANT_REDUCE_OP_SET, /* output[i] = quantize(input[i]) */
    QUANT_REDUCE_OP_ADD, /* output[i] += quantize(input[i]) */
} piquant_reduce_op_t;

typedef enum piquant_dtype_t {
    QUANT_F32,
    QUANT_UINT8,
    QUANT_UINT4
} piquant_dtype_t;

extern QUANT_EXPORT piquant_context_t* piquant_context_create(size_t num_threads);
extern QUANT_EXPORT void piquant_context_destroy(piquant_context_t* ctx);

extern QUANT_EXPORT void piquant_quantize(
    piquant_context_t* ctx,
    const void* in,
    piquant_dtype_t dtype_in,
    void* out,
    piquant_dtype_t dtype_out,
    size_t numel,
    float scale,
    int32_t zero_point,
    piquant_round_mode_t mode
);

extern QUANT_EXPORT void piquant_dequantize(
    piquant_context_t* ctx,
    const void* in,
    piquant_dtype_t dtype_in,
    void* out,
    piquant_dtype_t dtype_out,
    size_t numel,
    float scale,
    int32_t zero_point,
    piquant_reduce_op_t op
);

#ifdef __cplusplus
}
#endif
#endif
