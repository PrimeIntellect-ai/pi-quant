/* Minimal C99 API */

#ifndef PIQUANT_H
#define PIQUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#define PIQUANT_EXPORT __declspec(dllexport)
#else
#define PIQUANT_EXPORT __attribute__((visibility("default")))
#endif

/* computes and returns {scale, zero_point} derived from the data's mean and stddev. */
extern PIQUANT_EXPORT void piquant_compute_quant_config_from_data(const float* x, size_t n, float* out_scale, int32_t* out_zero_point);

typedef struct piquant_context_t piquant_context_t; /* Opaque context ptr */

typedef enum piquant_round_mode_t {
    PIQUANT_NEAREST,
    PIQUANT_STOCHASTIC
} piquant_round_mode_t;

typedef enum piquant_reduce_op_t {
    PIQUANT_REDUCE_OP_SET, /* output[i] = quantize(input[i]) */
    PIQUANT_REDUCE_OP_ADD, /* output[i] += quantize(input[i]) */
} piquant_reduce_op_t;

typedef enum piquant_dtype_t {
    PIQUANT_DTYPE_F32,
    PIQUANT_DTYPE_UINT8,
    PIQUANT_DTYPE_UINT4
} piquant_dtype_t;

extern PIQUANT_EXPORT piquant_context_t* piquant_context_create(size_t num_threads);
extern PIQUANT_EXPORT void piquant_context_destroy(piquant_context_t* ctx);

extern PIQUANT_EXPORT void piquant_quantize(
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

extern PIQUANT_EXPORT void piquant_dequantize(
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
