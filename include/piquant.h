/* Minimal C99 API used from the Python CFFI bindings but also useable from normal C.
 * For docs / a more complete C++ API, see piquant.hpp.
 */

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

typedef struct piquant_context_t piquant_context_t;

typedef enum piquant_round_mode_t {
    PIQUANT_NEAREST,
    PIQUANT_STOCHASTIC
} piquant_round_mode_t;

typedef enum piquant_reduce_op_t {
    PIQUANT_REDUCE_OP_SET,
    PIQUANT_REDUCE_OP_ADD,
} piquant_reduce_op_t;

typedef enum piquant_dtype_t {
    PIQUANT_DTYPE_F32 = 0,
    PIQUANT_DTYPE_BF16,

    PIQUANT_DTYPE_UINT2,
    PIQUANT_DTYPE_UINT4,
    PIQUANT_DTYPE_UINT8
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
    int64_t zero_point,
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
    int64_t zero_point,
    piquant_reduce_op_t op
);

extern PIQUANT_EXPORT void piquant_compute_quant_params_float32(
    piquant_context_t* ctx,
    const float* x,
    size_t n,
    piquant_dtype_t target_quant_dtype,
    float* out_scale,
    int64_t* out_zero_point
);

extern PIQUANT_EXPORT void piquant_compute_quant_params_bfloat16(
    piquant_context_t* ctx,
    const uint16_t* x,
    size_t n,
    piquant_dtype_t target_quant_dtype,
    float* out_scale,
    int64_t* out_zero_point
);

#ifdef __cplusplus
}
#endif
#endif
