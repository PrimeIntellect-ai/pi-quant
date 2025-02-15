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

typedef struct quant_context_t quant_context_t; /* Opaque context ptr */

typedef enum quant_round_mode_t {
    QUANT_NEAREST = 1,
    QUANT_STOCHASTIC = 0
} quant_round_mode_t;

extern QUANT_EXPORT quant_context_t* quant_context_create(size_t num_threads);
extern QUANT_EXPORT void quant_context_destroy(quant_context_t* ctx);

extern QUANT_EXPORT void quant_uint8(
    quant_context_t* ctx,
    const float* in,
    uint8_t* out,
    size_t numel,
    float scale,
    int32_t zero_point,
    quant_round_mode_t mode
);

extern QUANT_EXPORT void quant_uint4(
    quant_context_t* ctx,
    const float* in,
    uint8_t* out,
    size_t numel,
    float scale,
    int32_t zero_point,
    quant_round_mode_t mode
);

#ifdef __cplusplus
}
#endif
#endif
