from pathlib import Path
from typing import List, Tuple
from cffi import FFI
import sys

MAG_LIBS: List[Tuple[str, str]] = [
    ('win32', 'piquant.dll'),
    ('linux', 'libpiquant.so'),
    ('darwin', 'libpiquant.dylib'),
]

DECLS: str = """
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
    PIQUANT_DTYPE_UINT4,
    PIQUANT_DTYPE_INT4,
    PIQUANT_DTYPE_UINT8,
    PIQUANT_DTYPE_INT8,
    PIQUANT_DTYPE_UINT16,
    PIQUANT_DTYPE_INT16,
    PIQUANT_DTYPE_UINT32,
    PIQUANT_DTYPE_INT32,
    PIQUANT_DTYPE_UINT64,
    PIQUANT_DTYPE_INT64,
    PIQUANT_DTYPE_F32,
    PIQUANT_DTYPE_F64
} piquant_dtype_t;

extern  piquant_context_t* piquant_context_create(size_t num_threads);
extern  void piquant_context_destroy(piquant_context_t* ctx);

extern  void piquant_quantize(
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

extern  void piquant_dequantize(
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

/* computes and returns {scale, zero_point} derived from the data's mean and stddev. */
extern  void piquant_compute_quant_config_from_data(
    piquant_context_t* ctx,
    const float* x,
    size_t n,
    piquant_dtype_t target_quant_dtype,
    float* out_scale,
    int64_t* out_zero_point
);
"""


def load_native_module() -> Tuple[FFI, object]:
    platform = sys.platform
    lib_name = next((lib for os, lib in MAG_LIBS if platform.startswith(os)), None)
    assert lib_name, f'Unsupported platform: {platform}'

    # Locate the library in the package directory
    pkg_path = Path(__file__).parent
    lib_path = pkg_path / lib_name
    assert lib_path.exists(), f'piquant shared library not found: {lib_path}'

    ffi = FFI()
    ffi.cdef(DECLS)  # Define the _C declarations
    lib = ffi.dlopen(str(lib_path))  # Load the shared library
    return ffi, lib
