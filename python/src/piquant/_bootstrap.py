from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from cffi import FFI

import sys

_NATIVE_MODULES: List[Tuple[str, str]] = [
    ('win32', 'piquant.dll'),
    ('linux', 'libpiquant.so'),
    ('darwin', 'libpiquant.dylib'),
]

_CDECLS: str = """

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

extern piquant_context_t* piquant_context_create(size_t num_threads);
extern void piquant_context_destroy(piquant_context_t* ctx);

extern void piquant_quantize(
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

extern void piquant_dequantize(
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

extern void piquant_compute_quant_params_float32(
    piquant_context_t* ctx,
    const float* x,
    size_t n,
    piquant_dtype_t target_quant_dtype,
    float* out_scale,
    int64_t* out_zero_point
);

extern void piquant_compute_quant_params_bfloat16(
    piquant_context_t* ctx,
    const uint16_t* x,
    size_t n,
    piquant_dtype_t target_quant_dtype,
    float* out_scale,
    int64_t* out_zero_point
);
"""


def _load_native_module() -> Tuple[FFI, object]:
    platform = sys.platform
    lib_name = next((lib for os, lib in _NATIVE_MODULES if platform.startswith(os)), None)
    assert lib_name, f'Unsupported platform: {platform}'

    # Locate the library in the package directory
    pkg_path = Path(__file__).parent
    lib_path = pkg_path / lib_name
    assert lib_path.exists(), f'piquant shared library not found: {lib_path}'

    ffi = FFI()
    ffi.cdef(_CDECLS)  # Define the _C declarations
    lib = ffi.dlopen(str(lib_path))  # Load the shared library
    return ffi, lib


ffi, C = _load_native_module()
