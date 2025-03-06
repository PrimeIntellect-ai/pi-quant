from pathlib import Path
from typing import List, Tuple
from cffi import FFI
import sys

MAG_LIBS: List[Tuple[str, str]] = [
    ('win32', 'quant.dll'),
    ('linux', 'libquant.so'),
    ('darwin', 'libquant.dylib'),
]

DECLS: str = """


/* computes and returns {scale, zero_point} derived from the data's mean and stddev. */
extern  void compute_quant_config_from_data(const float* x, size_t n, float* out_scale, int32_t* out_zero_point);

typedef struct quant_context_t quant_context_t; /* Opaque context ptr */

typedef enum quant_round_mode_t {
    QUANT_NEAREST,
    QUANT_STOCHASTIC
} quant_round_mode_t;

typedef enum quant_reduce_op_t {
    QUANT_REDUCE_OP_SET, /* output[i] = quantize(input[i]) */
    QUANT_REDUCE_OP_ADD, /* output[i] += quantize(input[i]) */
} quant_reduce_op_t;

extern  quant_context_t* quant_context_create(size_t num_threads);
extern  void quant_context_destroy(quant_context_t* ctx);

extern  void quant_uint8(
    quant_context_t* ctx,
    const float* in,
    uint8_t* out,
    size_t numel,
    float scale,
    int32_t zero_point,
    quant_round_mode_t mode,
    quant_reduce_op_t op
);

extern  void dequant_uint8(
    quant_context_t* ctx,
    const uint8_t* in,
    float* out,
    size_t numel,
    float scale,
    int32_t zero_point
);

extern  void quant_uint4(
    quant_context_t* ctx,
    const float* in,
    uint8_t* out,
    size_t numel,
    float scale,
    int32_t zero_point,
    quant_round_mode_t mode,
    quant_reduce_op_t op
);

"""

def load_native_module() -> Tuple[FFI, object]:
    platform = sys.platform
    lib_name = next((lib for os, lib in MAG_LIBS if platform.startswith(os)), None)
    assert lib_name, f'Unsupported platform: {platform}'

    # Locate the library in the package directory
    pkg_path = Path(__file__).parent
    lib_path = pkg_path / lib_name
    assert lib_path.exists(), f'quant shared library not found: {lib_path}'

    ffi = FFI()
    ffi.cdef(DECLS)  # Define the C declarations
    lib = ffi.dlopen(str(lib_path))  # Load the shared library
    return ffi, lib