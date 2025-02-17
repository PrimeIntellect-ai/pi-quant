from pathlib import Path
from cffi import FFI
import sys

MAG_LIBS: list[tuple[str, str]] = [
    ('win32', 'quant.dll'),
    ('linux', 'libquant.so'),
    ('darwin', 'libquant.dylib'),
]

DECLS: str = """

typedef struct quant_context_t quant_context_t;

typedef enum quant_round_mode_t {
    QUANT_NEAREST = 1,
    QUANT_STOCHASTIC = 0
} quant_round_mode_t;

extern quant_context_t* quant_context_create(size_t num_threads);
extern void quant_context_destroy(quant_context_t* ctx);

extern void quant_uint8(
    quant_context_t* ctx,
    const float* in,
    uint8_t* out,
    size_t numel,
    float scale,
    int32_t zero_point,
    quant_round_mode_t mode
);

extern void quant_uint4(
    quant_context_t* ctx,
    const float* in,
    uint8_t* out,
    size_t numel,
    float scale,
    int32_t zero_point,
    quant_round_mode_t mode
);

"""

def load_native_module() -> tuple[FFI, object]:
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