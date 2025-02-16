import multiprocessing
from enum import Enum, unique

import cffi

ffi = cffi.FFI()
ffi.cdef("""

typedef struct quant_context_t quant_context_t; /* Opaque context ptr */

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

""")

C = ffi.dlopen("./libquant.so")


@unique
class RoundMode(Enum):
    NEAREST = C.QUANT_NEAREST
    STOCHASTIC = C.QUANT_STOCHASTIC


class Context:
    def __init__(self, num_threads: int | None = None) -> None:
        """Initialize a quantization context with a given number of threads. If num_threads is None, the number of threads is set to the number of available CPUs minus 1."""
        if num_threads is None:
            num_threads = max(multiprocessing.cpu_count() - 1, 1)
        self.__num_threads = num_threads
        self.__ctx = C.quant_context_create(self.__num_threads)

    def quant_uint8(
        self,
        ptr_in: int,
        ptr_out: int,
        numel: int,
        scale: float,
        zero_point: int,
        mode: RoundMode,
    ) -> None:
        """
            Quantize a float tensor to uint8 tensor.
            :param ptr_in: input tensor (must point to a valid, contiguous memory region of type float (in C float*))
            :param ptr_out: output tensor (must point to a valid, contiguous memory region of type uint8_t (in C uint8_t*))
            :param numel: number of elements in the tensor
            :param scale: quantization scale
            :param zero_point: quantization zero point
            :param mode: rounding mode
        """
        ptr_in: ffi.CData = ffi.cast("float*", ptr_in)
        ptr_out: ffi.CData = ffi.cast("uint8_t*", ptr_out)
        C.quant_uint8(self.__ctx, ptr_in, ptr_out, numel, scale, zero_point, mode.value)

    def quant_uint4(
        self,
        ptr_in: int,
        ptr_out: int,
        numel: int,
        scale: float,
        zero_point: int,
        mode: RoundMode,
    ) -> None:
        """
           Quantize a float tensor to uint8 tensor.
           :param ptr_in: input tensor (must point to a valid, contiguous memory region of type float (in C float*))
           :param ptr_out: output tensor (must point to a valid, contiguous memory region of type uint8_t (in C uint8_t*))
           :param numel: number of elements in the tensor
           :param scale: quantization scale
           :param zero_point: quantization zero point
           :param mode: rounding mode
        """
        ptr_in: ffi.CData = ffi.cast("float*", ptr_in)
        ptr_out: ffi.CData = ffi.cast("uint8_t*", ptr_out)
        C.quant_uint4(self.__ctx, ptr_in, ptr_out, numel, scale, zero_point, mode.value)

    def __del__(self) -> None:
        """Destroy the quantization context."""
        C.quant_context_destroy(self.__ctx)
