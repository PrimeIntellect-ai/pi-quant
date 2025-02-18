from dataclasses import dataclass
import multiprocessing
from enum import Enum, unique
from typing import Union, TYPE_CHECKING
from functools import lru_cache

from quant._loader import load_native_module
import importlib.util


if TYPE_CHECKING:
    # the code won't fail if torch and numpy are not installed
    import torch
    import numpy as np
else:
    torch = None
    np = None

    if importlib.util.find_spec("torch") is not None:
        import torch
        
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np


ffi, C = load_native_module()

@unique
class RoundMode(Enum):
    NEAREST = C.QUANT_NEAREST
    STOCHASTIC = C.QUANT_STOCHASTIC
    
class QuantDtype(Enum):
    INT8 = "int8"
    INT4 = "int4"



class Context:
    def __init__(self, num_threads: Union[int, None] = None) -> None:
        """Initialize a quantization context with a given number of threads. If num_threads is None, the number of threads is set to the number of available CPUs minus 1."""
        if num_threads is None:
            num_threads = max(multiprocessing.cpu_count() - 1, 1)
        self.__num_threads = num_threads
        self.__ctx = C.quant_context_create(self.__num_threads)

    def ptr_quant_uint8(
        self,
        ptr_in: int,
        ptr_out: int,
        *,
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

    def ptr_quant_uint4(
        self,
        ptr_in: int,
        ptr_out: int,
        *,
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


@dataclass
class QuantConfig:
    scale: float = 1.0
    zero_point: int = 0
    mode: RoundMode = RoundMode.NEAREST
    output_dtype: QuantDtype = QuantDtype.INT8


@lru_cache(maxsize=1)
def get_default_context():
    """Default context for quantization.
    This is a singleton that is used to avoid creating multiple contexts.
    """
    return Context()

def quant_torch(tensor: "torch.Tensor", out: Union["torch.Tensor", None] = None, *, config: QuantConfig = QuantConfig(), ctx: Union[Context, None] = None) -> "torch.Tensor":
    """
    Quantize a torch tensor.
        :param tensor: input tensor
        :param out: output tensor
        :param config: quantization configuration, allow to change the output dtype as well as the rounding mode
        :param ctx: quantization context, if None a singleton context is used. If you are using multiprocessing, you should create a new context for each process.
    """
    if torch is None:
        raise ImportError("torch is not installed")
    
    if ctx is None:
        ctx = get_default_context()

    assert tensor.is_contiguous(), "Input tensor must be contiguous"
        
    if config.output_dtype == QuantDtype.INT8:
        if out is None:
            out = torch.empty_like(tensor, dtype=torch.int8)
        elif out.dtype != torch.int8:
            raise ValueError("Output tensor must be of type int8")
    elif config.output_dtype == QuantDtype.INT4:
        raise NotImplementedError("Quantization to int4 is not implemented yet for torch Tensor")

    assert out.is_contiguous(), "Output tensor must be contiguous"

    ctx.ptr_quant_uint8(tensor.data_ptr(), out.data_ptr(), numel=tensor.numel(), scale=config.scale, zero_point=config.zero_point, mode=config.mode)
    return out

def quant_numpy(tensor: np.ndarray, out: Union[np.ndarray, None] = None, *, config: QuantConfig = QuantConfig(), ctx: Union[Context, None] = None) -> np.ndarray:
    """
    Quantize a numpy array.
        :param tensor: input tensor
        :param out: output tensor
        :param config: quantization configuration, allow to change the output dtype as well as the rounding mode
        :param ctx: quantization context, if None a singleton context is used. If you are using multiprocessing, you should create a new context for each process.
    """
    if np is None:
        raise ImportError("numpy is not installed")
    
    if ctx is None:
        ctx = get_default_context()
        
    assert tensor.is_contiguous(), "Input tensor must be contiguous"
    
    if config.output_dtype == QuantDtype.INT8:
        if out is None:
            out = np.empty_like(tensor, dtype=np.int8)
        elif out.dtype != np.int8:
            raise ValueError("Output tensor must be of type int8")
        
    elif config.output_dtype == QuantDtype.INT4:
        raise NotImplementedError("Quantization to int4 is not implemented yet for torch Tensor")

    ctx.ptr_quant_uint8(tensor.ctypes.data, out.ctypes.data, numel=tensor.size, scale=config.scale, zero_point=config.zero_point, mode=config.mode)
    return out
