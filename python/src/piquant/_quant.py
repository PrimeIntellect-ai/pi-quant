from dataclasses import dataclass
import multiprocessing
from enum import Enum, unique
from typing import Union, Tuple
from functools import lru_cache

from piquant._loader import load_native_module

_ffi, _C = load_native_module()


@unique
class RoundMode(Enum):
    NEAREST = _C.PIQUANT_NEAREST
    STOCHASTIC = _C.PIQUANT_STOCHASTIC


@unique
class ReduceOp(Enum):
    SET = _C.PIQUANT_REDUCE_OP_SET
    ADD = _C.PIQUANT_REDUCE_OP_ADD


@unique
class QuantDtype(Enum):
    UINT4 = _C.PIQUANT_DTYPE_UINT4
    INT4 = _C.PIQUANT_DTYPE_INT4
    UINT8 = _C.PIQUANT_DTYPE_UINT8
    INT8 = _C.PIQUANT_DTYPE_INT8
    UINT16 = _C.PIQUANT_DTYPE_UINT16
    INT16 = _C.PIQUANT_DTYPE_INT16
    UINT32 = _C.PIQUANT_DTYPE_UINT32
    INT32 = _C.PIQUANT_DTYPE_INT32
    UINT64 = _C.PIQUANT_DTYPE_UINT64
    INT64 = _C.PIQUANT_DTYPE_INT64
    F32 = _C.PIQUANT_DTYPE_F32
    F64 = _C.PIQUANT_DTYPE_F64

    def bit_size(self) -> int:
        if self in (QuantDtype.UINT4, QuantDtype.INT4):
            return 4
        elif self in (QuantDtype.UINT8, QuantDtype.INT8):
            return 8
        elif self in (QuantDtype.UINT16, QuantDtype.INT16):
            return 16
        elif self in (QuantDtype.UINT32, QuantDtype.INT32, QuantDtype.F32):
            return 32
        elif self in (QuantDtype.UINT64, QuantDtype.INT64, QuantDtype.F64):
            return 64
        else:
            raise ValueError(f'Unsupported dtype: {self}')

    def byte_size(self) -> int:
        return min(8, self.bit_size()) >> 3


@dataclass
class QuantConfig:
    scale: float = 1.0
    zero_point: int = 0
    mode: RoundMode = RoundMode.NEAREST
    output_dtype: QuantDtype = QuantDtype.UINT8


@dataclass
class DequantConfig:
    scale: float = 1.0
    zero_point: int = 0
    reduce_op: ReduceOp = ReduceOp.SET


class Context:
    def __init__(self, num_threads: Union[int, None] = None) -> None:
        """Initialize a quantization context with a given number of threads. If num_threads is None, the number of threads is set to the number of available CPUs minus 1."""
        if num_threads is None:
            num_threads = max(multiprocessing.cpu_count() - 1, 1)
        self.__num_threads = num_threads
        self._ctx = _C.piquant_context_create(self.__num_threads)

    def __del__(self) -> None:
        _C.piquant_context_destroy(self._ctx)

    @staticmethod
    @lru_cache(maxsize=1)
    def default() -> 'Context':
        """
        Default context for quantization.
        This is a singleton that is used to avoid creating multiple contexts.
        """
        return Context()

    def quantize_raw_ptr(
        self,
        ptr_in: int,
        dtype_in: QuantDtype,
        ptr_out: int,
        dtype_out: QuantDtype,
        numel: int,
        scale: float,
        zero_point: int,
        round_mode: RoundMode,
    ) -> None:
        assert ptr_in != 0, 'Input arr pointer must not be null'
        assert ptr_out != 0, 'Output arr pointer must not be null'
        ptr_in: _ffi.CData = _ffi.cast('const void*', ptr_in)
        ptr_out: _ffi.CData = _ffi.cast('void*', ptr_out)
        _C.piquant_quantize(
            self._ctx, ptr_in, dtype_in.value, ptr_out, dtype_out.value, numel, scale, zero_point, round_mode.value
        )

    def dequantize_raw_ptr(
        self,
        ptr_in: int,
        dtype_in: QuantDtype,
        ptr_out: int,
        dtype_out: QuantDtype,
        numel: int,
        scale: float,
        zero_point: int,
        reduce_op: ReduceOp,
    ) -> None:
        assert ptr_in != 0, 'Input arr pointer must not be null'
        assert ptr_out != 0, 'Output arr pointer must not be null'
        ptr_in: _ffi.CData = _ffi.cast('const void*', ptr_in)
        ptr_out: _ffi.CData = _ffi.cast('void*', ptr_out)
        _C.piquant_dequantize(
            self._ctx, ptr_in, dtype_in.value, ptr_out, dtype_out.value, numel, scale, zero_point, reduce_op.value
        )

    def compute_quant_config_raw_ptr(self, ptr: int, target_quant_dtype: QuantDtype, numel: int) -> Tuple[float, int]:
        """
        Compute the scale and zero point of a arr.
        :param ptr: p input arr data pointer (must point to a valid, contiguous memory region of type float (in _C float*))
        :param numel: number of elements in the arr
        """
        ptr: _ffi.CData = _ffi.cast('float*', ptr)
        scale: _ffi.CData = _ffi.new('float*')
        zero_point: _ffi.CData = _ffi.new('int64_t*')
        _C.piquant_compute_quant_config_from_data(self._ctx, ptr, numel, target_quant_dtype.value, scale, zero_point)
        return scale[0], zero_point[0]
