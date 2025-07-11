import weakref
from dataclasses import dataclass
import multiprocessing
from enum import Enum, unique
from typing import Union, Tuple
from functools import lru_cache

from piquant._bootstrap import ffi, C


@unique
class RoundMode(Enum):
    NEAREST = C.PIQUANT_NEAREST
    STOCHASTIC = C.PIQUANT_STOCHASTIC


@unique
class ReduceOp(Enum):
    SET = C.PIQUANT_REDUCE_OP_SET
    ADD = C.PIQUANT_REDUCE_OP_ADD


@unique
class DataType(Enum):
    F32 = C.PIQUANT_DTYPE_F32
    F64 = C.PIQUANT_DTYPE_F64
    UINT2 = C.PIQUANT_DTYPE_UINT2
    INT2 = C.PIQUANT_DTYPE_INT2
    UINT4 = C.PIQUANT_DTYPE_UINT4
    INT4 = C.PIQUANT_DTYPE_INT4
    UINT8 = C.PIQUANT_DTYPE_UINT8
    INT8 = C.PIQUANT_DTYPE_INT8
    UINT16 = C.PIQUANT_DTYPE_UINT16
    INT16 = C.PIQUANT_DTYPE_INT16
    UINT32 = C.PIQUANT_DTYPE_UINT32
    INT32 = C.PIQUANT_DTYPE_INT32
    UINT64 = C.PIQUANT_DTYPE_UINT64
    INT64 = C.PIQUANT_DTYPE_INT64

    def bit_size(self) -> int:
        if self in (DataType.UINT2, DataType.INT2):
            return 2
        elif self in (DataType.UINT4, DataType.INT4):
            return 4
        elif self in (DataType.UINT8, DataType.INT8):
            return 8
        elif self in (DataType.UINT16, DataType.INT16):
            return 16
        elif self in (DataType.UINT32, DataType.INT32, DataType.F32):
            return 32
        elif self in (DataType.UINT64, DataType.INT64, DataType.F64):
            return 64
        else:
            raise ValueError(f'Unsupported dtype: {self}')

    def byte_size(self) -> int:
        return min(8, self.bit_size()) >> 3


class Context:
    def __init__(self, num_threads: Union[int, None] = None) -> None:
        """Initialize a quantization context with a given number of threads. If num_threads is None, the number of threads is set to the number of available CPUs minus 1."""
        if num_threads is None:
            num_threads = max(multiprocessing.cpu_count() - 1, 1)
        self._num_threads = num_threads
        self._ctx = C.piquant_context_create(self._num_threads)
        self._finalizer = weakref.finalize(self, C.piquant_context_destroy, self._ctx)

    @staticmethod
    @lru_cache(maxsize=1)
    def get() -> 'Context':
        """
        Default context for quantization.
        This is a singleton that is used to avoid creating multiple contexts.
        """
        return Context()

    def quantize_raw_ptr(
        self,
        ptr_in: int,
        dtype_in: DataType,
        ptr_out: int,
        dtype_out: DataType,
        numel: int,
        scale: float,
        zero_point: int,
        round_mode: RoundMode,
    ) -> None:
        assert ptr_in != 0, 'Input arr pointer must not be null'
        assert ptr_out != 0, 'Output arr pointer must not be null'
        ptr_in: ffi.CData = ffi.cast('const void*', ptr_in)
        ptr_out: ffi.CData = ffi.cast('void*', ptr_out)
        C.piquant_quantize(
            self._ctx, ptr_in, dtype_in.value, ptr_out, dtype_out.value, numel, scale, zero_point, round_mode.value
        )

    def dequantize_raw_ptr(
        self,
        ptr_in: int,
        dtype_in: DataType,
        ptr_out: int,
        dtype_out: DataType,
        numel: int,
        scale: float,
        zero_point: int,
        reduce_op: ReduceOp,
    ) -> None:
        assert ptr_in != 0, 'Input arr pointer must not be null'
        assert ptr_out != 0, 'Output arr pointer must not be null'
        ptr_in: ffi.CData = ffi.cast('const void*', ptr_in)
        ptr_out: ffi.CData = ffi.cast('void*', ptr_out)
        C.piquant_dequantize(
            self._ctx, ptr_in, dtype_in.value, ptr_out, dtype_out.value, numel, scale, zero_point, reduce_op.value
        )

    def compute_quant_config_raw_ptr(self, ptr: int, target_quant_dtype: DataType, numel: int) -> Tuple[float, int]:
        """
        Compute the scale and zero point of a arr.
        :param ptr: p input arr data pointer (must point to a valid, contiguous memory region of type float (in C float*))
        :param numel: number of elements in the arr
        """
        ptr: ffi.CData = ffi.cast('float*', ptr)
        scale: ffi.CData = ffi.new('float*')
        zero_point: ffi.CData = ffi.new('int64_t*')
        C.piquant_compute_quant_config_from_data(self._ctx, ptr, numel, target_quant_dtype.value, scale, zero_point)
        return scale[0], zero_point[0]
