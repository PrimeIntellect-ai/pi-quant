from __future__ import annotations

__version__ = '0.1.0'
__author__ = 'Mario Sieg'
__email__ = 'mario.sieg.64@gmail.com'
__author_email__ = 'mario.sieg.64@gmail.com'

import importlib.util

import weakref
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
    BF16 = C.PIQUANT_DTYPE_BF16
    UINT2 = C.PIQUANT_DTYPE_UINT2
    UINT4 = C.PIQUANT_DTYPE_UINT4
    UINT8 = C.PIQUANT_DTYPE_UINT8

    @property
    def bit_size(self) -> int:
        _BIT_SIZES = {
            self.F32: 32,
            self.BF16: 16,
            self.UINT2: 2,
            self.UINT4: 4,
            self.UINT8: 8,
        }
        return _BIT_SIZES[self]

    @property
    def is_quantized(self) -> bool:
        return self in (self.UINT2, self.UINT4, self.UINT8)

    @property
    def is_dequantized(self) -> bool:
        return self in (self.F32, self.BF16)

    @property
    def stride(self) -> int:
        return min(8, self.bit_size) >> 3


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
    def get() -> Context:
        """
        Default context for quantization.
        This is a singleton that is used to avoid creating multiple contexts.
        """
        return Context()

    def quantize_ptr(
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
        assert dtype_in.is_dequantized, f'Input dtype must be a dequantized type, but is: {dtype_in}'
        assert dtype_out.is_quantized, f'Output dtype must be a quantized type, but is: {dtype_out}'
        assert ptr_in != 0, 'Input arr pointer must not be NULL'
        assert ptr_out != 0, 'Output arr pointer must not be NULL'
        ptr_in: ffi.CData = ffi.cast('const void*', ptr_in)
        ptr_out: ffi.CData = ffi.cast('void*', ptr_out)
        C.piquant_quantize(
            self._ctx, ptr_in, dtype_in.value, ptr_out, dtype_out.value, numel, scale, zero_point, round_mode.value
        )

    def dequantize_ptr(
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
        assert dtype_in.is_quantized, f'Input dtype must be a quantized type, but is: {dtype_in}'
        assert dtype_out.is_dequantized, f'Output dtype must be a dequantized type, but is: {dtype_out}'
        assert ptr_in != 0, 'Input arr pointer must not be NULL'
        assert ptr_out != 0, 'Output arr pointer must not be NULL'
        ptr_in: ffi.CData = ffi.cast('const void*', ptr_in)
        ptr_out: ffi.CData = ffi.cast('void*', ptr_out)
        C.piquant_dequantize(
            self._ctx, ptr_in, dtype_in.value, ptr_out, dtype_out.value, numel, scale, zero_point, reduce_op.value
        )

    def compute_quant_params_ptr_float32(self, ptr: int, target_quant_dtype: DataType, numel: int) -> Tuple[float, int]:
        assert target_quant_dtype.is_quantized, f'Target dtype must be a quantized type, but is: {target_quant_dtype}'
        assert ptr != 0, 'Input arr pointer must not be NULL'
        ptr: ffi.CData = ffi.cast('const float*', ptr)
        scale: ffi.CData = ffi.new('float*')
        zero_point: ffi.CData = ffi.new('int64_t*')
        C.piquant_compute_quant_params_float32(self._ctx, ptr, numel, target_quant_dtype.value, scale, zero_point)
        return scale[0], zero_point[0]

    def compute_quant_params_ptr_bfloat16(
        self, ptr: int, target_quant_dtype: DataType, numel: int
    ) -> Tuple[float, int]:
        assert target_quant_dtype.is_quantized, f'Target dtype must be a quantized type, but is: {target_quant_dtype}'
        assert ptr != 0, 'Input arr pointer must not be NULL'
        ptr: ffi.CData = ffi.cast('const uint16_t*', ptr)
        scale: ffi.CData = ffi.new('float*')
        zero_point: ffi.CData = ffi.new('int64_t*')
        C.piquant_compute_quant_params_bfloat16(self._ctx, ptr, numel, target_quant_dtype.value, scale, zero_point)
        return scale[0], zero_point[0]


if importlib.util.find_spec('torch') is not None:
    try:
        from . import torch
    except ImportError:
        pass
