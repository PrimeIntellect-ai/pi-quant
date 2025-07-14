from __future__ import annotations

import math

from . import *

import torch

_TORCH_DTYPE_MAP: dict[torch.dtype, DataType] = {
    torch.float32: DataType.F32,
    torch.bfloat16: DataType.BF16,
    torch.quint2x4: DataType.UINT2,
    torch.quint4x2: DataType.UINT4,
    torch.quint8: DataType.UINT8,
    torch.uint8: DataType.UINT8,
}

_QUANT_TYPES: set[torch.dtype] = {
    torch.quint2x4,
    torch.quint4x2,
    torch.quint8,
    torch.uint8,
}

_DEQUANT_TYPES: set[torch.dtype] = {
    torch.float32,
    torch.bfloat16,
}

_ROUND_MODES: dict[str, RoundMode] = {
    'nearest': RoundMode.NEAREST,
    'stochastic': RoundMode.STOCHASTIC,
}

_REDUCE_OPS: dict[str, ReduceOp] = {
    'set': ReduceOp.SET,
    'add': ReduceOp.ADD,
}

def torch_to_piquant_dtype(dtype: torch.dtype) -> DataType:
    if dtype not in _TORCH_DTYPE_MAP:
        raise ValueError(f'Unsupported quant_dtype: {dtype}')
    return _TORCH_DTYPE_MAP[dtype]


def piquant_to_torch_dtype(dtype: DataType) -> torch.dtype:
    for dtype, piquant_dtype in _TORCH_DTYPE_MAP.items():
        if piquant_dtype == dtype:
            return dtype
    raise ValueError(f'Unsupported quantized dtype: {dtype}')


def compute_quant_params(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    ctx: Context = Context.get()
) -> Tuple[float, int]:
    assert dtype in _QUANT_TYPES, f'Unsupported quantized dtype: {dtype}. Must be one of {list(_QUANT_TYPES)}'

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    if tensor.dtype == torch.bfloat16:
        return ctx.compute_quant_params_ptr_bfloat16(tensor.data_ptr(), torch_to_piquant_dtype(dtype), tensor.numel())
    else:
        return ctx.compute_quant_params_ptr_float32(tensor.data_ptr(), torch_to_piquant_dtype(dtype), tensor.numel())


def quantize(
    tensor: torch.Tensor,
    *,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    round_mode: str = 'nearest',
    ctx: Context = Context.get(),
) -> torch.Tensor:
    assert dtype in _QUANT_TYPES, f'Unsupported quantized dtype: {dtype}. Must be one of {list(_QUANT_TYPES)}'

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    dtype_in = torch_to_piquant_dtype(tensor.dtype)
    dtype_out = torch_to_piquant_dtype(dtype)

    out = torch.empty(tensor.shape, dtype=dtype)

    ctx.quantize_ptr(
        tensor.data_ptr(),
        dtype_in,
        out.data_ptr(),
        dtype_out,
        numel=tensor.numel(),
        scale=scale,
        zero_point=zero_point,
        round_mode=_ROUND_MODES[round_mode],
    )
    return out


def dequantize(
    tensor: torch.Tensor,
    *,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    reduce_op: str = 'set',
    ctx: Context = Context.get(),
) -> torch.Tensor:
    if dtype not in _DEQUANT_TYPES:
        raise ValueError(f'Unsupported dequantized dtype: {dtype}. Must be one of {list(_DEQUANT_TYPES)}')

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    out = torch.empty(tensor.shape, dtype=dtype)

    ctx.dequantize_ptr(
        tensor.data_ptr(),
        torch_to_piquant_dtype(tensor.dtype),
        out.data_ptr(),
        torch_to_piquant_dtype(out.dtype),
        numel=tensor.numel(),
        scale=scale,
        zero_point=zero_point,
        reduce_op=_REDUCE_OPS[reduce_op],
    )
    return out
