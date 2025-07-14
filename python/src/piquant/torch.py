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

def torch_to_piquant_dtype(dtype: torch.dtype) -> DataType:
    if dtype not in _TORCH_DTYPE_MAP:
        raise ValueError(f'Unsupported quant_dtype: {dtype}')
    return _TORCH_DTYPE_MAP[dtype]

def piquant_to_torch_dtype(dtype: DataType) -> torch.dtype:
    for dtype, piquant_dtype in _TORCH_DTYPE_MAP.items():
        if piquant_dtype == dtype:
            return dtype
    raise ValueError(f'Unsupported quantized dtype: {dtype}')

def compute_quant_config(
    tensor: torch.Tensor, quant_dtype: torch.dtype, *, ctx: Context = Context.get()
) -> Tuple[float, int]:
    assert quant_dtype in _QUANT_TYPES, f'Unsupported quantized dtype: {quant_dtype}. Must be one of {list(_QUANT_TYPES)}'

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    if tensor.dtype == torch.bfloat16:
        return ctx.compute_quant_params_ptr_bfloat16(tensor.data_ptr(), torch_to_piquant_dtype(quant_dtype), tensor.numel())
    else:
        return ctx.compute_quant_params_ptr_float32(tensor.data_ptr(), torch_to_piquant_dtype(quant_dtype), tensor.numel())

def quantize(
    tensor: torch.Tensor,
    quant_dtype: torch.dtype,
    *,
    scale: float,
    zero_point: int,
    round_mode: RoundMode = RoundMode.NEAREST,
    ctx: Context = Context.get(),
) -> torch.Tensor:
    assert quant_dtype in _QUANT_TYPES, f'Unsupported quantized dtype: {quant_dtype}. Must be one of {list(_QUANT_TYPES)}'

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    dtype_in = torch_to_piquant_dtype(tensor.dtype)
    dtype_out = torch_to_piquant_dtype(quant_dtype)

    out = torch.empty(tensor.shape, dtype=quant_dtype)

    ctx.quantize_ptr(
        tensor.data_ptr(),
        dtype_in,
        out.data_ptr(),
        dtype_out,
        numel=tensor.numel(),
        scale=scale,
        zero_point=zero_point,
        round_mode=round_mode,
    )
    return out


def dequantize(
    tensor: torch.Tensor,
    dequant_dtype: torch.dtype,
    *,
    scale: float,
    zero_point: int,
    reduce_op: ReduceOp = ReduceOp.SET,
    ctx: Context = Context.get(),
) -> torch.Tensor:
    if dequant_dtype not in _DEQUANT_TYPES:
        raise ValueError(f'Unsupported dequantized dtype: {dequant_dtype}. Must be one of {list(_DEQUANT_TYPES)}')

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    out = torch.empty(tensor.shape, dtype=dequant_dtype)

    ctx.dequantize_ptr(
        tensor.data_ptr(),
        torch_to_piquant_dtype(tensor.dtype),
        out.data_ptr(),
        torch_to_piquant_dtype(out.dtype),
        numel=tensor.numel(),
        scale=scale,
        zero_point=zero_point,
        reduce_op=reduce_op,
    )
    return out
