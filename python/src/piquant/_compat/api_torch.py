from .._core import *

import torch

_TORCH_DTYPE_MAP: dict[torch.dtype, DataType] = {
    torch.quint2x4: DataType.UINT2,
    torch.quint4x2: DataType.UINT4,
    torch.qint8: DataType.INT8,
    torch.quint8: DataType.UINT8,
    torch.uint8: DataType.UINT8,
    torch.int8: DataType.INT8,
    torch.uint16: DataType.UINT16,
    torch.int16: DataType.INT16,
    torch.uint32: DataType.UINT32,
    torch.int32: DataType.INT32,
    torch.uint64: DataType.UINT64,
    torch.int64: DataType.INT64,
    torch.float32: DataType.F32,
    torch.float64: DataType.F64,
}

_TORCH_QUANT_STORAGE_MAP: dict[DataType, torch.dtype] = {
    DataType.UINT2: torch.uint8,
    DataType.INT2: torch.int8,
    DataType.UINT4: torch.uint8,
    DataType.INT4: torch.int8,
}

def torch_to_piquant_dtype(dtype: torch.dtype) -> DataType:
    if dtype not in _TORCH_DTYPE_MAP:
        raise ValueError(f'Unsupported target_quant_dtype: {dtype}')
    return _TORCH_DTYPE_MAP[dtype]

def piquant_to_torch_dtype(dtype: DataType) -> torch.dtype:
    for torch_dtype, piquant_dtype in _TORCH_DTYPE_MAP.items():
        if piquant_dtype == dtype:
            return torch_dtype
    raise ValueError(f'Unsupported quantized dtype: {dtype}')

def _get_quantized_storage_dtype(dtype: DataType) -> torch.dtype:
    if dtype in _TORCH_QUANT_STORAGE_MAP:
        return _TORCH_QUANT_STORAGE_MAP[dtype]
    else:
        return piquant_to_torch_dtype(dtype)

def compute_quant_config_torch(
    tensor: torch.Tensor, *, target_quant_dtype: DataType, ctx: Context = Context.get()
) -> Tuple[float, int]:
    """
    Compute the scale and zero point of a arr.
        :param tensor: Input arr, must be of type float32.
        :param target_quant_dtype: Data type which the arr will be quantized to
        :param ctx: Context to use for computation, if None, the default context will be used.
    """
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    assert tensor.dtype == torch.float32, f'Expected arr of type float32, got {tensor.dtype}'
    return ctx.compute_quant_config_raw_ptr(tensor.data_ptr(), target_quant_dtype, tensor.numel())

def quantize_torch(
    in_tensor: torch.Tensor,
    *,
    scale: float,
    zero_point: int,
    output_dtype: DataType,
    round_mode: RoundMode = RoundMode.NEAREST,
    ctx: Context = Context.get(),
    out_tensor: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    """
    Quantize a tensor using the given configuration.
    :param in_tensor: Input tensor, must be of type float32.
    :param out_tensor: Quantized output tensor, if None, a new tensor will be created.
    :param config: Quantization configuration, including scale, zero point, and round mode.
    :param ctx: Context to use for quantization, if None, the default context will be used.
    :return: Quantized tensor.
    """

    if in_tensor.dtype != torch.float32:
        in_tensor = in_tensor.float()

    if out_tensor is None:
        out_tensor = torch.empty(in_tensor.shape, dtype=_get_quantized_storage_dtype(output_dtype))

    if not in_tensor.is_contiguous():
        in_tensor = in_tensor.contiguous()

    if not out_tensor.is_contiguous():
        out_tensor = out_tensor.contiguous()

    if in_tensor.numel() != out_tensor.numel():
        raise ValueError(
            f'Input and output tensors must have the same number of elements: {in_tensor.numel()} != {out_tensor.numel()}'
        )

    ctx.quantize_raw_ptr(
        in_tensor.data_ptr(),
        torch_to_piquant_dtype(in_tensor.dtype),
        out_tensor.data_ptr(),
        torch_to_piquant_dtype(out_tensor.dtype),
        numel=in_tensor.numel(),
        scale=scale,
        zero_point=zero_point,
        round_mode=round_mode,
    )
    return out_tensor


def dequantize_torch(
    in_tensor: torch.Tensor,
    *,
    scale: float,
    zero_point: int,
    reduce_op: ReduceOp,
    out_tensor: Union[torch.Tensor, None] = None,
    ctx: Context = Context.get(),
) -> torch.Tensor:
    """
    Dequantize a tensor using the given configuration.
    :param in_tensor: Input tensor. Must be in a quantized format (e.g., uint8).
    :param out_tensor: Dequantized output tensor in a dequantized format (e.g. float32). If None, a new tensor will be created.
    :param config: Dequantization configuration, including scale, zero point, and reduce operation.
    :param ctx: Context to use for dequantization, if None, the default context will be used.
    :return: Dequantized tensor.
    """

    if out_tensor is None:
        out_tensor = torch.empty_like(in_tensor, dtype=torch.float32)

    if not in_tensor.is_contiguous():
        in_tensor = in_tensor.contiguous()

    if not out_tensor.is_contiguous():
        out_tensor = out_tensor.contiguous()

    if in_tensor.numel() != out_tensor.numel():
        raise ValueError(
            f'Input and output tensors must have the same number of elements: {in_tensor.numel()} != {out_tensor.numel()}'
        )

    ctx.dequantize_raw_ptr(
        in_tensor.data_ptr(),
        torch_to_piquant_dtype(in_tensor.dtype),
        out_tensor.data_ptr(),
        torch_to_piquant_dtype(out_tensor.dtype),
        numel=in_tensor.numel(),
        scale=scale,
        zero_point=zero_point,
        reduce_op=reduce_op,
    )
    return out_tensor
