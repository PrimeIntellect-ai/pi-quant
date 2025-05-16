import importlib
from typing import TYPE_CHECKING, Tuple, Dict
from piquant._quant import *
import torch


def _torch_to_piquant_dtype(dtype: 'torch.target_quant_dtype') -> QuantDtype:
    if torch is None:
        raise ImportError('torch is not installed')
    _dtype_map: Dict['torch.target_quant_dtype', QuantDtype] = {
        torch.uint8: QuantDtype.UINT8,
        torch.int8: QuantDtype.INT8,
        torch.uint16: QuantDtype.UINT16,
        torch.int16: QuantDtype.INT16,
        torch.uint32: QuantDtype.UINT32,
        torch.int32: QuantDtype.INT32,
        torch.uint64: QuantDtype.UINT64,
        torch.int64: QuantDtype.INT64,
        torch.float32: QuantDtype.F32,
        torch.float64: QuantDtype.F64,
    }
    if not dtype in _dtype_map:
        raise ValueError(f'Unsupported target_quant_dtype: {dtype}')
    return _dtype_map[dtype]


def compute_quant_config_torch(
    tensor: 'torch.Tensor', *, target_quant_dtype: QuantDtype, ctx: Union[Context, None] = None
) -> Tuple[float, int]:
    """
    Compute the scale and zero point of a arr.
        :param tensor: Input arr, must be of type float32.
        :param target_quant_dtype: Data type which the arr will be quantized to
        :param ctx: Context to use for computation, if None, the default context will be used.
    """
    if torch is None:
        raise ImportError('torch is not installed')
    if ctx is None:
        ctx = Context.default()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    assert tensor.dtype == torch.float32, f'Expected arr of type float32, got {tensor.dtype}'
    return ctx.compute_quant_config_raw_ptr(tensor.data_ptr(), target_quant_dtype, tensor.numel())


def quantize_torch(
    in_tensor: 'torch.Tensor',
    out_tensor: Union['torch.Tensor', None] = None,
    *,
    config: QuantConfig = QuantConfig(),
    ctx: Union[Context, None] = None,
) -> 'torch.Tensor':
    """
    Quantize a tensor using the given configuration.
    :param in_tensor: Input tensor, must be of type float32.
    :param out_tensor: Quantized output tensor, if None, a new tensor will be created.
    :param config: Quantization configuration, including scale, zero point, and round mode.
    :param ctx: Context to use for quantization, if None, the default context will be used.
    :return: Quantized tensor.
    """
    if torch is None:
        raise ImportError('torch is not installed')

    if ctx is None:
        ctx = Context.default()

    if in_tensor.dtype != torch.float32:
        in_tensor = in_tensor.float()

    if out_tensor is None:
        out_tensor = torch.empty_like(in_tensor, dtype=torch.uint8)

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
        _torch_to_piquant_dtype(in_tensor.dtype),
        out_tensor.data_ptr(),
        _torch_to_piquant_dtype(out_tensor.dtype),
        numel=in_tensor.numel(),
        scale=config.scale,
        zero_point=config.zero_point,
        round_mode=config.mode,
    )
    return out_tensor


def dequantize_torch(
    in_tensor: 'torch.Tensor',
    out_tensor: Union['torch.Tensor', None] = None,
    *,
    config: DequantConfig = DequantConfig(),
    ctx: Union[Context, None] = None,
) -> 'torch.Tensor':
    """
    Dequantize a tensor using the given configuration.
    :param in_tensor: Input tensor. Must be in a quantized format (e.g., uint8).
    :param out_tensor: Dequantized output tensor in a dequantized format (e.g. float32). If None, a new tensor will be created.
    :param config: Dequantization configuration, including scale, zero point, and reduce operation.
    :param ctx: Context to use for dequantization, if None, the default context will be used.
    :return: Dequantized tensor.
    """

    if torch is None:
        raise ImportError('torch is not installed')

    if ctx is None:
        ctx = Context.default()

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
        _torch_to_piquant_dtype(in_tensor.dtype),
        out_tensor.data_ptr(),
        _torch_to_piquant_dtype(out_tensor.dtype),
        numel=in_tensor.numel(),
        scale=config.scale,
        zero_point=config.zero_point,
        reduce_op=config.reduce_op,
    )
    return out_tensor
