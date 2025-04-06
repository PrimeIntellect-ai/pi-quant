import importlib
from typing import TYPE_CHECKING, Tuple, Dict
from piquant._quant import *

if TYPE_CHECKING:
    import torch
else:
    torch = None
    if importlib.util.find_spec('torch') is not None:
        import torch

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
    torch.float64: QuantDtype.F64
}

def compute_quant_config_torch(tensor: 'torch.Tensor', *, target_quant_dtype: QuantDtype,  ctx: Union[Context, None] = None) -> Tuple[float, int]:
    """
    Compute the scale and zero point of a tensor.
        :param tensor: input tensor
    """
    if torch is None:
        raise ImportError('torch is not installed')
    if ctx is None:
        ctx = Context.default()
    assert tensor.is_contiguous()
    assert tensor.dtype == torch.float32
    return ctx.compute_quant_config_raw_ptr(tensor.data_ptr(), target_quant_dtype, tensor.numel())

def torch_to_piquant_dtype(dtype: 'torch.target_quant_dtype') -> QuantDtype:
    if torch is None:
        raise ImportError('torch is not installed')
    assert dtype in _dtype_map, f'Unsupported target_quant_dtype: {dtype}'
    return _dtype_map[dtype]

def quantize_torch(
    in_tensor: 'torch.Tensor',
    out_tensor: Union['torch.Tensor', None] = None,
    *,
    config: QuantConfig = QuantConfig(),
    ctx: Union[Context, None] = None
) -> 'torch.Tensor':
    if torch is None:
        raise ImportError('torch is not installed')

    if ctx is None:
        ctx = Context.default()

    if in_tensor.dtype in [torch.float16, torch.bfloat16, torch.quint8, torch.quint2x4, torch.quint4x2]:  # we don't have a native implementation for these dtypes yet
        in_tensor = in_tensor.float()

    if out_tensor is None:
        out_tensor = torch.empty_like(in_tensor, dtype=torch.uint8)

    assert in_tensor.is_contiguous()
    assert out_tensor.is_contiguous()
    assert in_tensor.numel() == out_tensor.numel()

    ctx.quantize_raw_ptr(
        in_tensor.data_ptr(),
        torch_to_piquant_dtype(in_tensor.dtype),
        out_tensor.data_ptr(),
        torch_to_piquant_dtype(out_tensor.dtype),
        numel=in_tensor.numel(),
        scale=config.scale,
        zero_point=config.zero_point,
        round_mode=config.mode
    )
    return out_tensor


def dequantize_torch(
    in_tensor: 'torch.Tensor',
    out_tensor: Union['torch.Tensor', None] = None,
    *,
    config: DequantConfig = DequantConfig(),
    ctx: Union[Context, None] = None
) -> 'torch.Tensor':
    if torch is None:
        raise ImportError('torch is not installed')

    if ctx is None:
        ctx = Context.default()

    if out_tensor is None:
        out_tensor = torch.empty_like(in_tensor, dtype=torch.float32)
    assert out_tensor.dtype == torch.float32

    assert in_tensor.is_contiguous()
    assert out_tensor.is_contiguous()
    assert in_tensor.numel() == out_tensor.numel()

    ctx.dequantize_raw_ptr(
        in_tensor.data_ptr(),
        torch_to_piquant_dtype(in_tensor.dtype),
        out_tensor.data_ptr(),
        torch_to_piquant_dtype(out_tensor.dtype),
        numel=in_tensor.numel(),
        scale=config.scale,
        zero_point=config.zero_point,
        reduce_op=config.reduce_op
    )
    return out_tensor
