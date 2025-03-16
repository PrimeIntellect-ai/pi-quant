import importlib
from typing import TYPE_CHECKING, Tuple, Dict
from piquant._quant import *

if TYPE_CHECKING:
    import torch
else:
    torch = None
    if importlib.util.find_spec('torch') is not None:
        import torch

def compute_quant_config_torch_f32(tensor: 'torch.Tensor') -> Tuple[float, int]:
    """
    Compute the scale and zero point of a tensor.
        :param tensor: input tensor
    """
    if torch is None:
        raise ImportError('torch is not installed')
    assert tensor.is_contiguous()
    assert tensor.dtype == torch.float32
    return compute_quant_config_f32_raw_ptr(tensor.data_ptr(), tensor.numel())

__torch_dtype_map: Dict['torch.dtype', QuantDtype] = {
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

def torch_to_piquant_dtype(dtype: 'torch.dtype') -> QuantDtype:
    if torch is None:
        raise ImportError('torch is not installed')
    assert dtype in __torch_dtype_map, f'Unsupported dtype: {dtype}'
    return __torch_dtype_map[dtype]

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

    if in_tensor.dtype in [torch.float16, torch.bfloat16]:  # we don't have a native implementation for these dtypes yet
        in_tensor = in_tensor.float()
    elif in_tensor.dtype == torch.float32:
        pass
    else:
        raise ValueError(f'unsupported dtype: {in_tensor.dtype}')

    if config.output_dtype == QuantDtype.UINT8:
        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor, dtype=torch.uint8)
        assert out_tensor.dtype == torch.uint8
    elif config.output_dtype == QuantDtype.UINT4:
            raise NotImplementedError('quantization to int4 is not yet implemented for torch tensors')

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

    if in_tensor.dtype == torch.uint8:
        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor, dtype=torch.float32)
        assert out_tensor.dtype == torch.float32
    elif in_tensor.dtype == torch.int4:
        raise NotImplementedError('dequantization from int4 is not yet implemented for torch tensors')
    else:
        raise ValueError(f'unsupported dtype: {in_tensor.dtype}')

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
