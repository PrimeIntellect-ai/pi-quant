import numpy as np
import pytest
import torch
from piquant import *

def test_ptr_dequant_config_compute():
    tensor = torch.rand(32)

    scale, zero_point = compute_quant_config_raw_ptr(tensor.data_ptr(), QuantDtype.UINT8, tensor.numel())
    assert scale > 0
    assert zero_point >= 0

def test_torch_dequant_config_compute():
    tensor = torch.rand(8192)
    scale, zero_point = compute_quant_config_torch(tensor, target_quant_dtype=QuantDtype.UINT8)
    assert scale > 0
    assert zero_point >= 0

def test_ptr_quant_int8():
    tensor = torch.rand(32)

    ctx = Context()
    quantized_tensor = torch.empty(tensor.numel(), dtype=torch.uint8)
    scale = 0.00784
    zero_point = 128
    ctx.quantize_raw_ptr(tensor.data_ptr(), QuantDtype.F32, quantized_tensor.data_ptr(), QuantDtype.UINT8, numel=tensor.numel(), scale=scale,
                    zero_point=zero_point, round_mode=RoundMode.NEAREST)

def test_quant_torch():
    tensor = torch.rand(32)
    
    quantized_tensor = quantize_torch(tensor, config=QuantConfig(output_dtype=QuantDtype.UINT8))
    
    assert quantized_tensor.dtype == torch.uint8
    assert quantized_tensor.numel() == tensor.numel()
    
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quant_torch_half_precision(dtype):
    tensor = torch.rand(32).bfloat16()
    
    quantized_tensor = quantize_torch(tensor, config=QuantConfig(output_dtype=QuantDtype.UINT8))
    
    assert quantized_tensor.dtype == torch.uint8
    assert quantized_tensor.numel() == tensor.numel()

def test_custom_quant_vs_torch_uint8():
    tensor = torch.rand(8192)
    scale, zero_point = compute_quant_config_torch(tensor, target_quant_dtype=QuantDtype.UINT8)
    torch_quant = torch.quantize_per_tensor(tensor, scale=scale, zero_point=zero_point, dtype=torch.quint8).int_repr()
    fast_quant = quantize_torch(tensor, config=QuantConfig(output_dtype=QuantDtype.UINT8, scale=scale, zero_point=zero_point))
    assert torch_quant.dtype == fast_quant.dtype
    assert torch_quant.numel() == tensor.numel()
    assert torch_quant.numel() == fast_quant.numel()
    for i in range(tensor.numel()):
        assert torch_quant[i].item() == fast_quant[i].item()

def test_custom_quant_vs_torch_decomposed_uint8():
    from torch.ao.quantization.fx._decomposed import quantize_per_tensor
    tensor = torch.rand(8192)
    scale, zero_point = compute_quant_config_torch(tensor, target_quant_dtype=QuantDtype.UINT8)
    torch_quant = quantize_per_tensor(tensor, scale=scale, zero_point=zero_point, quant_min=0, quant_max=255, dtype=torch.uint8)
    fast_quant = quantize_torch(tensor, config=QuantConfig(output_dtype=QuantDtype.UINT8, scale=scale, zero_point=zero_point))
    assert torch_quant.dtype == fast_quant.dtype
    assert torch_quant.numel() == tensor.numel()
    assert torch_quant.numel() == fast_quant.numel()
    for i in range(tensor.numel()):
        assert torch_quant[i].item() == fast_quant[i].item()

def test_custom_dequant_vs_torch_uint8_reduce_set():
    tensor = torch.rand(8192)
    scale, zero_point = compute_quant_config_torch(tensor, target_quant_dtype=QuantDtype.UINT8)
    torch_quant = torch.quantize_per_tensor(tensor, scale=scale, zero_point=zero_point, dtype=torch.quint8)
    torch_dequant = torch.dequantize(torch_quant)
    assert torch_dequant.dtype == torch.float32
    fast_quant = quantize_torch(tensor, config=QuantConfig(output_dtype=QuantDtype.UINT8, scale=scale, zero_point=zero_point))
    fast_dequant = dequantize_torch(fast_quant, None, config=DequantConfig(scale, zero_point))
    assert fast_dequant.dtype == torch.float32
    assert torch_dequant.numel() == fast_dequant.numel()
    assert torch_dequant.dtype == fast_dequant.dtype
    for i in range(tensor.numel()):
        assert torch_dequant[i].item() == fast_dequant[i].item()

def test_custom_dequant_vs_torch_uint8_reduce_add():
    tensor = torch.rand(8192)
    scale, zero_point = compute_quant_config_torch(tensor, target_quant_dtype=QuantDtype.UINT8)
    torch_quant = torch.quantize_per_tensor(tensor, scale=scale, zero_point=zero_point, dtype=torch.quint8)
    torch_dequant = torch.dequantize(torch_quant) + 3.1415
    assert torch_dequant.dtype == torch.float32
    fast_quant = quantize_torch(tensor, config=QuantConfig(output_dtype=QuantDtype.UINT8, scale=scale, zero_point=zero_point))
    fast_dequant = torch.full(size=fast_quant.shape, fill_value=3.1415, dtype=torch.float32)
    dequantize_torch(fast_quant, fast_dequant, config=DequantConfig(scale, zero_point, ReduceOp.ADD))
    assert fast_dequant.dtype == torch.float32
    assert torch_dequant.numel() == fast_dequant.numel()
    assert torch_dequant.dtype == fast_dequant.dtype
    for i in range(tensor.numel()):
        assert torch_dequant[i].item() == fast_dequant[i].item()
