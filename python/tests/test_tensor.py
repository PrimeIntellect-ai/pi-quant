import numpy as np
import pytest
import torch
from quant import *

def test_ptr_factors_compute():
    tensor = torch.rand(32)

    ctx = Context()
    scale, zero_point = compute_config_properties_from_data(tensor.data_ptr(), tensor.numel())
    assert scale > 0
    assert zero_point >= 0

def test_ptr_quant_int8():
    tensor = torch.rand(32)

    ctx = Context()
    quantized_tensor = torch.empty(tensor.numel(), dtype=torch.uint8)
    scale = 0.00784
    zero_point = 128
    ctx.ptr_quant_uint8(tensor.data_ptr(), quantized_tensor.data_ptr(), numel=tensor.numel(), scale=scale,
                    zero_point=zero_point, mode=RoundMode.NEAREST)

def test_quant_torch():
    tensor = torch.rand(32)
    
    quantized_tensor = quant_torch(tensor, config=QuantConfig(output_dtype=QuantDtype.INT8))
    
    assert quantized_tensor.dtype == torch.int8
    assert quantized_tensor.numel() == tensor.numel()
    

def test_quant_numpy():
    tensor = np.random.rand(32).astype(np.float32)

    quantized_tensor = quant_numpy(tensor, config=QuantConfig(output_dtype=QuantDtype.INT8))
    
    assert quantized_tensor.dtype == np.int8
    assert quantized_tensor.shape == tensor.shape
    
    
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quant_torch_half_precision(dtype):
    tensor = torch.rand(32).bfloat16()
    
    quantized_tensor = quant_torch(tensor, config=QuantConfig(output_dtype=QuantDtype.INT8))
    
    assert quantized_tensor.dtype == torch.int8
    assert quantized_tensor.numel() == tensor.numel()
    
def test_quant_numpy_fp16():
    tensor = np.random.rand(32).astype(np.float16)
    
    quantized_tensor = quant_numpy(tensor, config=QuantConfig(output_dtype=QuantDtype.INT8))
    
    assert quantized_tensor.dtype == np.int8
    assert quantized_tensor.shape == tensor.shape

"""
def test_custom_quant_vs_torch():
    tensor = torch.rand(32)

    scale, zero_point = compute_config_properties_from_data_torch(tensor)
    torch_quant = torch.quantize_per_tensor(tensor, scale=scale, zero_point=zero_point, dtype=torch.quint8)

    custom_quant = quant_torch(tensor, config=QuantConfig(
        output_dtype=QuantDtype.INT8,
        scale=scale,
        zero_point=zero_point
    ))

    torch_int = torch_quant.int_repr()
        custom_int = custom_quant.int_repr() if hasattr(custom_quant, "int_repr") else custom_quant
    
        assert torch.allclose(torch_int.float(), custom_int.float(), atol=1)

    torch_dequant = torch_quant.dequantize()
    custom_dequant = torch.tensor([x - zero_point * scale for x in custom_quant.tolist()])
    assert torch.allclose(torch_dequant, custom_dequant, atol=scale)
"""