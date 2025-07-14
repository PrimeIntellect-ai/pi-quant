import random

import pytest
import math
import torch
import piquant

gen = torch.manual_seed(128)
random.seed(128)

TORCH_FLOAT_TYPES: set[torch.dtype] = {
    torch.bfloat16, torch.float32
}

TORCH_QUANT_TYPES: set[torch.dtype] = {
    torch.quint8,
    torch.quint4x2,
    #torch.quint2x4,
}

def numel() -> int:
    return random.randint(1, 128)

@pytest.mark.parametrize('dtype_in', TORCH_FLOAT_TYPES)
@pytest.mark.parametrize('dtype_quantized', TORCH_QUANT_TYPES)
def test_compute_quant_config(dtype_in: torch.dtype, dtype_quantized: torch.dtype) -> None:
    tensor = torch.empty(numel(), numel(), numel(), numel(), dtype=dtype_in)
    tensor.uniform_(-1.0, 1.0, generator=gen)
    scale, zero_point = piquant.torch.compute_quant_config(tensor, dtype_quantized)
    assert scale > 0
    zero_point != 0
    assert not math.isnan(scale)
    assert not math.isinf(scale)


@pytest.mark.parametrize('dtype_in', TORCH_FLOAT_TYPES)
@pytest.mark.parametrize('dtype_quantized', TORCH_QUANT_TYPES)
def test_quantize_roundtrip(dtype_in: torch.dtype, dtype_quantized: torch.dtype) -> None:
    input = torch.empty(numel(), numel(), numel(), numel(), dtype=dtype_in)
    input.uniform_(-1.0, 1.0, generator=gen)
    scale, zero_point = piquant.torch.compute_quant_config(input, dtype_quantized)
    quantized_torch = torch.quantize_per_tensor(
        input.float(), scale=scale, zero_point=zero_point, dtype=dtype_quantized
    )
    quantized_pi = piquant.torch.quantize(input, dtype_quantized, zero_point=zero_point, scale=scale)

    # now dequantize both
    dequantized_torch = quantized_torch.dequantize().to(dtype_in)
    dequantized_pi = piquant.torch.dequantize(quantized_pi, dtype_in, scale=scale, zero_point=zero_point)
    assert dequantized_torch.dtype == dequantized_pi.dtype
    assert dequantized_pi.dtype == input.dtype
    assert torch.allclose(dequantized_torch, dequantized_pi, atol=1e-3)
    assert torch.allclose(dequantized_torch, input, atol=scale/2 + 1e-3)
    assert torch.allclose(dequantized_pi, input, atol=scale/2 + 1e-3)
