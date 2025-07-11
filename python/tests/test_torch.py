import random

import pytest
import math
import torch
from piquant import *

TORCH_FLOAT_TYPES: set[torch.dtype] = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64
}

TORCH_QUANT_TYPES: set[torch.dtype] = {
    torch.quint8,
    torch.qint8,
    torch.quint4x2,
    torch.quint2x4,
}

def numel() -> int:
    return random.randint(1, 8192)

@pytest.mark.parametrize("dtype_in", TORCH_FLOAT_TYPES)
@pytest.mark.parametrize("dtype_quantized", TORCH_QUANT_TYPES)
def test_compute_quant_config(dtype_in: torch.dtype, dtype_quantized: torch.dtype) -> None:
    tensor = torch.rand(numel(), dtype=dtype_in)
    scale, zero_point = compute_quant_config_torch(tensor, target_quant_dtype=torch_to_piquant_dtype(dtype_quantized))
    assert scale > 0
    zero_point != 0
    assert not math.isnan(scale)
    assert not math.isinf(scale)

@pytest.mark.parametrize("dtype_quantized", TORCH_QUANT_TYPES)
def test_quantize(dtype_quantized: torch.dtype) -> None:
    input = torch.rand(numel(), dtype=torch.float32)
    scale, zero_point = compute_quant_config_torch(input, target_quant_dtype=torch_to_piquant_dtype(dtype_quantized))
    quantized_torch = torch.quantize_per_tensor(input, scale=scale, zero_point=zero_point, dtype=dtype_quantized)
    quantized_pi = quantize_torch(input, scale=scale, zero_point=zero_point, output_dtype=torch_to_piquant_dtype(dtype_quantized))
