import random
import numpy as np
import math
import pytest
from piquant import *


def np_quant(x: np.array, scale: float, zero_point: int, dtype: np.dtype) -> np.array:
    return (np.round(x / scale) + zero_point).astype(dtype)


def np_dequant(x: np.array, scale: float, zero_point: int) -> np.array:
    return (x - zero_point).astype(np.float32) * scale


NP_FLOAT_TYPES: set[np.dtype] = {np.float16, np.float32, np.float64}

NP_QUANT_TYPES: set[np.dtype] = {
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
}


def numel() -> int:
    return random.randint(1, 8192)


@pytest.mark.parametrize('dtype_in', NP_FLOAT_TYPES)
@pytest.mark.parametrize('dtype_quantized', NP_QUANT_TYPES)
def test_compute_quant_config(dtype_in: np.dtype, dtype_quantized: np.dtype) -> None:
    tensor = np.random.rand(numel()).astype(dtype_in)
    scale, zero_point = compute_quant_config_numpy(tensor, quant_dtype=numpy_to_piquant_dtype(dtype_quantized))
    assert scale > 0
    zero_point != 0
    assert not math.isnan(scale)
    assert not math.isinf(scale)


@pytest.mark.parametrize('dtype_in', NP_FLOAT_TYPES)
@pytest.mark.parametrize('dtype_quantized', NP_QUANT_TYPES)
def test_quantize_roundtrip(dtype_in: np.dtype, dtype_quantized: np.dtype) -> None:
    input = np.random.rand(numel()).astype(dtype_in)
    scale, zero_point = compute_quant_config_numpy(input, quant_dtype=numpy_to_piquant_dtype(dtype_quantized))
    quantized_numpy = np_quant(input.astype(np.float32), scale=scale, zero_point=zero_point, dtype=dtype_quantized)
    quantized_pi = quantize_numpy(
        input, scale=scale, zero_point=zero_point, quant_dtype=numpy_to_piquant_dtype(dtype_quantized)
    )

    # now dequantize both
    dequantized_numpy = np_dequant(quantized_numpy, scale=scale, zero_point=zero_point)
    dequantized_pi = dequantize_numpy(quantized_pi, scale=scale, zero_point=zero_point)
    assert dequantized_numpy.dtype == dequantized_pi.dtype
    assert dequantized_pi.dtype == np.float32
    assert np.allclose(dequantized_numpy, dequantized_pi, atol=1e-6)
    assert np.allclose(dequantized_pi, input.astype(np.float32))
