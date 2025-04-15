import numpy
import pytest
import math
import numpy as np
from piquant import *

assert numpy is not None, "Numpy is not installed. Please install numpy to run these tests."

def test_dequant_config_compute_numpy():
    arr = np.random.rand(8192).astype(np.float32)
    scale, zero_point = compute_quant_config_numpy(arr, target_quant_dtype=QuantDtype.UINT8)
    assert scale > 0
    assert zero_point >= 0
    assert not math.isnan(scale)
    assert not math.isinf(scale)


def test_quant_numpy():
    array = np.random.rand(8192).astype(np.float32)

    quantized_array = quantize_numpy(array, config=QuantConfig(output_dtype=QuantDtype.UINT8))

    assert quantized_array.dtype == numpy.uint8
    assert quantized_array.size == array.size


def test_quant_numpy_half_precision():
    array = np.random.rand(8192).astype(np.float16)

    quantized_array = quantize_numpy(array, config=QuantConfig(output_dtype=QuantDtype.UINT8))

    assert quantized_array.dtype == numpy.uint8
    assert quantized_array.size == array.size

def test_quant_vs_numpy_uint8():
    array = np.random.rand(8192).astype(np.float32)
    scale, zero_point = compute_quant_config_numpy(array, target_quant_dtype=QuantDtype.UINT8)
    numpy_quant = (np.round(array / scale) + zero_point).astype(np.uint8)
    fast_quant = quantize_numpy(array, config=QuantConfig(output_dtype=QuantDtype.UINT8, scale=scale, zero_point=zero_point))
    assert numpy_quant.dtype == fast_quant.dtype
    assert numpy_quant.size == array.size
    assert numpy_quant.size == fast_quant.size
    assert np.allclose(numpy_quant, fast_quant)

def test_dequant_vs_torch_uint8_reduce_set():
    array = np.random.rand(8192).astype(np.float32)
    scale, zero_point = compute_quant_config_numpy(array, target_quant_dtype=QuantDtype.UINT8)
    numpy_quant = (np.round(array / scale) + zero_point).astype(np.uint8)
    numpy_dequant = (numpy_quant - zero_point).astype(np.float32) * scale
    assert numpy_dequant.dtype == numpy.float32
    fast_dequant = dequantize_numpy(quantize_numpy(array, config=QuantConfig(output_dtype=QuantDtype.UINT8, scale=scale, zero_point=zero_point)), None, config=DequantConfig(scale, zero_point))
    assert fast_dequant.dtype == numpy.float32
    assert numpy_dequant.size == fast_dequant.size
    assert numpy_dequant.dtype == fast_dequant.dtype
    assert np.allclose(numpy_dequant, fast_dequant, rtol=1e-5, atol=1e-5)
