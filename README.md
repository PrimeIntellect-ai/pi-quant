# pi-quant: Prime Intellect Fast Quantization Library
![logo.png](https://raw.githubusercontent.com/PrimeIntellect-ai/pi-quant/main/media/logo.png)

## Overview

**Fast, multithreaded CPU quantization kernels** with various rounding modes, outperforming PyTorch’s built-in quantization routines by **more than 2 times** on all tested hardware.
The kernels are optimized with SIMD intrinsics for different CPU architectures, including **AMD64** (SSE4.2, AVX2, AVX512F) and **ARM64** (Neon). The most optimal kernel is selected at runtime using runtime CPU detection.

## What is Quantization?

Quantization is the process of mapping continuous values into a finite, discrete set of values. In machine learning and signal processing, it is commonly used to **reduce the precision of numerical data**, lowering memory usage and improving computational efficiency while maintaining acceptable accuracy.

## Features

✅ **Parallel De/Quantization**: Efficiently quantizes and de-quantizes data using multiple threads.

✅ **Rich Datatype Support:** Provides f32, f64 ↔ (u)int4/8/16/32/64.

✅ **Modern Python API:** Use the library from Python with PyTorch, numpy or standalone.

✅ **Architecture-Specific Optimizations**: The kernels are optimized with SIMD intrinsics for different CPU architectures, including **AMD64** (SSE4.2, AVX2, AVX512F) and **ARM64** (Neon).

✅ **Thread Pool**: Reuses threads for minimal overhead.

✅ **Flexible Rounding Modes**: Supports both **nearest** and **stochastic** rounding modes.

✅ **C99 API**: Provides a C99 API for C projects or foreign language bindings (see `quant.h`).

✅ **Store Operators:** Supports multiple store modes (SET, ADD) during dequantization — useful for ring-reduction operations.

✅ **Quantization Parameters:** Efficient SIMD-parallel computation of quantization scale and zero point from input data.

## Installation

To install pi-quant from PyPI, run the following command:
```bash
pip install pypiquant
```

## Examples
piquant is torch compatible. Here are some examples of how to use it with PyTorch:

```python
import torch
import piquant

# Quantize and back with: bfloat16 -> uint4 -> bfloat16
# In torch, quint4x2 means two 4-bit quantized integers per byte.
tensor = torch.rand(1000, dtype=torch.bfloat16, device='cpu')

# Compute quantization parameters for uint4 (needed for quantization and dequantization)
scale, zero_point = piquant.torch.compute_quant_params(tensor, dtype=torch.quint4x2)

# Quantize the tensor to uint4
quantized = piquant.torch.quantize(tensor, scale=scale, zero_point=zero_point, dtype=torch.quint4x2)

# Dequantize back to bfloat16
dequantized = piquant.torch.dequantize(quantized, scale=scale, zero_point=zero_point, dtype=torch.bfloat16)

# Check if the dequantized tensor is close to the original tensor
assert torch.allclose(dequantized, tensor, atol=scale*0.5 + 1e-3), "Dequantization did not match original tensor"

# Print parts of original and dequantized tensors for verification
print("Original tensor (first 10 elements):", tensor[:10].tolist())
print("Dequant  tensor (first 10 elements):", dequantized[:10].tolist())
```

## Benchmark

The benchmarks were run on a variety of hardware. We benchmark against PyTorch’s [**torch.quantize_per_tensor**](https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html) and  **[torch.ao.quantization.fx._decomposed.quantize_per_tensor**.](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py) Each benchmark quantized float32 to uint8 across **1000 runs**. The number of elements and other details can be seen in the [benchmark code](https://github.com/PrimeIntellect-ai/quantization-kernels/blob/main/python/benchmark/benchmark.py).

### Benchmark 1 (AMD EPYC 9654, 360 vCPUs)

1000 runs with numel 27264000<br>
CPU:  AMD EPYC 9654 96-Core Processor, Runtime: AVX512-F<br>
Memory: 1485 GB<br>
Linux: 6.8.0-57-generic<br>

![bench1.png](https://raw.githubusercontent.com/PrimeIntellect-ai/pi-quant/main/media/bench1.png)
**Torch FX Quant** refers to  **[torch.ao.quantization.fx._decomposed.quantize_per_tensor](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py),**
**Torch Builtin Quant**  to [**torch.quantize_per_tensor](https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)** and **Fast Quant** to **pi-quant’s [piquant.quantize_torch](https://github.com/PrimeIntellect-ai/piquant/blob/4bcf6ebc69bf9b44f89b13965f010a1d025a59f6/python/src/piquant/_torch.py#L52).**

### Benchmark 2 (AMD EPYC 7742, 128 vCPUs)

1000 runs with numel 27264000<br>
CPU:  AMD EPYC 7742 64-Core Processor, Runtime: AVX2<br>
Memory: 528 GB<br>
Linux: 6.8.0-1023-nvidia<br>
![bench2.png](https://raw.githubusercontent.com/PrimeIntellect-ai/pi-quant/main/media/bench2.png)

### Benchmark 3 (Apple M3 Pro)

1000 runs with numel 27264000<br>
CPU:  Apple M3 Pro, Runtime: Neon<br>
Memory: 18 GB<br>
OSX: 15.4 (24E248)<br>
![bench3.png](https://raw.githubusercontent.com/PrimeIntellect-ai/pi-quant/main/media/bench3.png)
