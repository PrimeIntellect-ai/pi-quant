# Prime Fast Quantization Library

## Overview

This project provides **multithreaded CPU SIMD int8 and int4 quantization kernels** with various rounding modes. The kernels are optimized for different CPU architectures, including **AMD64** (SSE4.2, AVX2, AVX512F) and **ARM64**(Neon). The most optimal kernel is selected at runtime.

## What is Quantization?

Quantization is the process of mapping continuous values into a finite, discrete set of values. In machine learning and signal processing, it is commonly used to **reduce the precision of numerical data**, lowering memory usage and improving computational efficiency while maintaining acceptable accuracy.

## Features

✅ **Parallel De/Quantization**: Efficiently quantizes and dequantizes data using multiple threads.

✅ **Multiple Datatypes:**  Support for f32 ↔ uint8 and f32 ↔ uint4 quantization. **(uint4 is still WIP)**

✅ **Modern Python API:** Use the library from Python with PyTorch, numpy or standalone.

✅ **Architecture-Specific Optimizations**: Includes optimizations for AMD64 with SSE4.2, AVX2, and AVX512 instruction sets and ARM64 with NEON.

✅ **Thread Pool**: Reuses threads for minimal overhead.

✅ **Flexible Rounding Modes**: Supports both **nearest** and **stochastic** rounding modes.

✅ **C99 API**: Provides a C99 API for C projects or foreign language bindings (see `quant.h`).

✅ **Store Operators:** Multiple store operators for dequantization, useful for ring reduces.

## Benchmark

The benchmarks were run on a variety of hardware. We benchmark against torch quint8 quantize_per_tensor and also against the torch.fx quantize_per_tensor. Benchmarked was float32 to uint8 quantization with 1000 runs. The numel and other properties can be see in the [benchmark code](https://github.com/PrimeIntellect-ai/quantization-kernels/blob/main/python/benchmark/benchmark.py).

In the charts, “Torch FX Quant” refers to **torch.ao.quantization.fx._decomposed.quantize_per_tensor**.

“Torch Builtin Quant” referes to **torch.quantize_per_tensor** and Fast Quant to our own library **quant.quant_torch.**

### Benchmark 1 (Threadripper 3970X 32-Core Processor, 64 CPUs)

* 1000 runs with numel 27264000
* CPU:  AMD Ryzen Threadripper 3970X 32-Core Processor, Runtime: AVX2
* Memory: 128 GB
* Linux: 6.1.0-30-amd64

![image.png](https://i.imgur.com/rjurbfB.png)

### Benchmark 2 (Apple M3 Pro, 11 CPUs)

* 1000 runs with numel 27264000
* CPU: Apple M3 Pro, 11 CPUs, Runtime: ARM Neon
* Memory: 18GB
* OSX: 15.3.1 (24D70

![image.png](https://i.imgur.com/aMCvInY.png)

### Benchmark 3 (Xeon Platinum 8470, 104 vCPUs)

* 1000 runs with numel 27264000
* CPU:  Intel(R) Xeon(R) Platinum 8470, 104 vCPUs, Runtime: AVX512-F
* Memory: 752 GB
* Linux: 5.15.0-112-generic

![image.png](https://i.imgur.com/GreULz2.png)