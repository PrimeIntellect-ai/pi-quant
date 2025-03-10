# Quantization Kernels

This project provides multithreaded SIMD int8 and int4 quantization kernels with various rounding modes.<br>
The kernels are optimized for different CPU architectures, including AMD64 (SSE 4.2, AVX2, AVX512F) and ARM64 (Neon).<br>
The most optimal kernel is selected at runtime.

## Features

- **Parallel Quantization**: Efficiently quantizes data using multiple threads.
- **Architecture-Specific Optimizations**: Includes optimizations for AMD64 with SSE4.2, AVX2, and AVX512 instruction sets and ARM64 with NEON.
- **Thread Pool**: Reuses threads for minimum overhead.
- **Flexible Rounding Modes**: Supports both nearest and stochastic rounding modes.
- **C99 API**: Provides a C99 API for easy integration with plain C (see capi.h).

## Building the Project

### Prerequisites

- **CMake**: Version 3.20 or higher is required.
- **C++ Compiler**: A compiler that supports C++20 is necessary.

### Build Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd quantization-kernels
   ```

2. **Create a Build Directory**:
   ```bash
   mkdir build
   cd build
   ```

3. **Run CMake**:
   ```bash
   cmake ..
   ```

4. **Build the Project**:
   ```bash
   cmake --build .
   ```

### Running Benchmarks and Tests

- **Benchmarks**: After building, you can run the benchmark executable to evaluate performance.
  ```bash
  ./bench
  ```

- **Tests**: Run the test executable to verify functionality.
  ```bash
  ./test
  ```

## Code Structure

- **src/**: Contains the source code for the quantization kernels.
- **benchmark/**: Contains benchmarking code.
- **test/**: Contains test code.

## Usage

The main interface is provided through the `piquant::context` class, which allows for quantization of data using the `quantize_uint8` and `quantize_uint4` methods. These methods require input data, output buffers, a scale factor, a zero point, and a rounding mode.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
