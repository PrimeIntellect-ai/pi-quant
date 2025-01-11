cmake_minimum_required(VERSION 3.20)

project(quant_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(quant_kernels STATIC src/quant.hpp src/quant.cpp src/q8_kernel.inl)
target_compile_options(quant_kernels PRIVATE -msse2 -msse3 -mssse3 -msse4.1 -fno-rtti -fno-exceptions)

# release mode opt
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(quant_kernels PRIVATE -O3)
endif()

add_executable(bench benchmark/bench.cpp)
target_link_libraries(bench PRIVATE quant_kernels)

add_executable(test test/test.cpp)
target_link_libraries(test PRIVATE quant_kernels)
