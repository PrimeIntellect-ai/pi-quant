cmake_minimum_required(VERSION 3.20)

project(quant_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(quant_kernels STATIC quant.hpp quant.cpp)
target_compile_options(quant_kernels PRIVATE -msse2 -msse3 -mssse3 -msse4.1)

add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE quant_kernels)
