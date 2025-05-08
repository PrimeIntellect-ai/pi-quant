cmake_minimum_required(VERSION 3.20)

project(piquant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(QUANT_BUILD_TESTS "Build tests" ON)
option(QUANT_BUILD_BENCHMARKS "Build benchmarks" ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(IS_AMD64 TRUE)
else()
    set(IS_AMD64 FALSE)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(IS_ARM64 TRUE)
else()
    set(IS_ARM64 FALSE)
endif()

message(STATUS "Building for ${CMAKE_SYSTEM_PROCESSOR}")
file(GLOB QUANT_SOURCES include/*.hpp src/*.cpp src/*.hpp src/*.inl)
if (${IS_AMD64})
    function(set_file_opts filename posix_arch msvc_arch)
        message(STATUS "BLAS CPU permutation ${filename} ${posix_arch} / ${msvc_arch}")
        if (WIN32)
            set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/${filename}" APPEND PROPERTY COMPILE_FLAGS "${msvc_arch}")
        else()
            set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/${filename}" APPEND PROPERTY COMPILE_FLAGS "${posix_arch}")
        endif()
    endfunction()

    set(QUANT_SOURCES_AMD64
        src/amd64/kernel_amd64_sse42.cpp
        src/amd64/kernel_amd64_avx2.cpp
        src/amd64/kernel_amd64_avx512f.cpp
    )
    set(QUANT_SOURCES ${QUANT_SOURCES} ${QUANT_SOURCES_AMD64})

    set_file_opts("amd64/kernel_amd64_sse42.cpp" "-mtune=nehalem -msse4.2"  "/arch:SSE4.2")
    set_file_opts("amd64/kernel_amd64_avx2.cpp" "-mtune=skylake -mavx -mavx2 -mfma -mf16c"  "/arch:AVX2")
    set_file_opts("amd64/kernel_amd64_avx512f.cpp" "-mtune=cannonlake -mavx -mavx2 -mfma -mf16c -mavx512f -mavx512bw" "/arch:AVX512")

endif()

if (QUANT_COMPILE_STATIC)
    add_library(piquant STATIC ${QUANT_SOURCES})
else()
    add_library(piquant SHARED ${QUANT_SOURCES})
    target_compile_definitions(piquant PRIVATE QUANT_BUILD_SHARED)
    add_compile_options(-fPIC)
endif ()

target_compile_options(piquant PRIVATE -fomit-frame-pointer -fno-rtti)
target_include_directories(piquant PUBLIC include)

# add threadpool library
if (NOT TARGET threadpool)
    add_subdirectory(third_party/threadpool)
endif()

target_link_libraries(piquant PUBLIC threadpool)

# release mode opt
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(piquant PRIVATE -O3)
endif()

if (${QUANT_BUILD_TESTS})
    enable_testing()
    add_subdirectory(test)
endif()

if (${QUANT_BUILD_BENCHMARKS})
    add_subdirectory(benchmark)
endif()
