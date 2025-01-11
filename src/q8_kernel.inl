#ifndef Q8_KERNEL_IMPL
#error "impl is not defined"
#endif

#include <algorithm>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif
