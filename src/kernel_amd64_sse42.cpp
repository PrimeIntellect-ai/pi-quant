#ifndef __SSE4_2__
#error "Spec flag not enabled"
#endif
#ifdef __AVX__
#error "Spec level too high"
#endif

#define Q8_KERNEL_IMPL f32_q8_amd64_sse42
#define Q4_KERNEL_IMPL f32_q4_amd64_sse42
#include "kernels.inl"
#undef Q8_KERNEL_IMPL
#undef Q4_KERNEL_IMPL