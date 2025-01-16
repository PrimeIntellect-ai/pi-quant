#ifndef __AVX2__
#error "Spec flag not enabled"
#endif
#ifdef __AVX512F__
#error "Spec level too high"
#endif

#define Q8_KERNEL_IMPL f32_q8_amd64_avx2
#define Q4_KERNEL_IMPL f32_q4_amd64_avx2
#include "kernels.inl"
#undef Q8_KERNEL_IMPL
#undef Q4_KERNEL_IMPL