#if !defined(__AVX512F__) || !defined(__AVX512BW__)
#error "Spec flag not enabled"
#endif
#ifdef __AVX10__
#error "Spec level too high"
#endif

#define Q8_KERNEL_IMPL f32_q8_amd64_avx512f
#define Q4_KERNEL_IMPL f32_q4_amd64_avx512f
#include "kernels.inl"
#undef Q8_KERNEL_IMPL
#undef Q4_KERNEL_IMPL