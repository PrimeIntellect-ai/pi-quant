#if !defined(__AVX512F__) || !defined(__AVX512BW__)
#error "Spec flag not enabled"
#endif
#ifdef __AVX10__
#error "Spec level too high"
#endif

#define QUANT_KERNEL_IMPL install_quant_amd64_avx512f
#include "../kernels.inl"
#undef QUANT_KERNEL_IMPL
