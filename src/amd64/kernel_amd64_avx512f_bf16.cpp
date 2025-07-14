#if !defined(__AVX512F__) || !defined(__AVX512BW__) || !defined(__AVX512BF16__)
#error "Spec flag not enabled"
#endif
#ifdef __AVX10__
#error "Spec level too high"
#endif

#define QUANT_KERNEL_IMPL install_quant_amd64_avx512f_bf16
#include "../kernels/kernels.inl"
#undef QUANT_KERNEL_IMPL
