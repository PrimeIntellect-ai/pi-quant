#ifndef __AVX2__
#error "Spec flag not enabled"
#endif
#ifdef __AVX512F__
#error "Spec level too high"
#endif

#define QUANT_KERNEL_IMPL install_quant_amd64_avx2
#include "../kernels/kernels.inl"
#undef QUANT_KERNEL_IMPL
