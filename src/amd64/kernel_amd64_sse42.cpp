#ifndef _MSC_VER
#ifndef __SSE4_2__
#error "Spec flag not enabled"
#endif
#endif
#ifdef __AVX__
#error "Spec level too high"
#endif

#define QUANT_KERNEL_IMPL install_quant_amd64_sse42
#include "../kernels.inl"
#undef QUANT_KERNEL_IMPL
