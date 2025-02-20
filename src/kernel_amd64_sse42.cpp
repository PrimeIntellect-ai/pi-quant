#ifndef __SSE4_2__
#error "Spec flag not enabled"
#endif
#ifdef __AVX__
#error "Spec level too high"
#endif

#define QUANT8_KERNEL_IMPL f32_quant8_amd64_sse42
#define QUANT4_KERNEL_IMPL f32_quant4_amd64_sse42
#define DEQUANT8_KERNEL_IMPL f32_dequant8_amd64_sse42
#define DEQUANT4_KERNEL_IMPL f32_dequant4_amd64_sse42
#include "kernels.inl"
#undef QUANT8_KERNEL_IMPL
#undef QUANT4_KERNEL_IMPL
#undef DEQUANT8_KERNEL_IMPL
#undef DEQUANT4_KERNEL_IMPL