#if !defined(__AVX512F__) || !defined(__AVX512BW__)
#error "Spec flag not enabled"
#endif
#ifdef __AVX10__
#error "Spec level too high"
#endif

#define QUANT8_KERNEL_IMPL f32_quant8_amd64_avx512f
#define QUANT4_KERNEL_IMPL f32_quant4_amd64_avx512f
#define DEQUANT8_KERNEL_IMPL f32_dequant8_amd64_avx512f
#define DEQUANT4_KERNEL_IMPL f32_dequant4_amd64_avx512f
#include "kernels.inl"
#undef QUANT8_KERNEL_IMPL
#undef QUANT4_KERNEL_IMPL
#undef DEQUANT8_KERNEL_IMPL
#undef DEQUANT4_KERNEL_IMPL
