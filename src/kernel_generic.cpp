#define QUANT8_KERNEL_IMPL f32_quant8_generic
#define QUANT4_KERNEL_IMPL f32_quant4_generic
#define DEQUANT8_KERNEL_IMPL f32_dequant8_generic
#define DEQUANT4_KERNEL_IMPL f32_dequant4_generic
#include "kernels.inl"
#undef QUANT8_KERNEL_IMPL
#undef QUANT4_KERNEL_IMPL
#undef DEQUANT8_KERNEL_IMPL
#undef DEQUANT4_KERNEL_IMPL