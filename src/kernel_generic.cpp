#define Q8_KERNEL_IMPL f32_q8_generic
#define Q4_KERNEL_IMPL f32_q4_generic
#include "kernels.inl"
#undef Q8_KERNEL_IMPL
#undef Q4_KERNEL_IMPL