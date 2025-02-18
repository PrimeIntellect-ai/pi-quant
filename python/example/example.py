import torch
import quant

tensor = torch.rand(8192*8192*32)
print(tensor)

ctx = quant.Context()
quantized_tensor = torch.empty(tensor.numel(), dtype=torch.uint8)
scale = 0.00784
zero_point = 128
ctx.ptr_quant_uint8(tensor.data_ptr(), quantized_tensor.data_ptr(), numel=tensor.numel(), scale=scale, zero_point=zero_point, mode=quant.RoundMode.NEAREST)
print(quantized_tensor)
