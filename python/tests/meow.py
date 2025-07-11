import torch
from piquant import *

input = torch.tensor([0.978606, 0.57, 0.480873, -0.571152, -0.463621, -0.578745, 0.583622])

def get_q_scale_and_zero_point(tensor, q_min, q_max):
    r_min, r_max = tensor.min().item(), tensor.max().item()
    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_min - (r_min / scale)
    zero_point = max(min(int(round(zero_point)), q_max), q_min)

    return scale, zero_point

scale, zero_point = get_q_scale_and_zero_point(input, 0, 3)
print(f'MinMax Scale and Zero Point: {scale}, {zero_point}')
quantized_torch = torch.quantize_per_tensor(input, scale=scale, zero_point=zero_point, dtype=torch.quint2x4)
quantized_pi = quantize_torch(input, scale=scale, zero_point=zero_point, output_dtype=torch_to_piquant_dtype(torch.quint2x4))

# now dequantize both
dequantized_torch = quantized_torch.dequantize()
dequantized_pi = dequantize_torch(quantized_pi, scale=scale, zero_point=zero_point)
print(f'I {input.tolist()}')
print(f'T: {dequantized_torch.tolist()}')
print(f'P: {dequantized_pi.tolist()}')

