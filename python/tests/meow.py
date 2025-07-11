import torch
from piquant import *

tensor = torch.tensor([0.978606, 0.57, 0.480873, -0.571152, -0.463621, -0.578745, 0.583622])

scale, zero_point = compute_quant_config_torch(tensor, target_quant_dtype=QuantDtype.UINT2)
zero_point = 0
torch_dequant = torch.dequantize(
    torch.quantize_per_tensor(tensor, scale=scale, zero_point=zero_point, dtype=torch.quint2x4)
)
assert torch_dequant.dtype == torch.float32
pi_dequant = dequantize_torch(
    quantize_torch(tensor, config=QuantConfig(output_dtype=QuantDtype.UINT2, scale=scale, zero_point=zero_point)),
    None,
    config=DequantConfig(scale, zero_point),
)
print(f'PI: {pi_dequant}')
print(f'Torch: {torch_dequant}')

