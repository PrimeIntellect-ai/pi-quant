import torch
import piquant

# Quantize and back with: bfloat16 -> uint4 -> bfloat16
# In torch, quint4x2 means two 4-bit quantized integers per byte.
tensor = torch.rand(1000, dtype=torch.bfloat16, device='cpu')

# Compute quantization parameters for uint4 (needed for quantization and dequantization)
scale, zero_point = piquant.torch.compute_quant_params(tensor, dtype=torch.quint4x2)

# Quantize the tensor to uint4
quantized = piquant.torch.quantize(tensor, scale=scale, zero_point=zero_point, dtype=torch.quint4x2)

# Dequantize back to bfloat16
dequantized = piquant.torch.dequantize(quantized, scale=scale, zero_point=zero_point, dtype=torch.bfloat16)

# Check if the dequantized tensor is close to the original tensor
assert torch.allclose(dequantized, tensor, atol=scale/2 + 1e-3), "Dequantization did not match original tensor"

# Print parts of original and dequantized tensors for verification
print("Original tensor (first 10 elements):", tensor[:10].tolist())
print("Dequant  tensor (first 10 elements):", dequantized[:10].tolist())

