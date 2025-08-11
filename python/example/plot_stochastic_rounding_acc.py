import numpy as np
import torch
import matplotlib.pyplot as plt
import piquant

torch.manual_seed(42)

def compute_uint4_params(x: torch.Tensor):
    scale, zero_point = piquant.torch.compute_quant_params(x, dtype=torch.quint4x2)
    return scale, zero_point

def quantize_with_mode(x: torch.Tensor, scale: float, zp: int, mode: str):
    qt = piquant.torch
    kwargs_common = dict(scale=scale, zero_point=zp, dtype=torch.quint4x2)
    return qt.quantize(x, **kwargs_common, round_mode=mode)

def dequantize_to_bf16(q: torch.Tensor, scale: float, zp: int) -> torch.Tensor:
    return piquant.torch.dequantize(q, scale=scale, zero_point=zp, dtype=torch.bfloat16)

def cdf_values(x: torch.Tensor):
    s = torch.sort(x.flatten()).values.cpu().numpy()
    y = np.linspace(0.0, 1.0, num=s.size, endpoint=False)
    return s, y


tensor = torch.rand(1000, dtype=torch.bfloat16, device="cpu")
scale, zero_point = compute_uint4_params(tensor)

quant_near = quantize_with_mode(tensor, scale, zero_point, mode="nearest")
dq_near = dequantize_to_bf16(quant_near, scale, zero_point)

quant_sto = quantize_with_mode(tensor, scale, zero_point, mode="stochastic")
dq_sto = dequantize_to_bf16(quant_sto, scale, zero_point)

t32 = tensor.to(torch.float32)
err_near = (dq_near.to(torch.float32) - t32).abs()
err_sto  = (dq_sto.to(torch.float32)  - t32).abs()

mae_near = err_near.mean().item()
mse_near = (err_near ** 2).mean().item()

mae_sto = err_sto.mean().item()
mse_sto = (err_sto ** 2).mean().item()

print(f"scale={scale:.8g}  zero_point={zero_point}")
print(f"Nearest : MAE={mae_near:.6e}  MSE={mse_near:.6e}")
print(f"Stochastic : MAE={mae_sto:.6e}  MSE={mse_sto:.6e}")

step = float(scale)
tol = step/2 + 1e-3
print(f"Sanity tol: {tol:.6g}")
print("Allclose-nearest?", torch.allclose(dq_near, tensor, atol=tol))
print("Allclose-stochastic?", torch.allclose(dq_sto,  tensor, atol=tol))

s_near, y_near = cdf_values(err_near)
s_sto,  y_sto  = cdf_values(err_sto)

plt.figure()
plt.plot(s_near, y_near, label=f"Nearest (MAE={mae_near:.3e}, MSE={mse_near:.3e})")
plt.plot(s_sto,  y_sto,  label=f"Stochastic (MAE={mae_sto:.3e}, MSE={mse_sto:.3e})")
plt.xlabel("Absolute error")
plt.ylabel("CDF")
plt.title("uint4 Quantization: Nearest vs Stochastic (Dequant error CDF)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("quant_error_cdf.png", dpi=160)
plt.show()

print("Original (10):  ", tensor[:10].tolist())
print("Nearest  (10):  ", dq_near[:10].tolist())
print("Stochastic (10):", dq_sto[:10].tolist())