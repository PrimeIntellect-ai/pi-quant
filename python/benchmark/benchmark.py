import os
import timeit
import multiprocessing

from piquant import QuantDtype

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU usage
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())

import torch
from torch.ao.quantization.fx._decomposed import quantize_per_tensor
import piquant
import matplotlib.pyplot as plt

num_runs = 1000
numel = 27264000  #  Value from realistic test


def quantize_torch_fx(tensor: torch.Tensor, scale: int, zero_point: float) -> None:
    return quantize_per_tensor(
        tensor, scale=scale, zero_point=zero_point, quant_min=0, quant_max=255, dtype=torch.uint8
    )


def quantize_torch_builtin(tensor: torch.Tensor, scale: int, zero_point: float) -> None:
    return torch.quantize_per_tensor(tensor, scale=scale, zero_point=zero_point, dtype=torch.quint8).int_repr()


def quantize_fast(tensor: torch.Tensor, scale: int, zero_point: float) -> None:
    return piquant.quantize_torch(
        tensor, config=piquant.QuantConfig(output_dtype=piquant.QuantDtype.UINT8, scale=scale, zero_point=zero_point)
    )


tensor = torch.rand(numel, device='cpu')
scale, zero_point = piquant.compute_quant_config_torch(tensor, target_quant_dtype=QuantDtype.UINT8)


def benchmark_torch_fx_quant() -> None:
    quantize_torch_fx(tensor, scale, zero_point)


def benchmark_torch_quant() -> None:
    quantize_torch_builtin(tensor, scale, zero_point)


def benchmark_fast_quant() -> None:
    quantize_fast(tensor, scale, zero_point)


time_torch_fx = timeit.timeit(benchmark_torch_fx_quant, number=num_runs)
time_torch = timeit.timeit(benchmark_torch_quant, number=num_runs)
time_fast = timeit.timeit(benchmark_fast_quant, number=num_runs)

print(f'Torch FX quantization time for {num_runs} runs: {time_torch_fx:.6f} seconds')
print(f'Torch quantization time for {num_runs} runs: {time_torch:.6f} seconds')
print(f'Fast quantization time for {num_runs} runs: {time_fast:.6f} seconds')

labels = ['torch.ao.quantization.fx._decomposed.quantize_per_tensor', 'torch.quantize_per_tensor', 'piquant.quantize_torch']
times = [time_torch_fx, time_torch, time_fast]

plt.figure(figsize=(6, 4))
plt.bar(labels, times)
plt.ylabel(f'Time (seconds) for {num_runs} runs')
plt.title('Quantization Benchmark')
plt.savefig('benchmark.png', dpi=300)
plt.show()
