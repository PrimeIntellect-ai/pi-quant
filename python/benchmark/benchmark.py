import os
import timeit

os.environ["CUDA_VISIBLE_DEVICES"]="" # Force CPU usage

import torch
import quant
import matplotlib.pyplot as plt

num_runs = 10000
numel = 27264000 #  Value from realistic test

def quantize_torch_builtin(tensor, scale, zero_point):
    return torch.quantize_per_tensor(tensor, scale=scale, zero_point=zero_point, dtype=torch.quint8).int_repr()

def quantize_fast(tensor, scale, zero_point):
    return quant.quant_torch(tensor, config=quant.QuantConfig(output_dtype=quant.QuantDtype.UINT8, scale=scale, zero_point=zero_point))

tensor = torch.rand(numel, device='cpu')
scale, zero_point = quant.compute_config_properties_from_data_torch(tensor)

def benchmark_torch_quant():
    quantize_torch_builtin(tensor, scale, zero_point)

def benchmark_fast_quant():
    quantize_fast(tensor, scale, zero_point)

time_torch = timeit.timeit(benchmark_torch_quant, number=num_runs)
time_fast = timeit.timeit(benchmark_fast_quant, number=num_runs)

print(f"Torch quantization time for {num_runs} runs: {time_torch:.6f} seconds")
print(f"Fast quantization time for {num_runs} runs: {time_fast:.6f} seconds")

labels = ['Torch Quant', 'Fast Quant']
times = [time_torch, time_fast]

plt.figure(figsize=(6, 4))
plt.bar(labels, times)
plt.ylabel(f'Time (seconds) for {num_runs} runs')
plt.title('Quantization Benchmark')
plt.show()
