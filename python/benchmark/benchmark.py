import os
import timeit
import multiprocessing

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())  # OpenMP
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())  # MKL

import piquant
import torch
import numpy as np
import matplotlib.pyplot as plt

NUM_RUNS: int = 1_000
NUMEL: int = 100_000_000

QUANT_DTYPES_TO_BENCH: list[torch.dtype] = [
    torch.quint8,
    torch.quint4x2,
]


def quantize_torch(t: torch.Tensor, scale: float, zp: int, dtype: torch.dtype) -> None:
    torch.quantize_per_tensor(t, scale=scale, zero_point=zp, dtype=dtype).int_repr()


def quantize_piquant(t: torch.Tensor, scale: float, zp: int, dtype: torch.dtype) -> None:
    piquant.torch.quantize(t,  scale=scale, zero_point=zp, dtype=dtype)


dtype_labels: list[str] = []
torch_times: list[float] = []
piquant_times: list[float] = []

tensor = torch.rand(NUMEL, dtype=torch.float32, device='cpu')

for torch_d in QUANT_DTYPES_TO_BENCH:
    scale, zp = piquant.torch.compute_quant_params(tensor, dtype=torch_d)
    zp = int(zp)

    def _bench_torch() -> None:
        quantize_torch(tensor, scale, zp, torch_d)

    def _bench_piquant() -> None:
        quantize_piquant(tensor, scale, zp, torch_d)

    # Warmup runs
    _bench_torch()
    _bench_piquant()

    torch_time = timeit.timeit(_bench_torch, number=NUM_RUNS)
    piquant_time = timeit.timeit(_bench_piquant, number=NUM_RUNS)
    dtype_labels.append(str(torch_d).replace('torch.', ''))
    torch_times.append(torch_time)
    piquant_times.append(piquant_time)
    print(f'{dtype_labels[-1]:<10} | torch: {torch_time:.6f}s | piquant: {piquant_time:.6f}s')

x = np.arange(len(dtype_labels))
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, torch_times, width, label='torch')
plt.bar(x + width / 2, piquant_times, width, label='piquant')
plt.ylabel(f'Total time for {NUM_RUNS} runs (s)')
plt.xticks(x, dtype_labels)
plt.title('Quantization Benchmark: PyTorch vs. piquant')
plt.legend()
plt.tight_layout()
plt.savefig('quant_benchmark.png', dpi=300)
plt.show()
