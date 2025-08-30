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

torch.set_num_threads(multiprocessing.cpu_count())

NUM_RUNS: int = 1_000
NUMEL: int = 1000000

QUANT_DTYPES_TO_BENCH: list[torch.dtype] = [
    torch.quint8,
    torch.quint4x2,
    torch.quint2x4
]

def quantize_torch(t: torch.Tensor, scale: float, zp: int, dtype: torch.dtype) -> torch.tensor:
    return torch.quantize_per_tensor(t, scale=scale, zero_point=zp, dtype=dtype)


def quantize_piquant(t: torch.Tensor, scale: float, zp: int, dtype: torch.dtype) -> torch.tensor:
    return piquant.torch.quantize(t,  scale=scale, zero_point=zp, dtype=dtype)


dtype_labels: list[str] = []
torch_times: list[float] = []
piquant_times: list[float] = []

for torch_d in QUANT_DTYPES_TO_BENCH:
    tensor = torch.rand(NUMEL, dtype=torch.float32, device='cpu')
    torch_results = []
    results_piquant = []

    scale, zp = piquant.torch.compute_quant_params(tensor, dtype=torch_d)
    zp = int(zp)

    def _bench_torch() -> None:
        torch_results.append(quantize_torch(tensor, scale, zp, torch_d))

    def _bench_piquant() -> None:
        results_piquant.append(quantize_piquant(tensor, scale, zp, torch_d))

    # Warmup runs
    _bench_torch()
    _bench_piquant()

    torch_time = timeit.timeit(_bench_torch, number=NUM_RUNS)
    piquant_time = timeit.timeit(_bench_piquant, number=NUM_RUNS)
    dtype_labels.append(str(torch_d).replace('torch.', ''))
    torch_times.append(torch_time)
    piquant_times.append(piquant_time)

    # Verify that the results are the same
    for i in range(NUM_RUNS): # We compare dequantized results, because .int_repr() is implemented for packed types in torch
        dq_torch = torch_results[i].dequantize()
        dq_piquant = piquant.torch.dequantize(results_piquant[i], scale=scale, zero_point=zp, dtype=torch.float32)
        assert dq_torch.numel() == dq_piquant.numel()
        assert dq_torch.dtype == dq_piquant.dtype
        if not torch.allclose(dq_torch, dq_piquant, atol=1e-1):
            print(f"Results differ for dtype {torch_d} at run {i}")
            for j in range(dq_torch.numel()):
                if not torch.isclose(dq_torch[j], dq_piquant[j], atol=1e-1):
                    print(f"  Index {j}: torch={dq_torch[j]}, piquant={dq_piquant[j]}")
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
