import os, time, multiprocessing as mp
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(mp.cpu_count())

import torch, piquant
torch.set_num_threads(mp.cpu_count())

TOTAL_GIB = 32
SRC_DTYPE = torch.bfloat16
Q_DTYPE = torch.quint4x2
OUT_DTYPE = torch.bfloat16

B_PER_ELEM = torch.tensor([], dtype=SRC_DTYPE).element_size()
TOTAL_BYTES = TOTAL_GIB * (1 << 30)
N_ELEMS = TOTAL_BYTES // B_PER_ELEM
print(f'Numel = {N_ELEMS}')

with torch.inference_mode():
    x = torch.rand(N_ELEMS, dtype=SRC_DTYPE)
    SCALE, ZP = piquant.torch.compute_quant_params(x, dtype=Q_DTYPE)

t0 = time.perf_counter()
with torch.inference_mode():
    q = piquant.torch.quantize(x, scale=SCALE, zero_point=int(ZP), dtype=Q_DTYPE)
t1 = time.perf_counter()
print(f"Quant time (bf16 -> uint4) for {TOTAL_GIB} GiB: {t1 - t0:.3f} s")

t0 = time.perf_counter()
with torch.inference_mode():
    y = piquant.torch.dequantize(q, scale=SCALE, zero_point=int(ZP), dtype=OUT_DTYPE)
t1 = time.perf_counter()
print(f"Dequant time (uint4 -> bf16) for {TOTAL_GIB} GiB source: {t1 - t0:.3f} s")