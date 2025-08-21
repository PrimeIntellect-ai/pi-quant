import os, time, multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())

import torch, piquant
torch.set_num_threads(mp.cpu_count())

TOTAL_GIB = 4
ITERATIONS = 10
SRC_DTYPE = torch.bfloat16
Q_DTYPE = torch.quint4x2
OUT_DTYPE = torch.bfloat16

B_PER_ELEM = torch.tensor([], dtype=SRC_DTYPE).element_size()
TOTAL_BYTES = TOTAL_GIB * (1<<30)
N_ELEMS = TOTAL_BYTES // B_PER_ELEM

with torch.inference_mode():
    x = torch.rand(N_ELEMS, dtype=SRC_DTYPE)
    SCALE, ZP = piquant.torch.compute_quant_params(x, dtype=Q_DTYPE)

with torch.inference_mode():
    q = piquant.torch.quantize(x, scale=SCALE, zero_point=int(ZP), dtype=Q_DTYPE)
    _ = piquant.torch.dequantize(q, scale=SCALE, zero_point=int(ZP), dtype=OUT_DTYPE)

quant_results, dequant_results = [], []
uint4_source_gib = (N_ELEMS // 2) / (1<<30)

for i in range(ITERATIONS):
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = piquant.torch.quantize(x, scale=SCALE, zero_point=int(ZP), dtype=Q_DTYPE)
    t1 = time.perf_counter()
    quant_results.append(TOTAL_GIB / (t1 - t0))

    with torch.inference_mode():
        q = piquant.torch.quantize(x, scale=SCALE, zero_point=int(ZP), dtype=Q_DTYPE)
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = piquant.torch.dequantize(q, scale=SCALE, zero_point=int(ZP), dtype=OUT_DTYPE)
    t1 = time.perf_counter()
    dequant_results.append(uint4_source_gib / (t1 - t0))

print(f"Quant bf16->uint4  avg throughput: {sum(quant_results)/ITERATIONS:.2f} GiB/s")
print(f"Dequant uint4->bf16 avg throughput: {sum(dequant_results)/ITERATIONS:.2f} GiB/s")