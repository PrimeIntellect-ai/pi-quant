import os, time, multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())

import torch, piquant
torch.set_num_threads(mp.cpu_count())

TOTAL_GIB = 32
ITERATIONS = 10

def measure_throughput(dq_type: torch.dtype, q_type: torch.dtype):
    B_PER_ELEM = torch.tensor([], dtype=dq_type).element_size()
    TOTAL_BYTES = TOTAL_GIB * (1<<30)
    N_ELEMS = TOTAL_BYTES // B_PER_ELEM
    with torch.inference_mode():
        x = torch.rand(N_ELEMS, dtype=dq_type)
        SCALE, ZP = piquant.torch.compute_quant_params(x, dtype=q_type)
    with torch.inference_mode():
        q = piquant.torch.quantize(x, scale=SCALE, zero_point=int(ZP), dtype=q_type)
        _ = piquant.torch.dequantize(q, scale=SCALE, zero_point=int(ZP), dtype=dq_type)
    quant_results, dequant_results = [], []
    uint4_source_gib = (N_ELEMS // 2) / (1<<30)
    for i in range(ITERATIONS):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = piquant.torch.quantize(x, scale=SCALE, zero_point=int(ZP), dtype=q_type)
        t1 = time.perf_counter()
        quant_results.append(TOTAL_GIB / (t1 - t0))
        with torch.inference_mode():
            q = piquant.torch.quantize(x, scale=SCALE, zero_point=int(ZP), dtype=q_type)
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = piquant.torch.dequantize(q, scale=SCALE, zero_point=int(ZP), dtype=dq_type)
        t1 = time.perf_counter()
        dequant_results.append(uint4_source_gib / (t1 - t0))

    print(f'Quant {dq_type} -> {q_type} avg throughput: {sum(quant_results)/ITERATIONS:.2f} GiB/s')
    print(f'Dequant {q_type} -> {dq_type} avg throughput: {sum(dequant_results)/ITERATIONS:.2f} GiB/s')

measure_throughput(torch.bfloat16, torch.quint4x2)
measure_throughput(torch.bfloat16, torch.quint2x4)
