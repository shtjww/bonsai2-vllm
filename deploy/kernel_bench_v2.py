"""Batched (T>1) bandwidth/accuracy sweep for the v2 kernel.
Usage: python kernel_bench_v2.py [tensor_name] [T_list]
Validates numerics vs numpy dequant reference, then sweeps BLOCK_T/BLOCK_OUT/SPLIT_K.
"""
import sys
import time

sys.path.insert(0, "/root/bonsai2-vllm")

import numpy as np
import torch

from src.gguf_reader import GGUFReader
from src.prism_dequant import DEQUANT
from vllm_bonsai.ternary_kernel import pack_ptq1, ternary_matmul, ptq1_gemm_kernel

GGUF = "/root/autodl-tmp/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
NAME = sys.argv[1] if len(sys.argv) > 1 else "blk.0.ffn_down.weight"
Ts = [int(t) for t in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["2", "8", "16", "32"])]

with GGUFReader(GGUF) as r:
    t = r.tensors[NAME]
    out_dim, in_dim = t.shape[1], t.shape[0]
    print(f"{NAME}: torch ({out_dim}, {in_dim})", flush=True)
    buf = r.tensor_bytes(NAME)
    ref = DEQUANT[t.ggml_type](bytes(buf), t.n_elements).reshape(
        t.shape[::-1]).astype(np.float32)
    packed = pack_ptq1(bytes(buf), out_dim, in_dim)

rng = np.random.default_rng(0)
Tmax = max(Ts)
x = rng.normal(size=(Tmax, in_dim)).astype(np.float32)
y_ref_all = x @ ref.T  # (T, out)
x_t = torch.from_numpy(x).cuda()

gb = packed.numel() * 4 / 1e9


def v1_run(xr, T, BO, SK):
    y1 = torch.zeros(T, out_dim, dtype=torch.float32, device="cuda")
    grid = ((out_dim + BO - 1) // BO, SK, T)
    ptq1_gemm_kernel[grid](packed, xr, y1, in_dim, out_dim, SPLIT_K=SK, BLOCK_OUT=BO)
    return y1


for T in Ts:
    xr = x_t[:T].contiguous()
    yr = y_ref_all[:T]
    # numerics (default dispatch config)
    y = ternary_matmul(packed, xr)
    torch.cuda.synchronize()
    err = np.abs(y.cpu().numpy() - yr).max() / max(np.abs(yr).max(), 1e-9)
    print(f"T={T}: default-config rel_err={err:.2e}", flush=True)
    # v1 baseline (current production kernel, GEMV x T)
    for BO, SK in ((128, 16), (256, 8)):
        v1_run(xr, T, BO, SK)  # warmup/compile
        torch.cuda.synchronize()
        n_iter = 100
        t0 = time.time()
        for _ in range(n_iter):
            y1 = v1_run(xr, T, BO, SK)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n_iter
        print(f"  [v1] T={T} BO={BO} SK={SK}: {dt*1e3:.2f} ms, "
              f"traffic={gb*T:.2f}GB -> {gb*T/dt:.0f} GB/s", flush=True)
    # v2 sweep (warmup each config before timing)
    for BT in (4, 8, 16, 32):
        if BT > T:
            continue
        for BO, SK in ((64, 8), (128, 4), (128, 8), (256, 4), (256, 8)):
            try:
                ternary_matmul(packed, xr, split_k=SK, block_out=BO, block_t=BT)
                torch.cuda.synchronize()
                n_iter = 100
                t0 = time.time()
                for _ in range(n_iter):
                    y = ternary_matmul(packed, xr, split_k=SK, block_out=BO, block_t=BT)
                torch.cuda.synchronize()
                dt = (time.time() - t0) / n_iter
                eff_gb = gb * ((T + BT - 1) // BT)
                print(f"  [v2] T={T} BT={BT} BO={BO} SK={SK}: {dt*1e3:.2f} ms, "
                      f"traffic={eff_gb:.2f}GB -> {eff_gb/dt:.0f} GB/s", flush=True)
            except Exception as e:
                print(f"  [v2] T={T} BT={BT} BO={BO} SK={SK}: FAIL {type(e).__name__}", flush=True)
print("V2_BENCH_DONE", flush=True)
