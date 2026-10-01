"""v3 (tensor-core tl.dot) kernel validation + honest benchmark.

Honest methodology: cycle through MANY distinct layer tensors (working set > L2)
so neither kernel gets a fake L2-resident advantage — approximates end-to-end
layer streaming. Compares v1 (production) vs v3 at T=16/32.

Usage: python kernel_bench_v3.py
"""
import sys
import time

sys.path.insert(0, "/root/bonsai2-vllm")

import numpy as np
import torch

from src.gguf_reader import GGUFReader
from src.prism_dequant import DEQUANT
from vllm_bonsai.ternary_kernel import (
    pack_ptq1, ptq1_gemm_kernel, ptq1_gemm_v3_kernel)

GGUF = "/root/autodl-tmp/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
# distinct layers -> ~8 x 20MB working set, busts L2 (~126MB on this GPU? still streams)
NAMES = [f"blk.{i}.ffn_down.weight" for i in range(0, 16, 2)]

packeds = []
refs = []
with GGUFReader(GGUF) as r:
    for nm in NAMES:
        t = r.tensors[nm]
        out_dim, in_dim = t.shape[1], t.shape[0]
        buf = r.tensor_bytes(nm)
        packeds.append(pack_ptq1(bytes(buf), out_dim, in_dim))
        if nm == NAMES[0]:
            refs.append(DEQUANT[t.ggml_type](bytes(buf), t.n_elements).reshape(
                t.shape[::-1]).astype(np.float32))
out_dim, in_dim = 5120, 17408
print(f"tensors: {len(packeds)} x ffn_down ({out_dim}x{in_dim}), "
      f"working set {sum(p.numel() * 4 for p in packeds) / 1e9:.2f} GB", flush=True)

for T in (16, 32):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(T, in_dim)).astype(np.float32)
    x_t = torch.from_numpy(x).cuda()

    # numerics vs numpy (v3, first tensor)
    y = torch.zeros(T, out_dim, dtype=torch.float32, device="cuda")
    grid = ((out_dim + 63) // 64, 8, (T + 15) // 16)
    ptq1_gemm_v3_kernel[grid](packeds[0], x_t, y, T, in_dim, out_dim,
                              SPLIT_K=8, BLOCK_OUT=64, BLOCK_T=16, num_stages=1)
    torch.cuda.synchronize()
    y_ref = x @ refs[0].T
    err = np.abs(y.cpu().numpy() - y_ref).max() / np.abs(y_ref).max()
    print(f"T={T}: v3 rel_err={err:.2e}", flush=True)

    def run_v1():
        for p in packeds:
            yy = torch.zeros(T, out_dim, dtype=torch.float32, device="cuda")
            g = ((out_dim + 255) // 256, 8, T)
            ptq1_gemm_kernel[g](p, x_t, yy, in_dim, out_dim, SPLIT_K=8, BLOCK_OUT=256)

    def run_v3(BO, SK, BT, NW, NS=1):
        for p in packeds:
            yy = torch.zeros(T, out_dim, dtype=torch.float32, device="cuda")
            g = ((out_dim + BO - 1) // BO, SK, (T + BT - 1) // BT)
            ptq1_gemm_v3_kernel[g](p, x_t, yy, T, in_dim, out_dim,
                                   SPLIT_K=SK, BLOCK_OUT=BO, BLOCK_T=BT,
                                   num_warps=NW, num_stages=NS)

    for label, fn in [("v1 BO=256 SK=8", run_v1),
                      ("v3 BO=64  SK=8 BT=16 NW=4 NS=1", lambda: run_v3(64, 8, 16, 4, 1)),
                      ("v3 BO=64  SK=8 BT=16 NW=8 NS=1", lambda: run_v3(64, 8, 16, 8, 1)),
                      ("v3 BO=128 SK=4 BT=16 NW=8 NS=1", lambda: run_v3(128, 4, 16, 8, 1)),
                      ("v3 BO=128 SK=4 BT=16 NW=4 NS=1", lambda: run_v3(128, 4, 16, 4, 1)),
                      ("v3 BO=128 SK=2 BT=32 NW=8 NS=1", lambda: run_v3(128, 2, 32, 8, 1))]:
        if "BT=32" in label and T < 32:
            continue
        try:
            fn()  # warmup + compile
            torch.cuda.synchronize()
            n_iter = 30
            t0 = time.time()
            for _ in range(n_iter):
                fn()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / n_iter / len(packeds)
            print(f"  T={T} [{label}]: {dt*1e3:.3f} ms/tensor", flush=True)
        except Exception as e:
            print(f"  T={T} [{label}]: FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)

print("V3_BENCH_DONE", flush=True)
