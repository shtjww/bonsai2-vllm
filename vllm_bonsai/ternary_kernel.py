"""Triton PTQ1_0 ternary GEMV/GEMM (validated vs numpy: rel err ~5e-7).

y[t, o] = sum_j w[o, j] * x[t, j]   with w dequantized in-kernel.

Packed layout (preprocessed at load): uint32 P[n_blk][7][n_out]
  (word-major, out contiguous -> fully coalesced per-word loads)
PTQ1_0 block = 128 weights / 28 bytes = 7 uint32 words:
  words 0..5 = qs[24], word 6 = qh[2] + fp16 scale
Trit extraction (fork C semantics, uint8 wraparound):
  trit(byte, n) = (((byte * 3^n) & 0xFF) * 3 >> 8) - 1
  stage c=16: byte m (word m//4)      -> pos n*16+m,   n=0..4
  stage c=8 : byte 16+m (word 4+m//4) -> pos 80+n*8+m, n=0..4   (m = 4wi+j!)
  qh byte h (word 6, j=h)             -> pos 120+n*2+h, n=0..3
  scale = fp16 bits (word6 >> 16)

Split-K via atomic fp32 add (y zeroed before launch).
"""
import math
import os

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def ptq1_gemm_kernel(
    P,               # uint32 (n_blk, 7, n_out) — coalesced layout
    x_ptr,           # (T, n_in) fp32
    y_ptr,           # (T, n_out) fp32 (zeroed)
    n_in: tl.constexpr,
    n_out: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_k = tl.program_id(1)
    t_idx = tl.program_id(2)
    outs = pid_o * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    om = outs < n_out
    n_blk = n_in // 128
    bps = (n_blk + SPLIT_K - 1) // SPLIT_K
    b_lo = pid_k * bps
    b_hi = tl.minimum(b_lo + bps, n_blk)

    xr = x_ptr + t_idx * n_in
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    for b in range(b_lo, b_hi):
        row = P + (b * 7 * n_out) + outs
        w0 = tl.load(row, mask=om, other=0)
        w1 = tl.load(row + n_out, mask=om, other=0)
        w2 = tl.load(row + 2 * n_out, mask=om, other=0)
        w3 = tl.load(row + 3 * n_out, mask=om, other=0)
        w4 = tl.load(row + 4 * n_out, mask=om, other=0)
        w5 = tl.load(row + 5 * n_out, mask=om, other=0)
        w6 = tl.load(row + 6 * n_out, mask=om, other=0)

        part = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
        # stage c=16: word wi (0..3), byte j -> m=4wi+j, pos n*16+m
        ws16 = [w0, w1, w2, w3]
        for n in tl.static_range(5):
            pw = 3 ** n
            for wi in tl.static_range(4):
                wv = ws16[wi]
                for j in tl.static_range(4):
                    byte = (wv >> (8 * j)) & 0xFF
                    t = (((byte * pw) & 0xFF) * 3) >> 8
                    tf = t.to(tl.float32) - 1.0
                    xv = tl.load(xr + b * 128 + n * 16 + 4 * wi + j)
                    part += tf * xv
        # stage c=8: word wi (4..5), byte j -> m=4wi+j, pos 80+n*8+m
        ws8 = [w4, w5]
        for n in tl.static_range(5):
            pw = 3 ** n
            for wi in tl.static_range(2):
                wv = ws8[wi]
                for j in tl.static_range(4):
                    byte = (wv >> (8 * j)) & 0xFF
                    t = (((byte * pw) & 0xFF) * 3) >> 8
                    tf = t.to(tl.float32) - 1.0
                    xv = tl.load(xr + b * 128 + 80 + n * 8 + 4 * wi + j)
                    part += tf * xv
        # qh: word 6, byte h -> pos 120+n*2+h
        for n in tl.static_range(4):
            pw = 3 ** n
            for h in tl.static_range(2):
                byte = (w6 >> (8 * h)) & 0xFF
                t = (((byte * pw) & 0xFF) * 3) >> 8
                tf = t.to(tl.float32) - 1.0
                xv = tl.load(xr + b * 128 + 120 + n * 2 + h)
                part += tf * xv

        sc = ((w6 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        acc += part * sc.to(tl.float32)

    tl.atomic_add(y_ptr + t_idx * n_out + outs, acc, mask=om)


@triton.jit
def ptq1_gemm_v2_kernel(
    P,               # uint32 (n_blk, 7, n_out) — coalesced layout
    XT,              # (n_in, T) fp32 — x transposed, contiguous per position
    y_ptr,           # (T, n_out) fp32 (zeroed)
    T,
    n_in: tl.constexpr,
    n_out: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Batched-over-T variant: each weight block is loaded ONCE and applied to
    BLOCK_T tokens -> weight traffic divided by BLOCK_T vs the GEMV kernel."""
    pid_o = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_t = tl.program_id(2)
    outs = pid_o * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    om = outs < n_out
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tm = t_offs < T
    n_blk = n_in // 128
    bps = (n_blk + SPLIT_K - 1) // SPLIT_K
    b_lo = pid_k * bps
    b_hi = tl.minimum(b_lo + bps, n_blk)

    acc = tl.zeros((BLOCK_T, BLOCK_OUT), dtype=tl.float32)
    for b in range(b_lo, b_hi):
        row = P + (b * 7 * n_out) + outs
        w0 = tl.load(row, mask=om, other=0)
        w1 = tl.load(row + n_out, mask=om, other=0)
        w2 = tl.load(row + 2 * n_out, mask=om, other=0)
        w3 = tl.load(row + 3 * n_out, mask=om, other=0)
        w4 = tl.load(row + 4 * n_out, mask=om, other=0)
        w5 = tl.load(row + 5 * n_out, mask=om, other=0)
        w6 = tl.load(row + 6 * n_out, mask=om, other=0)

        part = tl.zeros((BLOCK_T, BLOCK_OUT), dtype=tl.float32)
        xb = XT + b * 128 * T + t_offs
        # stage c=16: word wi (0..3), byte j -> m=4wi+j, pos n*16+m
        ws16 = [w0, w1, w2, w3]
        for n in tl.static_range(5):
            pw = 3 ** n
            for wi in tl.static_range(4):
                wv = ws16[wi]
                for j in tl.static_range(4):
                    byte = (wv >> (8 * j)) & 0xFF
                    t = (((byte * pw) & 0xFF) * 3) >> 8
                    tf = t.to(tl.float32) - 1.0
                    xv = tl.load(xb + (n * 16 + 4 * wi + j) * T, mask=tm, other=0.0)
                    part += xv[:, None] * tf[None, :]
        # stage c=8: word wi (4..5), byte j -> m=4wi+j, pos 80+n*8+m
        ws8 = [w4, w5]
        for n in tl.static_range(5):
            pw = 3 ** n
            for wi in tl.static_range(2):
                wv = ws8[wi]
                for j in tl.static_range(4):
                    byte = (wv >> (8 * j)) & 0xFF
                    t = (((byte * pw) & 0xFF) * 3) >> 8
                    tf = t.to(tl.float32) - 1.0
                    xv = tl.load(xb + (80 + n * 8 + 4 * wi + j) * T, mask=tm, other=0.0)
                    part += xv[:, None] * tf[None, :]
        # qh: word 6, byte h -> pos 120+n*2+h
        for n in tl.static_range(4):
            pw = 3 ** n
            for h in tl.static_range(2):
                byte = (w6 >> (8 * h)) & 0xFF
                t = (((byte * pw) & 0xFF) * 3) >> 8
                tf = t.to(tl.float32) - 1.0
                xv = tl.load(xb + (120 + n * 2 + h) * T, mask=tm, other=0.0)
                part += xv[:, None] * tf[None, :]

        sc = ((w6 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        acc += part * sc.to(tl.float32)[None, :]

    tl.atomic_add(y_ptr + t_offs[:, None] * n_out + outs[None, :], acc,
                  mask=tm[:, None] & om[None, :])


@triton.jit
def ptq1_gemm_v3_kernel(
    P,               # uint32 (n_blk, 7, n_out)
    X,               # (T, n_in) fp32
    y_ptr,           # (T, n_out) fp32 (zeroed)
    T,
    n_in: tl.constexpr,
    n_out: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Tensor-core (tl.dot) variant: decode a full 128-weight block into a
    (128, BLOCK_OUT) fp16 tile with vectorized index math, then MMA against a
    (BLOCK_T, 128) x tile. Decode runs on CUDA cores, MMA on tensor cores."""
    pid_o = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_t = tl.program_id(2)
    outs = pid_o * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    om = outs < n_out
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tm = t_offs < T
    n_blk = n_in // 128
    bps = (n_blk + SPLIT_K - 1) // SPLIT_K
    b_lo = pid_k * bps
    b_hi = tl.minimum(b_lo + bps, n_blk)

    # ---- per-position unpack indices (128 positions within a PTQ1_0 block) ----
    pos = tl.arange(0, 128)
    # stage c=16: pos 0..79   -> word m//4 (m=pos%16), byte m%4, n=pos//16
    # stage c=8 : pos 80..119 -> word 4+m//4 (m=(pos-80)%8), byte m%4, n=(pos-80)//8
    # qh        : pos 120..127-> word 6, byte h=(pos-120)%2, n=(pos-120)//2
    p16 = pos % 16
    p8 = (pos - 80) % 8
    word_idx = tl.where(pos < 80, p16 // 4,
                tl.where(pos < 120, 4 + p8 // 4, 6))
    byte_idx = tl.where(pos < 80, p16 % 4,
                tl.where(pos < 120, p8 % 4, (pos - 120) % 2))
    n_idx = tl.where(pos < 80, pos // 16,
             tl.where(pos < 120, (pos - 80) // 8, (pos - 120) // 2))
    pw = tl.where(n_idx == 0, 1,
         tl.where(n_idx == 1, 3,
         tl.where(n_idx == 2, 9,
         tl.where(n_idx == 3, 27, 81)))).to(tl.int32)
    byte_shift = (byte_idx * 8)[:, None]
    pw_col = pw[:, None]

    acc = tl.zeros((BLOCK_T, BLOCK_OUT), dtype=tl.float32)
    for b in range(b_lo, b_hi):
        # gather the right word of the block for each of 128 positions: (128, BLOCK_OUT)
        wwords = tl.load(P + b * 7 * n_out + word_idx[:, None] * n_out + outs[None, :],
                         mask=om[None, :], other=0)
        bytes_ = (wwords >> byte_shift) & 0xFF
        tt = (((bytes_ * pw_col) & 0xFF) * 3) >> 8
        w_tile = tt.to(tl.float16) - 1.0                     # (128, BLOCK_OUT)
        x_tile = tl.load(X + t_offs[:, None] * n_in + b * 128 + pos[None, :],
                         mask=tm[:, None], other=0.0)         # (BLOCK_T, 128)
        part = tl.dot(x_tile.to(tl.float16), w_tile, out_dtype=tl.float32)
        w6 = tl.load(P + (b * 7 + 6) * n_out + outs, mask=om, other=0)
        sc = ((w6 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        acc += part * sc.to(tl.float32)[None, :]

    tl.atomic_add(y_ptr + t_offs[:, None] * n_out + outs[None, :], acc,
                  mask=tm[:, None] & om[None, :])


def pack_ptq1(buf, out_dim: int, in_dim: int) -> torch.Tensor:
    """(out, n_blk, 28) uint8 -> uint32 cuda (n_blk, 7, out) coalesced."""
    p = np.frombuffer(buf, dtype=np.uint8).reshape(out_dim, in_dim // 128, 28)
    p32 = np.ascontiguousarray(p).view(np.uint32).reshape(out_dim, -1, 7)
    p32 = np.ascontiguousarray(p32.transpose(1, 2, 0))
    return torch.from_numpy(p32).cuda()


def ternary_matmul(packed: torch.Tensor, x: torch.Tensor,
                   split_k: int | None = None, block_out: int | None = None,
                   block_t: int | None = None) -> torch.Tensor:
    """packed (n_blk,7,n_out) uint32 cuda; x (T, n_in) fp32 -> y (T, n_out) fp32.
    Block sizes tuned on RTX PRO 6000 (see PROGRESS.md / BENCH_20261001.md):
    big tensors prefer wide blocks, small ones need more split-K parallelism.
    T >= 2 dispatches to the batched v2 kernel (weight traffic / BLOCK_T)."""
    T, n_in = x.shape
    n_blk, _, n_out = packed.shape
    y = torch.zeros(T, n_out, dtype=torch.float32, device=x.device)
    if T >= 16 and os.environ.get("BONSAI_KERNEL_V3", "1") == "1":
        # v3 tensor-core kernel: default for T>=16 since 2026-10-01.
        # Measured (8-layer L2-busting sweep, ffn_down 5120x17408):
        #   T=16: 0.043 ms/tensor vs v1 0.209 (4.9x); T=32: 0.074 vs 0.591 (8.0x)
        # rel_err ~2e-4 (x cast to fp16 for MMA) — serving-grade.
        # Winning config: BT=16 BO=64 SK=8 NW=4 NS=1. Set BONSAI_KERNEL_V3=0 to revert.
        if block_t is None:
            block_t = 16
        if block_out is None:
            block_out, split_k = 64, 8
        grid = ((n_out + block_out - 1) // block_out, split_k,
                (T + block_t - 1) // block_t)
        ptq1_gemm_v3_kernel[grid](packed, x, y, T, n_in, n_out,
                                  SPLIT_K=split_k, BLOCK_OUT=block_out,
                                  BLOCK_T=block_t, num_stages=1)
        return y
    # v2 batched kernel: measured 2026-10-01 — wins in L2-resident micro-bench
    # (0.27 vs 0.39 ms @T=32) but LOSES end-to-end (126 vs 226 tok/s @bs=16).
    # v1's GEMV-per-token gets near-DRAM-peak effective bandwidth via L2 reuse
    # across concurrent token programs; v2 is compute-bound at ~250GB/s effective.
    # Kept for future work (needs tl.dot/tensor-core path to beat fork's GEMM).
    # Opt-in only: BONSAI_KERNEL_V2=1
    if T >= 12 and os.environ.get("BONSAI_KERNEL_V2") == "1":
        # v2 batched kernel (measured on RTX PRO 6000, BENCH log 2026-10-01):
        # T>=24: BT=16 wins (0.27ms vs v1 0.39ms on 5120x17408);
        # 12<=T<24: BT=8 wins; below that v1's L2 residency beats batching.
        if block_t is None:
            block_t = 16 if T >= 24 else 8
        if block_out is None:
            block_out, split_k = 256, 8
        xt = x.t().contiguous()          # (n_in, T)
        grid = ((n_out + block_out - 1) // block_out, split_k,
                (T + block_t - 1) // block_t)
        ptq1_gemm_v2_kernel[grid](packed, xt, y, T, n_in, n_out,
                                  SPLIT_K=split_k, BLOCK_OUT=block_out,
                                  BLOCK_T=block_t)
        return y
    if block_out is None:
        if n_out * n_in >= 5e8:      # lm_head class (248320x5120)
            block_out, split_k = 256, 4
        elif n_out >= 16384:         # ffn_gate/up, ssm_out class
            block_out, split_k = 256, 8
        else:                        # qkv/o_proj class
            block_out, split_k = 128, 16
    grid = ((n_out + block_out - 1) // block_out, split_k, T)
    ptq1_gemm_kernel[grid](packed, x, y, n_in, n_out,
                           SPLIT_K=split_k, BLOCK_OUT=block_out)
    return y


# ---- torch custom op: opaque to dynamo -> vLLM compile/cudagraph compatible ----

import math

_H_CACHE: dict[tuple, torch.Tensor] = {}


def _get_H(bs: int, device) -> torch.Tensor:
    key = (bs, str(device))
    if key not in _H_CACHE:
        import numpy as np
        idx = np.arange(bs, dtype=np.uint32)
        p = idx[:, None] & idx[None, :]
        p ^= p >> 16
        p ^= p >> 8
        p ^= p >> 4
        p ^= p >> 2
        p ^= p >> 1
        H = (1.0 - 2.0 * (p & 1).astype(np.float32)) / math.sqrt(bs)
        _H_CACHE[key] = torch.from_numpy(H).to(device)
    return _H_CACHE[key]


@torch.library.custom_op("bonsai::ternary_hadamard_linear", mutates_args=())
def _ternary_hadamard_linear(packed: torch.Tensor, x: torch.Tensor,
                             signs: torch.Tensor, block_size: int) -> torch.Tensor:
    """y = W_q^T (H (signs * x)); packed PTQ1_0 weights, in-kernel dequant."""
    T = x.shape[0]
    x2 = x.reshape(T, -1).float()
    bs = block_size
    H = _get_H(bs, x.device)
    if H.device != x.device:
        H = H.to(x.device)
        _H_CACHE[(bs, str(x.device))] = H
    if not hasattr(_ternary_hadamard_linear, "_dbg"):
        print(f"[ternary-op] x={x.device} signs={signs.device} H={H.device} "
              f"packed={packed.device}", flush=True)
        _ternary_hadamard_linear._dbg = True
    xh = (x2 * signs).reshape(T, -1, bs) @ H
    return ternary_matmul(packed, xh.reshape(T, -1))


@_ternary_hadamard_linear.register_fake
def _(packed, x, signs, block_size):
    return x.new_empty(x.shape[0], packed.shape[2], dtype=torch.float32)
