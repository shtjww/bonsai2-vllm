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


def pack_ptq1(buf, out_dim: int, in_dim: int) -> torch.Tensor:
    """(out, n_blk, 28) uint8 -> uint32 cuda (n_blk, 7, out) coalesced."""
    p = np.frombuffer(buf, dtype=np.uint8).reshape(out_dim, in_dim // 128, 28)
    p32 = np.ascontiguousarray(p).view(np.uint32).reshape(out_dim, -1, 7)
    p32 = np.ascontiguousarray(p32.transpose(1, 2, 0))
    return torch.from_numpy(p32).cuda()


def ternary_matmul(packed: torch.Tensor, x: torch.Tensor,
                   split_k: int | None = None, block_out: int | None = None) -> torch.Tensor:
    """packed (n_blk,7,n_out) uint32 cuda; x (T, n_in) fp32 -> y (T, n_out) fp32.
    Block sizes tuned on RTX PRO 6000 (see PROGRESS.md): big tensors prefer
    wide blocks, small ones need more split-K parallelism."""
    T, n_in = x.shape
    n_blk, _, n_out = packed.shape
    if block_out is None:
        if n_out * n_in >= 5e8:      # lm_head class (248320x5120)
            block_out, split_k = 256, 4
        elif n_out >= 16384:         # ffn_gate/up, ssm_out class
            block_out, split_k = 256, 8
        else:                        # qkv/o_proj class
            block_out, split_k = 128, 16
    y = torch.zeros(T, n_out, dtype=torch.float32, device=x.device)
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
