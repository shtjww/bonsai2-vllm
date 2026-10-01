"""Validate conversion of the three untested tensor classes against runtime semantics:
  1. token_embd (inverse path): h = s * (H z)
  2. output.weight (lm_head, forward): y = W_q^T (H (s x))
  3. in_proj_qkv repack: vLLM group-split must recover llama.cpp contiguous q/k/v
Run on server: python debug_validate.py
"""
import sys

sys.path.insert(0, "/root/bonsai2-vllm/src")
import numpy as np

from gguf_reader import GGUFReader
from hadamard import fwht, hadamard_matrix, apply_hadamard_rows
from convert_to_bf16 import load_signs
from prism_dequant import DEQUANT

GGUF = "/root/autodl-tmp/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"

with GGUFReader(GGUF) as r:
    md = r.metadata
    bs = md["prism.hadamard.block_size"]
    signs = load_signs(md)
    H = hadamard_matrix(bs)

    def deq(name):
        t = r.tensors[name]
        return DEQUANT[t.ggml_type](r.tensor_bytes(name), t.n_elements).reshape(t.shape)

    # ---------- 1. embedding inverse (8192列子集，数学等价) ----------
    E = deq("token_embd.weight")[:, :8192]  # (5120, 8192) rotated-basis stored
    s = signs[5120]
    tok = 1000
    z = E[:, tok].astype(np.float32)
    h_runtime = s * fwht(z.reshape(-1, bs)).reshape(-1)          # s * (H z)
    E_conv = apply_hadamard_rows(E, H) * s[:, None]              # our conversion
    h_conv = E_conv[:, tok]
    err = np.abs(h_runtime - h_conv).max() / np.abs(h_runtime).max()
    print(f"1. embedding inverse: rel err = {err:.2e}", "OK" if err < 1e-4 else "FAIL")

    # ---------- 2. lm_head forward (8192列子集) ----------
    W = deq("output.weight")[:, :8192]    # (5120, 8192)
    x = np.random.default_rng(0).normal(size=5120).astype(np.float32)
    y_runtime = W.T @ (fwht((s * x).reshape(-1, bs)).reshape(-1))
    W_conv = apply_hadamard_rows(W, H) * s[:, None]
    y_conv = W_conv.T @ x
    err = np.abs(y_runtime - y_conv).max() / np.abs(y_runtime).max()
    print(f"2. lm_head forward:   rel err = {err:.2e}", "OK" if err < 1e-4 else "FAIL")

    # ---------- 3. qkv repack vs vLLM group split ----------
    Wqkv = deq("blk.0.attn_qkv.weight")   # (5120, 10240) contiguous [q,k,v]
    s5120 = signs[5120]
    y_llama = Wqkv.T.astype(np.float32) @ (fwht((s5120 * x).reshape(-1, bs)).reshape(-1))
    q_ref, k_ref, v_ref = y_llama[:2048], y_llama[2048:4096], y_llama[4096:]

    W_conv = apply_hadamard_rows(Wqkv, H) * s5120[:, None]
    # correct repack: per group [q_g(128), k_g(128), v_g(384)] — v ALSO interleaved
    q = W_conv[:, :2048].reshape(5120, 16, 128)
    k = W_conv[:, 2048:4096].reshape(5120, 16, 128)
    v = W_conv[:, 4096:].reshape(5120, 16, 384)
    W_rep = np.concatenate([q, k, v], 2).reshape(5120, 10240)
    mixed = W_rep.T @ x                                  # vLLM input, 10240
    # vLLM group split: view (16 groups, 640) -> [128(q),128(k),384(v)]
    mixed_g = mixed.reshape(16, 640)
    q_vllm = mixed_g[:, :128].reshape(-1)
    k_vllm = mixed_g[:, 128:256].reshape(-1)
    v_vllm = mixed_g[:, 256:].reshape(-1)
    e_q = np.abs(q_ref - q_vllm).max() / max(np.abs(q_ref).max(), 1e-9)
    e_k = np.abs(k_ref - k_vllm).max() / max(np.abs(k_ref).max(), 1e-9)
    e_v = np.abs(v_ref - v_vllm).max() / max(np.abs(v_ref).max(), 1e-9)
    ok = max(e_q, e_k, e_v) < 1e-4
    print(f"3. qkv repack: q={e_q:.2e} k={e_k:.2e} v={e_v:.2e}", "OK" if ok else "FAIL")
