"""Hadamard (normalized Sylvester Walsh) transform utilities.

Matches fork-llama.cpp src/llama-model.cpp rotation generation:
  H[row, col] = (-1)^popcount(row & col) / sqrt(n)

H is symmetric and self-inverse up to normalization: H @ H = I.
Weights in the checkpoint live in the rotated basis; the runtime applies H to
activations blockwise before matmul (llama_mul_mat_hadamard). To convert a
weight back to the standard basis (for vLLM):

  ggml mul_mat convention: y = W^T x, W shape (ne0=in_dim, ne1=out_dim)
  fork computes:          y = W_q^T (H x) = (H W_q)^T x
  => W_standard = H_applied_blockwise_along_input_dim(W_q)

For "inverse" tables (e.g. token embedding): rotation is applied AFTER the
lookup, x = H e_stored  =>  E_standard = H_applied_blockwise_along_row(E).
"""
from __future__ import annotations

import numpy as np


def hadamard_matrix(n: int) -> np.ndarray:
    """Sylvester-order normalized Walsh-Hadamard matrix, float32.

    Signs are NOT folded into H: the runtime applies them as a separate
    full-width diagonal (see convert pipeline).  H[row,col] = (-1)^popcount(row&col)/sqrt(n)
    """
    assert n & (n - 1) == 0, "block size must be a power of two"
    idx = np.arange(n, dtype=np.uint32)
    parity = idx[:, None] & idx[None, :]
    p = parity.astype(np.uint32)
    p ^= p >> 16
    p ^= p >> 8
    p ^= p >> 4
    p ^= p >> 2
    p ^= p >> 1
    return np.where(p & 1, -1.0, 1.0).astype(np.float32) / np.float32(np.sqrt(n))


def apply_hadamard_rows(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Blockwise H along the input dim (axis 0 of the (in, out) matrix): H W_q per block."""
    n = H.shape[0]
    in_dim = W.shape[0]
    assert in_dim % n == 0
    Wr = W.reshape(in_dim // n, n, -1)
    out = np.einsum("ij,bjo->bio", H.astype(np.float32), Wr.astype(np.float32))
    return out.reshape(W.shape)


def unpermute_gdn_rows(W: np.ndarray, n_k: int, rep: int) -> np.ndarray:
    """Undo the runtime feature permutation for gdn_v_grouped ssm_out weights.

    Runtime permutes activation features: tiled [hd, nk, rep] -> grouped [hd, rep, nk]
    BEFORE applying signs+H.  Since W_std = P^T diag(s) H W_q, after applying H and
    signs we must un-permute rows back to the tiled order.
    """
    hd = W.shape[0] // (n_k * rep)
    Wr = W.reshape(hd, rep, n_k, -1)          # rows currently in grouped order
    return Wr.transpose(0, 2, 1, 3).reshape(W.shape)


def fwht(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard along the last axis, normalized (matches runtime FWHT kernels)."""
    n = x.shape[-1]
    assert n & (n - 1) == 0
    y = x.astype(np.float32).copy()
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = np.stack([a + b, a - b], axis=-2).reshape(*x.shape)
        h *= 2
    return (y / np.float32(np.sqrt(n))).reshape(x.shape)
