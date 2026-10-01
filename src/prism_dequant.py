"""Vectorized dequantizers for PrismML's PQ2_0 (142) and PTQ1_0 (143).

Ported line-by-line from the PrismML llama.cpp fork:
  ggml/src/ggml-quants.c :: dequantize_row_pq2_0 / dequantize_row_ptq1_0
  ggml/src/ggml-common.h :: block_pq2_0 / block_ptq1_0

Block layouts (group = 128 weights):
  PQ2_0  : d(fp16) + qs[32]   (2 bits/weight, code in {0,1,2} -> (code-1)*d)  34 B
  PTQ1_0 : qs[24] + qh[2] + d(fp16)  (base-3 trit packing, 5 trits/byte)      28 B

PTQ1_0 trit packing (from quantize_row_ptq1_0_ref):
  qs is consumed in stages [32, 16, 8]; with 24 bytes only stages 16 and 8 run:
    bytes  0..15 -> stage c=16: byte m holds trits for positions n*16+m, n=0..4  (pos 0..79)
    bytes 16..23 -> stage c=8 : byte m holds trits for positions 80 + n*8+m      (pos 80..119)
  qh (2 bytes): byte h holds trits for positions 120 + n*2 + h, n=0..3           (pos 120..127)
  Trit extraction from packed byte q for stage index n:  ((q * 3**n * 3) >> 8) - 1
"""
from __future__ import annotations

import numpy as np

POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint32)

QK = 128
PTQ1_0_BLOCK_BYTES = 28
PQ2_0_BLOCK_BYTES = 34

# uint32 view of one fp16 as stored (little-endian) -> float32
def _fp16_to_f32(u16: np.ndarray) -> np.ndarray:
    return u16.astype(np.uint16).view(np.float16).astype(np.float32)


def dequant_pq2_0(buf: memoryview | bytes, n_elements: int) -> np.ndarray:
    """PQ2_0 raw block bytes -> float32 weights (ggml dim order)."""
    assert n_elements % QK == 0
    nb = n_elements // QK
    raw = np.frombuffer(buf, dtype=np.uint8, count=nb * PQ2_0_BLOCK_BYTES)
    raw = raw.reshape(nb, PQ2_0_BLOCK_BYTES)

    d = _fp16_to_f32(raw[:, 0:2].copy().view(np.uint16).reshape(nb))      # (nb,)
    qs = raw[:, 2:34]                                                       # (nb, 32) uint8

    # 2-bit codes: byte j, element j*4+e at bit offset e*2
    codes = np.stack([(qs >> (2 * e)) & 0x3 for e in range(4)], axis=-1)    # (nb, 32, 4)
    w = (codes.astype(np.float32) - 1.0) * d[:, None, None]
    return w.reshape(n_elements)


def dequant_ptq1_0(buf: memoryview | bytes, n_elements: int) -> np.ndarray:
    """PTQ1_0 raw block bytes -> float32 weights (ggml dim order)."""
    assert n_elements % QK == 0
    nb = n_elements // QK
    raw = np.frombuffer(buf, dtype=np.uint8, count=nb * PTQ1_0_BLOCK_BYTES)
    raw = raw.reshape(nb, PTQ1_0_BLOCK_BYTES)

    qs = raw[:, 0:24].astype(np.uint32)          # (nb, 24)
    qh = raw[:, 24:26].astype(np.uint32)         # (nb, 2)
    d = _fp16_to_f32(raw[:, 26:28].copy().view(np.uint16).reshape(nb))  # (nb,)

    out = np.empty((nb, QK), dtype=np.float32)

    # trit n of byte q: ((uint8_t)(q * pow3[n]) * 3 >> 8) - 1
    # NOTE the uint8 wraparound BEFORE the *3 shift — matching the C code exactly
    def trits(q: np.ndarray, n: int) -> np.ndarray:
        return ((((q * POW3[n]) & 0xFF) * 3) >> 8).astype(np.float32) - 1.0

    # stage c=16: bytes 0..15 -> positions n*16+m
    for n in range(5):
        out[:, n * 16:(n + 1) * 16] = trits(qs[:, 0:16], n)
    # stage c=8: bytes 16..23 -> positions 80 + n*8+m
    for n in range(5):
        out[:, 80 + n * 8:80 + (n + 1) * 8] = trits(qs[:, 16:24], n)
    # qh: bytes 0..1 -> positions 120 + n*2 + h
    for n in range(4):
        out[:, 120 + n * 2:120 + (n + 1) * 2] = trits(qh, n)

    out *= d[:, None]
    return out.reshape(n_elements)


DEQUANT = {
    142: dequant_pq2_0,
    143: dequant_ptq1_0,
}
