"""Roundtrip self-test for prism_dequant.py.

Implements quantize_row_ptq1_0_ref / quantize_row_pq2_0_ref in Python
(ported from the fork's ggml-quants.c), packs random data, dequantizes with
the vectorized implementation, and checks that every element matches the
scalar reference.

Run: python test_dequant.py
"""
import numpy as np

from prism_dequant import dequant_pq2_0, dequant_ptq1_0, QK

RNG = np.random.default_rng(0)


def f16(x: float) -> int:
    return np.float16(x).view(np.uint16).item()


def quant_ptq1_0_block(x: np.ndarray) -> bytes:
    """Scalar reference port of quantize_row_ptq1_0_ref for one 128-block."""
    assert x.shape == (QK,)
    amax = np.abs(x).max()
    d = np.float16(amax).item()
    idv = 1.0 / d if d else 0.0

    qs = bytearray(24)
    qh = bytearray(2)
    xpos = 0
    j = 0
    for c in (32, 16, 8):
        while j + c <= 24:
            for m in range(c):
                q = 0
                for n in range(5):
                    xi = int(round(float(x[xpos + m + n * c]) * idv)) + 1
                    q = q * 3 + xi
                q = (q * 256 + 242) // 243
                qs[j + m] = q
            xpos += 5 * c
            j += c
    for h in range(2):
        q = 0
        for m in range(4):
            xi = int(round(float(x[xpos + h + m * 2]) * idv)) + 1
            q = q * 3 + xi
        q *= 3
        q = (q * 256 + 242) // 243
        qh[h] = q
    xpos += 8
    assert xpos == QK, xpos
    return bytes(qs) + bytes(qh) + np.uint16(f16(amax)).tobytes()


def quant_pq2_0_block(x: np.ndarray) -> bytes:
    """Scalar reference port of quantize_row_pq2_0_ref for one 128-block."""
    assert x.shape == (QK,)
    amax = np.abs(x).max()
    d = np.float16(amax).item()
    idv = 1.0 / d if d else 0.0
    qs = bytearray(32)
    for j, w in enumerate(x):
        q = int(round(float(w) * idv)) + 1
        q = min(max(q, 0), 3)
        qs[j // 4] |= q << ((j % 4) * 2)
    return np.uint16(f16(amax)).tobytes() + bytes(qs)


def scalar_dequant_ptq1_0(block: bytes) -> np.ndarray:
    """Straight loop port of dequantize_row_ptq1_0 for cross-checking."""
    qs, qh = block[:24], block[24:26]
    d = np.frombuffer(block[26:28], dtype=np.float16)[0].astype(np.float32)
    pow3 = [1, 3, 9, 27, 81]
    y = []
    j = 0
    for c in (32, 16, 8):
        while j + c <= 24:
            for n in range(5):
                for m in range(c):
                    q = (qs[j + m] * pow3[n]) & 0xFF
                    xi = ((q * 3) >> 8)
                    y.append((xi - 1) * d)
            j += c
    for n in range(4):
        for h in range(2):
            q = (qh[h] * pow3[n]) & 0xFF
            xi = ((q * 3) >> 8)
            y.append((xi - 1) * d)
    return np.array(y, dtype=np.float32)


def test_ptq1_0(nblocks=50):
    blocks = b"".join(quant_ptq1_0_block(RNG.normal(size=QK).astype(np.float32))
                      for _ in range(nblocks))
    vec = dequant_ptq1_0(blocks, nblocks * QK).reshape(nblocks, QK)
    for i in range(nblocks):
        ref = scalar_dequant_ptq1_0(blocks[i * 28:(i + 1) * 28])
        if not np.allclose(vec[i], ref, rtol=0, atol=0):
            bad = np.nonzero(vec[i] != ref)[0]
            raise AssertionError(f"block {i} mismatch at positions {bad[:10]}")
    print(f"PTQ1_0 roundtrip OK ({nblocks} blocks, scalar cross-check passed)")


def test_pq2_0(nblocks=50):
    xs = [RNG.normal(size=QK).astype(np.float32) for _ in range(nblocks)]
    blocks = b"".join(quant_pq2_0_block(x) for x in xs)
    vec = dequant_pq2_0(blocks, nblocks * QK).reshape(nblocks, QK)
    for i, x in enumerate(xs):
        amax = np.float16(np.abs(x).max()).item()
        ref = np.array([min(max(int(round(float(w) / amax)) + 1, 0), 3) - 1 for w in x],
                       dtype=np.float32) * np.float32(amax)
        if not np.allclose(vec[i], ref, rtol=0, atol=0):
            raise AssertionError(f"block {i} mismatch")
    print(f"PQ2_0 roundtrip OK ({nblocks} blocks)")


if __name__ == "__main__":
    test_ptq1_0()
    test_pq2_0()
