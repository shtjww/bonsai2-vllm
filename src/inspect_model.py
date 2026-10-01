"""Inspect a Bonsai 2 GGUF checkpoint: metadata census + tensor census + dequant sanity.

Usage:
  python inspect_model.py <model.gguf> [--dequant-check]
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from gguf_reader import GGUFReader
from prism_dequant import DEQUANT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dequant-check", action="store_true",
                    help="dequantize one ternary tensor and print value stats")
    args = ap.parse_args()

    with GGUFReader(args.model) as r:
        print(f"=== file: {args.model}")
        print(f"gguf version: {r.version}, tensors: {len(r.tensors)}, metadata keys: {len(r.metadata)}")

        print("\n=== general metadata ===")
        for k in sorted(r.metadata):
            if k.startswith("prism."):
                continue
            v = r.metadata[k]
            s = str(v)
            print(f"  {k} = {s[:120]}{'...' if len(s) > 120 else ''}")

        hc = r.hadamard_config
        print("\n=== prism.hadamard config ===")
        if hc is None:
            print("  (none — stock model, no rotated basis)")
        else:
            for k, v in hc.items():
                s = str(v)
                print(f"  {k} = {s[:200]}{'...' if len(s) > 200 else ''}")

        print("\n=== tensor census by type ===")
        cnt = Counter(t.type_name for t in r.tensors.values())
        bytes_by = Counter()
        for t in r.tensors.values():
            bytes_by[t.type_name] += t.n_bytes
        for name, n in cnt.most_common():
            print(f"  {name:8s} tensors={n:4d}  bytes={bytes_by[name]/1e9:.3f} GB")

        print("\n=== sample tensors ===")
        for name in list(r.tensors)[:15]:
            t = r.tensors[name]
            print(f"  {t.type_name:8s} {str(t.shape):24s} {name}")

        if args.dequant_check:
            print("\n=== dequant sanity check ===")
            for name, t in r.tensors.items():
                if t.ggml_type in DEQUANT:
                    w = DEQUANT[t.ggml_type](r.tensor_bytes(name), t.n_elements)
                    vals, counts = np.unique(np.sign(w[w != 0]), return_counts=True)
                    frac = {int(v): int(c) for v, c in zip(vals, counts)}
                    zeros = int((w == 0).sum())
                    print(f"  tensor: {name} ({t.type_name}, {t.shape})")
                    print(f"    abs max={np.abs(w).max():.6f}  zeros={zeros} ({zeros/w.size:.1%})")
                    print(f"    sign histogram (nonzero): {frac}")
                    print(f"    ternary purity: unique |w|/d ratios should collapse to {{0,1}}")
                    break


if __name__ == "__main__":
    main()
