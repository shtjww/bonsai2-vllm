"""Convert a Bonsai 2 GGUF (PTQ1_0/PQ2_0, rotated basis) to standard-basis weights.

Runtime path in the fork (build_lora_mm / build_embd_rows):
  forward weights : y = W_q^T ( H (s . P(x)) )     => W_std = P^T diag(s) H W_q
  inverse (embd)  : h = s . (H z) after lookup     => E_std = diag(s) H E_stored
where H = blockwise normalized Sylvester-Walsh-Hadamard (block_size=1024),
      s = full-width +/-1 diagonal (per input width 5120/6144/17408),
      P = GDN v-grouped feature permutation (ssm_out weights only).

Usage:
  python convert_to_bf16.py <model.gguf> <out_dir> [--tensors name1,name2] [--dtype bf16|f32]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gguf_reader import GGUFReader
from hadamard import apply_hadamard_rows, hadamard_matrix, unpermute_gdn_rows
from prism_dequant import DEQUANT


def load_signs(md: dict) -> dict[int, np.ndarray]:
    """Split prism.hadamard.sign_values by sign_widths -> {width: ±1 vector}."""
    widths = md.get("prism.hadamard.sign_widths", [])
    values = md.get("prism.hadamard.sign_values", [])
    out, off = {}, 0
    for w in widths:
        out[int(w)] = np.array(values[off:off + w], dtype=np.float32)
        off += w
    assert off == len(values), f"sign_values length {len(values)} != sum(widths) {off}"
    return out


def convert(model_path: str, out_dir: str, only: set[str] | None, dtype):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with GGUFReader(model_path) as r:
        md = r.metadata
        folded = set(md.get("prism.hadamard.weight_names", []))
        inverses = set(md.get("prism.hadamard.inverse_weight_names", []))
        block_size = md.get("prism.hadamard.block_size", 0)
        gdn_grouped = bool(md.get("prism.hadamard.gdn_v_grouped", False))
        signs = load_signs(md) if md.get("prism.hadamard.sign_mode") == "explicit" else {}

        H = hadamard_matrix(block_size) if block_size else None
        # GDN permutation geometry (from llama-model.cpp: n_v=ssm_dt_rank, n_k=ssm_n_group)
        n_v = int(md.get("qwen35.ssm.time_step_rank", 0))
        n_k = int(md.get("qwen35.ssm.group_count", 0))
        rep = n_v // n_k if (n_v and n_k) else 0
        print(f"[hadamard] block={block_size} folded={len(folded)} inverse={len(inverses)} "
              f"gdn_grouped={gdn_grouped} (hd*nk*rep, nk={n_k}, rep={rep})")

        tensors, stats = {}, []
        for name, t in r.tensors.items():
            if only and name not in only:
                continue
            if t.ggml_type in DEQUANT:
                w = DEQUANT[t.ggml_type](r.tensor_bytes(name), t.n_elements).reshape(t.shape)
            else:
                w = r.tensor_f16(name)

            kind = "float"
            if H is not None and (name in folded or name in inverses):
                width = t.shape[0]
                s = signs.get(width)
                assert s is not None, f"no sign vector for width {width} ({name})"
                w2 = w.reshape(width, -1).astype(np.float32)
                w2 = apply_hadamard_rows(w2, H)          # H W_q
                w2 = w2 * s[:, None]                      # diag(s) ...
                if gdn_grouped and ".ssm_out." in name:
                    w2 = unpermute_gdn_rows(w2, n_k, rep)  # P^T ...
                    kind = "folded+gdn"
                else:
                    kind = "folded" if name in folded else "inverse"
                w = w2.reshape(t.shape)

            tensors[name] = w.astype(dtype)
            stats.append((name, t.type_name, kind, t.shape))

        stem = Path(model_path).stem
        if only:
            stem += ".partial"
        np.savez(out_dir / (stem + ".npz"), **tensors)
        meta = {
            "source": str(model_path),
            "hadamard": {"block_size": block_size,
                         "gdn_v_grouped": gdn_grouped,
                         "sign_widths": list(signs)},
            "tensors": {n: {"ggml_type": ty, "path": k, "shape": list(map(int, sh))}
                        for n, ty, k, sh in stats},
            "note": "STANDARD basis, ggml dim order (ne0 fastest). Names not yet HF-mapped.",
        }
        (out_dir / (stem + ".meta.json")).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        total = sum(v.nbytes for v in tensors.values())
        print(f"[done] {len(tensors)} tensors, {total/1e9:.2f} GB -> {out_dir / (stem + '.npz')}")
        for n, ty, k, sh in stats[:20]:
            print(f"  {k:11s} {ty:8s} {str(sh):22s} {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--tensors", default=None, help="comma-separated subset for quick tests")
    ap.add_argument("--dtype", default="f32", choices=["f32", "bf16"])
    args = ap.parse_args()
    only = set(args.tensors.split(",")) if args.tensors else None
    dt = np.float32 if args.dtype == "f32" else None
    if args.dtype == "bf16":
        import ml_dtypes  # numpy-compatible bf16
        dt = ml_dtypes.bfloat16
    convert(args.model, args.out_dir, only, dt)
