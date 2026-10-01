"""Streaming converter: Bonsai 2 GGUF -> standard-basis BF16 safetensors (HF names).

Single pass, one tensor at a time: dequant -> un-rotate (H/signs/GDN perm) ->
transpose (ggml (in,out) -> torch (out,in)) -> bf16 -> append to shard.
Peak memory = one tensor (max ~2.5 GB f32 for lm_head).

Usage:
  python convert_to_safetensors.py <model.gguf> <out_dir> [--tensors a,b,c]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from convert_to_bf16 import load_signs
from gguf_reader import GGUFReader
from hadamard import apply_hadamard_rows, hadamard_matrix, unpermute_gdn_rows
from name_map import ggml_to_hf
from prism_dequant import DEQUANT

SHARD_BYTES = 5 * 1024**3


def to_bf16_bytes(arr_f32: np.ndarray) -> bytes:
    """float32 -> bfloat16 with round-to-nearest-even, returned as raw bytes."""
    u = arr_f32.view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    u16 = ((u + rounding) >> 16).astype(np.uint16)
    return u16.tobytes()


class ShardWriter:
    """Minimal safetensors shard writer (bf16 only)."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.shard_idx = 0
        self.entries: dict[str, dict] = {}   # name -> {dtype, shape, data_offsets, shard}
        self._buf = bytearray()
        self._shards: list[str] = []

    def _flush(self):
        if not self._buf:
            return
        fname = f"model-{self.shard_idx:05d}-of-XXXXX.safetensors"
        self._shards.append(fname)
        header = {}
        off = 0
        for name, e in self.entries.items():
            if e["shard"] == self.shard_idx:
                header[name] = {"dtype": "BF16", "shape": e["shape"],
                                "data_offsets": [e["beg"], e["end"]]}
        hj = json.dumps(header).encode()
        pad = (8 - (8 + len(hj)) % 8) % 8
        hj += b" " * pad
        with open(self.out_dir / fname, "wb") as f:
            f.write(len(hj).to_bytes(8, "little"))
            f.write(hj)
            f.write(self._buf)
        self.shard_idx += 1
        self._buf = bytearray()

    def add(self, name: str, arr_f32: np.ndarray):
        bf = to_bf16_bytes(arr_f32)
        if len(self._buf) + len(bf) > SHARD_BYTES:
            self._flush()
        beg = len(self._buf)
        self._buf += bf
        self.entries[name] = {"shape": list(arr_f32.shape), "beg": beg,
                              "end": beg + len(bf), "shard": self.shard_idx}

    def close(self, extra_files: dict[str, Path] | None = None):
        self._flush()
        # rename shards with final count
        total = len(self._shards)
        final_names = {}
        for i, old in enumerate(self._shards):
            new = f"model-{i+1:05d}-of-{total:05d}.safetensors"
            (self.out_dir / old).rename(self.out_dir / new)
            final_names[i] = new
        index = {"metadata": {"total_size": sum(e["end"] - e["beg"] for e in self.entries.values())},
                 "weight_map": {n: final_names[e["shard"]] for n, e in self.entries.items()}}
        (self.out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        if extra_files:
            for dst, src in extra_files.items():
                shutil.copy(src, self.out_dir / dst)


def convert(model_path: str, out_dir: str, only: set[str] | None = None,
            config_src: str | None = None):
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
        n_v = int(md.get("qwen35.ssm.time_step_rank", 0))
        n_k = int(md.get("qwen35.ssm.group_count", 0))
        rep = n_v // n_k if (n_v and n_k) else 0

        w = ShardWriter(out_dir)
        names = [n for n in r.tensors if not only or n in only]
        for i, name in enumerate(names):
            t = r.tensors[name]
            if t.ggml_type in DEQUANT:
                arr = DEQUANT[t.ggml_type](r.tensor_bytes(name), t.n_elements).reshape(t.shape)
            else:
                arr = r.tensor_f16(name)

            if H is not None and (name in folded or name in inverses):
                width = t.shape[0]
                arr = arr.reshape(width, -1).astype(np.float32)
                arr = apply_hadamard_rows(arr, H)
                arr = arr * signs[width][:, None]
                if gdn_grouped and ".ssm_out." in name:
                    arr = unpermute_gdn_rows(arr, n_k, rep)
                arr = arr.reshape(t.shape)

            # ggml (in, out[, ...]) -> torch Linear (out, in); 1D tensors stay
            if arr.ndim == 2:
                arr = np.ascontiguousarray(arr.T)
            w.add(ggml_to_hf(name), arr.astype(np.float32, copy=False))
            if (i + 1) % 50 == 0 or i + 1 == len(names):
                print(f"[convert] {i+1}/{len(names)}", flush=True)

        extra = {}
        if config_src:
            extra["config.json"] = Path(config_src)
        w.close(extra)
        print(f"[done] {len(names)} tensors -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--tensors", default=None)
    ap.add_argument("--config", default=None, help="path to HF config.json to copy in")
    args = ap.parse_args()
    only = set(args.tensors.split(",")) if args.tensors else None
    convert(args.model, args.out_dir, only, args.config)
