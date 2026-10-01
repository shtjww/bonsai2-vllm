"""Hidden-state dump hooks for debugging. Installed post-load when
BONSAI_DUMP_DIR is set (serve_bonsai.py also forces --enforce-eager then).

Saves the FIRST prefill pass (num_tokens > 1) to {dump_dir}/vllm_dump.npz:
  ids        (T,) int32
  embed      (T, H) embed_tokens output
  layer_i    (T, H) decoder layer i output (hidden only, residual dropped)
  final_norm (T, H)
Then disarms itself so decode steps are untouched.
"""
from __future__ import annotations

import os

import numpy as np
import torch

_store: dict[str, np.ndarray] = {}
_armed = True


def _to_np(t) -> np.ndarray:
    if isinstance(t, (tuple, list)):
        t = t[0]
    return t.detach().float().cpu().numpy()


def _maybe_save(dump_dir: str):
    global _armed
    if "final_norm" in _store and "embed" in _store:
        np.savez(os.path.join(dump_dir, "vllm_dump.npz"), **_store)
        print(f"[bonsai-dump] saved {len(_store)} arrays -> {dump_dir}/vllm_dump.npz",
              flush=True)
        _armed = False


def install_dump_hooks(model: torch.nn.Module, dump_dir: str):
    os.makedirs(dump_dir, exist_ok=True)
    embed = None
    layers = None
    norm = None
    for name, mod in model.named_modules():
        if name.endswith("embed_tokens") and embed is None:
            embed = mod
        if name.endswith("language_model.layers") or (
                layers is None and name.endswith("model.layers")):
            layers = mod
        if name.endswith("language_model.norm") or (
                norm is None and name.endswith("model.norm")):
            norm = mod
    assert embed is not None and layers is not None, "dump: embed/layers not found"

    def embed_hook(m, args, out):
        if _armed and out.shape[-2] > 1:
            _store["ids"] = args[0].detach().cpu().numpy().astype(np.int32)
            _store["embed"] = _to_np(out)

    embed.register_forward_hook(embed_hook)

    for i, layer in enumerate(layers):
        def mk(i):
            def hook(m, args, out):
                if _armed and "embed" in _store and f"layer_{i}" not in _store:
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if t.shape[-2] == _store["embed"].shape[-2]:
                        _store[f"layer_{i}"] = _to_np(t)
            return hook
        layer.register_forward_hook(mk(i))

    if norm is not None:
        def norm_hook(m, args, out):
            if _armed and f"layer_{len(layers) - 1}" in _store:
                _store["final_norm"] = _to_np(out)
                _maybe_save(dump_dir)
        norm.register_forward_hook(norm_hook)

    # fine-grained: capture sub-module I/O of layer 0 (GDN) and layer 3 (attn)
    sub_filter = os.environ.get("BONSAI_DUMP_SUB", "layers.0.,layers.3.")
    for name, mod in model.named_modules():
        if not any(f in name for f in sub_filter.split(",")):
            continue
        depth = name.count(".")
        if depth < 3 or depth > 5:   # layers.N.{child} .. layers.N.linear_attn.{op}
            continue

        def mk_sub(key):
            def hook(m, args, out):
                if _armed and "embed" in _store and key not in _store:
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if torch.is_tensor(t) and t.shape[-2] == _store["embed"].shape[-2]:
                        _store[key] = _to_np(t)
            return hook
        mod.register_forward_hook(mk_sub(f"sub::{name}"))

    print(f"[bonsai-dump] hooks installed: {len(layers)} layers -> {dump_dir}",
          flush=True)
