"""Build packed-ternary registry from the GGUF and swap vLLM modules.

Enabled when env BONSAI_TERNARY=1 (GGUF path from BONSAI_GGUF_PATH).
Runs inside the EngineCore process after model load (called from loader.py).

Module targets (vLLM qwen3_5 tree, matched by name suffix):
  linear_attn.in_proj_qkvz <- [attn_qkv, attn_gate]      (both folded)
  linear_attn.out_proj     <- [ssm_out]                  (folded, NO unperm)
  self_attn.qkv_proj       <- [attn_q, attn_k, attn_v]   (folded)
  self_attn.o_proj         <- [attn_output]              (folded)
  mlp.gate_up_proj         <- [ffn_gate, ffn_up]         (folded)
  mlp.down_proj            <- [ffn_down]                 (folded)
  lm_head                  <- [output.weight]            (folded)
Embedding stays fp16 (lookup reads 10KB/token — bandwidth-neutral).
"""
from __future__ import annotations

import os
import sys

import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "src"))

from .ternary_kernel import pack_ptq1
from .ternary_layers import TernaryHadamardLinear, TernaryLMHead

# ggml suffix -> (module suffix, slot). Only folded tensors.
GGML_TO_MODULE = {
    "attn_qkv.weight": ("linear_attn.in_proj_qkvz", 0),
    "attn_gate.weight": ("linear_attn.in_proj_qkvz", 1),
    "ssm_out.weight": ("linear_attn.out_proj", 0),
    "ffn_gate.weight": ("mlp.gate_up_proj", 0),
    "ffn_up.weight": ("mlp.gate_up_proj", 1),
    "ffn_down.weight": ("mlp.down_proj", 0),
    "attn_q.weight": ("self_attn.qkv_proj", 0),
    "attn_k.weight": ("self_attn.qkv_proj", 1),
    "attn_v.weight": ("self_attn.qkv_proj", 2),
    "attn_output.weight": ("self_attn.o_proj", 0),
}

NK, NV, HD = 16, 48, 128
REP = NV // NK
VPERM = [j // REP + NK * (j % REP) for j in range(NV)]  # grouped -> tiled


def _vperm_packed(packed: torch.Tensor, v_lo: int) -> torch.Tensor:
    """permute v-head row-blocks (128 rows) along the out axis of packed
    (n_blk, 7, out). Same grouped->tiled perm as weights.py::_perm_v_heads.
    (via int32 view: torch has no uint32 advanced indexing on CUDA)"""
    nb, _, n_out = packed.shape
    p32 = packed.view(torch.int32)
    head, v = p32[:, :, :v_lo], p32[:, :, v_lo:]
    v = v.reshape(nb, 7, NV, HD)[:, :, VPERM, :].reshape(nb, 7, -1)
    return torch.cat([head, v], dim=2).contiguous().view(torch.uint32)


def build_registry(gguf_path: str) -> dict:
    """module suffix -> sorted [(slot, packed_tensor)]; plus lm_head entry."""
    from src.gguf_reader import GGUFReader

    registry: dict[str, list] = {}
    with GGUFReader(gguf_path) as r:
        md = r.metadata
        folded = set(md.get("prism.hadamard.weight_names", []))
        for name, t in r.tensors.items():
            if name == "output.weight":
                key, slot = "lm_head", 0
            elif name.startswith("blk."):
                rest = name[4:]
                layer_str, suffix = rest.split(".", 1)
                if suffix not in GGML_TO_MODULE:
                    continue
                mod_suffix, slot = GGML_TO_MODULE[suffix]
                key = f"layers.{int(layer_str)}.{mod_suffix}"
            else:
                continue
            if name not in folded:
                continue
            in_dim, out_dim = t.shape[0], t.shape[1]   # ggml (in, out)
            packed = pack_ptq1(r.tensor_bytes(name), out_dim, in_dim)
            # v-head grouped->tiled perm (vLLM pairs j//rep, ckpt pairs j%nk)
            if name.endswith("attn_qkv.weight"):
                packed = _vperm_packed(packed, 2 * NK * HD)
            elif name.endswith("attn_gate.weight"):
                packed = _vperm_packed(packed, 0)
            registry.setdefault(key, []).append((slot, packed))
    # merge multi-part modules into ONE packed tensor (same in_dim & n_blk)
    # -> single kernel launch, no cat
    merged: dict[str, list] = {}
    for k, parts in registry.items():
        parts.sort(key=lambda x: x[0])
        packed_all = torch.cat([p for _, p in parts], dim=2).contiguous()
        merged[k] = [(0, packed_all)]
    return merged


def swap_modules(model: torch.nn.Module, gguf_path: str):
    from src.gguf_reader import GGUFReader
    from src.convert_to_bf16 import load_signs

    with GGUFReader(gguf_path) as r:
        md = r.metadata
        block_size = md["prism.hadamard.block_size"]
        signs = load_signs(md)

    registry = build_registry(gguf_path)
    print(f"[ternary] packed {sum(len(v) for v in registry.values())} tensors "
          f"for {len(registry)} modules", flush=True)

    modules = dict(model.named_modules())
    swapped = 0
    for key, parts in registry.items():
        matches = [n for n in modules if n.endswith(key)]
        if not matches:
            print(f"[ternary] WARN no module matches {key}", flush=True)
            continue
        name = matches[0]
        out_dtype = next(model.parameters()).dtype
        if key == "lm_head":
            new = TernaryLMHead(parts[0][1], torch.from_numpy(signs[5120]),
                                block_size, out_dtype)
        else:
            in_dim = 5120  # all parts share input dim; ssm_out is 6144, ffn_down 17408
            # infer from packed: n_blk * 128
            in_dim = parts[0][1].shape[0] * 128
            new = TernaryHadamardLinear([p for _, p in parts],
                                        torch.from_numpy(signs[in_dim]),
                                        block_size, out_dtype)
        parent_name, _, child = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        old = getattr(parent, child)
        del old                      # drop fp16 weight reference
        setattr(parent, child, new)
        swapped += 1
    torch.cuda.empty_cache()
    print(f"[ternary] swapped {swapped} modules; "
          f"GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
