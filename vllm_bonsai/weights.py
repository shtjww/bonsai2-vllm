"""Core weight iterator: stream HF-named, standard-basis FP16 tensors from a
Bonsai 2 GGUF file. Framework-level code (torch) — vLLM-agnostic.

GPU-accelerated: dequant (numpy) -> un-rotate (torch batched GEMM on CUDA)
-> transpose -> fp16. Load time target: minutes, not tens of minutes.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "src"))

from src.gguf_reader import GGUFReader          # noqa: E402
from src.convert_to_bf16 import load_signs       # noqa: E402
from src.hadamard import hadamard_matrix         # noqa: E402
from src.name_map import ggml_to_hf              # noqa: E402
from src.prism_dequant import DEQUANT            # noqa: E402

VISUAL_BLOCK_MAP = {
    "attn_qkv":  "attn.qkv",
    "attn_out":  "attn.proj",
    "ffn_up":    "mlp.linear_fc1",
    "ffn_down":  "mlp.linear_fc2",
    "ln1":       "norm1",
    "ln2":       "norm2",
}

# vLLM runs these norms as GemmaRMSNorm: y = x * (1 + w), so its checkpoints
# store zero-centered weights. llama.cpp GGUF stores 1-centered weights
# (applied directly as y = x * w). Subtract 1 to match vLLM's convention.
# NOT applied to linear_attn.norm (vLLM RMSNormGated is standard x * w)
# nor to vision LayerNorms.
_GEMMA_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)
_FINAL_NORM = "model.language_model.norm.weight"


def _iter_visual_weights(mmproj_path: str, dtype):
    """mmproj GGUF (BF16/F32, no rotation) -> model.visual.* HF tensors."""
    with GGUFReader(mmproj_path) as r:
        tensors = r.tensors

        # per-block mappings
        for name in sorted(tensors):
            if not name.startswith("v.blk."):
                continue
            rest = name[len("v.blk."):]
            layer_str, suffix = rest.split(".", 1)
            base, kind = suffix.rsplit(".", 1)      # e.g. attn_qkv + weight
            hf = f"model.visual.blocks.{int(layer_str)}.{VISUAL_BLOCK_MAP[base]}.{kind}"
            # tensor_torch: axes already reversed to HF orientation — no .t()
            arr = r.tensor_torch(name)
            t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32, copy=False)))
            yield hf, t.to(dtype)

        # patch embedding: ggml (16,16,3,1152) reversed -> (1152,3,16,16);
        # two temporal slices stacked -> Conv3d (1152,3,2,16,16)
        pe0 = torch.from_numpy(np.ascontiguousarray(r.tensor_torch("v.patch_embd.weight").astype(np.float32)))
        pe1 = torch.from_numpy(np.ascontiguousarray(r.tensor_torch("v.patch_embd.weight.1").astype(np.float32)))
        pe = torch.stack([pe0, pe1], dim=2).contiguous()
        yield "model.visual.patch_embed.proj.weight", pe.to(dtype)
        pb = torch.from_numpy(np.ascontiguousarray(r.tensor_f16("v.patch_embd.bias").astype(np.float32)))
        yield "model.visual.patch_embed.proj.bias", pb.to(dtype)

        # positional embedding: ggml (1152, 2304) reversed -> HF (2304, 1152)
        pos = torch.from_numpy(np.ascontiguousarray(r.tensor_torch("v.position_embd.weight").astype(np.float32)))
        yield "model.visual.pos_embed.weight", pos.to(dtype)

        # merger: mm.0 -> linear_fc1, mm.2 -> linear_fc2
        for src, hf in (("mm.0", "linear_fc1"), ("mm.2", "linear_fc2")):
            w = torch.from_numpy(np.ascontiguousarray(r.tensor_torch(f"{src}.weight").astype(np.float32)))
            yield f"model.visual.merger.{hf}.weight", w.to(dtype)
            b = torch.from_numpy(np.ascontiguousarray(r.tensor_f16(f"{src}.bias").astype(np.float32)))
            yield f"model.visual.merger.{hf}.bias", b.to(dtype)
        # merger norm: llama.cpp stores it as v.post_ln (1152,)
        nw = torch.from_numpy(np.ascontiguousarray(r.tensor_f16("v.post_ln.weight").astype(np.float32)))
        yield "model.visual.merger.norm.weight", nw.to(dtype)
        nb = torch.from_numpy(np.ascontiguousarray(r.tensor_f16("v.post_ln.bias").astype(np.float32)))
        yield "model.visual.merger.norm.bias", nb.to(dtype)

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_H_CACHE: dict[int, torch.Tensor] = {}


def _perm_v_heads(w: torch.Tensor, n_k: int, rep: int, hd: int, v_lo: int) -> torch.Tensor:
    """GDN v-head order: checkpoint grouped (fork kernel pairs v-head j with
    k-head j % n_k) -> tiled (vLLM/HF pair v-head j with k-head j // rep).
    new[j] = old[j // rep + n_k * (j % rep)].  v_lo = leading non-v rows."""
    n_v = n_k * rep
    idx = torch.tensor([j // rep + n_k * (j % rep) for j in range(n_v)],
                       device=w.device)
    head, v = w[:v_lo], w[v_lo:]
    v = v.reshape(n_v, hd, *w.shape[1:])[idx]
    return torch.cat([head, v.reshape(-1, *w.shape[1:])], dim=0)


def _get_H(block_size: int) -> torch.Tensor:
    if block_size not in _H_CACHE:
        _H_CACHE[block_size] = torch.from_numpy(hadamard_matrix(block_size)).to(_DEV)
    return _H_CACHE[block_size]


def _unrotate(arr: np.ndarray, in_width: int, H: torch.Tensor,
              signs: np.ndarray, gdn_unperm: tuple[int, int] | None) -> torch.Tensor:
    """Undo Hadamard folding along the INPUT dim (LAST axis after ggml->torch
    axis reversal). arr: (..., in_width). Returns torch fp32 on _DEV.

    W_std = P^T diag(s) H W_q  =>  per block: w = w @ H (H symmetric), then * s,
    then undo the GDN feature permutation if needed."""
    w = torch.from_numpy(np.ascontiguousarray(
        arr.astype(np.float32, copy=False))).to(_DEV)
    head_shape = w.shape[:-1]
    w = w.reshape(-1, in_width)
    bs = H.shape[0]
    w = w.reshape(w.shape[0], in_width // bs, bs)
    w = torch.matmul(w, H)                       # batched GEMM on GPU (H symmetric)
    w = w.reshape(w.shape[0], in_width)
    s = torch.from_numpy(signs).to(_DEV)
    w = w * s[None, :]
    # NOTE: no GDN feature un-permutation. The checkpoint stores ssm_out input
    # features in tiled [hd, nk, rep] order — which is exactly what HF/vLLM
    # produce (repeat_interleave pairing: v-head j <-> k-head j//rep).
    # llama.cpp's runtime perm exists only to adapt ITS grouped (j%nk) kernel;
    # undoing it in the weights was wrong for HF/vLLM. (Verified: layer-0
    # numpy trace matches fork node sums to 5e-4 with NO weight perm.)
    return w.reshape(*head_shape, in_width)


def iter_bonsai_weights(gguf_path: str, dtype=torch.float16):
    """Yield (hf_name, torch.Tensor[dtype]) in standard basis, HF layout."""
    with GGUFReader(gguf_path) as r:
        md = r.metadata
        folded = set(md.get("prism.hadamard.weight_names", []))
        inverses = set(md.get("prism.hadamard.inverse_weight_names", []))
        block_size = md.get("prism.hadamard.block_size", 0)
        gdn_grouped = bool(md.get("prism.hadamard.gdn_v_grouped", False))
        signs = load_signs(md) if md.get("prism.hadamard.sign_mode") == "explicit" else {}
        H = _get_H(block_size) if block_size else None
        n_v = int(md.get("qwen35.ssm.time_step_rank", 0))
        n_k = int(md.get("qwen35.ssm.group_count", 0))
        rep = n_v // n_k if (n_v and n_k) else 0

        for name, t in r.tensors.items():
            if t.ggml_type in DEQUANT:
                flat = DEQUANT[t.ggml_type](r.tensor_bytes(name), t.n_elements)
            else:
                flat = r.tensor_f16(name).reshape(-1)

            # ggml layout: ne[0] is fastest-varying, so the flat stream in
            # C-order has shape REVERSED vs ggml ne[]. For 2D linear weights
            # this yields (out, in) — torch orientation directly, NO transpose.
            # (The old reshape(t.shape)+transpose scrambled every 2D tensor.)
            arr = flat.reshape(t.shape[::-1])
            in_width = t.shape[0]                # ggml ne0 = input/features dim

            if H is not None and (name in folded or name in inverses):
                out = _unrotate(arr, in_width, H, signs[in_width], None)
            else:
                out = torch.from_numpy(np.ascontiguousarray(
                    arr.astype(np.float32, copy=False))).to(_DEV)

            # NOTE: NO qkv repack here. vLLM's Qwen3.5 GDN path
            # (gqa_interleaved_layout=False) expects plain contiguous
            # [q | k | v] — same as llama.cpp. The group-interleaved repack
            # was a wrong assumption borrowed from Qwen3-Next.

            # GGUF ssm_a stores -exp(A_log) (fork multiplies it directly);
            # vLLM/HF expect raw A_log and apply -exp() themselves.
            if name.endswith("ssm_a"):
                out = torch.log(-out)

            # conv1d: ggml (kernel, channels) reversed -> (channels, kernel)
            # -> torch (channels, 1, kernel). Channels already contiguous
            # [q|k|v] — no reorder.
            if "conv1d" in name and out.ndim == 2:
                out = out.unsqueeze(1).contiguous()

            # GDN v-head permutation (grouped -> tiled), see _perm_v_heads.
            # ssm_out is NOT permuted: its input features are stored tiled
            # already (verified against fork node sums).
            if n_v and rep:
                n_k_, hd_ = n_k, 128
                if name.endswith("attn_qkv.weight"):
                    out = _perm_v_heads(out, n_k_, rep, hd_, 2 * n_k_ * hd_)
                elif name.endswith("attn_gate.weight"):
                    out = _perm_v_heads(out, n_k_, rep, hd_, 0)
                elif "conv1d" in name:
                    out = _perm_v_heads(out, n_k_, rep, hd_, 2 * n_k_ * hd_)
                elif name.endswith(("ssm_alpha.weight", "ssm_beta.weight")):
                    idx = [j // rep + n_k_ * (j % rep) for j in range(n_v)]
                    out = out[torch.tensor(idx, device=out.device)]
                elif name.endswith(("ssm_a", "ssm_dt.bias")):
                    idx = [j // rep + n_k_ * (j % rep) for j in range(n_v)]
                    out = out[torch.tensor(idx, device=out.device)]

            hf_name = ggml_to_hf(name)
            if hf_name.endswith(_GEMMA_NORM_SUFFIXES) or hf_name == _FINAL_NORM:
                out = out - 1.0
            yield hf_name, out.to(dtype).cpu()

    # vision tower (mmproj) if present alongside the language model
    mmproj = glob.glob(os.path.join(os.path.dirname(gguf_path), "*mmproj*.gguf"))
    if mmproj:
        for hf, t in _iter_visual_weights(mmproj[0], dtype):
            yield hf, t.cpu()
