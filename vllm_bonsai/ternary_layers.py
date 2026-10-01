"""Ternary Hadamard linear layers: packed PTQ1_0 weights stay resident (~1.75 bpw),
activation gets the Hadamard/sign treatment at runtime: y = W_q^T (H (s ⊙ x)).

This mirrors the fork's runtime math exactly (validated layer-0 node sums to 5e-4).
Compile-compatible: packed weights are registered buffers (dynamo lifts them as
real graph inputs) and the math runs inside a torch custom op (opaque node).
"""
from __future__ import annotations

import torch
from torch import nn

_ternary_op = torch.ops.bonsai.ternary_hadamard_linear


class TernaryHadamardLinear(nn.Module):
    """Drop-in for vLLM column/row-parallel linears (returns (y, dummy_bias)).

    parts: list with ONE packed uint32 cuda tensor (n_blk, 7, out)
    (registry pre-merges multi-part fused modules into a single packed tensor).
    """

    def __init__(self, parts: list[torch.Tensor], signs: torch.Tensor,
                 block_size: int, out_dtype: torch.dtype):
        super().__init__()
        packed = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2).contiguous()
        self.register_buffer("packed_w", packed)              # uint32, cuda
        self.register_buffer("signs", signs.float().cuda())   # (in_dim,)
        self.block_size = block_size
        self.out_dtype = out_dtype
        self.in_dim = signs.numel()
        self.out_dim = packed.shape[2]
        # vLLM/dynamo compat: real (empty) Parameter so param lifting works
        self.register_parameter(
            "weight",
            nn.Parameter(torch.zeros(0, dtype=out_dtype, device="cuda"),
                         requires_grad=False))
        self.bias = None

    def forward(self, x, *args, **kwargs):
        y = _ternary_op(self.packed_w, x, self.signs, self.block_size)
        # second tuple element = the "bias" slot vLLM unpacks and discards;
        # must be a tensor (None breaks inductor runtime wrappers)
        return y.to(self.out_dtype), y.new_empty(0)


class _TernaryLMHeadMethod:
    """quant_method shim for vLLM's LogitsProcessor: apply(module, hidden, bias)."""

    def apply(self, module, hidden_states, bias=None):
        return module.ternary_forward(hidden_states)


class TernaryLMHead(nn.Module):
    """Drop-in for lm_head (VocabParallelEmbedding): logits = W^T (H (s ⊙ h))."""

    def __init__(self, packed: torch.Tensor, signs: torch.Tensor,
                 block_size: int, out_dtype: torch.dtype):
        super().__init__()
        self.register_buffer("packed_w", packed)
        self.register_buffer("signs", signs.float().cuda())
        self.block_size = block_size
        self.out_dtype = out_dtype
        self.quant_method = _TernaryLMHeadMethod()
        self.register_parameter(
            "weight",
            nn.Parameter(torch.zeros(0, dtype=out_dtype, device="cuda"),
                         requires_grad=False))
        self.bias = None

    def ternary_forward(self, hidden_states):
        y = _ternary_op(self.packed_w, hidden_states, self.signs, self.block_size)
        return y.to(self.out_dtype)
