"""GGUF (qwen35, llama.cpp naming) -> HF (qwen3_5_text, vLLM) tensor name mapping.

Conventions:
  ggml:  weight shape (in_dim, out_dim), ne0 = input
  HF:    Linear weight shape (out_dim, in_dim)  => TRANSPOSE on conversion

Layer types alternate: 3x linear_attention (GDN) + 1x full_attention.
GGUF drops MTP and vision tower is a separate mmproj file; HF names below are
text-model only (model.language_model.*).
"""

GLOBAL_MAP = {
    "token_embd.weight":  "model.language_model.embed_tokens.weight",
    "output.weight":      "lm_head.weight",
    "output_norm.weight": "model.language_model.norm.weight",
}

# per-layer suffix map for linear_attention (GDN) layers
LINEAR_ATTN_MAP = {
    "attn_norm.weight":            "input_layernorm.weight",
    "post_attention_norm.weight":  "post_attention_layernorm.weight",
    "attn_qkv.weight":             "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight":            "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight":            "linear_attn.in_proj_a.weight",
    "ssm_beta.weight":             "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight":           "linear_attn.conv1d.weight",
    "ssm_dt.bias":                 "linear_attn.dt_bias",
    "ssm_a":                       "linear_attn.A_log",
    "ssm_norm.weight":             "linear_attn.norm.weight",
    "ssm_out.weight":              "linear_attn.out_proj.weight",
    "ffn_gate.weight":             "mlp.gate_proj.weight",
    "ffn_up.weight":               "mlp.up_proj.weight",
    "ffn_down.weight":             "mlp.down_proj.weight",
}

# per-layer suffix map for full_attention layers
FULL_ATTN_MAP = {
    "attn_norm.weight":            "input_layernorm.weight",
    "post_attention_norm.weight":  "post_attention_layernorm.weight",
    "attn_q.weight":               "self_attn.q_proj.weight",
    "attn_k.weight":               "self_attn.k_proj.weight",
    "attn_v.weight":               "self_attn.v_proj.weight",
    "attn_output.weight":          "self_attn.o_proj.weight",
    "attn_q_norm.weight":          "self_attn.q_norm.weight",
    "attn_k_norm.weight":          "self_attn.k_norm.weight",
    "ffn_gate.weight":             "mlp.gate_proj.weight",
    "ffn_up.weight":               "mlp.up_proj.weight",
    "ffn_down.weight":             "mlp.down_proj.weight",
}

FULL_ATTN_INTERVAL = 4  # layers 3, 7, 11, ... are full attention


def is_full_attention(layer: int) -> bool:
    return (layer + 1) % FULL_ATTN_INTERVAL == 0


def ggml_to_hf(name: str) -> str:
    if name in GLOBAL_MAP:
        return GLOBAL_MAP[name]
    assert name.startswith("blk."), f"unknown tensor: {name}"
    rest = name[4:]
    layer_str, suffix = rest.split(".", 1)
    layer = int(layer_str)
    table = FULL_ATTN_MAP if is_full_attention(layer) else LINEAR_ATTN_MAP
    assert suffix in table, f"unknown suffix in {name}"
    return f"model.language_model.layers.{layer}.{table[suffix]}"
