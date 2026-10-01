"""Install the bonsai_gguf load format into vLLM's loader registry.

Call install() BEFORE creating any LLM/engine object.
"""
from __future__ import annotations


def install():
    from .loader import register
    register()
