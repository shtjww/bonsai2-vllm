"""vLLM custom model loader: serve Bonsai 2 GGUF directly (dequant-at-load).

Verified against vLLM 0.26.0 internals:
  - DefaultModelLoader (not DefaultLoader)
  - _get_weights_iterator(self, source: Source)
  - registry dict: vllm.model_executor.model_loader._LOAD_FORMAT_TO_MODEL_LOADER
"""
from __future__ import annotations

import glob
import os

import torch

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from .weights import iter_bonsai_weights

LOAD_FORMAT = "bonsai_gguf"


class BonsaiGGUFLoader(DefaultModelLoader):
    """DefaultModelLoader with a weights iterator fed by our GGUF pipeline."""

    def _get_weights_iterator(self, source):
        path = source.model_or_path
        if os.path.isdir(path):
            hits = glob.glob(os.path.join(path, "*PTQ1_0*.gguf")) or \
                   glob.glob(os.path.join(path, "*.gguf"))
            assert hits, f"no .gguf found in {path}"
            path = hits[0]
        target_dtype = torch.float16  # ternary values x fp16 scales are exact in fp16
        return iter_bonsai_weights(path, dtype=target_dtype)

    def load_model(self, vllm_config, model_config):
        model = super().load_model(vllm_config, model_config)
        _post_load(model)
        return model


def _post_load(model):
    """Shared post-load hooks: hidden-state dump + ternary swap."""
    dump_dir = os.environ.get("BONSAI_DUMP_DIR")
    if dump_dir and not getattr(model, "_bonsai_dump_installed", False):
        from .dump import install_dump_hooks
        install_dump_hooks(model, dump_dir)
        model._bonsai_dump_installed = True
    if os.environ.get("BONSAI_TERNARY") == "1" and \
            not getattr(model, "_bonsai_ternary", False):
        from .ternary_swap import swap_modules
        swap_modules(model, os.environ["BONSAI_GGUF_PATH"])
        model._bonsai_ternary = True


def register():
    from vllm.model_executor.model_loader import _LOAD_FORMAT_TO_MODEL_LOADER
    _LOAD_FORMAT_TO_MODEL_LOADER[LOAD_FORMAT] = BonsaiGGUFLoader

    # hooks for ANY load format (safetensors included): wrap the base
    # DefaultModelLoader so --from-safetensors runs also get post-load hooks
    orig_load_model = DefaultModelLoader.load_model

    def _wrapped(self, *args, **kwargs):
        model = orig_load_model(self, *args, **kwargs)
        _post_load(model)
        return model

    if os.environ.get("BONSAI_DUMP_DIR") or os.environ.get("BONSAI_TERNARY") == "1":
        DefaultModelLoader.load_model = _wrapped
