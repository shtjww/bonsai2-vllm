#!/usr/bin/env python3
"""Serve Bonsai 2 27B via vLLM with the bonsai_gguf loader (dequant-at-load).

Run anywhere:
    python serve_bonsai.py --gguf-dir models/serve-bonsai2 --ternary --port 8000

Then benchmark with the usual run_bench.sh against http://127.0.0.1:8000.
"""
import argparse
import os

import vllm_bonsai
vllm_bonsai.install()  # must happen before vllm engine creation

from vllm.entrypoints.openai.api_server import run_server  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", required=True,
                    help="directory containing Ternary-Bonsai-2-27B-PTQ1_0.gguf "
                         "plus config.json / tokenizer files copied from the official repo")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92,
                    help="lower this when sharing the GPU (e.g. with fork llama-server)")
    ap.add_argument("--dtype", default=None,
                    help="e.g. float16 — ternary x fp16 scales are exact in fp16, bf16 loses scale precision")
    ap.add_argument("--from-safetensors", action="store_true",
                    help="gguf-dir is actually a pre-converted safetensors dir "
                         "(skip bonsai_gguf loader, ~1min startup)")
    ap.add_argument("--ternary", action="store_true",
                    help="keep folded weights packed PTQ1_0 (~5.9GB resident), "
                         "dequant in-kernel (torch.compile / CUDA Graph compatible)")
    args = ap.parse_args()

    if args.ternary:
        os.environ["BONSAI_TERNARY"] = "1"
        os.environ["BONSAI_GGUF_PATH"] = os.path.join(
            args.gguf_dir if os.path.isdir(args.gguf_dir) else os.path.dirname(args.gguf_dir),
            "Ternary-Bonsai-2-27B-PTQ1_0.gguf")
        if not os.path.exists(os.environ["BONSAI_GGUF_PATH"]):
            raise SystemExit(
                f"GGUF not found at {os.environ['BONSAI_GGUF_PATH']} — "
                "make sure Ternary-Bonsai-2-27B-PTQ1_0.gguf is inside --gguf-dir "
                "(see README quickstart step 1).")

    # delegate to vLLM's OpenAI server with our load format
    import sys
    sys.argv = [
        "vllm-serve",
        "--model", args.gguf_dir,
        "--served-model-name", "bonsai2-27b",
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", "256",   # Mamba cache blocks are the binding constraint (427 here)
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    if not args.from_safetensors:
        sys.argv += ["--load-format", "bonsai_gguf"]
    if args.dtype:
        sys.argv += ["--dtype", args.dtype]
    if os.environ.get("BONSAI_DUMP_DIR"):
        sys.argv.append("--enforce-eager")  # forward hooks need eager mode
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    parser = make_arg_parser(FlexibleArgumentParser())
    vllm_args = parser.parse_args()
    import asyncio
    asyncio.run(run_server(vllm_args))


if __name__ == "__main__":
    main()
