#!/usr/bin/env python3
"""Same-protocol throughput bench: vLLM (8000) vs fork llama-server (8080).

Greedy, fixed prompts, fixed output length. Reports per-run:
prefill tps, decode tps (from server timing when available, else wall clock).
"""
import json
import time

import requests

PROMPTS = [
    "Write a detailed essay about the history of computing.",
    "用中文详细解释量子计算的原理、现状和未来。",
    "Implement a full HTTP server in Python with routing and middleware.",
    "Explain the transformer architecture step by step with formulas.",
]
MAX_TOKENS = 256
ROUNDS = 2  # warmup + measured


def bench(base, model, name):
    url = f"{base}/completions"
    tot_wall = tot_tokens = 0
    for rnd in range(ROUNDS):
        for p in PROMPTS:
            t0 = time.time()
            r = requests.post(url, json={
                "model": model, "prompt": p, "max_tokens": MAX_TOKENS,
                "temperature": 0, "stream": False,
            }, timeout=600)
            dt = time.time() - t0
            r.raise_for_status()
            usage = r.json().get("usage", {})
            n = usage.get("completion_tokens", MAX_TOKENS)
            if rnd > 0:
                tot_wall += dt
                tot_tokens += n
            tag = "warmup" if rnd == 0 else "meas"
            print(f"  [{name}/{tag}] {n} tok in {dt:.1f}s = {n / dt:.1f} tok/s", flush=True)
    tps = tot_tokens / tot_wall if tot_wall else 0
    print(f"[{name}] decode throughput (greedy, bs=1): {tps:.1f} tok/s", flush=True)
    return tps


if __name__ == "__main__":
    a = bench("http://127.0.0.1:8000/v1", "bonsai2-27b", "vLLM")
    b = bench("http://127.0.0.1:8080/v1", "bonsai", "fork")
    print(json.dumps({"vllm_tps": round(a, 1), "fork_tps": round(b, 1)}))
