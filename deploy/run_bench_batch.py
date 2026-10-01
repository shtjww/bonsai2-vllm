#!/usr/bin/env python3
"""Batch scaling bench: N concurrent greedy requests, measure aggregate decode tps."""
import sys
import threading
import time

import requests

PROMPTS = [
    "Write a detailed essay about the history of computing.",
    "用中文详细解释量子计算的原理、现状和未来。",
    "Implement a full HTTP server in Python with routing and middleware.",
    "Explain the transformer architecture step by step with formulas.",
    "Describe the Byzantine generals problem and its solutions.",
    "写一首关于秋天的长诗。",
    "Explain how gradient descent works with math.",
    "Design a rate limiter and explain tradeoffs.",
]
MAX_TOKENS = 256


def one(base, model, prompt, res, idx):
    t0 = time.time()
    r = requests.post(f"{base}/completions", json={
        "model": model, "prompt": prompt, "max_tokens": MAX_TOKENS,
        "temperature": 0}, timeout=900)
    dt = time.time() - t0
    r.raise_for_status()
    n = r.json().get("usage", {}).get("completion_tokens", MAX_TOKENS)
    res[idx] = (n, dt)


def bench(base, model, name, bs):
    # warmup with 2
    for p in PROMPTS[:2]:
        one(base, model, p, {}, 0)
    res = {}
    t0 = time.time()
    threads = [threading.Thread(target=one,
                                args=(base, model, PROMPTS[i % len(PROMPTS)], res, i))
               for i in range(bs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    tot = sum(n for n, _ in res.values())
    print(f"[{name} bs={bs}] {tot} tok in {wall:.1f}s = {tot / wall:.1f} tok/s aggregate",
          flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "vllm"
    for bs in (1, 4, 8):
        if which in ("vllm", "both"):
            bench("http://127.0.0.1:8000/v1", "bonsai2-27b", "vLLM-ternary", bs)
        if which in ("fork", "both"):
            bench("http://127.0.0.1:8080/v1", "bonsai", "fork", bs)
