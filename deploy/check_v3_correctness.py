"""v3 correctness, production workload shape: 32-way concurrent x 256 tokens.

Compares each concurrent output against the single-stream greedy output of the
same prompt. Byte-identical is ideal; late divergence = batch fp16-MMA numerics.
"""
import concurrent.futures as cf
import os.path

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


def gen(prompt):
    r = requests.post("http://127.0.0.1:8000/v1/completions", json={
        "model": "bonsai2-27b", "prompt": prompt,
        "max_tokens": 256, "temperature": 0}, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


refs = [gen(p) for p in PROMPTS]              # single-stream (T=1)
with cf.ThreadPoolExecutor(32) as ex:
    outs = list(ex.map(gen, PROMPTS * 4))      # 32-way (T>=16 -> v3)

n_same = 0
for i, o in enumerate(outs):
    ref = refs[i % 8]
    if o == ref:
        n_same += 1
    else:
        prefix = len(os.path.commonprefix([o, ref]))
        print(f"[{i}] prompt#{i % 8} diverge at char {prefix}/{len(ref)}", flush=True)
print(f"V3_CORRECTNESS_32WAY: {n_same}/32 byte-identical to single-stream", flush=True)
