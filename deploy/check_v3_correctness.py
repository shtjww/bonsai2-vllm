"""v3 correctness check: 16-way concurrent greedy outputs must match single-stream."""
import concurrent.futures as cf
import os.path

import requests

PROMPTS = [
    "Write a detailed essay about the history of computing.",
    "用中文详细解释量子计算的原理。",
    "Implement a rate limiter in Python.",
    "Explain gradient descent with math.",
]


def gen(prompt):
    r = requests.post("http://127.0.0.1:8000/v1/completions", json={
        "model": "bonsai2-27b", "prompt": prompt,
        "max_tokens": 128, "temperature": 0}, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


refs = [gen(p) for p in PROMPTS]          # single-stream (T=1, v1 path)
with cf.ThreadPoolExecutor(16) as ex:
    outs = list(ex.map(gen, PROMPTS * 4))  # 16-way (T>=16, v3 path)

ok = True
for i, o in enumerate(outs):
    ref = refs[i % 4]
    if o != ref:
        ok = False
        prefix = len(os.path.commonprefix([o, ref]))
        print(f"[{i}] DIVERGE at char {prefix}/{len(ref)}", flush=True)
print("V3_CORRECTNESS:", "PASS (16-way outputs identical to single-stream)" if ok
      else "DIVERGENCE FOUND", flush=True)
