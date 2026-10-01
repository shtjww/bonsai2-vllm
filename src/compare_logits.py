#!/usr/bin/env python3
"""Correctness anchor: compare vLLM (bonsai_gguf) vs PrismML fork (llama-server).

Both expose OpenAI-compatible chat completions with logprobs. Greedy decode,
fixed prompts. We compare:
  1. top-1 token agreement rate (per position)
  2. mean |Δ logprob| on positions where both agree
  3. first divergence position per prompt

Usage: python compare_logits.py --vllm http://127.0.0.1:8000/v1 --fork http://127.0.0.1:8080/v1
"""
import argparse
import json

import requests

PROMPTS = [
    "用一句话解释什么是KV cache。",
    "Write a Python function to compute the nth Fibonacci number iteratively.",
    "The capital of France is",
    "请解释大模型推理中prefill和decode的区别。",
    "What is 17 * 23? Think step by step.",
    "List three prime numbers greater than 100.",
    "Translate 'The quick brown fox jumps over the lazy dog' into Chinese.",
    "def quicksort(arr):",
]


def query(base, model, prompt, max_tokens=64):
    r = requests.post(f"{base}/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens,
        "logprobs": True, "top_logprobs": 5,
    }, timeout=300)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    content = choice["message"]["content"]
    lp = (choice.get("logprobs") or {}).get("content") or []
    toks = [(t["token"], t["logprob"], [tt["token"] for tt in t.get("top_logprobs", [])])
            for t in lp]
    return content, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--fork", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--vllm-model", default="bonsai2-27b")
    ap.add_argument("--fork-model", default="bonsai")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default="compare_result.json")
    args = ap.parse_args()

    report = []
    total_pos = agree_pos = 0
    deltas = []
    for p in PROMPTS:
        v_content, v_toks = query(args.vllm, args.vllm_model, p, args.max_tokens)
        f_content, f_toks = query(args.fork, args.fork_model, p, args.max_tokens)
        n = min(len(v_toks), len(f_toks))
        agree, first_div = 0, None
        for i in range(n):
            if v_toks[i][0] == f_toks[i][0]:
                agree += 1
                deltas.append(abs(v_toks[i][1] - f_toks[i][1]))
            elif first_div is None:
                first_div = i
        total_pos += n
        agree_pos += agree
        rec = {"prompt": p[:40], "positions": n, "agree": agree,
               "first_divergence": first_div,
               "vllm_head": v_content[:80], "fork_head": f_content[:80]}
        report.append(rec)
        print(f"[{agree}/{n}] div@{first_div} | {p[:40]}", flush=True)

    summary = {
        "top1_agreement": agree_pos / max(total_pos, 1),
        "mean_abs_dlogprob": sum(deltas) / max(len(deltas), 1),
        "positions": total_pos,
    }
    print(json.dumps(summary, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "report": report}, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
