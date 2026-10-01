#!/bin/bash
# compare vLLM (8000) vs fork (8080) on same prompts, greedy
export PATH=/root/miniconda3/bin:$PATH
for PROMPT in "用一句话解释什么是KV cache" "The capital of France is" "请解释prefill和decode的区别"; do
  echo "=== Q: $PROMPT"
  echo "--- vLLM:"
  curl -s --max-time 120 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"bonsai2-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":0,\"max_tokens\":64}" \
    | python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][:300])"
  echo "--- fork:"
  curl -s --max-time 120 http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"bonsai\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":0,\"max_tokens\":64}" \
    | python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][:300])"
done
