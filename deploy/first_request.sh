#!/bin/bash
curl -s --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai2-27b","messages":[{"role":"user","content":"用一句话解释什么是KV cache"}],"temperature":0,"max_tokens":96}' \
  | /root/miniconda3/bin/python -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); u=d.get('usage',{}); print('---'); print('usage:', u)"
