#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
echo "=== torch下载状态 ==="
grep -E "torch" /root/vllm_install.log | tail -3
echo "=== 日志尾部 ==="
tail -3 /root/vllm_install.log
echo "=== uv缓存 ==="
du -sh /root/.cache/uv
echo "=== vllm? ==="
python -c 'import vllm; print("VLLM_OK", vllm.__version__)' 2>&1 | tail -1
