#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
echo "=== ifaces ==="; ls /sys/class/net
for d in $(ls /sys/class/net | grep -v lo); do
  echo "$d rx=$(cat /sys/class/net/$d/statistics/rx_bytes)"
done
echo "=== pip proc ==="; ps -eo pid,etime,stat,cmd | grep 'pip install' | grep -v grep
echo "=== pip cache ==="; du -sh /root/.cache/pip 2>/dev/null
echo "=== vllm? ==="; pip show vllm 2>/dev/null | head -2 || echo none
echo "=== tmp big files ==="; ls -la /tmp/*.whl /tmp/pip-* 2>/dev/null | head -5
