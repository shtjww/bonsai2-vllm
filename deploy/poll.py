"""One-shot server status probe (run locally with: uv run --with paramiko python deploy/poll.py)
Requires env: AUTODL_SSH_HOST / AUTODL_SSH_PORT / AUTODL_SSH_PASS.
"""
import os
import sys
import time

import paramiko

host = os.environ.get("AUTODL_SSH_HOST", "")
port = int(os.environ.get("AUTODL_SSH_PORT", "22"))
password = os.environ.get("AUTODL_SSH_PASS", "")
if not host or not password:
    sys.exit("Set AUTODL_SSH_HOST / AUTODL_SSH_PORT / AUTODL_SSH_PASS env vars first.")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, port=port, username="root", password=password, timeout=20)


def q(cmd):
    _, o, _ = c.exec_command(cmd, timeout=30)
    return o.read().decode().strip()


def rx():
    return int(q("cat /sys/class/net/eth0/statistics/rx_bytes"))


a = rx()
time.sleep(15)
b = rx()
print(f"下载速率: {(b - a) / 15 / 1e6:.1f} MB/s")
print("uv缓存:", q("du -sh /root/.cache/uv | cut -f1"))
print("torch下完没:", q("grep -c 'Downloaded torch' /root/vllm_install.log"))
print("uv进程数:", q("pgrep -fc 'uv pip' || echo 0"))
print("vllm:", q("/root/miniconda3/bin/pip show vllm 2>/dev/null | head -1") or "未装完")
print("权重:", q("ls -la /root/autodl-tmp/models/*.gguf 2>/dev/null || du -sh /root/autodl-tmp/models/*.parts 2>/dev/null || echo 未开始"))
c.close()
