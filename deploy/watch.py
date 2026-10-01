"""Watch server until vllm installed AND gguf downloaded. Prints progress each poll.
Run: uv run --with paramiko python deploy/watch.py
Requires env: AUTODL_SSH_HOST / AUTODL_SSH_PORT / AUTODL_SSH_PASS.
"""
import os
import sys
import time

import paramiko

HOST = os.environ.get("AUTODL_SSH_HOST", "")
PORT = int(os.environ.get("AUTODL_SSH_PORT", "22"))
PASS = os.environ.get("AUTODL_SSH_PASS", "")
if not HOST or not PASS:
    sys.exit("Set AUTODL_SSH_HOST / AUTODL_SSH_PORT / AUTODL_SSH_PASS env vars first.")


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASS,
              timeout=20, banner_timeout=30)
    return c


c = connect()


def q(cmd):
    _, o, _ = c.exec_command(cmd, timeout=60)
    return o.read().decode().strip()


while True:
    try:
        vllm = q("/root/miniconda3/bin/pip show vllm 2>/dev/null | head -1")
        cache = q("du -sh /root/.cache/uv 2>/dev/null | cut -f1")
        gguf = q("ls -la /root/autodl-tmp/models/*.gguf 2>/dev/null | awk '{print $5}'")
        parts = q("du -sb /root/autodl-tmp/models/*.parts 2>/dev/null | cut -f1")
        gguflog = q("tail -1 /root/gguf_dl.log 2>/dev/null")
        print(f"[{time.strftime('%H:%M:%S')}] vllm={vllm or '...'} uv_cache={cache} "
              f"gguf={gguf or parts or '...'} | {gguflog}", flush=True)
        if vllm.startswith("Version: 0.26.0") and gguf == "5946648928":
            print("ALL_DONE", flush=True)
            break
    except Exception as e:
        print("poll error:", e, flush=True)
        try:
            c.close()
        except Exception:
            pass
        c = connect()
    time.sleep(120)
c.close()
