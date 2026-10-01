"""Remote driver for the AutoDL instance (paramiko). Usage:
  uv run --with paramiko python deploy/remote.py <cmd...>
  uv run --with paramiko python deploy/remote.py --upload
  uv run --with paramiko python deploy/remote.py --script <local_file>
"""
import sys
import tarfile
import io
import os

import paramiko

HOST = os.environ.get("AUTODL_SSH_HOST", "")
PORT = int(os.environ.get("AUTODL_SSH_PORT", "22"))
USER = "root"
PASS = os.environ.get("AUTODL_SSH_PASS", "")
if not HOST or not PASS:
    sys.exit("Set AUTODL_SSH_HOST / AUTODL_SSH_PORT / AUTODL_SSH_PASS env vars first.")
LOCAL_PROJ = r"D:\2026年工作\bonsai2-vllm"
REMOTE_PROJ = "/root/bonsai2-vllm"

INCLUDE_DIRS = ["src", "vllm_bonsai", "deploy", "reference"]
INCLUDE_FILES = ["README.md"]


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)
    return c


def run(c, cmd, timeout=600):
    print(f"$ {cmd}", flush=True)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    if out.strip():
        print(out[-3000:], flush=True)
    if err.strip():
        print("[stderr]", err[-1000:], flush=True)
    return stdout.channel.recv_exit_status(), out, err


def upload(c):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in INCLUDE_DIRS:
            tar.add(os.path.join(LOCAL_PROJ, d), arcname=d)
        for f in INCLUDE_FILES:
            p = os.path.join(LOCAL_PROJ, f)
            if os.path.exists(p):
                tar.add(p, arcname=f)
    data = buf.getvalue()
    print(f"uploading {len(data)/1e6:.1f} MB ...", flush=True)
    run(c, f"mkdir -p {REMOTE_PROJ}")
    sftp = c.open_sftp()
    with sftp.file("/root/proj.tar.gz", "wb") as f:
        f.write(data)
    sftp.close()
    rc, _, _ = run(c, f"cd {REMOTE_PROJ} && tar xzf /root/proj.tar.gz && rm /root/proj.tar.gz && ls")
    print("upload rc =", rc, flush=True)


if __name__ == "__main__":
    c = connect()
    if sys.argv[1] == "--upload":
        upload(c)
    elif sys.argv[1] == "--script":
        with open(sys.argv[2], encoding="utf-8") as f:
            run(c, f.read(), timeout=3600)
    else:
        run(c, " ".join(sys.argv[1:]))
    c.close()
