"""Run a command on the AutoDL instance. Creds ONLY from env vars.
Usage: $env:AUTODL_SSH_PASS='...'; python deploy\_ssh.py "cmd"
"""
import os
import sys

import paramiko

HOST = os.environ.get("AUTODL_SSH_HOST", "connect.westb.seetacloud.com")
PORT = int(os.environ.get("AUTODL_SSH_PORT", "18990"))
PASS = os.environ.get("AUTODL_SSH_PASS")
if not PASS:
    sys.exit("set AUTODL_SSH_PASS first")

cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read().lstrip("﻿")
timeout = int(os.environ.get("SSH_CMD_TIMEOUT", "120"))

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PASS, timeout=20)
_, out, err = c.exec_command(cmd, timeout=timeout)
rc = out.channel.recv_exit_status()
sys.stdout.write(out.read().decode(errors="replace"))
e = err.read().decode(errors="replace")
if e.strip():
    sys.stderr.write(e)
c.close()
sys.exit(rc)
