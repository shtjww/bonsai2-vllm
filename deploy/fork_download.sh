#!/bin/bash
# robust fork binary download: try many proxies with resume
set -x
BIN=/root/bonsai2-vllm/fork-bin
mkdir -p $BIN
REL=prism-b10709-9a9394a
ASSET="llama-${REL}-bin-linux-cuda-13.3-x64.tar.gz"
GH="https://github.com/PrismML-Eng/llama.cpp/releases/download/$REL/$ASSET"
for U in "$GH" \
         "https://ghfast.top/$GH" \
         "https://gh-proxy.com/$GH" \
         "https://github.moeyy.xyz/$GH" \
         "https://gh.llkk.cc/$GH" \
         "https://ghproxy.net/$GH"; do
  echo "=== $U"
  curl -L --fail --connect-timeout 10 --max-time 600 -C - "$U" -o /tmp/llama.tgz && break
done
ls -la /tmp/llama.tgz
tar -xzf /tmp/llama.tgz -C $BIN --strip-components=1 || tar -xzf /tmp/llama.tgz -C $BIN
rm -f /tmp/llama.tgz
$BIN/llama-cli --version
