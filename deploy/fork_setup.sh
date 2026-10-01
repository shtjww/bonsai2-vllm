#!/bin/bash
# fork侧标准答案环境：下载PrismML预编译llama.cpp CUDA二进制 + 启动llama-server
# 在服务器上执行：bash fork_setup.sh
set -ex
MODELS=/root/autodl-tmp/models
BIN=/root/bonsai2-vllm/fork-bin
mkdir -p $BIN

# 检测driver CUDA版本选构建tag（与官方download_binaries.sh同逻辑）
CUDA_VER=$(nvidia-smi | sed -n 's/.*CUDA[ A-Z]*Version:[[:space:]]*\([0-9]*\.[0-9]*\).*/\1/p')
MAJOR=${CUDA_VER%%.*}; MINOR=${CUDA_VER#*.}
if [ "$MAJOR" -gt 13 ] || { [ "$MAJOR" -eq 13 ] && [ "$MINOR" -ge 3 ]; }; then TAG="13.3";
elif [ "$MAJOR" -eq 13 ] || { [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 8 ]; }; then TAG="12.8";
else TAG="12.4"; fi
echo "driver CUDA $CUDA_VER -> build tag $TAG"

REL=prism-b10709-9a9394a
ASSET="llama-${REL}-bin-linux-cuda-${TAG}-x64.tar.gz"
if [ ! -x $BIN/llama-server ]; then
  for URL in "https://github.com/PrismML-Eng/llama.cpp/releases/download/$REL/$ASSET" \
             "https://ghproxy.net/https://github.com/PrismML-Eng/llama.cpp/releases/download/$REL/$ASSET" \
             "https://gh-proxy.com/https://github.com/PrismML-Eng/llama.cpp/releases/download/$REL/$ASSET"; do
    echo "trying $URL"
    curl -L --fail --connect-timeout 15 --progress-bar "$URL" -o /tmp/llama.tgz && break || true
  done
  tar -xzf /tmp/llama.tgz -C $BIN --strip-components=1 || tar -xzf /tmp/llama.tgz -C $BIN
  rm -f /tmp/llama.tgz
fi
$BIN/llama-cli --version || $BIN/llama-server --help | head -3

# 启动fork参考服务（:8080），PTQ1_0权重
if ! curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
  nohup $BIN/llama-server -m $MODELS/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    --port 8080 -c 8192 --alias bonsai > /root/fork-server.log 2>&1 &
  echo "fork server starting, log: /root/fork-server.log"
fi
echo "FORK_SETUP_DONE"
