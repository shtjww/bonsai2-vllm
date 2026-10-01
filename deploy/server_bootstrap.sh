#!/bin/bash
# server bootstrap on RTX6000D (torch 2.12.1+cu130 preinstalled)
set -ex
export PATH=/root/miniconda3/bin:$PATH
PROJ=/root/bonsai2-vllm
MODELS=/root/autodl-tmp/models
mkdir -p $MODELS
PY=/root/miniconda3/bin/python

# 1. vLLM: NO proxy, aliyun mirror
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
$PY -m pip install vllm==0.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
$PY -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
$PY -c "import flashinfer; print('flashinfer', flashinfer.__version__)" || echo "NO_FLASHINFER"

# 2. weights: academic turbo proxy, huggingface.co DIRECT
source /etc/network_turbo
if [ ! -f $MODELS/Ternary-Bonsai-2-27B-PTQ1_0.gguf ]; then
  $PY $PROJ/src/download_parallel.py \
    https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    $MODELS/Ternary-Bonsai-2-27B-PTQ1_0.gguf 32
fi
ls -la $MODELS/*.gguf

# 3. serve dir
unset http_proxy https_proxy
SERVE=$MODELS/serve-bonsai2
mkdir -p $SERVE
ln -sf $MODELS/Ternary-Bonsai-2-27B-PTQ1_0.gguf $SERVE/
cp $PROJ/reference/config_serve.json $SERVE/config.json
cp $PROJ/reference/tokenizer.json $PROJ/reference/tokenizer_config.json \
   $PROJ/reference/vocab.json $PROJ/reference/merges.txt $SERVE/
ls -la $SERVE
echo "BOOTSTRAP_DONE"
