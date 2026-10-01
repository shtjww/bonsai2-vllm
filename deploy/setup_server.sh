#!/bin/bash
# AutoDL 4090 实例引导脚本：Bonsai2-vLLM 正确性验证环境
# 用法: bash setup_server.sh
set -e
PROJ=/root/bonsai2-vllm
MODELS=/root/autodl-tmp/models
mkdir -p $MODELS

echo "=== [1/4] vLLM 环境（版本冻结 0.26.0，沿用验收纪律）==="
pip install vllm==0.26.0 2>&1 | tail -1
# 注意坑清单：GDN内核强依赖flashinfer，0.26自带；若缺：pip install flashinfer-python

echo "=== [2/4] 项目代码（本目录整个rsync/scp上传后应已就位）==="
test -f $PROJ/vllm_bonsai/weights.py || { echo "项目代码未就位：先把本地项目上传到 $PROJ"; exit 1; }

echo "=== [3/4] 下载三元权重（hf-mirror + 并行分块下载器，断点续传）==="
export HF_ENDPOINT=https://hf-mirror.com
if [ ! -f $MODELS/Ternary-Bonsai-2-27B-PTQ1_0.gguf ]; then
  python3 $PROJ/src/download_parallel.py \
    https://hf-mirror.com/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    $MODELS/Ternary-Bonsai-2-27B-PTQ1_0.gguf 24
fi

echo "=== [4/4] 组装服务目录（config + tokenizer + gguf软链）==="
SERVE=$MODELS/serve-bonsai2
mkdir -p $SERVE
ln -sf $MODELS/Ternary-Bonsai-2-27B-PTQ1_0.gguf $SERVE/
cp $PROJ/reference/config_serve.json $SERVE/config.json
cp $PROJ/reference/tokenizer.json $PROJ/reference/tokenizer_config.json \
   $PROJ/reference/vocab.json $PROJ/reference/merges.txt \
   $PROJ/reference/chat_template.jinja $PROJ/reference/generation_config.json $SERVE/ 2>/dev/null || true
ls -la $SERVE

echo "=== 完成。启动服务：==="
echo "cd $PROJ && python vllm_bonsai/serve_bonsai.py --gguf-dir $SERVE --port 8000"
