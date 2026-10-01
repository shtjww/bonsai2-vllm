#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
which python python3
python3 --version
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)' 2>&1 | tail -1
python3 -c 'import vllm; print("vllm", vllm.__version__)' 2>&1 | tail -1
python3 -c 'import flashinfer; print("flashinfer", flashinfer.__version__)' 2>&1 | tail -1
nvidia-smi --query-gpu=driver_version --format=csv,noheader
free -g | head -2
