#!/bin/bash
# fetch wheel URLs from tuna simple index for the pinned big packages
for pkg in torch/2.11.0 torchvision/0.26.0 torchaudio/2.11.0 vllm/0.26.0 flashinfer-python/0.6.14; do
  name=${pkg%/*}; ver=${pkg#*/}
  url=$(curl -s --max-time 20 "https://pypi.tuna.tsinghua.edu.cn/simple/$name/" \
    | grep -o "href=\"[^\"]*${name//-/_}-${ver}-cp312-cp312-manylinux[^\"]*\.whl\"" | head -1 | sed 's/href="//;s/"$//')
  if [ -z "$url" ]; then
    url=$(curl -s --max-time 20 "https://pypi.tuna.tsinghua.edu.cn/simple/$name/" \
      | grep -o "href=\"[^\"]*${name//-/_}-${ver}-cp38-abi3-manylinux[^\"]*\.whl\"" | head -1 | sed 's/href="//;s/"$//')
  fi
  case "$url" in
    http*) echo "$url" | sed 's/#.*//';;
    *) echo "https://pypi.tuna.tsinghua.edu.cn/simple/$name/${url}" | sed 's/#.*//';;
  esac
done
