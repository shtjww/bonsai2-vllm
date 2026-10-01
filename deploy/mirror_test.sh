#!/bin/bash
# speed test: download first 40MB of the vllm wheel from 3 mirrors
URL_PATH="pypi/packages/vllm-0.26.0"
# find real wheel urls via simple index
for M in "https://mirrors.aliyun.com/pypi/simple/vllm/" "https://mirrors.cloud.tencent.com/pypi/simple/vllm/" "https://pypi.tuna.tsinghua.edu.cn/simple/vllm/"; do
  echo "=== $M"
  WHEEL=$(curl -s --max-time 15 "$M" | grep -o 'href="[^"]*vllm-0.26.0-cp38-abi3-manylinux[^"]*"' | tail -1 | sed 's/href="//;s/"$//')
  if [ -z "$WHEEL" ]; then echo "  no wheel link found"; continue; fi
  case "$WHEEL" in
    http*) URL="$WHEEL";;
    /*)    URL="$(echo $M | sed 's|\(https://[^/]*\)/.*|\1|')$WHEEL";;
    *)     URL="${M}${WHEEL}";;
  esac
  URL="${URL%%#*}"
  echo "  url: ${URL:0:120}"
  curl -sL --max-time 25 -r 0-41943039 -o /dev/null -w "  speed: %{speed_download} B/s (http %{http_code})\n" "$URL"
done
