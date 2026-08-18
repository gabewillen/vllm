#!/bin/bash
# usage: launch.sh <yaml> <log> [extra env assignments via env]
HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 PYTHONHASHSEED=8013 exec /shared/vllm/.venv-qwen38/bin/vllm serve --host 127.0.0.1 --config "$1" > "$2" 2>&1 < /dev/null
