#!/bin/bash
# usage: sgl_launch.sh <log> [extra args...]
LOG=$1; shift
cd /shared/vllm
HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 exec .venv-sglang/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B-FP8 --served-model-name Qwen3.8-27B --tp 4 --port 8013 --host 127.0.0.1 \
  --context-length 262144 --mem-fraction-static 0.88 --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --max-running-requests 96 \
  "$@" > "$LOG" 2>&1 < /dev/null
