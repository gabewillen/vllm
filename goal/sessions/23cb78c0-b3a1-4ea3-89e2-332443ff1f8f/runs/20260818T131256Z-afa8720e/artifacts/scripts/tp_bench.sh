#!/bin/bash
# usage: tp_bench.sh <tag> [max_concurrency] [num_prompts] [in] [out]
S=$(cd "$(dirname "$0")" && pwd)
RD=$(cd "$S/../.." && pwd); TAG=$1; MC=${2:-}; NP=${3:-128}; IN=${4:-1024}; OUT=${5:-1024}
ARGS=""; [ -n "$MC" ] && ARGS="--max-concurrency $MC"
HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 /shared/vllm/.venv-qwen38/bin/vllm bench serve --host localhost --port 8013 --model Qwen3.8-27B --tokenizer Qwen/Qwen3.8-27B-FP8 --dataset-name random --random-input-len $IN --random-output-len $OUT --num-prompts $NP --ignore-eos $ARGS --save-result --result-dir $RD/artifacts/logs --result-filename bench-$TAG.json > $RD/artifacts/logs/bench-$TAG.log 2>&1
grep -E "Successful|Output token throughput|Mean TTFT|Median TTFT|Median TPOT|P99 TPOT|Duration|Acceptance length" $RD/artifacts/logs/bench-$TAG.log | sed "s/^/[$TAG] /"
