#!/bin/bash
RD=/shared/vllm/goal/sessions/ea737f9a-4156-40c9-b1b8-1e900c68c3ee/runs/20260815T021827Z-475a50cd
V=/shared/vllm/.venv-qwen38/bin
echo "PROBE_START"
$V/python $RD/artifacts/long_ctx_probe.py --port 8012 --target-tokens 200000 > $RD/artifacts/logs/long-ctx-probe-iter9.log 2>&1
PE=$?
echo "PROBE_EXIT=$PE"
[ $PE -ne 0 ] && { echo "GATE_FAILED"; exit 1; }
echo "BENCH_START"
cd /tmp/claude-1000/-shared-vllm/ea737f9a-4156-40c9-b1b8-1e900c68c3ee/scratchpad
HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 $V/vllm bench serve --host localhost --port 8012 --model Qwen3.8-27B --tokenizer Qwen/Qwen3.8-27B-FP8 --dataset-name random --random-input-len 1024 --random-output-len 1024 --num-prompts 128 --ignore-eos --save-result --result-dir $RD/artifacts/logs --result-filename bench-iter9-batch16k.json > $RD/artifacts/logs/bench-iter9-batch16k.log 2>&1
echo "BENCH_EXIT=$?"
grep -E "Output token throughput|Mean TPOT|Benchmark duration" $RD/artifacts/logs/bench-iter9-batch16k.log
echo "MEASURE_DONE"
