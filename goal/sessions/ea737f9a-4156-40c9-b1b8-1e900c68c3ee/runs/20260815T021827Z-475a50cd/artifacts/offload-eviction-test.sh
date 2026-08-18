#!/bin/bash
RD=/shared/vllm/goal/sessions/ea737f9a-4156-40c9-b1b8-1e900c68c3ee/runs/20260815T021827Z-475a50cd
P=/shared/vllm/.venv-qwen38/bin/python
probe() { $P $RD/artifacts/long_ctx_probe.py --port 8012 --target-tokens 200000 --salt "$1" 2>&1 | grep -oE "latency: [0-9.]+s|retrieved: \w+" | tr '\n' ' '; }
echo "A_COLD: $(probe A)"
echo "A_WARM: $(probe A)"
for s in B C D E F G H; do echo "FILL_$s: $(probe $s)"; done
echo "A_AFTER_EVICTION: $(probe A)"
echo "EVICTION_TEST_DONE"
