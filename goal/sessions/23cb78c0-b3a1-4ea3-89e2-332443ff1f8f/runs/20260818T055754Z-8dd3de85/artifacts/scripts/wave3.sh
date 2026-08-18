#!/bin/bash
S=/tmp/claude-1000/-shared-vllm/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/scratchpad; RD=/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85; L=/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs; P=/shared/vllm/.venv-qwen38/bin/python
run() { tag=$1; yaml=$2; shift 2; $S/down.sh >/dev/null; rm -f /dev/shm/vllm_offload_*.mmap
  env "$@" NCCL_P2P_LEVEL=SYS $S/up.sh $S/$yaml $L/server-$tag.log > /dev/null || { echo "[$tag] startup failed"; return 1; }
  until curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8013/health 2>/dev/null | grep -q 200; do sleep 5; pgrep -f "[.]venv-qwen38/bin/vllm serve" >/dev/null || { echo "[$tag] died"; return 1; }; done; echo "[$tag] up"; }
sudo -n systemctl stop vllm-qwen38; sleep 5
run lat_final_r3 lat_final_v1.yaml VLLM_USE_V2_MODEL_RUNNER=1 && $P $S/logprob_agreement.py 8013 $L/logprobs-lat_final_v2.json $S/agree_texts.json
run lat_base_r3 lat_base.yaml && $P $S/logprob_agreement.py 8013 $L/logprobs-lat_base.json $S/agree_texts.json && $P $S/compare_logprobs.py $L/logprobs-lat_base.json $L/logprobs-lat_final_v2.json | tee $L/logprob-agreement-latency.log
run tp_final_r3 tp_final_nomw.yaml NCCL_MAX_NCHANNELS=1 && $P $S/logprob_agreement.py 8013 $L/logprobs-tp_final.json $S/agree_texts.json
run tp_base_r3 tp_base.yaml && $P $S/logprob_agreement.py 8013 $L/logprobs-tp_base.json $S/agree_texts.json && $P $S/compare_logprobs.py $L/logprobs-tp_base.json $L/logprobs-tp_final.json | tee $L/logprob-agreement-throughput.log
# also: same config twice? compare lat_base run vs itself is trivial; instead compare tp_base vs lat_base (both V1, no DBO; MTP on/off) as a reference for cross-config noise
$P $S/compare_logprobs.py $L/logprobs-lat_base.json $L/logprobs-tp_base.json | tee $L/logprob-agreement-reference-latbase-vs-tpbase.log
$S/down.sh > /dev/null; rm -f /dev/shm/vllm_offload_*.mmap
# restore production (new hardened unit) and re-run live proof
sudo -n cp /shared/vllm/serve-configs/systemd/vllm-qwen38.service /shared/vllm/serve-configs/systemd/vllm-qwen38-throughput.service /etc/systemd/system/ && sudo -n systemctl daemon-reload && sudo -n systemctl start vllm-qwen38
until curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8012/health 2>/dev/null | grep -q 200; do sleep 10; systemctl is-active --quiet vllm-qwen38 || { echo "prod unit died"; break; }; done; echo "prod up: $(systemctl is-active vllm-qwen38)"
echo WAVE3 DONE
