#!/bin/bash
set -u
S=/tmp/claude-1000/-shared-vllm/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/scratchpad; RD=/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85; L=/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs; P=/shared/vllm/.venv-qwen38/bin/python
sudo -n systemctl stop vllm-qwen38; sleep 5
run() { tag=$1; yaml=$2; shift 2; $S/down.sh >/dev/null; rm -f /dev/shm/vllm_offload_*.mmap
  env "$@" NCCL_P2P_LEVEL=SYS $S/up.sh $S/$yaml $L/server-$tag.log > /dev/null || { echo "[$tag] startup failed"; return 1; }
  until curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8013/health 2>/dev/null | grep -q 200; do sleep 5; pgrep -f "[.]venv-qwen38/bin/vllm serve" >/dev/null || { echo "[$tag] died"; return 1; }; done; echo "[$tag] up"; }
# 1. final latency profile (V2), no middleware for the bench client
run lat_final_r2 lat_final_v1.yaml VLLM_USE_V2_MODEL_RUNNER=1 && {
  grep -h "Draft lm_head" $L/server-lat_final_r2.log | head -1 | cut -c60-200
  $P $S/ss_bench.py 8013 > $L/ss-lat_final_r2.log 2>&1; cat $L/ss-lat_final_r2.log
  { echo -n "[lat_final_v2 37k seed61] "; $P $S/ttft_probe.py 8013 32000 61; echo -n "[lat_final_v2 90k seed62] "; $P $S/ttft_probe.py 8013 80000 62; } 2>&1 | tee $L/ttft-lat_final_v2.log
  $P $S/burst_client.py 8013 128 1024 512 2>&1 | tee $L/burst-lat_final_v2.log
  $P $S/outputs_capture.py 8013 $L/outputs-lat_final_v2.json > /dev/null 2>&1; echo outputs captured
}
# 2. baseline latency config (pre-change prod yaml minus offload), V1
run lat_base_r2 lat_base.yaml && {
  { echo -n "[lat_base_v1 37k seed61] "; $P $S/ttft_probe.py 8013 32000 61; echo -n "[lat_base_v1 90k seed62] "; $P $S/ttft_probe.py 8013 80000 62; } 2>&1 | tee $L/ttft-lat_base_v1.log
  $P $S/outputs_capture.py 8013 $L/outputs-lat_base.json > /dev/null 2>&1; echo outputs captured
  $P $S/compare_outputs.py $L/outputs-lat_base.json $L/outputs-lat_final_v2.json 2>&1 | tee $L/greedy-compare-latency.log
}
# 3. final throughput profile (DBO, offload), no middleware
run tp_final_r2 tp_final_nomw.yaml NCCL_MAX_NCHANNELS=1 && {
  { for n in 8000 16000 32000; do echo -n "[tp_final dbo n=$n] "; $P $S/ttft_probe.py 8013 $n $((70+n/1000)); done; } 2>&1 | tee $L/ttft-tp_final_dbo.log
  $P $S/outputs_capture.py 8013 $L/outputs-tp_final.json > /dev/null 2>&1; echo outputs captured
}
# 4. baseline throughput config (old max yaml, no DBO), V1
run tp_base_r2 tp_base.yaml && {
  { for n in 8000 16000 32000; do echo -n "[tp_base n=$n] "; $P $S/ttft_probe.py 8013 $n $((70+n/1000)); done; } 2>&1 | tee $L/ttft-tp_base.log
  $P $S/outputs_capture.py 8013 $L/outputs-tp_base.json > /dev/null 2>&1; echo outputs captured
  $P $S/compare_outputs.py $L/outputs-tp_base.json $L/outputs-tp_final.json 2>&1 | tee $L/greedy-compare-throughput.log
}
$S/down.sh > /dev/null; rm -f /dev/shm/vllm_offload_*.mmap
echo WAVE2 DONE
