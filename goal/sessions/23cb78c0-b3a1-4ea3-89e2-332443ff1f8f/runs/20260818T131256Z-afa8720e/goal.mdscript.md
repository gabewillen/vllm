---
active: false
status: completed
goal: >-
  Starting from the verified latency and throughput profiles, push 4x L4
  Qwen3.8-27B FP8 to the hardware limit. Latency: beat 65/55/99/144 tok/s
  (reason/prose/write/edit) without worsening TTFT, quality, 262k context or
  stability. Throughput: beat 828 tok/s at 128 concurrency while also improving
  32/64-way. Measure every change against the current best; reject
  regressions; prefer fundamental fixes; stop when profiling shows the rest is
  hardware-bound or negligible.
conversation_id: 23cb78c0-b3a1-4ea3-89e2-332443ff1f8f
run_id: 20260818T131256Z-afa8720e
run_dir: goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T131256Z-afa8720e
previous_run: goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85
proof_kind: default
live_proof: required
primary_user_action: "Serve each profile on the real 4x L4 stack and benchmark (ss_bench / vllm bench serve / needle) against the current best"
resume_heading: pursue-goal
iteration: 3
started_at: 2026-08-18T13:12:56Z
completed_at: 2026-08-18T16:32:59Z
completion_reason: remaining limits hardware-bound (GEMM at HBM streaming ceiling; NCCL bound by PCIe x8 slots) or negligible
skip_hooks: true
loop_driver: harness-goal
review_loop: waived by user (2026-08-18)
baseline_latency: "prod 8012 (V2, adaptive K, int4 draft head): 65.2/54.6/98.6/144.3; 8013 no-middleware 68/59/106/143"
baseline_throughput: "c128 820.8-827.9, c64 676-680, c32 477-480 (tp_final yaml, DBO, ch1)"
best_latency: "live 8012: 66.2/64.5/105.1/160.0 (8013 no-mw ss2 mean 101); baseline 65.2/54.6/98.6/144.3"
best_throughput: "c128 1170.2, c64 912.3, c32 615.0 (TPOT 100/68/51 ms); baseline 828/676/477"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Goal Contract

* same two deliverables (serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml + vllm-qwen38; qwen3_8_27b_fp8_max.yaml + vllm-qwen38-throughput); venv patches under serve-configs/patches
* review loop waived by the user; proof = benchmarks vs current best + live check on 8012 + needle/stress; commit each kept change
* candidate levers (from profiles): in_proj_ba skinny GEMM (1.5 ms/step), target lm_head fp8 (1.1 ms, quality-measured via logprob agreement), block-fp8 GEMM configs at M=64-128, GDN state dtype/IO at batch, decode-only batch-128 profile, DBO tuning, idle gaps

## Resume Goal

* read front matter, tail progress.jsonl, systemctl is-active vllm-qwen38; [Pursue Goal](#pursue-goal)

## Pursue Goal

* iteration 0: decode-only profile at batch 128 (throughput) + block-fp8 GEMM M-sweep microbench; then levers in evidence order
* keep/reject by measurement; append progress.jsonl; commit kept changes; restore prod between windows

## Complete Goal

* set active false, status completed; append logs; report bests vs baselines

## Manual Stop

* set active false, status stopped/blocked; append goal_stopped

## Stop Hook Resume Command

* mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T131256Z-afa8720e/goal.mdscript.md#pursue-goal
