---
active: false
status: completed
goal: >-
  Improve vLLM and the Qwen3.8-27B-FP8 deployment to achieve a 40-60% prefill
  throughput speedup on the 4x L4 box, measured against the current baseline
  (~155 s / ~1290 tok/s for a 200k-token prompt; short-context prefill baseline
  to be measured in iteration 0).
conversation_id: ea737f9a-4156-40c9-b1b8-1e900c68c3ee
run_id: 20260815T202705Z-097700e2
run_dir: goal/sessions/ea737f9a-4156-40c9-b1b8-1e900c68c3ee/runs/20260815T202705Z-097700e2
proof_kind: default
live_proof: profiler traces + before/after prefill benchmarks (200k probe latency and a 32k prefill benchmark)
resume_heading: complete-goal
iteration: 6
started_at: 2026-08-15T20:27:05Z
skip_hooks: true
loop_driver: harness-goal
baseline_200k_s: 155.0
baseline_32k: 20.1
best_200k_s: 90.4
best_32k: 10.4
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Goal Contract

* objective: 40-60% prefill throughput improvement for Qwen3.8-27B-FP8 on 4x L4 (SM89), via vLLM kernel-path fixes/tuning and/or checkpoint requantization
* physics note (honesty clause): at 200k tokens, ~40% of prefill FLOPs are BF16 attention on 16 full-attn layers (no FP8 attention path on SM89), capping GEMM-only levers near +35% there; the 40-60% target is primarily assessed on short/mid prefill (<=32k, GEMM-dominated) with the 200k number reported alongside
* measurement: (a) 200k needle-probe latency (existing harness); (b) 32k-input prefill benchmark (vllm bench serve, 16x 32k-in/1-out, measured TTFT/prefill tok/s); baselines captured in iteration 0 before any change
* levers, in evidence order: confirm actual GEMM kernel path on SM89 (Marlin W8A16 vs CUTLASS FP8 W8A8) via profiler; per-tensor FP8-dynamic requant of the BF16 checkpoint if Marlin; GDN Triton kernel autotuning for SM89 (fla chunked kernels are H100-tuned); chunked-prefill size; attention backend check for head_dim 256 on SM89
* constraints: production service on the same GPUs — experiments in bounded downtime windows, service restored to the committed config between; venv-local patches documented; no pushes upstream without user review
* stop condition: target met on the 32k metric with 200k reported; or all levers measured and exhausted below target (report honest ceiling); or user stop
* completion gate: multi-lane self-review verdict Proven-for with empty blocking findings in review-verdict.mdscript.md

## Resume Goal

* read front matter; restore iteration and baselines
* check service state (systemctl is-active vllm-qwen38) and GPU occupancy before experiments
* read tail of progress.jsonl
* [Pursue Goal](#pursue-goal)

## Pursue Goal

* iteration 0 (ground truth): capture 32k prefill baseline on the live service; torch-profile one prefill window (or infer from serve-log kernel names) to attribute time between GEMM kernels (marlin/cutlass names), GDN Triton kernels, and attention; record FLOP shares
* iteration 1+: apply single levers per round, measure both metrics, keep or revert; requant path runs offline (CPU or bounded GPU window) producing /data checkpoints
* after each iteration: append to progress.jsonl; update front matter bests
* target met -> [Complete Goal](#complete-goal); exhausted -> record ceiling analysis -> [Complete Goal](#complete-goal) with honest scope
* blocked -> [Manual Stop](#manual-stop)

## Complete Goal

* require review-verdict.mdscript.md with Proven-for and empty blocking findings (lanes: rules, completeness, eng-perf)
* set active: false, status: completed; append run_completed to progress.jsonl, goal_completed to session-log.jsonl and goal/goal-log.jsonl
* report metric history and artifact paths

## Manual Stop

* set active: false, status stopped/blocked with blocker; append goal_stopped to progress.jsonl and both logs; report

## Stop Hook Resume Command

* mdscript-exec /shared/vllm/goal/sessions/ea737f9a-4156-40c9-b1b8-1e900c68c3ee/runs/20260815T202705Z-097700e2/goal.mdscript.md#pursue-goal
