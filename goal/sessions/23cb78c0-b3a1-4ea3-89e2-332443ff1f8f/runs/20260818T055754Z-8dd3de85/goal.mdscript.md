---
active: false
status: completed
goal: >-
  Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles:
  (1) LATENCY: max single-stream decode + min TTFT, preserving K7 MTP, 262k ctx,
  stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high
  concurrency, preserving 262k ctx, stability, quality (MTP optional).
  Profile continuously, optimize current bottleneck, benchmark + stress-test each
  change vs best, stop only when remaining gains are negligible or hardware-bound.
conversation_id: 23cb78c0-b3a1-4ea3-89e2-332443ff1f8f
run_id: 20260818T055754Z-8dd3de85
run_dir: goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85
proof_kind: default
live_proof: required
primary_user_action: >-
  Serve each profile config on the real 4x L4 stack (vllm serve --config ...),
  run bench_single_stream.py (latency) and vllm bench serve 128x1k/1k + a
  long-context/stress probe (throughput), and record tok/s vs the prior best.
resume_heading: complete-goal
iteration: 9
started_at: 2026-08-18T05:57:54Z
skip_hooks: true
loop_driver: harness-goal
review_round: 2
orchestrator_model: claude-fable-5
baseline_latency: "MTP K7 + P2P + TRITON_ATTN (git 68dfda8): reason/prose/write/edit 51/42/81/115 tok/s"
baseline_throughput: "plain TP4 + P2P: 620 tok/s (128x1k/1k); MTP prod profile 424.7 (pre-P2P)"
best_latency: "V2 + adaptive K(m2) + int4 draft head + offload (gpu-mem 0.92): ss 62-68/56/95-106/143 (base 44/39/77/112); live 8012 65.3/54.7/98.7/144.4; cold TTFT 37k 17.8s (base 17.5), 90k 58.6 (57.0); 200k needle OK; burst 128/128; logprob agreement within config noise"
best_throughput: "seqs128 + DBO + TRITON + NCCL ch1 + offload: c128 828 (base 622), c64 680 (633), c32 480 (453); 200k needle OK 158s"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Goal Contract

* two deliverables: serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml (LATENCY profile, prod) and a THROUGHPUT profile (serve-configs/qwen3_8_27b_fp8_max.yaml or successor)
* metrics: latency = bench_single_stream.py 4 prompts (decode tok/s, TTFT) + cold long-prefill TTFT; throughput = vllm bench serve 128x1k/1k output tok/s at max concurrency + KV pool tokens + stability under burst
* invariants: 262k ctx, K7 MTP in latency profile, quality unchanged (greedy outputs / needle probe), no crashes under 128-burst stress
* harness /goal drives multi-round continuation; self-goal hooks skipped (loop_driver harness-goal)
* constraints: production service shares GPUs — experiments on port 8013 with prod stopped in bounded windows; restore prod between; venv patches documented under serve-configs/patches
* stop condition: all lever families measured, remaining deltas < ~3% or hardware-bound; multi-lane review Proven-for

## Resume Goal

* read front matter, tail progress.jsonl, check systemctl is-active vllm-qwen38 and nvidia-smi
* [Pursue Goal](#pursue-goal)

## Pursue Goal

* iteration 0: capture fresh baselines for both profiles on current stack (post-P2P) into artifacts/logs
* iterate: profile the bottleneck (torch profiler / nsys-lite / metrics), apply one lever, benchmark vs best, stress, keep or revert; append progress.jsonl each iteration
* levers: comm (P2P done; custom allreduce >2 GPUs, NCCL algo/proto tuning), TP/PP layouts, batching (chunk size, seqs, cudagraph sizes), KV/GDN state allocation, attention backends, kernel fusion passes, SM89 kernels
* when exhausted -> [Complete Goal](#complete-goal); blocked -> [Manual Stop](#manual-stop)

## Complete Goal

* require review-verdict Proven-for with empty blocking findings; set active false, status completed; append logs

## Manual Stop

* set active false, status stopped/blocked; append goal_stopped to progress.jsonl and both logs

## Done So Far

* iter0: baselines on 8013 (no offload): ss 44.4/39.2/77.4/111.6; profile: fp8 block GEMM 32ms/step @281GB/s (hw-bound), bf16 lm_head 20ms/step (8x636MB reads), nccl 7ms, misc; NCCL env sweep -> defaults optimal; vLLM custom allreduce 10x slower over PCIe P2P (dead end)
* iter1: venv patch: SpeculativeConfig.adaptive_draft_length (+ema alpha/margin/min) — scheduler EMA of accepted tokens picks K per step (config/speculative.py, v1/core/sched/scheduler.py, config/vllm.py) -> prose 39->48.7
* iter2: venv patch: draft_lm_head_dtype fp8|int4 (v1/spec_decode/draft_lm_head.py; wired in llm_base_proposer.py and V2 eagle/utils.py); margin sweep 1/2/3 -> 2; int4 head best: 62-63.5/53.6/89.4/136.4
* finding: draft_tensor_parallel_size ignored for MTP (eagle path uses target TP); CPU launch cost of the eager GDN op (~0.6ms/call, 48/step) + draft loop makes ranks desync (nccl wait 27%) — CPU-bound now
* iter3: V2 runner works; patched V2 to honor per-step draft count (fused draft graph set per K) + cudagraph_utils adaptive query lens (guarded to target/draft-prefill managers): 67.5/54.8/95.3/142.7
* iter4-5: throughput baseline 622/633/453 (c128/c64/c32); PP2 408 (rejected); seqs128 757 (+22%); triton=flashinfer; marlin worse; V2 plain = V1
* iter6: dense-TP DBO patch (0006): prefill 16k 7.8->5.7s with NCCL_MAX_NCHANNELS=1; found+fixed split-request mamba state slot bug (mamba_state_seq_lens)
* iter7-8: final configs validated: latency (V2, 0.92 mem, offload+keepalive) ss 62-66/56/95/144, 200k needle OK, burst OK; throughput c128 828 / c64 680 / c32 480 (offload on); keepalive middleware: SSE early-commit for stream=true; vllm bench client mis-parses SSE comment pings (bench w/o middleware)
* wrote serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml + _max.yaml, systemd units (latency: V2 env; throughput unit: NCCL_MAX_NCHANNELS=1), patches 0005/0006 + README (verified to reproduce venv state)
* review round 1: 6 lanes, none signed off (see progress.jsonl review_rejected); fix wave: config validators + DP guard, V2 propose returns drafted columns, production Marlin path + hw preconditions, pure functions + tests (serve-configs/tests, 26 pass), DBO overlap gated to dense models, middleware cap/typing/OTEL+prom counters/SSE-wrap, unit hardening, docs rationale-only, manifest venv python, scripts self-contained; re-measured (wave2/3): TTFT V2=V1, burst 128/128, greedy 2/8 identical -> claim reworded, logprob agreement 0.079/3.2% vs reference 0.092/3.5%; commits split into 6 atomic product commits + evidence

## Next Steps

* superseded: user waived the review loop and set a follow-on goal (push to hardware limit) -> new run

* review round 2 (packet + 6 lanes) on head after the fix wave; if Proven-for -> complete goal; else next fix wave
* remaining small levers noted, not taken: in_proj_ba bf16 GEMM cuBLAS pick (~1 ms/step), fp8 target lm_head (quality-touching), DBO for V2 (V2 rejects dbo)

## Stop Hook Resume Command

* mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/goal.mdscript.md#pursue-goal
