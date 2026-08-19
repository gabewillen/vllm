# vLLM patches for the Qwen3.8-27B deployment

**These patches are now commits on this branch.** `qwen3.8-27B` is upstream
`acb0f1dcd` (`vllm 0.27.2rc1.dev110+gacb0f1dcd`) with the nine changes below
applied in order, so the deployment installs directly:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install git+https://github.com/gabewillen/vllm@qwen3.8-27B
```

No venv-local patching and no lane branches are needed. The patch files and
`apply-to-venv.sh` are kept for reference and for porting onto a different
upstream base.

| Commit | Patch |
|---|---|
| `1916ceabb` | 0001 kv-offload flock-serialized, row-aligned chunked cudaHostRegister |
| `587ba5725` | 0002 kv-offload chunked host unregister on cleanup |
| `8ba5662cf` | 0003 offload connector detects the MTP draft group by name |
| `fa5f18714` | 0004 draft online quant tolerates a callable hf_overrides |
| `4f69ab1ef` | 0005 adaptive draft length + quantized draft lm_head + V2 runner support |
| `6195d9351` | 0006 dense-TP dual batch overlap for the prefill all-reduce |
| `084ccc5b9` | 0007 GDN in_proj_ba skinny GEMM |
| `4b3a669a2` | 0008 qwen3_5 quantized target lm_head |
| `5a083a845` | 0009 dynamic reasoning effort - telemetry, V2 budget updates, controller |
| `5e67ad3e4` | l4-configs: L4-tuned block-fp8 GEMM configs for the Qwen3.8-27B TP4 shapes |

The rest of this file documents what each change does and why.

---


None of these nine changes is in any vLLM release. Installing this branch
carries all of them. If you instead run a stock wheel in a venv, apply them
with `apply-to-venv.sh` after every venv rebuild, which silently drops them.

| Patch | Fixes | Symptom without it |
|---|---|---|
| 0001 gpu_worker.py | flock-serialized, row-stride-aligned chunked `cudaHostRegister` of the /dev/shm offload region | startup deadlock with `cpu_bytes_to_use` >= ~16 GiB at TP4 (all ranks pin the whole region concurrently); naive 1 GiB chunking then breaks `cuMemcpyBatchAsync` (error 1) |
| 0002 shared_offload_region.py | unregister the same chunks on cleanup | leaked pinned mappings / errors at shutdown after 0001 |
| 0003 offloading/scheduler.py | detect the MTP draft tower by `mtp.` layer prefix instead of marking every KV group EAGLE-volatile | with `speculative-config` MTP enabled, tiered-offload restore never converges (fs tier hits, promotes, then full-recompute); resume of a 32k chat 20 s instead of 2-3 s |
| 0005 config/speculative.py, config/vllm.py, v1/core/sched/scheduler.py, v1/spec_decode/{draft_lm_head.py (new), llm_base_proposer.py}, v1/worker/gpu/{cudagraph_utils.py, model_runner.py, spec_decode/{speculator,eagle/utils,autoregressive/speculator,dflash/speculator,multi_module_mtp/speculator}.py} | speculative-config `adaptive_draft_length` (+`adaptive_draft_ema_alpha`/`_margin`/`_min_tokens`, validated: K >= 2, min <= K, disabled with a warning under DP > 1): scheduler EMA of accepted draft tokens picks the per-step draft count; V1 drafts that many; the V2 speculator runs that many steps (one fused draft graph set per count) and returns only the drafted columns so the scheduler verifies exactly those; cudagraph_utils captures the adaptive query lengths only for the batch-size range that can use them and only for managers whose decode query covers the verify tokens. `draft_lm_head_dtype: fp8|int4`: quantized copy of the shared target lm_head (CUTLASS fp8 per-channel / Marlin int4 group-128 via the production GPTQ repack path) for the drafter's argmax only; target head untouched; hardware support checked at load | latency profile: single-stream 44/39/77/112 -> 62-68/55-56/95-106/143 tok/s (reason/prose/write/edit); target distribution untouched (per-token logprob agreement with the previous config: mean |dlogprob| 0.079, top-1 disagreement 3.2%, vs 0.092 / 3.5% between the two pre-existing prod configs). Without it: every step drafts K=7 (each draft step re-reads the 636 MB bf16 lm_head shard) and V2 + dynamic SD divides by zero building draft-decode graphs. Observability: mean draft length = spec_decode_num_draft_tokens_total / spec_decode_num_drafts_total |
| 0006 config/vllm.py, distributed/parallel_state.py, v1/worker/{gpu_model_runner.py, gpu_ubatch_wrapper.py, ubatch_utils.py, ubatching.py}, v1/attention/backend.py, v1/attention/backends/{gdn_attn,mamba_attn,linear_attn}.py | dense-TP dual batch overlap: `--enable-dbo` on a non-MoE model (all2all assert relaxed for dense models), micro-batching decided locally when DP=1, `dp_metadata` may be None, and - only for dense models (`ubatching.set_overlap_tp_all_reduce`, MoE keeps its upstream schedule) - every TP all-reduce yields to the other micro-batch while it runs on the comm stream; a request split across micro-batches keeps ONE mamba/GDN state slot (`CommonAttentionMetadata.mamba_state_seq_lens`). No per-step ubatch counter exists in the model runner (documented exception: owner gwillen, revisit when vLLM adds runner-level OTEL metrics; the effect is visible in prefill throughput / TTFT metrics) | throughput profile (with `NCCL_MAX_NCHANNELS=1`, offload on): cold prefill 9k 3.65 -> 2.88 s, 16k 6.85 -> 5.97 s, 37k 14.4 -> 13.6 s vs the previous config; 128x1k/1k 757 -> 828 tok/s at 128 concurrent. Without the state-slot fix a split prompt returns garbage/EOS (the second half read the pre-copied old state) |
| 0007 model_executor/layers/mamba/gdn/{qwen_gdn_linear_attn.py, skinny_linear.py (new)} | GDN `in_proj_ba` routed through a split-K Triton kernel (`torch.ops.vllm.gdn_skinny_linear`) for 2..16-row decode batches; cuBLAS keeps M==1 and M>16 | cuBLAS picks a 27 us kernel for the bf16 [8 x 5120] x [5120 x 24] gate projection on Ada (4 us at M=1/32); 48 layers -> ~1 ms per decode step (-2..3% step time on the latency profile) |
| 0008 model_executor/models/qwen3_5.py | `--additional-config '{"lm_head_dtype": "fp8", "lm_head_free_bf16": false}'`: the target model serves logits from a per-channel fp8 copy of lm_head (same `QuantizedDraftLMHead` as 0005); `lm_head_free_bf16: true` releases the bf16 shard (only when no drafter shares it - the throughput profile) | -1 ms per verify step; per-token logprob agreement with the bf16 head 0.035 mean / 2.5% top-1 (below the 0.076 / 3.2% kernel-path noise floor); needle 200k OK |
| 0009 config/reasoning.py, entrypoints/openai/chat_completion/{dynamic_effort.py (new), protocol.py, serving.py}, outputs.py, v1/core/sched/{effort_controller.py (new), scheduler.py, output.py}, v1/engine/{__init__.py, output_processor.py}, v1/metrics/{loggers.py, stats.py}, v1/outputs.py, v1/sample/{effort_signals.py (new), sampler.py, metadata.py, rejection_sampler.py, thinking_budget_state.py}, v1/worker/{gpu_input_batch.py, gpu_model_runner.py}, v1/worker/gpu/{async_utils.py, model_runner.py, sample/{effort.py (new), output.py, sampler.py, thinking_budget.py}, spec_decode/rejection_sampler.py} | Dynamic reasoning effort (`docs/dynamic-reasoning.claude.md`): (a) P0 telemetry — per-request entropy (÷log V) and top1-top2 margin at the canonical stage (after penalties, before budget force/temperature/top-k/p), committed rows only, both runners, opt-in per request via `vllm_xargs.effort_telemetry`, JSONL sink `VLLM_EFFORT_TELEMETRY=/path`; zero work when no request is flagged. (b) Versioned thinking-budget updates `SchedulerOutput.thinking_budget_updates` (req -> (revision, budget)) applied by both runners with acks in `ModelRunnerOutput.thinking_budget_acks`; V2 fix: greedy budgeted requests were never forced (`_requires_logits_processing` ignored the budget); CPU torch reference of the V2 Triton budget kernels. (c) `reasoning_effort: "dynamic"` — rendered as template `medium` + the Qwen `low` sentence appended to the last user turn (block 0 stays effort-free, §2b), `thinking_token_budget = ladder[0]`, scheduler-side controller (`--reasoning-config '{"dynamic_effort": {...}}'`: ladder 1k/4k/16k/64k, checks at 75/90 % of the cap, entropy/margin/trend/MTP score vs θ per rung, stall clamp, batch-size rung cap, max_tokens headroom, deadline), `effort` object on the chat response, Prometheus `vllm:effort_*` | Without it: no per-request uncertainty telemetry, no mid-request budget raise, `dynamic` is rejected by the template. GPU-validated 2026-08-18 on the latency profile (static caps force at exactly N tokens under MTP K=7; `dynamic` escalates 0->1 on an open-ended prompt and stays capped on a confident grind; acks land, no `late`; Triton-vs-torch budget kernels 300/300 identical; docs §2c); CPU tests: `serve-configs/tests/test_effort_telemetry.py`, `test_v2_thinking_budget.py`, `test_effort_controller.py` (124 total pass) |
| l4-configs/*.json (data, copied by apply-to-venv.sh) | L4-tuned Triton block-fp8 GEMM configs for the five per-rank TP4 shapes of Qwen3.8-27B (N,K = 4096/8704/3584x5120, 5120x4352, 5120x1536), keyed by M; from an HBM-realistic sweep (weights rotated past the 48 MB L2) | vLLM has no NVIDIA_L4 entries and falls back to BLOCK_M=64/N=128 for every M; the tuned tiles are +5-18% per GEMM (M<=32 and 96-128), measured in-model: single-stream 68/59/106/143 -> 72/63/112/150 tok/s |
| 0004 model_loader/weight_utils.py | `get_quant_config` tolerates a callable `hf_overrides` (draft models compose one) instead of raising | `speculative-config` with `"quantization"` on a dspark/eagle draft crashes at load with "hf_overrides must be a dict" (note: online-fp8 for the DSpark drafter still fails later in the DFlash precompute path on SM89 — Marlin-repacked weight; keep the drafter bf16) |

Evidence, repros and benchmarks: `goal/sessions/ea737f9a-4156-40c9-b1b8-1e900c68c3ee/runs/20260815T021827Z-475a50cd/artifacts/offload-debug/`
(0001-0004) and `goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/`
(0005-0006: progress.jsonl, artifacts/logs/ss-*.log, bench-*.json, prof_*/).
Tests for the CPU-checkable parts: `serve-configs/tests/` (see its README).
All are unreported upstream as of 2026-08-18 and are candidates for PRs
(human-owned per AGENTS.md). 0005/0006 are the most upstream-worthy: adaptive
draft length + quantized draft head are model-agnostic MTP/EAGLE wins on
memory-bound GPUs; dense-TP DBO generalizes DBO beyond MoE all2all.
