---
active: false
status: completed
goal: >-
  Serve Qwen/Qwen3.8-27B (FP8) on the 4x L4 box via vLLM with the full 262144-token
  context enabled, maximize serving throughput, and optimize until diminishing
  returns (<5% primary-metric gain over two consecutive optimization rounds).
conversation_id: ea737f9a-4156-40c9-b1b8-1e900c68c3ee
run_id: 20260815T021827Z-475a50cd
run_dir: goal/sessions/ea737f9a-4156-40c9-b1b8-1e900c68c3ee/runs/20260815T021827Z-475a50cd
proof_kind: default
live_proof: server startup log showing max-model-len 262144 + benchmark result JSON/logs per iteration
resume_heading: complete-goal
iteration: 9
started_at: 2026-08-15T02:18:27Z
skip_hooks: true
loop_driver: harness-goal
primary_metric: aggregate output tok/s, vllm bench serve, 128 prompts x 1k-in/1k-out, and a 200k-token-prompt long-context probe must still pass
baseline_tok_s: 42.55
best_tok_s: 474.61
best_config: artifacts/qwen3_8_27b_fp8_iter8a_seqs96.yaml
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Goal Contract

* objective: max-throughput vLLM serving of Qwen/Qwen3.8-27B-FP8 on 4x NVIDIA L4 (23 GiB, SM89, PCIe) with `max-model-len: 262144` accepted at startup and usable end to end
* success: benchmark-proven throughput plateau — two consecutive optimization rounds each improving the primary metric by <5% — with the winning config committed to `serve-configs/` and proof logs under `artifacts/logs/`
* primary metric: aggregate output tok/s from `vllm bench serve` (128 prompts, 1024-in/1024-out, concurrency swept); secondary gate: a 200k-token-prompt request completes successfully (262k capability is real, not just configured)
* proof_kind: default — logs sufficient (server startup logs, benchmark JSON, progress.jsonl)
* constraints: FP8 checkpoint (30.9 GB) needs >=2 GPUs per replica; candidate topologies TP4, TP2x2 replicas (throughput comparison is part of the loop; 262k must remain servable in the winning topology)
* engine base decision pending: upstream-fix research lane (GDN+MTP fixes between 0.23 base and 0.27.2 nightly) feeds venv strategy — fresh nightly venv vs local-tree editable install
* allowed: create venvs, download checkpoints to /data/huggingface, run/kill local vllm servers on spare ports, write serve-configs/*.yaml and benchmark scripts, append to run logs
* forbidden: pushing/publishing, modifying prior runs, touching other conversations' session dirs, deleting checkpoints
* stop condition: diminishing returns rule above, or user stop, or an unresolvable blocker (e.g. SM89 GDN kernel segfault with no workaround)

## Resume Goal

* read front matter; restore `{{iteration}}`, `{{best_tok_s}}`, `{{best_config}}`
* check for a running vllm server (`pgrep -af "vllm serve"`) and running background downloads before starting new ones
* read the tail of `progress.jsonl` for the last completed step
* [Pursue Goal](#pursue-goal)

## Pursue Goal

* iteration 0 (bring-up): ensure Qwen/Qwen3.8-27B-FP8 downloaded to /data/huggingface; decide engine base from upstream-fix research; create `.venv-qwen38`; first server smoke at TP2, `enforce-eager`, `max-model-len 262144`, small `max-num-seqs`; verify 200k-prompt probe passes (SM89 GDN Triton risk: triton#9939)
* iteration 1 (baseline): run primary-metric benchmark on the smoke config; record `baseline_tok_s` in front matter and progress.jsonl
* iterations 2+ (optimize, one lever per round, measure, keep or revert; user-directed order 2026-08-15: iter3 = TP2x2 replicas aggregate, iter4 = TP4, then remaining levers):
  * topology: TP4 single replica vs TP2x2 replicas (router = round-robin nginx or client-side); PP fallback only if needed
  * CUDA graphs on (drop enforce-eager), tune `max-num-batched-tokens` (4096/8192/16384), `max-num-seqs` vs GDN float32 state pool, `--kv-cache-dtype fp8`, prefix caching, async scheduling
  * MTP speculative decoding (`{"method":"mtp","num_speculative_tokens":2-3}`) — needs the GDN spec fix; measure at low and high concurrency separately
  * long-context gate re-checked after every kept change
* after each iteration: append `{"event":"iteration", "iter":N, "config":..., "tok_s":..., "kept":bool}` to progress.jsonl; update front matter `iteration`, `best_tok_s`, `best_config`
* if two consecutive rounds gain <5%: write winning config to `serve-configs/qwen3_8_27b_fp8_max.yaml`, copy proof logs to `artifacts/logs/`, [Complete Goal](#complete-goal)
* if blocked (kernel crash, OOM floor, unfixable): record blocker, [Manual Stop](#manual-stop)

## Complete Goal

* require multi-lane self-review verdict `Proven for` with empty blocking findings in `review-verdict.mdscript.md` (lanes: rules, security, completeness, eng-python/eng-config as applicable)
* set front matter `active: false`, `status: completed`, `resume_heading: complete-goal`
* append `run_completed` to progress.jsonl; `goal_completed` to session-log.jsonl and goal/goal-log.jsonl
* report winning config, metric history, and artifact paths

## Manual Stop

* set front matter `active: false`, `status: stopped` or `blocked` with blocker summary, `resume_heading: manual-stop`
* append `goal_stopped` with blocker to progress.jsonl and both logs
* report progress and blocker

## Stop Hook Resume Command

* mdscript-exec /shared/vllm/goal/sessions/ea737f9a-4156-40c9-b1b8-1e900c68c3ee/runs/20260815T021827Z-475a50cd/goal.mdscript.md#pursue-goal
