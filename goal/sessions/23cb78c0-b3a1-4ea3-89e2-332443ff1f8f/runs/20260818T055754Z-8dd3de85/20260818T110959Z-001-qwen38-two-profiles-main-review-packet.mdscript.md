---
artifact_kind: review-packet
artifact_stamp: 20260818T110959Z
subject: qwen38-two-profiles
owner_role: self-goal
review_round: 1
blocking_severities: all findings
status: open
goal: "Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles: (1) LATENCY: max single-stream decode + min TTFT preserving K7 MTP, 262k ctx, stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high concurrency preserving 262k ctx, stability, quality; profile continuously, optimize the bottleneck, benchmark + stress each change vs best, stop only when remaining gains are negligible or hardware-bound"
conversation_id: 23cb78c0-b3a1-4ea3-89e2-332443ff1f8f
run_id: 20260818T055754Z-8dd3de85
goal_mdscript: /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/goal.mdscript.md
proof_kind: default
live_proof: required
primary_user_action: "Serve each profile on the real 4x L4 stack (systemd units on port 8012 / experiment servers on 8013), run bench_single_stream / vllm bench serve / 200k needle, compare with the prior best"
proof_scope: live-proof
merge_target: master
merge_base: 68dfda8
review_head: b3c9e81
review_diff: /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-serve-configs.diff
review_diff_scope: "git diff 68dfda8 b3c9e81 -- serve-configs (product change); goal-run evidence files listed below are read-only artifacts of the same commit"
re_entry: /mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T110959Z-001-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Review This Change

* read the claim in [Claim Under Review](#claim-under-review)
* read only the paths listed in [In Scope](#in-scope)
* do not read another lane's sign-off, the author's repair narrative, or any preferred verdict
* run this lane's entrypoint and answer [Open Questions](#open-questions)
* write findings to the `{{signoff_path}}` the composer supplied

## Claim Under Review

* claim: commit b3c9e81 (repo /shared/vllm, branch master, base 68dfda8) delivers two production serving profiles for Qwen3.8-27B-FP8 on the 4x L4 box: the latency profile (serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml + systemd unit vllm-qwen38 with VLLM_USE_V2_MODEL_RUNNER=1) raises single-stream decode from 44/39/77/112 to 62-66/56/95/144 tok/s (reason/prose/code-write/code-edit, greedy) with unchanged outputs, 262k context intact (200k needle retrieved), K7 MTP kept; the throughput profile (serve-configs/qwen3_8_27b_fp8_max.yaml + unit vllm-qwen38-throughput with NCCL_MAX_NCHANNELS=1) raises 128x1k/1k aggregate from 622 to 828 tok/s at 128 concurrent (680 at 64, 480 at 32) with 262k context intact; both depend on venv-local vLLM patches serve-configs/patches/0005-*.patch and 0006-*.patch (applied to /shared/vllm/.venv-qwen38, verified to reproduce the venv state) documented in serve-configs/patches/README.md; the keepalive middleware now early-commits SSE for stream=true; live proof exists on production port 8012 for both units
* proof_scope: live-proof; merge_target: master; frozen head: b3c9e81 (merge_base 68dfda8)
* the artifact under review is the diff of serve-configs/ plus the run evidence; the venv-local vLLM source edits are represented by the two patch files (the venv is /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm, read-only for review)

## In Scope

* /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-serve-configs.diff (the product diff, serve-configs only)
* /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-stat.txt (full commit stat incl. evidence files)
* serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml
* serve-configs/qwen3_8_27b_fp8_max.yaml
* serve-configs/systemd/vllm-qwen38.service
* serve-configs/systemd/vllm-qwen38-throughput.service
* serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch
* serve-configs/patches/0006-dense-tp-dual-batch-overlap-prefill-allreduce.patch
* serve-configs/patches/README.md
* serve-configs/patches/apply-to-venv.sh
* serve-configs/middleware/vllm_keepalive.py
* /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/manifest.json (evidence index; every listed path may be read)
* /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/progress.jsonl and /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/goal.mdscript.md (run history)
* /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/ (live proof on port 8012), /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/ (bench JSON/logs), /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/scripts/, /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/configs/
* /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/ (patched vLLM sources, read-only; the *.orig0005 files next to patched files are the pre-patch originals)
* AGENTS.md, CLAUDE.md at /shared/vllm (repo rules; no .cursor/.vscode/.windsurf rule trees exist - verify)
* live services: `systemctl is-active vllm-qwen38`, `curl -s http://127.0.0.1:8012/health` (read-only checks are allowed; do NOT restart/stop services, do NOT run benchmarks that load the GPUs for more than a few requests, do NOT modify the venv)

## Proof Supplied

* /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/*-prod-8012-ss_bench.log — VLLM_API_KEY=... python artifacts/scripts/ss_bench.py 8012 against the running production unit (exit 0): 61.6/54.6/96.9/143.1 tok/s
* /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/*-prod-8012-throughput-c32.log/.json — vllm bench serve --port 8012 --max-concurrency 32 --num-prompts 128 against the running throughput unit (exit 0): 128/128 successful, 476 tok/s, TPOT 62.4 ms
* /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/ss-lat_base.log (baseline 44.4/39.2/77.4/111.6), ss-lat_final_v2_offload.log, ss2-lat_final_v2.log, needle200k-lat_final_v2.log (200k needle retrieved), needle200k-tp_final.log
* /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/bench-tp_base_c{128,64,32}.json (baseline), bench-tp_final3_c{128,64,32}.json (final: 827.9/680.2/479.6)
* /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/outputs-dbo.json (DBO output sanity), prof_*/profiler_out_0.txt (profiles)
* patch reproduction: applying 0005+0006 to the *.orig0005 originals in a temp tree reproduced the venv files byte-for-byte (cmp) — reproducible with the commands in the packet author's notes: for o in $(find vllm -name '*.orig0005') copy to temp tree; patch -p1 both; cmp each

## Proof Not Claimed

* no upstream PR; the vLLM patches are venv-local (README says candidates for PRs, human-owned per AGENTS.md)
* the throughput unit is installed but disabled (Conflicts= the latency unit); only one profile runs at a time by design; the 128-concurrent number was measured on port 8013 with the identical yaml (minus port/middleware), the live 8012 run is at 32 concurrent
* quality is asserted from spec-decode exactness (drafter-only changes) + greedy spot checks + 200k needle, not from a full eval suite
* the vllm bench client mis-parses SSE comment pings; benches were run without the middleware on 8013 (documented)
* remaining known small levers (in_proj_ba GEMM ~1 ms/step, fp8 target lm_head, DBO on V2) were measured/considered and not taken

## Open Questions

* does the diff or the run history violate AGENTS.md (no bare pip, uv/venv only, no low-value PRs, accountability wording in commits) or any other repo rule?
* are the two systemd units safe: can both start at once, does the throughput unit's env break the latency profile or vice versa, are secrets or private endpoints leaked in configs/patches/logs?
* is any claimed measurement not backed by an artifact in the manifest, or any manifest path missing on disk?
* do patches 0005/0006 introduce correctness risks the evidence does not cover (e.g. adaptive K with async scheduling, split micro-batch state slot, DP>1, non-MTP spec methods, LoRA)?
* does the middleware change alter behavior for non-streaming clients or leak request bodies?
* is the latency profile's 262k-context invariant actually preserved given gpu-memory-utilization dropped to 0.92 (KV pool 1.148M tokens vs 1.206M before)?

## Resume This Review

* run `/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T110959Z-001-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change` to enter this round's review
