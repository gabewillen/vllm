---
reviewer_id: eng-core
reviewer_lane: eng-core
rules_file: /home/gwillen/.agents/skills/self-review/references/engineering-rules/core.rules.md
extra_rules_files:
  - /home/gwillen/.agents/skills/self-review/references/engineering-rules/local.rules.md
lane_applicable: true
review_round: 1
goal: "Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles: (1) LATENCY: max single-stream decode + min TTFT preserving K7 MTP, 262k ctx, stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high concurrency preserving 262k ctx, stability, quality; profile continuously, optimize the bottleneck, benchmark + stress each change vs best, stop only when remaining gains are negligible or hardware-bound"
conversation_id: 23cb78c0-b3a1-4ea3-89e2-332443ff1f8f
proof_scope: live-proof
review_head: b3c9e81
merge_base: 68dfda8
signed_off: false
verifier_summary: "Blind eng-core attack on commit b3c9e81 (serve-configs diff, patches 0005/0006 traced through the patched venv sources, middleware, yaml, systemd). Rules attacked: CORE-DET/MEM/WORK/CONC/STATE/ERR/API/GEN/BOUND/CFG/SEC/BUILD/TEST/PERF/OBS-001/002/DOC/EXC and LOCAL-ARG/CUT/GIT-001. Live checks passed (unit active, /health 200, manifest paths present, no secrets in configs/logs). Blocking: V2 adaptive-K path leaves the verify length at K with stale draft columns (single-source-of-truth breach; exactness holds only for greedy drafts; acceptance stats misreport; unvalidated K==1 / min_tokens>K crash), one commit bundling many logical changes, positional/magic-literal call sites on owned signatures, dated change history in yaml/README comments, no telemetry on new control paths."
evidence:
  - "V2 speculator propose() returns self.draft_tokens[:num_reqs] (K columns) after running only _num_steps draft steps (/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:349-402); model_runner assigns it into req_states.draft_tokens[idx_mapping] (n,1 broadcasts to n,K when _num_steps==1) and DraftTokensHandler.set_draft_tokens reports num_draft_tokens = shape[1] = K (v1/worker/gpu/model_runner.py:1806-1833, v1/worker/gpu/spec_decode/utils.py:22-52); scheduler.update_draft_token_ids stores K placeholders and schedules K+1 verify tokens next step (v1/core/sched/scheduler.py:566-570,713-724,2249-2269). num_spec_tokens_to_schedule is consumed nowhere else in the V2 runner (grep)."
  - "Rejection sampler exactness with stale drafts holds only for draft_sample_method=greedy (one-hot q, rejection_sampler_utils.py:161-188; speculator.py:132-141 allocates draft_logits only for 'probabilistic'); probabilistic mode pairs batch-position-indexed stale tokens with req-state-indexed draft_logits -> biased acceptance test. No validator rejects adaptive+probabilistic, adaptive+K==1, or adaptive_draft_min_tokens>K (config/speculative.py:188-206); init_cudagraph_manager KeyErrors on the latter two (speculator.py:146-163)."
  - "git log 68dfda8..b3c9e81 is a single commit (104 files: patch 0005 = adaptive K + quantized draft head + V2 runner support, patch 0006 = dense-TP DBO + split-request mamba slot, keepalive SSE early-commit, new systemd unit, two yaml retunes, ~95 goal-run evidence files, goal/goal-log.jsonl); no remote configured (git remote -v empty), so history is not shared."
  - "Owned signatures called positionally: QuantizedDraftLMHead(target_language_model.lm_head, draft_head_dtype) (llm_base_proposer.py:1576, gpu/spec_decode/eagle/utils.py:97), self._update_accepted_ema(req_id, num_accepted) (scheduler.py:1854); bare literals marlin_quantize(..., 128, act_order=False), max(2, ...), max(1, min(...)), .get('more_body', False)."
  - "Dated change-history blocks added to serve-configs/qwen3_8_27b_fp8_max.yaml (lines 3-21) and qwen3_8_27b_fp8_mtp_latency.yaml (lines 60-87): '2026-08-18 (goal run ...): 622 -> 814', '44/39/77/112 -> 62-66/56/95/144', 'Measured and rejected: ...'."
  - "DBO pairing checked: yield_and_switch_from_compute_to_comm / _from_comm_to_compute record+wait events on shared compute/comm streams (ubatching.py:133-147, 200-235); split-request slot: both slices gather the same align-mode slot via state_seq_lens and slice-2 continuation runs after slice-1 on the same compute stream, has_initial_state derived from num_computed (gdn_attn.py:222-227, 402-419; ubatch_utils.py:175-207) - attack failed, logic sound for the deployed dense/DP=1 case."
  - "Live: systemctl is-active vllm-qwen38 = active, vllm-qwen38-throughput = inactive (Conflicts= wiring present in both units), curl /health = 200; manifest.json 25 paths all present; no API keys/tokens in artifacts, configs, or patches (grep)."
commands_run:
  - "git log --oneline 68dfda8..b3c9e81; git show --stat --format=%B b3c9e81; git remote -v"
  - "sed/grep over review-round1-serve-configs.diff, review-round1-stat.txt"
  - "grep -n num_spec_tokens_to_schedule/draft_tokens/decode_cudagraph_manager/_num_steps across .venv-qwen38 vllm/v1/worker/gpu/{model_runner.py,spec_decode/**}, v1/core/sched/scheduler.py"
  - "sed -n over ubatching.py, ubatch_utils.py, gpu_ubatch_wrapper.py, dp_utils.py, gdn_attn.py, backends/utils.py (mamba_get_block_table_tensor), rejection_sampler_utils.py, speculator.py, logits_processor.py, interfaces.py, marlin_utils(_test).py, _custom_ops.py"
  - "grep -rIn for VLLM_API_KEY=/Bearer/sk-/hf_ over run artifacts; python manifest path existence check"
  - "systemctl is-active vllm-qwen38 vllm-qwen38-throughput; curl -s http://127.0.0.1:8012/health"
attack_attempts:
  - "CORE-STATE-001/CORE-DET-001 against V2 adaptive draft length: FIRED - drafted count (_num_steps) and reported count (shape[1]=K) disagree; verify stays K+1, stale columns verified; exact only under greedy one-hot q."
  - "CORE-SEC-001 config validation: FIRED (folded into P1) - adaptive_draft_length with num_speculative_tokens==1 or adaptive_draft_min_tokens>K raises KeyError in init_cudagraph_manager; adaptive+probabilistic unguarded."
  - "CORE-CONC-001 against DBO all-reduce yield pairing (assert stream discipline, event record/wait, shared streams across ubatches, deterministic issue order across TP ranks with DP=1): did not fire for the deployed dense case; noted MoE+DBO now yields on every TP all-reduce (unmeasured path) as P2 under CORE-PERF-001."
  - "Split micro-batch mamba/GDN state slot (state_seq_lens + num_computed_tokens adjustment, same-stream ordering of slice-1 write before slice-2 read): did not fire."
  - "Quantized draft head math/shape: fp8 per-channel scale [N,1] with B=[K,N] column-major and cutlass assert; int4 marlin_quantize [K,N] group 128, apply_gptq_marlin_linear kwargs; __getattr__ fallback to target head for shard_indices/tp_size; Qwen3_5MTP.compute_logits uses self.lm_head so the quantized copy is actually exercised: did not fire (note: depends on marlin_utils_test, a test helper module)."
  - "cudagraph_utils query-length capture guard: decode_query_len>K distinguishes target/draft-prefill managers from draft-decode; ranged qlens vs rounded num_reqs only degrade to non-full-graph at range boundaries: no correctness finding; but the extra target-manager captures are dead under V2 because verify never shrinks (folded into P1)."
  - "Middleware body peeking: buffer bounded by request body the app already reads, JSON parse only on complete body, http.disconnect passthrough, non-stream clients unchanged, no body logged: no P0/P1; silent except->False is P3 under CORE-ERR-001."
  - "CORE-SEC-001 secrets: grep artifacts/configs/patches for keys/tokens: none found; /etc/vllm/qwen38.env is root-only (permission denied on read)."
  - "CORE-WORK-001 (EMA loop bounded by running reqs, capture loops bounded by schedule x sizes), CORE-MEM-001 (_accepted_ema freed in _free_request, managers owned by speculator), CORE-CFG-001 (env-only settings belong to third-party vLLM/NCCL, not owned), CORE-BUILD-001 (patches pinned to wheel 0.27.2rc1.dev110 and byte-reproduced), LOCAL-CUT-001 (no deprecated shims/aliases retained; decode_cudagraph_manager remains the live default handle): did not fire."
  - "LOCAL-GIT-001 atomic commits: FIRED - one commit carries >=5 independent logical changes plus evidence dump."
  - "LOCAL-ARG-001 named arguments: FIRED at owned call sites and bare literals."
  - "CORE-DOC-001 change history in comments: FIRED in both yaml headers."
  - "CORE-OBS-001/002 telemetry: FIRED - no telemetry on new control paths (adaptive K chosen, DBO local micro-batch decision, SSE early commit); no new OTEL dimensions so OBS-002 cardinality is n/a."
p_findings:
  - grade: P1
    location: "/shared/vllm/serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch (v1/worker/gpu/spec_decode/autoregressive/speculator.py propose(): 'return self.draft_tokens[:num_reqs]' after running _num_steps steps; v1/worker/gpu/model_runner.py:1806-1833; config/speculative.py adaptive_* fields; v1/worker/gpu/cudagraph_utils.py adaptive qlens)"
    summary: "V2 adaptive draft length halves draft compute but not the verify: propose returns the full K-wide buffer with stale columns (or an (n,1) tensor that broadcasts one token into all K columns), the handler reports K drafts, the scheduler stores K placeholders and schedules K+1 verify tokens; the extra target-manager cudagraph captures for K_adaptive+1 are dead memory (part of the 1.59 GiB that forced gpu-memory-utilization 0.92); spec-decode acceptance stats count garbage drafts; output exactness relies on draft_sample_method=greedy (one-hot q) and is biased for probabilistic drafts; adaptive with num_speculative_tokens==1 or adaptive_draft_min_tokens>K raises KeyError at init. Packet/README claim 'V2 speculator honors it' is only half true."
    contract: "CORE-STATE-001 MUST Single Source Of Truth (drafted count vs reported count); CORE-SEC-001 MUST Validate Untrusted Input (config cross-field validation)"
    remediation: "Return draft_tokens[:num_reqs, :self._num_steps]; write only those columns into req_states.draft_tokens and pass the narrowed tensor to set_draft_tokens so the scheduler schedules _num_steps drafts (then the ranged target-manager captures become live); add a SpeculativeConfig validator rejecting adaptive_draft_length with num_speculative_tokens<2, adaptive_draft_min_tokens>num_speculative_tokens, and (until stale-column handling is fixed) draft_sample_method=probabilistic; correct README/yaml wording; re-measure the latency profile and KV pool after the fix."
  - grade: P1
    location: "git commit b3c9e81 (104 files) on /shared/vllm master"
    summary: "One commit bundles independent logical changes: patch 0005 (itself three features: adaptive K, quantized draft head, V2 runner support), patch 0006 (dense-TP DBO + split-request mamba slot), keepalive SSE early-commit, new throughput systemd unit + latency unit env, two yaml retunes, and ~95 goal-run evidence files + goal/goal-log.jsonl; subject names two outcomes, not one change."
    contract: "LOCAL-GIT-001 MUST Atomic Commits"
    remediation: "History is unpushed (no remote), so split into per-change commits (adaptive-K/head patch, DBO patch, middleware, units+yaml, evidence) with subjects stating each change; keep evidence dumps out of product commits going forward."
  - grade: P1
    location: "patch 0005: llm_base_proposer.py:1576 and gpu/spec_decode/eagle/utils.py:97 QuantizedDraftLMHead(target_lm_head, dtype); scheduler.py:1854 self._update_accepted_ema(req_id, num_accepted); draft_lm_head.py marlin_quantize(..., 128, act_order=False); speculator.py max(2, ...), max(1, min(...)); middleware .get('more_body', False)"
    summary: "Owned multi-argument signatures are called positionally (including a string-enum dtype), and bare magic literals (group size 128, floor 2, floor 1, default False) are passed to builtin/third-party calls without a named binding."
    contract: "LOCAL-ARG-001 MUST Named Arguments At Call Sites"
    remediation: "Call QuantizedDraftLMHead(target_head=..., dtype=...) and _update_accepted_ema(req_id=..., num_accepted=...); bind MARLIN_GROUP_SIZE = 128, MIN_FUSED_DRAFT_STEPS = 2, MIN_DRAFT_STEPS = 1 and a named default for more_body."
  - grade: P1
    location: "/shared/vllm/serve-configs/qwen3_8_27b_fp8_max.yaml lines 3-21; /shared/vllm/serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml lines 60-87; serve-configs/patches/README.md 'as of 2026-08-18'"
    summary: "Comments carry dated change history (before->after numbers, run ids, rejected experiments) instead of only the contract that governs the config now."
    contract: "CORE-DOC-001 MUST NOT Keep Change History In Comments And Docs"
    remediation: "Keep the rationale per setting (why 128 seqs, why NCCL_MAX_NCHANNELS=1, why 0.92) and move dated before/after narrative and rejected-experiment lists to the commit message / goal-run artifacts."
  - grade: P1
    location: "patch 0005 scheduler._adaptive_num_spec_tokens / _update_accepted_ema; patch 0006 gpu_model_runner DP=1 should_ubatch branch and parallel_state._all_reduce_out_place DBO branch; serve-configs/middleware/vllm_keepalive.py early SSE commit"
    summary: "New control paths (chosen draft K per step, local micro-batch decision, SSE-vs-JSON early commit) emit no telemetry; the only observable signal, spec-decode acceptance stats, is distorted under V2 adaptive (see finding 1). No packet-carried CORE-EXC-001 exception."
    contract: "CORE-OBS-001 MUST OpenTelemetry Telemetry"
    remediation: "Surface chosen K and ubatch decisions through vLLM's existing stats/metrics path (SpecDecodingStats / prometheus-OTEL exporter) with bounded labels, count early commits by content type in the middleware, or record a documented exception with owner and expiry."
  - grade: P2
    location: "patch 0006 vllm/distributed/parallel_state.py _all_reduce_out_place"
    summary: "The yield-around-all-reduce is keyed on the global dbo_enabled(), so every MoE+DBO deployment now also yields on each TP all-reduce (o_proj/down_proj) - an unmeasured schedule change; the assert current_stream()==compute_stream would fire for any all-reduce issued while a ubatch is on its comm stream."
    contract: "CORE-PERF-001 MUST Measure Performance Claims (optimization decisions justified by benchmarks)"
    remediation: "Gate the branch on the dense-TP DBO configuration (e.g. not model_config.is_moe or an explicit parallel_config flag) or benchmark an MoE+DBO model before generalizing."
  - grade: P3
    location: "/shared/vllm/serve-configs/middleware/vllm_keepalive.py _receive except Exception: self.expect_sse = False"
    summary: "Body-peek failures are swallowed silently; a non-JSON or oversized body falls back to JSON commit with no trace."
    contract: "CORE-ERR-001 MUST Explicit Failure Handling"
    remediation: "Catch (ValueError, AttributeError, UnicodeDecodeError) and logger.debug the fallback reason."
rules_reviewed:
  - /home/gwillen/.agents/skills/self-review/references/engineering-rules/core.rules.md
  - /home/gwillen/.agents/skills/self-review/references/engineering-rules/local.rules.md
objectives_checked: [CORE-DET-001, CORE-MEM-001, CORE-WORK-001, CORE-CONC-001, CORE-STATE-001, CORE-ERR-001, CORE-API-001, CORE-GEN-001, CORE-BOUND-001, CORE-CFG-001, CORE-SEC-001, CORE-BUILD-001, CORE-TEST-001, CORE-PERF-001, CORE-OBS-001, CORE-OBS-002, CORE-DOC-001, CORE-EXC-001, LOCAL-ARG-001, LOCAL-CUT-001, LOCAL-GIT-001]
artifact_paths:
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T110959Z-001-qwen38-two-profiles-main-review-packet.mdscript.md
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-serve-configs.diff
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-stat.txt
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/manifest.json
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/
  - /shared/vllm/serve-configs/middleware/vllm_keepalive.py
  - /shared/vllm/serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch
  - /shared/vllm/serve-configs/patches/0006-dense-tp-dual-batch-overlap-prefill-allreduce.patch
  - /shared/vllm/serve-configs/patches/README.md
  - /shared/vllm/serve-configs/qwen3_8_27b_fp8_max.yaml
  - /shared/vllm/serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml
  - /shared/vllm/serve-configs/systemd/vllm-qwen38.service
  - /shared/vllm/serve-configs/systemd/vllm-qwen38-throughput.service
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/speculator.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/ubatching.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/ubatch_utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu_ubatch_wrapper.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/dp_utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/attention/backends/gdn_attn.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/attention/backends/utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/spec_decode/llm_base_proposer.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/eagle/utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/model_executor/models/interfaces.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_test.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/marlin_utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/_custom_ops.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/config/speculative.py
remaining_gaps:
  - "No automated tests accompany patches 0005/0006 (adaptive-K scheduling, quantized head equivalence, split-request mamba slot); quality is asserted from spec-decode exactness plus greedy spot checks and one needle probe."
  - "MoE+DBO and DP>1 behavior of the generalized all-reduce yield is unmeasured on this box."
  - "The 128-concurrent throughput number is from port 8013 with the middleware off; the live 8012 proof is at 32 concurrent (packet acknowledges)."
signed_off_at: 2026-08-18T11:40:00Z
repair_resume_command: "/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T110959Z-001-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Signoff

* lane `eng-core` verdict: `signed_off: false` against `/home/gwillen/.agents/skills/self-review/references/engineering-rules/core.rules.md` (+ `local.rules.md`) for commit b3c9e81 (base 68dfda8)
* P1 CORE-STATE-001 / CORE-SEC-001 — patch 0005 V2 speculator `propose()` returns the K-wide `draft_tokens` buffer after `_num_steps` steps; the handler and scheduler see K drafts, verify never shrinks, stale columns are verified, extra target-manager graphs are dead, acceptance stats misreport, exactness depends on greedy drafts, and adaptive with K==1 or min_tokens>K crashes at init. Remediation: return/write only `[:num_reqs, :_num_steps]`, pass the narrowed tensor to `set_draft_tokens`, add cross-field config validation, re-measure.
* P1 LOCAL-GIT-001 — commit b3c9e81 bundles >=5 logical changes plus ~95 evidence files. Remediation: split into per-change commits (history is unpushed).
* P1 LOCAL-ARG-001 — positional calls on owned signatures (`QuantizedDraftLMHead(head, dtype)` x2, `_update_accepted_ema(req_id, n)`) and bare literals (128, 2, 1, False). Remediation: keyword arguments and named constants.
* P1 CORE-DOC-001 — dated change-history blocks in both yaml headers and README. Remediation: keep per-setting rationale, move history to commits/run artifacts.
* P1 CORE-OBS-001 — no telemetry on adaptive-K choice, local ubatch decision, SSE early commit; the one existing signal is distorted. Remediation: emit via vLLM stats/metrics with bounded labels or record a CORE-EXC-001 exception.
* P2 CORE-PERF-001 — patch 0006 all-reduce yield keyed on global `dbo_enabled()` changes MoE+DBO schedules unmeasured. Remediation: gate on dense-TP config or benchmark MoE.
* P3 CORE-ERR-001 — middleware `except Exception` swallows body-peek failures. Remediation: narrow the except and log the fallback.

## Resume From Signoff

* `signed_off` is `false`: the next jump is `repair_resume_command` — `/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T110959Z-001-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change` after repair
* after repair, a fresh blind reviewer must re-run this lane from a new packet; never re-enter this lane's own review from this sign-off
* when a later round reaches `signed_off: true`, continue at `/mdscript-exec /home/gwillen/.agents/skills/self-review/workflows/triple-adversarial-blind-review.mdscript.md#aggregate-triple-signoffs`
