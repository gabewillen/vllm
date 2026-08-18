---
reviewer_id: eng-dbc
reviewer_lane: eng-dbc
rules_file: /home/gwillen/.agents/skills/self-review/references/engineering-rules/dbc.rules.md
lane_applicable: true
review_round: 1
goal: "Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles: (1) LATENCY: max single-stream decode + min TTFT preserving K7 MTP, 262k ctx, stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high concurrency preserving 262k ctx, stability, quality; profile continuously, optimize the bottleneck, benchmark + stress each change vs best, stop only when remaining gains are negligible or hardware-bound"
conversation_id: 23cb78c0-b3a1-4ea3-89e2-332443ff1f8f
run_id: 20260818T055754Z-8dd3de85
review_head: b3c9e81
merge_base: 68dfda8
proof_scope: live-proof
signed_off: false
verifier_summary: "Blind DBC-rules attack on patches 0005/0006 (venv vLLM), the keepalive middleware and the two yaml/systemd profiles. Verified: pydantic range validation of adaptive_draft_ema_alpha/margin/min_tokens and the draft_lm_head_dtype Literal fires; scheduler->async-scheduler->V2 runner->speculator num_spec_tokens_to_schedule contract is consistent (placeholders = num_spec_tokens_to_schedule, runner verifies scheduler-side counts, extra drafts ignored); mamba_state_seq_lens reaches every mamba_get_block_table_tensor caller and GDN derives has_initial_state on-device so split micro-batches share one slot; middleware early-commit contract behaves as its docstring says for stream=true, stream=false, non-JSON and chunked bodies. Standing MUST breaches: adaptive_draft_length precondition (K>1, min_tokens<=K) is not validated and crashes the V2 speculator with a KeyError; adaptive is not covered by the DP>1 dynamic-SD guard; no contract tests ship with the new public config fields / metadata field / middleware behavior."
evidence:
  - "probe_cfg.py (scratchpad, .venv-qwen38 python): alpha=1.5 -> ValidationError, alpha=1.0 -> ValidationError, margin=-1 -> ValidationError, min_tokens=0 -> ValidationError, draft_lm_head_dtype=bogus -> ValidationError; BUT adaptive_draft_length=True with num_speculative_tokens=1 -> OK and adaptive_draft_min_tokens=99 with K=3 -> OK (accepted silently)"
  - "/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:146-162: _draft_step_options() returns range(max(2,min_tokens), K+1); for K=1 or min_tokens>K the range is empty and self.decode_cudagraph_managers[self.num_speculative_steps] raises KeyError at load_model (internal defect instead of validation failure)"
  - "/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/config/vllm.py:932-947: _maybe_disable_dynamic_sd_for_data_parallel() tests only uses_dynamic_speculative_decoding(); adaptive_draft_length is not disabled for data_parallel_size>1 although it has the identical per-rank K divergence property (the patch did extend the sibling _maybe_override_dynamic_sd_cudagraph_mode at :908-916)"
  - "v1/core/sched/async_scheduler.py:22-25 sets placeholders = [-1]*scheduler_output.num_spec_tokens_to_schedule; v1/worker/gpu/model_runner.py:1116-1137 sizes verification from scheduler_output.scheduled_spec_decode_tokens; speculator.propose clamps _num_steps to [1,K]: contract consistent for K in 1..7. num_spec_tokens_to_schedule==0 is passed as None (model_runner.py:1819) so the V2 drafter still runs the full K chain for batches in the [33,96,0] range (wasted work, not incorrect)"
  - "grep of mamba_get_block_table_tensor( callers: linear_attn.py:72,197, gdn_attn.py:223, mamba_attn.py:527,813 all use state_seq_lens; gdn_attn.py:402 has_initial_state = compute_num_computed_tokens() (seq_lens - query_lens, on device) so the split-first slice sees initial state; ubatch_utils.py:180-206,257 threads mamba_state_seq_lens and bumps _num_computed_tokens_cpu; backend.py:640-664 unpadded() drops the field (spec-decode path only, not used under DBO)"
  - "probe_mw.py (scratchpad): stream=true slow app -> early 200 text/event-stream + ': keepalive' then app body; stream=false / non-JSON slow -> 200 application/json + '\n'; fast stream app -> untouched start; chunked body (2 http.request messages) parsed correctly; stream=true + app 400 JSON -> 200 text/event-stream with JSON body (logged warning, matches docstring)"
  - "systemctl is-active: vllm-qwen38 active, vllm-qwen38-throughput inactive; curl 127.0.0.1:8012/health -> 200 (read-only checks only)"
commands_run:
  - "sed/cat over review-round1-serve-configs.diff, review-round1-stat.txt, packet, dbc.rules.md, engineering-rules.mdscript.md"
  - "grep -n num_spec_tokens_to_schedule / spec_token_ids / num_output_placeholders in vllm/v1/core/sched/{scheduler,async_scheduler,output}.py and vllm/v1/worker/gpu/model_runner.py"
  - "grep -rn 'mamba_get_block_table_tensor(' and 'state_seq_lens' in vllm/v1; grep -rn 'CommonAttentionMetadata(' in vllm/v1"
  - "cd scratchpad && timeout 300 /shared/vllm/.venv-qwen38/bin/python probe_cfg.py (SpeculativeConfig validation probe)"
  - "cd scratchpad && timeout 120 /shared/vllm/.venv-qwen38/bin/python probe_mw.py (ASGI middleware early-commit probe)"
  - "systemctl is-active vllm-qwen38 vllm-qwen38-throughput; curl -s -m 5 http://127.0.0.1:8012/health"
attack_attempts:
  - "Range validation of adaptive_draft_ema_alpha/margin/min_tokens and draft_lm_head_dtype: attacked with out-of-range and bogus values -> pydantic rejects all (DBC-PRE-003 did NOT fire for ranges)"
  - "Cross-field precondition K>1 / min_tokens<=K for adaptive_draft_length: attacked with K=1 and min_tokens=99 -> accepted; traced to KeyError in speculator.load_model (DBC-PRE-003 / DBC-INV-002 FIRED)"
  - "DP>1 with adaptive_draft_length: read _maybe_disable_dynamic_sd_for_data_parallel -> adaptive not guarded (DBC-PRE-001 / DBC-CONFIG-001 FIRED)"
  - "Async-scheduling placeholder count vs drafted count (V2): placeholders = num_spec_tokens_to_schedule, drafter clamps to it, verify count from scheduler; sync-mode V2 reports full-K [-1] placeholders (get_draft_tokens) so stale columns are verified but rejection sampling stays exact -> no correctness breach (did not fire; perf note for schedule value 0)"
  - "cudagraph_utils adaptive query-len capture: for schedule [[1,8,7],[9,32,2],[33,96,0]] with min_tokens=1 the allowed (query_len,num_reqs) set covers K in 1..7 for <=8, {1,2} for 9..32, 0 for 33..96; missing graphs fall back to non-graph dispatch -> no contract breach"
  - "mamba_state_seq_lens across builders and reconstruction sites: all state-slot callers use the property; unpadded()/make_local_attention_virtual_batches drop it but are not on the DBO prefill path; DBO shares one compute stream (gpu_ubatch_wrapper.py:473,254) so ubatch0's state write precedes ubatch1's read per layer -> did not fire"
  - "Middleware body peek: chunked body, non-dict JSON, non-JSON, stream=true+error status all behave as the module docstring declares; body buffer bounded by request size -> did not fire (postcondition note kept as P3)"
  - "Test coverage of the new public contracts: searched the patch and repo diff for tests -> none (DBC-TEST-001 FIRED)"
p_findings:
  - severity: P1
    location: "serve-configs/patches/0005-*.patch -> vllm/config/speculative.py:188-207 (new fields, no cross-field validator) and vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:146-162"
    summary: "adaptive_draft_length=true with num_speculative_tokens==1, or adaptive_draft_min_tokens > num_speculative_tokens, passes SpeculativeConfig validation and then crashes the V2 speculator with KeyError(num_speculative_steps) in load_model; the scheduler silently disables adaptive for K<=1 while the speculator does not (constructor does not establish its invariant)"
    contract: "DBC-PRE-003 MUST Validate Untrusted Input At Boundaries; DBC-INV-002 MUST Establish Invariants Before Exposure"
    remediation: "In SpeculativeConfig._verify_args reject adaptive_draft_length with num_speculative_tokens<=1 and adaptive_draft_min_tokens>num_speculative_tokens (or clamp with a warning); make _draft_step_options always include num_speculative_steps"
  - severity: P1
    location: "serve-configs/patches/0005-*.patch -> vllm/config/vllm.py:932-947 (_maybe_disable_dynamic_sd_for_data_parallel)"
    summary: "adaptive_draft_length lets each DP-rank scheduler pick a different per-step K exactly like num_speculative_tokens_per_batch_size, but only the latter is disabled for data_parallel_size>1; the precondition DP==1 is undeclared and unenforced"
    contract: "DBC-PRE-001 MUST Define Caller Preconditions; DBC-CONFIG-001 MUST Define Configuration Contracts"
    remediation: "Extend the DP guard to adaptive_draft_length (disable with warning) or document DP==1 as a precondition and reject it in _verify_args"
  - severity: P1
    location: "serve-configs/patches/0005-*.patch, 0006-*.patch, serve-configs/middleware/vllm_keepalive.py (no accompanying tests anywhere in the commit)"
    summary: "New public contracts (speculative-config fields adaptive_draft_*/draft_lm_head_dtype, CommonAttentionMetadata.mamba_state_seq_lens/state_seq_lens, KeepAlive stream=true early-commit) ship without contract tests for accepted/rejected/boundary inputs; verification exists only as ad-hoc benches and this lane's scratch probes"
    contract: "DBC-TEST-001 MUST Test Public Contracts"
    remediation: "Add tests (repo tests/ or serve-configs/tests) covering SpeculativeConfig validation incl. the cross-field cases above, _make_metadata_with_slice state_seq_lens/num_computed bump for split requests, and the middleware early-commit paths (the probe_cfg.py/probe_mw.py scenarios are a ready starting point)"
  - severity: P2
    location: "serve-configs/patches/0005-*.patch -> vllm/config/speculative.py:188 draft_lm_head_dtype; vllm/v1/spec_decode/draft_lm_head.py"
    summary: "draft_lm_head_dtype declares no hardware/method precondition: fp8 needs SM89+ CUTLASS scaled_mm and int4 needs Marlin, and the option is silently ignored when the drafter has its own lm_head; failures surface at load as internal errors rather than validation failures"
    contract: "DBC-PRE-001 MUST Define Caller Preconditions; DBC-PRE-003 MUST Validate Untrusted Input At Boundaries"
    remediation: "Validate platform capability in _verify_args / at QuantizedDraftLMHead construction with a clear ValueError, and log a warning when the option is set but no head is shared"
  - severity: P3
    location: "serve-configs/patches/0005-*.patch -> vllm/v1/worker/gpu/model_runner.py:1819 (num_speculative_steps=num_spec_tokens_to_schedule or None)"
    summary: "Postcondition 'V2 speculator honors the scheduler's per-step draft count' does not hold for count 0: the drafter runs the full K-step chain for batches in the [33,96,0] range (extra tokens are ignored, so wasted work only)"
    contract: "DBC-POST-001 MUST Define Provider Guarantees"
    remediation: "Skip propose() (or pass 1 and drop the result) when num_spec_tokens_to_schedule == 0, or state the exception in the README/patch"
  - severity: P3
    location: "serve-configs/middleware/vllm_keepalive.py:160-186"
    summary: "After a stream=true early commit the SSE content-type is fixed even when the app later answers with a JSON error (client sees 200 text/event-stream carrying a JSON body); the docstring declares the 200-with-error-body outcome but not the content-type mismatch"
    contract: "DBC-POST-002 MUST Define Side Effect Outcomes"
    remediation: "State the mismatch in the module docstring (or wrap late JSON errors as an SSE data: event when is_sse)"
rules_reviewed:
  - /home/gwillen/.agents/skills/self-review/references/engineering-rules/dbc.rules.md
artifact_paths:
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T110959Z-001-qwen38-two-profiles-main-review-packet.mdscript.md
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-serve-configs.diff
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-stat.txt
  - /shared/vllm/serve-configs/middleware/vllm_keepalive.py
  - /shared/vllm/serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch
  - /shared/vllm/serve-configs/patches/0006-dense-tp-dual-batch-overlap-prefill-allreduce.patch
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/config/speculative.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/config/vllm.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/core/sched/async_scheduler.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/ubatch_utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu_ubatch_wrapper.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/attention/backend.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/attention/backends/gdn_attn.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/attention/backends/utils.py
  - /tmp/claude-1000/-shared-vllm/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/scratchpad/probe_cfg.py
  - /tmp/claude-1000/-shared-vllm/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/scratchpad/probe_mw.py
objectives_checked:
  - DBC-SCOPE-001
  - DBC-SCOPE-002
  - DBC-SOURCE-002
  - DBC-PRE-001
  - DBC-PRE-002
  - DBC-PRE-003
  - DBC-POST-001
  - DBC-POST-002
  - DBC-INV-001
  - DBC-INV-002
  - DBC-FAIL-001
  - DBC-CONFIG-001
  - DBC-SECRET-001
  - DBC-CONC-001
  - DBC-TEST-001
  - DBC-VERSION-001
  - DBC-DOC-001
remaining_gaps:
  - "The three P1 findings above (adaptive K>1/min_tokens precondition + KeyError, DP>1 guard, missing contract tests) must be repaired or explicitly waived by the contract owner before this lane can sign off"
  - "Not exercised live: adaptive K on the sync (non-async) scheduler path and structured-output drafts with a shrunk _num_steps (stale draft columns are copied to the scheduler); reasoning only, no runtime probe"
signed_off_at: "2026-08-18T11:19:07Z"
repair_resume_command: "/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T110959Z-001-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Signoff

* lane verdict: eng-dbc **signed_off: false** for commit b3c9e81 (base 68dfda8) against /home/gwillen/.agents/skills/self-review/references/engineering-rules/dbc.rules.md
* P1 DBC-PRE-003 / DBC-INV-002 — vllm/config/speculative.py:188-207 + v1/worker/gpu/spec_decode/autoregressive/speculator.py:146-162: adaptive_draft_length with num_speculative_tokens==1 or adaptive_draft_min_tokens>K is accepted and crashes V2 speculator load with KeyError; remediation: reject/clamp in _verify_args and make _draft_step_options always contain num_speculative_steps
* P1 DBC-PRE-001 / DBC-CONFIG-001 — vllm/config/vllm.py:932-947: adaptive_draft_length not covered by the DP>1 dynamic-SD guard; remediation: extend the guard or declare and reject DP>1
* P1 DBC-TEST-001 — patches 0005/0006 and vllm_keepalive.py: no contract tests for the new config fields, mamba_state_seq_lens slicing, or middleware early-commit; remediation: add tests (probe_cfg.py / probe_mw.py scenarios are a starting point)
* P2 DBC-PRE-001 / DBC-PRE-003 — draft_lm_head_dtype: no hardware/method precondition validation, silently ignored without a shared head; remediation: validate at config/construction time with a clear ValueError and warn when unused
* P3 DBC-POST-001 — v1/worker/gpu/model_runner.py:1819: num_spec_tokens_to_schedule==0 is passed as None so the drafter runs full K; remediation: skip propose or document
* P3 DBC-POST-002 — vllm_keepalive.py:160-186: SSE content-type fixed after early commit even for a later JSON error; remediation: document or wrap as SSE event
* verified without findings: pydantic range validation of the adaptive_* fields and the dtype Literal; scheduler/async-scheduler/V2-runner/speculator num_spec_tokens_to_schedule agreement; mamba_state_seq_lens reaching every state-slot caller with a single shared compute stream under DBO; middleware behavior for stream/non-stream/non-JSON/chunked bodies; both systemd units Conflicts= each other; live 8012 healthy

## Resume From Signoff

* signed_off is false: the next jump is `/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T110959Z-001-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change` after the composer repairs or waives the P1 findings
* a fresh blind eng-dbc reviewer must be spawned for the next round; never re-enter this lane's own review from this sign-off
* when a later round reaches signed_off true, continue at `/mdscript-exec /home/gwillen/.agents/skills/self-review/workflows/triple-adversarial-blind-review.mdscript.md#aggregate-triple-signoffs`
