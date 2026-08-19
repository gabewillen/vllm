---
reviewer_id: eng-dbc
reviewer_lane: eng-dbc
rules_file: /home/gwillen/.agents/skills/self-review/references/engineering-rules/dbc.rules.md
lane_applicable: true
review_round: 2
goal: "Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles: (1) LATENCY: max single-stream decode + min TTFT preserving K7 MTP, 262k ctx, stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high concurrency preserving 262k ctx, stability, quality; profile continuously, optimize the bottleneck, benchmark + stress each change vs best, stop only when remaining gains are negligible or hardware-bound"
conversation_id: 23cb78c0-b3a1-4ea3-89e2-332443ff1f8f
run_id: 20260818T055754Z-8dd3de85
review_head: 1f3c16e
merge_base: 68dfda8
proof_scope: live-proof
signed_off: false
verifier_summary: "Blind DBC lane, round 2. Attacked the runner<->speculator propose() contract, scheduler<->runner num_spec_tokens_to_schedule (async placeholders, sync path, K=0, fused-manager fallback), config validation + DP guard, QuantizedDraftLMHead preconditions and MTP shared_head routing, CommonAttentionMetadata.mamba_state_seq_lens across all mamba_get_block_table_tensor call sites and re-slicing helpers, and the middleware early-commit/SSE-wrap/body-peek contract. Ran the 26 CPU contract tests (pass). One P1: patch 0005 makes the V2 runner pass num_speculative_steps= to every speculator's propose(), but only AutoRegressiveSpeculator accepts it; BaseSpeculator/DFlash/DSpark/MultiModuleMTP raise TypeError on the first step, so the venv-shared DSpark profile (V2-only) is broken. Plus P2 (DP guard untested) and P3 contract-doc gaps."
evidence:
  - "inspect.signature over the venv: BaseSpeculator/DFlashSpeculator/DSparkSpeculator/MultiModuleMTPSpeculator.propose lack num_speculative_steps; AutoRegressiveSpeculator has it; sig.bind(..., num_speculative_steps=3) on DFlashSpeculator.propose -> TypeError: got an unexpected keyword argument 'num_speculative_steps'; call site /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py:1819 passes it unconditionally"
  - "config/vllm.py:612-625 (venv): DSpark forces the V2 model runner ('DSpark is implemented only by the V2 GPU model runner'); serve-configs/qwen3_8_27b_fp8_dspark_code.yaml is an existing opt-in profile in the same venv"
  - "cd /shared/vllm/serve-configs/tests && .venv-qwen38 python -m pytest -q . -> 26 passed (config validation, adaptive policy, capture policy, ubatch slice, middleware ASGI)"
  - "v1/core/sched/async_scheduler.py: placeholders = [-1]*scheduler_output.num_spec_tokens_to_schedule, the same value the runner drafts in that step -> async draft/verify counts agree; v1/worker/gpu/spec_decode/utils.py DraftTokensHandler.get_draft_tokens returns [-1]*draft_tokens.shape[1] on the sync path, so K=0 (speculator MIN_DRAFT_STEPS=1) schedules 1 spec token under sync scheduling, contradicting the propose() comment 'its token is simply not scheduled'"
  - "all four mamba_get_block_table_tensor call sites (mamba_attn.py:527,813; gdn_attn.py:223; linear_attn.py:72,197) use state_seq_lens; ubatch_utils.py:182,257 sets it; CommonAttentionMetadata.unpadded() (backend.py:641-670) does not propagate mamba_state_seq_lens"
  - "llm_base_proposer.py:1573-1594 and gpu/spec_decode/eagle/utils.py:90-110: quantized head assigned to draft lm_head, but MTP layer shared_head.head is re-pointed at the unquantized target head; qwen3_5_mtp.py:298 uses self.lm_head so the deployed model is unaffected"
  - "systemctl is-active vllm-qwen38 -> active; vllm-qwen38-throughput -> inactive; curl /health on 8012 -> 200; needle200k-tp_final.log and needle200k-lat_final_v2.log both 'needle retrieved: True'"
commands_run:
  - "sed/grep over the packet, review-round2-serve-configs.diff, patches/0005+0006, tests, middleware, and the patched venv sources under /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm"
  - "cd /shared/vllm/serve-configs/tests && /shared/vllm/.venv-qwen38/bin/python -m pytest -q . (26 passed)"
  - "/shared/vllm/.venv-qwen38/bin/python -c 'inspect.signature(...propose)' for BaseSpeculator, DFlashSpeculator, DSparkSpeculator, MultiModuleMTPSpeculator, AutoRegressiveSpeculator; sig.bind with num_speculative_steps=3 on DFlashSpeculator.propose"
  - "grep -rn mamba_get_block_table_tensor / CommonAttentionMetadata( / mamba_state_seq_lens over the venv"
  - "systemctl is-active vllm-qwen38 vllm-qwen38-throughput; curl -s http://127.0.0.1:8012/health"
attack_attempts:
  - "Runner->speculator contract (DBC-SUBTYPE-001/DBC-PRE-001): does every BaseSpeculator subtype accept the new propose(num_speculative_steps=) kwarg the V2 runner now always passes? FIRED: only AutoRegressiveSpeculator does; DFlash/DSpark/MultiModuleMTP -> TypeError at first execute_model."
  - "Scheduler<->runner async placeholder contract: can the drafted column count differ from the placeholder count? DID NOT FIRE: both derive from scheduler_output.num_spec_tokens_to_schedule of the same step; runner reads per-request counts from scheduled_spec_decode_tokens and slices req_states.draft_tokens[:, :num_drafted]."
  - "K=0 range and fused-manager fallback under sync scheduling: propose() returns 1 (or K) columns and DraftTokensHandler reports that count to the scheduler -> K=0 becomes K=1 and a missing fused manager (adaptive_draft_min_tokens > schedule K) verifies K instead of the requested count. FIRED at P3 (production uses async scheduling; documented postcondition is wrong for the sync caller)."
  - "Config contract: adaptive validation (K>=2, min<=K, alpha/margin bounds) tested; DP guard ordering (_maybe_disable_dynamic_sd_for_data_parallel before _maybe_override_dynamic_sd_cudagraph_mode) correct; but no test exercises the DP>1 disable path or the draft_lm_head_dtype no-op cases. FIRED at P2/P3."
  - "QuantizedDraftLMHead preconditions: hardware checks raise ValueError at load; _pack_rows_gptq raises on K % pack; MTP shared_head.head is re-pointed at the unquantized head so the fp8/int4 copy is silently unused for shared_head-style MTP models (not qwen3_5_mtp). FIRED at P3 (undocumented postcondition)."
  - "mamba_state_seq_lens invariant: all state-slot selection call sites use state_seq_lens; ubatch slice test covers first/continuation slice; CommonAttentionMetadata.unpadded() drops the field (spec-decode + DBO drafter path only). FIRED at P3."
  - "Middleware early-commit/SSE-wrap: fast responses untouched, stream=true committed as SSE, JSON error after SSE commit wrapped as one data: event, >8 MiB body falls back to JSON commit (documented), body buffer freed after last chunk and never logged, pinger cancelled in finally. DID NOT FIRE beyond the documented >8 MiB limitation; timing-based tests have ~0.5 s margin (SHOULD-level note only)."
p_findings:
  - severity: P1
    location: "serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch (vllm/v1/worker/gpu/model_runner.py hunk, venv model_runner.py:1819) vs vllm/v1/worker/gpu/spec_decode/speculator.py BaseSpeculator.propose, dflash/speculator.py, dspark/speculator.py, multi_module_mtp/speculator.py"
    summary: "The V2 runner now calls self.speculator.propose(..., num_speculative_steps=num_spec_tokens_to_schedule) for every speculator, but the abstract BaseSpeculator.propose contract and the DFlash/DSpark/MultiModuleMTP implementations were not extended; they raise TypeError on the first step. DSpark is V2-only, so the existing opt-in serve-configs/qwen3_8_27b_fp8_dspark_code.yaml profile in the same venv is broken by this patch regardless of adaptive_draft_length."
    contract: "DBC-SUBTYPE-001 MUST Preserve Behavioral Substitutability; DBC-PRE-001 MUST Define Caller Preconditions; DBC-SCOPE-001 MUST Define Contracts At Reliance Boundaries"
    remediation: "Add num_speculative_steps: int | None = None to BaseSpeculator.propose and every subtype (accepting and ignoring it where the drafter cannot vary), or pass it only when isinstance(self.speculator, AutoRegressiveSpeculator) / the speculator advertises support; add a CPU contract test asserting every BaseSpeculator subclass's propose() signature binds the runner's kwargs; regenerate 0005 and re-apply."
  - severity: P2
    location: "serve-configs/patches/0005 (vllm/config/vllm.py _maybe_disable_dynamic_sd_for_data_parallel) and serve-configs/tests/test_adaptive_draft_length.py"
    summary: "The DP>1 guard (adaptive_draft_length forced False with a warning) is a public config contract with no test; only the SpeculativeConfig field validation is covered."
    contract: "DBC-TEST-001 MUST Test Public Contracts"
    remediation: "Add a test that builds a VllmConfig (or calls the guard on a stub with data_parallel_size=2) and asserts adaptive_draft_length becomes False and the warning fires; keep DP=1 unchanged."
  - severity: P3
    location: "serve-configs/patches/0005 (autoregressive/speculator.py propose() comment 'its token is simply not scheduled'; model_runner.py num_drafted slicing) with vllm/v1/worker/gpu/spec_decode/utils.py DraftTokensHandler.get_draft_tokens"
    summary: "Under sync scheduling (async disabled) the scheduler learns the draft count from draft_tokens.shape[1]; K=0 (MIN_DRAFT_STEPS=1) schedules 1 spec token and the fused-manager fallback (num_spec_tokens_to_schedule < adaptive_draft_min_tokens) schedules K instead of the requested count. Production runs async scheduling, so the deployed profiles are unaffected, but the stated postcondition is wrong for the sync caller."
    contract: "DBC-POST-001 MUST Define Provider Guarantees; DBC-DOC-001 MUST Document Public Contracts"
    remediation: "Either return draft_tokens[:, :num_spec_tokens_to_schedule] (0 columns for K=0) so the sync path matches V1's llm_base_proposer K=0 behavior, or document that V2 sync scheduling verifies max(1, requested) drafts and capture a fused manager for adaptive_draft_min_tokens..K unconditionally."
  - severity: P3
    location: "serve-configs/patches/0005 (vllm/config/speculative.py draft_lm_head_dtype; v1/spec_decode/draft_lm_head.py; llm_base_proposer.py:1573-1594; gpu/spec_decode/eagle/utils.py:90-110)"
    summary: "draft_lm_head_dtype=fp8|int4 is a silent no-op for spec methods without a shared head and for MTP models that route logits through layer.shared_head.head (re-pointed at the unquantized head after quantization). Config contract does not state this failure mode."
    contract: "DBC-CONFIG-001 MUST Define Configuration Contracts; DBC-POST-001 MUST Define Provider Guarantees"
    remediation: "Warn (or reject) when draft_lm_head_dtype != auto and no shared head is quantized, and either point shared_head.head at the quantized copy too or document the qwen3_5_mtp-style self.lm_head requirement in the field docstring."
  - severity: P3
    location: "serve-configs/patches/0006 (vllm/v1/attention/backend.py CommonAttentionMetadata.mamba_state_seq_lens) vs CommonAttentionMetadata.unpadded() at backend.py:641-670"
    summary: "The new invariant 'every slice selects the state slot from state_seq_lens' is not carried through unpadded(), which rebuilds the dataclass without mamba_state_seq_lens (spec-decode drafter metadata path). Not reachable by the two reviewed profiles (throughput has no spec decode; latency has no DBO), but the invariant is not established at that boundary."
    contract: "DBC-INV-001 MUST Define Stable Invariants; DBC-INV-002 MUST Establish Invariants Before Exposure"
    remediation: "Propagate mamba_state_seq_lens (sliced to num_actual_reqs) in unpadded(), or assert it is None there and document DBO+mamba-drafter as unsupported."
rules_reviewed:
  - /home/gwillen/.agents/skills/self-review/references/engineering-rules/dbc.rules.md
artifact_paths:
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T122455Z-002-qwen38-two-profiles-main-review-packet.mdscript.md
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round2-serve-configs.diff
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round2-stat.txt
  - /shared/vllm/serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch
  - /shared/vllm/serve-configs/patches/0006-dense-tp-dual-batch-overlap-prefill-allreduce.patch
  - /shared/vllm/serve-configs/patches/README.md
  - /shared/vllm/serve-configs/middleware/vllm_keepalive.py
  - /shared/vllm/serve-configs/tests/test_adaptive_draft_length.py
  - /shared/vllm/serve-configs/tests/test_keepalive_middleware.py
  - /shared/vllm/serve-configs/tests/test_ubatch_split_metadata.py
  - /shared/vllm/serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml
  - /shared/vllm/serve-configs/qwen3_8_27b_fp8_max.yaml
  - /shared/vllm/serve-configs/systemd/vllm-qwen38.service
  - /shared/vllm/serve-configs/systemd/vllm-qwen38-throughput.service
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/speculator.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/core/sched/async_scheduler.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/config/vllm.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/config/speculative.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/spec_decode/llm_base_proposer.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/eagle/utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/attention/backend.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/ubatch_utils.py
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/needle200k-tp_final.log
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/needle200k-lat_final_v2.log
objectives_checked:
  - DBC-SCOPE-001
  - DBC-SCOPE-002
  - DBC-SOURCE-002
  - DBC-PRE-001
  - DBC-PRE-003
  - DBC-POST-001
  - DBC-POST-002
  - DBC-INV-001
  - DBC-INV-002
  - DBC-FAIL-001
  - DBC-SUBTYPE-001
  - DBC-CONFIG-001
  - DBC-SECRET-001
  - DBC-CONC-001
  - DBC-ASYNC-001
  - DBC-TEST-001
  - DBC-DOC-001
remaining_gaps:
  - "P1: extend BaseSpeculator.propose (and DFlash/DSpark/MultiModuleMTP) to accept num_speculative_steps, or gate the kwarg in the V2 runner; add a signature-binding contract test; regenerate patch 0005 and re-apply to the venv"
  - "P2: contract test for the DP>1 adaptive_draft_length guard"
  - "P3: V2 sync-scheduling K=0 / fused-fallback postcondition; draft_lm_head_dtype silent no-op cases; mamba_state_seq_lens through CommonAttentionMetadata.unpadded()"
signed_off_at: "2026-08-18T12:33:00Z"
repair_resume_command: "/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T122455Z-002-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Signoff

* lane `eng-dbc` verdict: **signed_off: false** against `/home/gwillen/.agents/skills/self-review/references/engineering-rules/dbc.rules.md` (round 2, head 1f3c16e, base 68dfda8)
* P1 — DBC-SUBTYPE-001 / DBC-PRE-001 / DBC-SCOPE-001 — patch 0005 V2 `model_runner.py:1819` passes `num_speculative_steps=` to every speculator, but `BaseSpeculator.propose`, `DFlashSpeculator`, `DSparkSpeculator`, `MultiModuleMTPSpeculator` do not accept it (TypeError on first step; DSpark is V2-only so `serve-configs/qwen3_8_27b_fp8_dspark_code.yaml` in the same venv breaks). Remediation: add the kwarg to the base contract and all subtypes (or gate it on speculator support), add a signature-binding contract test, regenerate and re-apply 0005.
* P2 — DBC-TEST-001 — `_maybe_disable_dynamic_sd_for_data_parallel` adaptive DP>1 guard has no test. Remediation: add a CPU test asserting adaptive_draft_length is forced False with DP=2.
* P3 — DBC-POST-001 / DBC-DOC-001 — under sync scheduling V2 K=0 drafts and schedules 1 token, and the fused-manager fallback schedules K; the propose() comment claims otherwise. Remediation: return the requested column count (0 for K=0) or document the sync behavior.
* P3 — DBC-CONFIG-001 / DBC-POST-001 — `draft_lm_head_dtype` silently no-op for non-shared-head methods and shared_head.head-routed MTP models. Remediation: warn/reject or quantize shared_head.head too; document.
* P3 — DBC-INV-001 / DBC-INV-002 — `CommonAttentionMetadata.unpadded()` drops `mamba_state_seq_lens`. Remediation: propagate or assert None with documented unsupported combo.
* contracts that held: async placeholder/draft-count agreement, config field validation (K>=2, min<=K, bounded alpha/margin), hardware preconditions on QuantizedDraftLMHead, state_seq_lens at all four state-slot call sites, middleware early-commit (SSE for stream=true, JSON otherwise), SSE wrapping of late non-SSE bodies, fast responses untouched, body not logged, pinger cancelled; 26 CPU contract tests pass; systemd units Conflicts= each other; live 8012 healthy.

## Resume From Signoff

* `signed_off` is false: the next jump is the repair, then a **fresh** blind reviewer — do not re-enter this lane's own review from this sign-off
* repair_resume_command: `/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T122455Z-002-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change`
* after repair the composer re-runs the lanes; aggregation lives at `/mdscript-exec /home/gwillen/.agents/skills/self-review/workflows/triple-adversarial-blind-review.mdscript.md#aggregate-triple-signoffs`
