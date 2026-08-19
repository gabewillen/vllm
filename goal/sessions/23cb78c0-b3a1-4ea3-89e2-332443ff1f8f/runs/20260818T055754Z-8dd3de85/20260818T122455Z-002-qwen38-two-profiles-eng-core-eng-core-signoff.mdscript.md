---
reviewer_id: eng-core
reviewer_lane: eng-core
rules_file: /home/gwillen/.agents/skills/self-review/references/engineering-rules/core.rules.md
extra_rules_files:
  - /home/gwillen/.agents/skills/self-review/references/engineering-rules/local.rules.md
lane_applicable: true
review_round: 2
goal: "Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles: (1) LATENCY: max single-stream decode + min TTFT preserving K7 MTP, 262k ctx, stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high concurrency preserving 262k ctx, stability, quality; profile continuously, optimize the bottleneck, benchmark + stress each change vs best, stop only when remaining gains are negligible or hardware-bound"
conversation_id: 23cb78c0-b3a1-4ea3-89e2-332443ff1f8f
review_head: 1f3c16e
merge_base: 68dfda8
signed_off: false
verifier_summary: "Blind eng-core round-2 review of 68dfda8..1f3c16e (7 commits, serve-configs diff + patches 0005/0006 in the venv). Attacked adaptive-K policy/validation, V2 speculator drafted-column contract, cudagraph capture policy, DBO all-reduce yield pairing and dense-only gate, split micro-batch mamba state metadata, keepalive body peek / early commit / SSE wrap / telemetry, commit atomicity, named args, comment hygiene against core.rules.md + local.rules.md. Core mechanics hold (yield pairing correct, dense gate correct, state-slot metadata consistent, patches reproduce the venv byte-for-byte, 26/26 CPU tests pass x3, live 8012 healthy, measurements backed by artifacts). Blocking findings: positional/bare-boolean call sites in new owned signatures (LOCAL-ARG-001), wall-clock-sleep keepalive tests not classified as integration tests (CORE-TEST-001), stale max-num-seqs comment contradicting the new value (CORE-DOC-001); plus a partial OTEL gap on the app-error-after-early-commit path (CORE-OBS-001) and a comment that overstates the 0-draft contract outside async scheduling."
evidence:
  - "git log --stat 68dfda8..1f3c16e: 48646ff patch0005+tests+README, a4703eb latency yaml+unit, af18ad7 patch0006+test, e1a099a throughput yaml+unit, 2815915 middleware+test, 4957ee2 unit hardening, 1f3c16e evidence; each subject states one change, ordering is bisectable (0006 lands before the yaml that enables DBO)."
  - "cd serve-configs/tests && .venv-qwen38/bin/python -m pytest -q . -> 26 passed (three consecutive runs, ~13.5 s each, all wall-clock)."
  - "Patch reproduction: copied every *.orig0005 into a scratch tree, applied 0005 then 0006 with patch -p1, cmp against the venv for all 19 files -> identical (n=0)."
  - "vllm/v1/worker/ubatching.py: yield_and_switch_from_compute_to_comm asserts current_stream()==compute_stream, records compute-done, yields, switches to comm, waits; yield_and_switch_from_comm_to_compute mirrors it; _all_reduce_out_place calls them in that order around device_communicator.all_reduce -> pairing correct; both GroupCoordinator.all_reduce branches (custom op and direct) reach _all_reduce_out_place; dbo_overlap_tp_all_reduce is gated by _OVERLAP_TP_ALL_REDUCE set only when model_config.is_moe is False (gpu_ubatch_wrapper.__init__)."
  - "vllm/v1/worker/ubatch_utils.py _make_metadata_with_slice: mamba_state_seq_lens = state_seq_lens[request_slice] (never reduced) while seq_lens is reduced for a split last request; num_computed_tokens_cpu[0] += tokens_skipped for a split first request; gdn/mamba/linear builders use m.state_seq_lens for mamba_get_block_table_tensor ((seq_len-1)//block_size) -> both slices address the same state block, has_initial_state derived from computed tokens -> consistent."
  - "Scheduler flow (venv v1/core/sched/scheduler.py + async_scheduler.py + engine/core.py): num_spec_tokens_to_schedule feeds AsyncScheduler placeholders; V2 speculator returns drafts[:, :_num_steps] with _num_steps = max(1, min(K, n)) >= n so req_states.draft_tokens always holds at least the scheduled columns; scheduler counts (not runner width) decide verification under async scheduling."
  - "Live read-only: systemctl is-active vllm-qwen38 -> active, vllm-qwen38-throughput -> inactive; curl /health -> 200; /metrics has 4 vllm_keepalive_* lines. Claimed numbers match artifacts: ss-lat_base 44.4/39.2/77.4/111.6, live r2 65.3/54.7/98.7/144.4; bench-tp_base_c128 621.7, tp_final3 c128/64/32 827.9/680.2/479.6."
  - "grep for VLLM_API_KEY=<value>/Bearer/sk- in run artifacts and serve-configs: no hits."
commands_run:
  - "git log --stat --format=... 68dfda8..1f3c16e"
  - "sed -n over review-round2-serve-configs.diff (all 1918 lines)"
  - "cd /shared/vllm/serve-configs/tests && /shared/vllm/.venv-qwen38/bin/python -m pytest -q . -p no:cacheprovider (x3)"
  - "cp *.orig0005 -> scratch tree; patch -p1 < 0005; patch -p1 < 0006; cmp each vs venv"
  - "grep -n num_spec_tokens_to_schedule / spec_token_ids / decode_cudagraph_manager / _THREAD_ID_TO_CONTEXT in the venv vllm tree; sed -n of scheduler.py 585-760, 1270-1300, 2274-2335; async_scheduler.py 1-60; engine/core.py 605-735; gpu/model_runner.py 1790-1850; speculator.py 240-380; ubatching.py 60-230; ubatch_utils.py 30-262; parallel_state.py 662-700; gpu_model_runner.py 4118-4160; attention/backends/utils.py mamba_get_block_table_tensor"
  - "systemctl is-active vllm-qwen38 vllm-qwen38-throughput; curl -s http://127.0.0.1:8012/health; curl -s http://127.0.0.1:8012/metrics | grep -c vllm_keepalive"
  - "grep -rIl 'VLLM_API_KEY=[A-Za-z0-9]|Bearer ...|sk-...' run artifacts and serve-configs"
  - "python check of artifacts/manifest.json paths + json read of bench-tp_*.json output_throughput"
attack_attempts:
  - "LOCAL-ARG-001 named arguments: FIRED. New owned signatures called positionally: speculator.py:362 self._multi_step_propose(input_batch, num_reqs, num_rejected, dummy_run, skip_attn_for_dummy_run, is_profile) passes three booleans positionally; draft_lm_head.py:158 _pack_rows_gptq(q_w, MARLIN_WEIGHT_TYPE.size_bits); cudagraph_utils.py:314 _query_len_allowed(decode_query_len, rounded_num_reqs); test_keepalive_middleware.py _slow_app(1.5, ...) / _slow_app(0.0) pass a bare number positionally to an owned helper. Named correctly elsewhere (adaptive_num_spec_tokens, update_accepted_ema, QuantizedDraftLMHead, maybe_quantize_shared_lm_head, decode_query_lens_for_spec, decode_query_len_allowed, check_ubatch_thresholds)."
  - "CORE-TEST-001 deterministic tests: FIRED. test_keepalive_middleware.py drives the real asyncio clock (asyncio.sleep 1.5 / 2.5 s app delays against the middleware's fixed 1.0 s poll loop; test_idle_stream_gets_comment_pings asserts >= 2 pings in a 2.5 s window where exactly 2 land at ~1 s and ~2 s). serve-configs/tests/README.md does not classify them as integration boundary tests. Passed 3/3 locally, but the timing margin is the assertion boundary."
  - "CORE-DOC-001 comment hygiene: FIRED. serve-configs/qwen3_8_27b_fp8_max.yaml:40-41 keeps 'Concurrency saturates ~seqs96 on L4 (seqs64->96 was +2.2%)' directly above the new 'max-num-seqs: 128' whose header rationale says 128 still scales with P2P; line 36 '5.3x full-length concurrency at seqs96' likewise. Other new comments state present contracts (checked patches, middleware, systemd, latency yaml)."
  - "CORE-OBS-001/002 telemetry: PARTIAL. New OTEL counters vllm.keepalive.pings / early_commits with a bounded content_type label are declared and the cardinality is stated in-source (OBS-002 satisfied); the failure path 'app responded >= 300 after early commit' (client sees 200 + wrapped error) and the body-peek fallback (oversize / non-JSON body) emit only stdlib log lines, no OTEL signal."
  - "Speculator 0-draft contract: PARTIAL. propose() comment says a request for 0 'is simply not scheduled'; that holds under async scheduling (AsyncScheduler placeholders use num_spec_tokens_to_schedule) but with V2 + --no-async-scheduling DraftTokensHandler.get_draft_tokens returns [-1]*num_drafted=1 and update_draft_token_ids schedules it, so batches in the [33,96,0] range would verify 1 draft/step. Not the deployed configuration."
  - "Adaptive-K policy/validation: did not fire. K>=2 and min<=K validated; DP>1 disables with warning_once before uses_dynamic check; EMA dict popped in _free_request (bounded by live requests); adaptive_num_spec_tokens returns full K when any request lacks history; the fused-graph fallback (missing manager) drafts K, a superset of the scheduled count, so req_states.draft_tokens always covers the verified columns."
  - "V2 width handling: did not fire. num_drafted = draft_tokens.shape[1] >= num_spec_tokens_to_schedule by construction (max(1, min(K, n))); set_draft_tokens receives the same width; adaptive_verification uses speculator confidences unchanged."
  - "Cudagraph capture policy: did not fire. decode_query_lens_for_spec skips managers with decode_query_len <= K (draft decode), unions schedule lengths with per-range adaptive lengths, and _query_len_allowed only admits adaptive lengths inside their batch-size range; test covers schedule [[1,8,7],[9,32,2],[33,96,0]]."
  - "DBO yield pairing / dense-only gate / cross-stream memory: did not fire. compute->comm yield records compute-done and waits before the all-reduce; comm->compute mirrors; the next comm-stream allocation cannot overtake compute consumers because every comm op waits on the preceding compute-done event; gate is set from model_config.is_moe at wrapper init (single writer, before threads)."
  - "Split micro-batch mamba metadata: did not fire (see evidence); CPU test asserts state_seq_lens [400] for the reduced first slice and num_computed 300 for the continuation."
  - "Middleware body peek / early commit / SSE wrap: did not fire on correctness. Peek bounded by _MAX_PEEK_BYTES (+ one chunk) and freed after decision; narrow except (ValueError incl. UnicodeDecodeError, AttributeError); expect_sse only read by the ping task on the same loop; SSE wrap prefixes 'data: ' once and terminates with a blank line; app SSE after SSE commit is not double-wrapped; JSON commit path unchanged for non-stream bodies. Noted, not graded: json.loads of up to 8 MiB on the event loop per request duplicates FastAPI's parse; RecursionError from a pathologically nested body escapes the narrow except (vLLM's own parse would fail on the same body)."
  - "LOCAL-GIT-001 atomic commits: did not fire. Seven commits, one subject each, no checkpoint messages, evidence isolated in 1f3c16e; the 2815915 typing of ASGI callables was judged part of introducing the Message/Receive aliases the new _receive needs rather than a separate refactor; apply-to-venv.sh cache-clearing generalization in 48646ff is required by the wider set of patched packages."
  - "LOCAL-CUT-001 hard cutover: not applicable (the patched vLLM tree and the middleware are deployed on production port 8012; decode_cudagraph_manager alias is the live single-step manager, not a legacy path)."
  - "CORE-SEC-001 secrets: did not fire (no key material in artifacts/configs; unit reads /etc/vllm/qwen38.env via EnvironmentFile)."
  - "CORE-PERF-001 measured claims: did not fire (baseline/final numbers in ss-lat_base.log, live ss_bench-r2.log, bench-tp_*_c{128,64,32}.json match the claim)."
  - "CORE-CFG-001 flags vs env: did not fire (NCCL_MAX_NCHANNELS / VLLM_USE_V2_MODEL_RUNNER are third-party settings; middleware settings reachable by constructor argument with env as override)."
p_findings:
  - severity: P1
    location: "/shared/vllm/serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch (venv vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:362-365; vllm/v1/spec_decode/draft_lm_head.py:158; vllm/v1/worker/gpu/cudagraph_utils.py:314) and /shared/vllm/serve-configs/tests/test_keepalive_middleware.py (_slow_app(1.5, ...), _slow_app(0.0))"
    summary: "New signatures owned by the change are called positionally, including three bare booleans (dummy_run, skip_attn_for_dummy_run, is_profile) to _multi_step_propose, a bare number to _pack_rows_gptq and _query_len_allowed, and a bare float delay to the test helper _slow_app."
    contract: "LOCAL-ARG-001 MUST Named Arguments At Call Sites"
    remediation: "Pass keyword arguments at these call sites (input_batch=..., dummy_run=..., num_bits=..., query_len=..., num_reqs=..., delay=...); regenerate patch 0005 and re-apply to the venv."
  - severity: P1
    location: "/shared/vllm/serve-configs/tests/test_keepalive_middleware.py (test_stream_request_early_commits_as_sse, test_json_request_early_commits_as_json, test_json_error_after_sse_commit_is_wrapped_as_sse_event, test_idle_stream_gets_comment_pings) and /shared/vllm/serve-configs/tests/README.md"
    summary: "Middleware tests depend on wall-clock sleeps against a hard-coded 1.0 s poll loop (test_idle_stream_gets_comment_pings asserts >= 2 pings where exactly 2 fit in the 2.5 s window) and are not classified as integration boundary tests."
    contract: "CORE-TEST-001 MUST Deterministic Tests"
    remediation: "Make the poll interval / clock injectable in _KeepAliveResponder (or drive the loop with a fake clock) and assert on message sequence rather than elapsed time; or classify the file as an integration boundary test in tests/README.md and widen the margins so the assertion is not on the boundary."
  - severity: P1
    location: "/shared/vllm/serve-configs/qwen3_8_27b_fp8_max.yaml:36,40-42"
    summary: "The comment 'Concurrency saturates ~seqs96 on L4 (seqs64->96 was +2.2%)' (and '5.3x full-length concurrency at seqs96') sits directly above the new max-num-seqs: 128 and contradicts the header rationale that 128 still scales with P2P."
    contract: "CORE-DOC-001 MUST NOT Keep Change History In Comments And Docs"
    remediation: "Rewrite the inline comment to state the current contract (128 concurrent, GDN state per seq is the scaling cost) and refresh the seqs96 concurrency figure or drop it."
  - severity: P2
    location: "/shared/vllm/serve-configs/middleware/vllm_keepalive.py:_send (early_committed branch, status >= 300 warning; wrap_body_as_sse) and _body_requests_stream fallback"
    summary: "The failure path where the app answers >= 300 after an early commit (client sees 200 with a wrapped error body) and the body-peek fallback are logged only; no OTEL counter or attribute records them, so operators cannot alert on lost status lines."
    contract: "CORE-OBS-001 MUST OpenTelemetry Telemetry (failure paths)"
    remediation: "Add a bounded counter (e.g. vllm.keepalive.late_errors with content_type in {sse,json}, or a status_class label on early_commits) and mirror it to prometheus like the others."
  - severity: P3
    location: "/shared/vllm/serve-configs/patches/0005-... (venv vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:262-265 propose() comment; vllm/v1/worker/gpu/model_runner.py:1821-1826)"
    summary: "The comment states a 0-draft request's token 'is simply not scheduled'; that is true only under async scheduling. With V2 + --no-async-scheduling the runner reports num_drafted=1 and the scheduler verifies it, so the [33,96,0] schedule range would run K=1 instead of plain decode (perf, not correctness; not the deployed configuration)."
    contract: "CORE-API-001 MUST Explicit API Contracts / CORE-DOC-001"
    remediation: "State the async-scheduling precondition in the comment, or clamp num_drafted to num_spec_tokens_to_schedule (0 -> pass an empty width to set_draft_tokens) in the runner."
rules_reviewed:
  - /home/gwillen/.agents/skills/self-review/references/engineering-rules/core.rules.md
  - /home/gwillen/.agents/skills/self-review/references/engineering-rules/local.rules.md
artifact_paths:
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T122455Z-002-qwen38-two-profiles-main-review-packet.mdscript.md
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round2-serve-configs.diff
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round2-commits.txt
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/manifest.json
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/ss-lat_base.log
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/20260818T122159Z-prod-8012-ss_bench-r2.log
  - /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/bench-tp_final3_c128.json
  - /shared/vllm/serve-configs/middleware/vllm_keepalive.py
  - /shared/vllm/serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch
  - /shared/vllm/serve-configs/patches/0006-dense-tp-dual-batch-overlap-prefill-allreduce.patch
  - /shared/vllm/serve-configs/patches/README.md
  - /shared/vllm/serve-configs/patches/apply-to-venv.sh
  - /shared/vllm/serve-configs/qwen3_8_27b_fp8_max.yaml
  - /shared/vllm/serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml
  - /shared/vllm/serve-configs/systemd/vllm-qwen38.service
  - /shared/vllm/serve-configs/systemd/vllm-qwen38-throughput.service
  - /shared/vllm/serve-configs/tests/README.md
  - /shared/vllm/serve-configs/tests/test_adaptive_draft_length.py
  - /shared/vllm/serve-configs/tests/test_keepalive_middleware.py
  - /shared/vllm/serve-configs/tests/test_ubatch_split_metadata.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/core/sched/async_scheduler.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/engine/core.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu/cudagraph_utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/spec_decode/draft_lm_head.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/ubatching.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/ubatch_utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/distributed/parallel_state.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/attention/backends/utils.py
  - /shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/config/vllm.py
objectives_checked:
  - CORE-DET-001
  - CORE-MEM-001
  - CORE-WORK-001
  - CORE-CONC-001
  - CORE-STATE-001
  - CORE-ERR-001
  - CORE-API-001
  - CORE-GEN-001
  - CORE-BOUND-001
  - CORE-CFG-001
  - CORE-SEC-001
  - CORE-BUILD-001
  - CORE-TEST-001
  - CORE-PERF-001
  - CORE-OBS-001
  - CORE-OBS-002
  - CORE-DOC-001
  - CORE-EXC-001
  - LOCAL-ARG-001
  - LOCAL-CUT-001
  - LOCAL-GIT-001
remaining_gaps: []
signed_off_at: "2026-08-18T12:37:45Z"
repair_resume_command: "/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T122455Z-002-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Signoff

* lane `eng-core` verdict for head 1f3c16e (base 68dfda8): `signed_off: false`
* rules file: `/home/gwillen/.agents/skills/self-review/references/engineering-rules/core.rules.md` plus `local.rules.md`
* the mechanics under attack held: DBO yield pairing and dense-only gate, split micro-batch state metadata, V2 drafted-column width vs scheduler counts, adaptive-K validation and DP gate, cudagraph capture policy, middleware peek/commit/wrap, patch reproducibility, measured claims, secrets
* P1 LOCAL-ARG-001 — patch 0005 speculator.py:362 `_multi_step_propose(...)` (three positional booleans), draft_lm_head.py:158 `_pack_rows_gptq(q_w, ...)`, cudagraph_utils.py:314 `_query_len_allowed(...)`, tests `_slow_app(1.5, ...)` — pass keyword arguments, regenerate 0005, re-apply to the venv
* P1 CORE-TEST-001 — serve-configs/tests/test_keepalive_middleware.py relies on wall-clock sleeps against the fixed 1.0 s poll loop with a boundary assertion (>= 2 pings in 2.5 s) and is not classified as an integration boundary test — inject the poll interval/clock or classify and widen the margins
* P1 CORE-DOC-001 — serve-configs/qwen3_8_27b_fp8_max.yaml:36,40-42 stale "saturates ~seqs96" comment above `max-num-seqs: 128` — restate the current contract
* P2 CORE-OBS-001 — vllm_keepalive.py app-error-after-early-commit / body-peek fallback paths have no OTEL signal — add a bounded counter or label and mirror to prometheus
* P3 CORE-API-001 — speculator.py propose() comment claims a 0-draft request is never scheduled; only true under async scheduling — state the precondition or clamp num_drafted in the runner

## Resume From Signoff

* `signed_off` is false: the next jump is the repair command `/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T122455Z-002-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change` after the composer's repair wave
* after repair, a fresh blind reviewer must run this lane's entrypoint against the new head; do not re-enter this lane's own review from this sign-off
* when a later round reaches `signed_off: true`, continue at `/mdscript-exec /home/gwillen/.agents/skills/self-review/workflows/triple-adversarial-blind-review.mdscript.md#aggregate-triple-signoffs`
