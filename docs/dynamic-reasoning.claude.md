# Dynamic reasoning effort — signals and implementation plan

Status: plan (2026-08-18). Target: the two Qwen3.8-27B-FP8 profiles on the
4x L4 box (`serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml`, V2 runner + MTP;
`serve-configs/qwen3_8_27b_fp8_max.yaml`, V1 runner + DBO). Ships as venv-local
patches `0009+` (`serve-configs/patches/`), same contract as 0005-0008.

## 1. What "dynamic" means here

`reasoning_effort: "dynamic"` = every request starts at the **lowest** rung and
the engine **escalates** one rung at a time, mid-generation, only when live
signals say more thinking is likely to pay. Effort never de-escalates within
a request; the only early exit is a stall/loop guard. Rungs are thinking-token
caps (the actuator that already exists) plus, in a later phase, a thinking
floor that stops the model from closing `</think>` too early.

Default ladder (tunable per config / per request):

| rung | name   | thinking cap (tokens) | prompt instruction (fixed at prefill) |
|-----:|--------|----------------------:|----------------------------------------|
| 0    | low    | 1 024                 | `low` sentence appended to the last user turn (see §2b) — start rung |
| 1    | medium | 4 096                 | n/a (cannot change after prefill) |
| 2    | high   | 16 384                | n/a |
| 3    | max    | min(65 536, max_tokens) | n/a |

## 2. What the engine already has (facts, with refs)

- `reasoning_effort` on `/v1/chat/completions` is only a chat-template kwarg
  (`vllm/entrypoints/openai/chat_completion/protocol.py:218-231`, `:482-491`).
  Qwen3.8's template turns it into a *system-prompt sentence*: `xhigh`
  (default!) and `low` add instructions, `medium` adds none, anything else
  raises (`chat_template.jinja` lines 45-56 in the HF snapshot). So the prompt
  half of "effort" is fixed at prefill; a `dynamic` value must be intercepted
  by the frontend; §2b measured where the sentence should go (tail of the
  prompt, not the system message).
- Static per-request `thinking_token_budget` exists (`SamplingParams`,
  `protocol.py:231,643`) and is enforced by
  `vllm/v1/sample/thinking_budget_state.py` (`ThinkingBudgetStateHolder`):
  it tracks `<think>`/`</think>` from `ReasoningConfig` token ids, counts
  thinking tokens, and forces the end sequence by writing `+1e9` into logits —
  spec-decode aware (bonus row vs target rows, rollback when a draft is
  rejected before the forced token). Called from
  `vllm/v1/sample/sampler.py:381-419` after penalties, before sampling. Held
  per worker in `vllm/v1/worker/gpu_input_batch.py:110,824,879,933`.
  **This is the cap actuator.** Its budget is a mutable dict field
  (`state["thinking_token_budget"]`), so raising it mid-request is trivial.
- **V2 runner** (`vllm/v1/worker/gpu/`, latency profile via
  `VLLM_USE_V2_MODEL_RUNNER=1`): the *upstream tree here* has no thinking
  budget, but the **deployed 0.27.2rc1.dev110 package already ships one**
  (`vllm/v1/worker/gpu/sample/thinking_budget.py`, `ThinkingBudgetState` +
  Triton kernels, applied on target rows in the sampler and the rejection
  sampler; the config gate is already lifted; `ReasoningConfig` there also
  splits forced vs natural end ids). Found by lane B (2026-08-18); patch 0009
  adds a CPU torch reference of those kernels, versioned budget updates, and
  fixes a gap where greedy budgeted requests were never forced.
- `ReasoningConfig` (`vllm/config/reasoning.py`) derives start/end token id
  *sequences* from the parser or from `--reasoning-config
  '{"reasoning_end_str": ...}'` (`vllm/engine/arg_utils.py:1551,2525`). A
  multi-token end string is forced token-by-token (`end_count` logic), which
  gives us Qwen's own "graceful close" recipe for free (see §5).
- Custom logits processors are **rejected when speculative decoding is on**
  (`vllm/v1/sample/logits_processor/__init__.py:43-45`), so the controller
  cannot be a plugin on the latency profile; it has to be built in like the
  budget holder.
- MTP acceptance per request per step is already computed in the scheduler
  (`vllm/v1/core/sched/scheduler.py:1553-1571`, `num_accepted`) and, with
  patch 0005, kept as a per-request EMA (`_accepted_ema`,
  `update_accepted_ema`) that picks the next draft length. Same shape as what
  we need: per-request signal → per-request knob sent in `SchedulerOutput`.
- Logprobs: V1 `Sampler.compute_logprobs/gather_logprobs`
  (`vllm/v1/sample/sampler.py:305-351`) do a full-vocab log-softmax + topk
  only when requested; V2 `compute_topk_logprobs` likewise
  (`vllm/v1/worker/gpu/sample/sampler.py:72-145`). No entropy anywhere in
  `vllm/` today. Rows already exist for every verified position under spec
  decode.
- Repetition detection exists as a *terminal stop*: `RepetitionDetectionParams`
  (`vllm/sampling_params.py:146`) checked in `check_stop`
  (`vllm/v1/core/sched/utils.py:30,119`). Reuse its evidence earlier as the
  stall signal; never weaken the stop.
- (upstream tree only) `vllm/config/vllm.py:2126` rejects V2 thinking budgets;
  the deployed package no longer has that gate.
- Chat serving is stateless; tool results only exist as `role: tool` messages
  in the *next* request. The Responses API keeps `num_reasoning_tokens` as a
  TODO (`vllm/entrypoints/openai/responses/context.py:181`).
- Prod runs a nightly wheel + patches; the source tree here is upstream. All
  refs above are upstream lines; the 0005 refs are patch-relative.

### 2a. The system-prompt lever and the KV prefix cache

`dynamic` itself never reaches the template (it raises on anything but
`xhigh`/`medium`/`low`); the frontend renders it as **one fixed value**. That
sentence sits in block 0 of the prompt, so:

- All dynamic requests share the same prefix → multi-turn cache hits work.
- A conversation that flips between renderings (`dynamic`→`low` vs a client
  that omits effort → template default `xhigh`) misses from block 0 → full
  re-prefill (3-14 s at 9-37k tokens, 130-150 s at ~200k on 4x L4). Rule:
  effort rendering is constant per conversation; log a metric when the same
  prompt prefix arrives with a different rendering.
- Because the prompt is fixed at prefill, **escalation is invisible to the
  model**: raising the cap only stops the cut-off; it cannot ask a model that
  was told to be brief to think harder. Rendering choice is therefore a real
  design decision to measure in P3:
  - `low` — model self-limits (cheap rung 0); escalation only helps
    requests that wanted to continue; early-but-wrong closes need the floor
    actuator (§5, P5).
  - `medium` — no sentence; the ladder is the only limiter; escalation is
    meaningful, rung 0 costs more on easy tasks.
  Prior: `medium` + ladder for accuracy, `low` + floor if token cost
  dominates. A template tweak that moves the sentence to the generation
  prompt (end of the prefix) would decouple effort from block 0 entirely —
  measured in §2b: tail placement wins.

### 2b. Measured 2026-08-18: where the effort sentence sits (P0, done)

Setup: live latency profile (MTP K=7, V2 runner), greedy, `max_tokens` 6k/20k.
`system` = stock template; `tail_user` = `reasoning_effort: medium` (no
template sentence) + the *same* sentence appended to the last user turn —
token-identical to a template that emits it at the tail, and it works today
without `--trust-request-chat-template` or a template file. `medium` = no
sentence. Cells are reasoning tokens (all answers correct on both grids).
Scripts/raw: `goal/sessions/9cb54694-02e7-4328-89f5-d9a7084f8f87/artifacts/effort-placement/`.

KV prefix cache, 2-turn conversation, ~9-10k-token shared history
(`vllm:prefix_cache_hits_total` deltas):

| placement | repeat, same effort | next turn switches effort |
|---|---:|---:|
| system (stock) | 6592 / 9298 hit | **0 hit — full re-prefill** |
| tail_user | 8240 / 10096 hit | **8240 hit — no extra cost** |

Behaviour, easy prompts (grid 1) and harder prompts (grid 2):

| prompt | xhigh/system | low/system | medium | xhigh/tail | low/tail |
|---|---:|---:|---:|---:|---:|
| arith | 81 | 42 | 48 | 91 | 31 |
| fact | 30 | 24 | 20 | 75 | 20 |
| code (is_prime) | 302 | 106 | 113 | 90 | 66 |
| logic | 78 | 67 | 66 | 69 | 45 |
| math (÷7 not ÷5) | 162 | 128 | 128 | 224 | 94 |
| prose | 36 | 54 | 171 | 101 | 46 |
| edit (median bug) | 435 | 302 | 730 | 1799 | 605 |
| aime (lcm pairs) | 409 | 566 | 666 | 540 | 652 |
| count (5 increasing digits) | 240 | 510 | 409 | 228 | 304 |
| algo (O(1) parentheses) | 3356 | 691 | 5138 | 1447 | 200 |
| debug (top-k heap) | 1765 | 1849 | 2431 | 6762 | 1960 |
| puzzle (5 houses) | 347 | 376 | 546 | 400 | 357 |

Read-outs (single greedy samples; run-to-run variance under MTP is real —
the same config re-run moved e.g. 163→302 tokens — so only ≥2× deltas count):

1. **Tail placement is cache-safe and steers at least as hard.** `low/tail`
   is the shortest thinking in 9 of 12 prompts; `xhigh/tail` produces the
   longest on the code tasks (edit 4×, debug 4× vs system placement). The
   sentence's influence is stronger nearer the generation point.
2. **The sentence is a weak, noisy dial on non-code tasks.** On aime/count/
   puzzle `xhigh` is often *shorter* than `low` (409 vs 566, 240 vs 510):
   the model's own difficulty estimate dominates the prompt hint. On code
   tasks it moves 3-5×. Any prompt-only "effort" is therefore not a
   reliable ladder — this is the argument for the engine-side cap/floor.
3. No accuracy signal at these difficulties (all correct); the accuracy cost
   of `low` needs harder sets (P3), not this grid.

Decisions taken from this: (a) the frontend renders `dynamic` as
`reasoning_effort: medium` + the `low` sentence appended to the last user
turn (rung 0 = "low/tail"; `medium` is not the effort level — it is the only
template value that emits *no* system-message sentence, so the tail sentence
is the sole instruction and block 0 stays effort-free), keeping block 0 identical for every effort and
letting a conversation change effort per turn without re-prefill; a
template-file change is optional; (b) the engine ladder is the primary
mechanism, the sentence is only the rung-0 prior; (c) `dynamic` in the
system message is off the table.

## 3. Signal inventory

Costs are per decode step for the batch; "free" = already materialised.

| # | signal | produced where | cost | tells us | caveats |
|---|--------|----------------|------|----------|---------|
| S1 | thinking phase + `think_count` (tokens since `<think>`) | budget holder (V1); scheduler can recompute from `req.output_token_ids` + `ReasoningConfig` ids | free | position on the ladder; the escalation trigger point (`think_count ≥ 0.75·cap`) | multi-token `<think>` sequences; prompt may already be mid-think (`continue_final_message`) |
| S2 | entropy of the target distribution at each sampled row | sampler, from the (temperature-scaled) logits row: `H = logsumexp(z) − Σ softmax(z)·z` | one fused reduction over vocab per row (152k × rows) — same order as the log-softmax vLLM already does when `logprobs` is set; negligible next to the 27B GEMMs | per-token uncertainty; high sustained H in the think phase = model is still exploring | must be computed on the same tensor on every TP rank (it is: logits are all-gathered) or decided rank-agnostically in the scheduler (§6) |
| S3 | logit margin `top1 − top2` | sampler, `topk(2)` per row | cheap | confidence of the greedy choice; robust to long flat tails where H is noisy | under top-k/top-p sampling use pre-mask logits |
| S4 | entropy trend | CPU/GPU EMA pair (fast α≈0.3 / slow α≈0.05) of S2 over the last N think tokens | free | falling = converging (let it finish or cap), rising/flat-high near the cap = still working (escalate) | needs ≥64 tokens to be meaningful; reset on `<think>` |
| S5 | sampled-token surprisal `−log p(sampled)` | same row (V1 `token_logprobs` path) | free when logprobs on, else same as S2 | complements S2 for stochastic decoding | greedy prod: equals S3-ish |
| S6 | MTP acceptance per request (accepted/drafted per pass, per-position counters) | scheduler `update_from_output` (`scheduler.py:1553-1571`), 0005 EMA | free | high acceptance = the 1-layer drafter predicts the target ⇒ text is low-surprise (boilerplate, restating) ⇒ do not escalate; low = hard tokens ⇒ escalate | latency profile only (spec off >32 seqs / off in max profile); adaptive draft length changes K per step, so use accepted/drafted, and normalise per request against its own first-256-token baseline; prose is inherently lower than code; measures draft/target mismatch, not difficulty — **corroborating evidence only, never a standalone escalation vote**; absent = unknown, not 0 |
| S7 | repetition / stall | scheduler: reuse the `RepetitionDetectionParams` detector's evidence before its terminal stop, plus (a) 16-gram repeated ≥3× in the last 512 think tokens, (b) rolling hash of 32-token windows seen before, (c) density of backtrack markers (`Wait`, `Hmm`, `Actually`, `Let me re-check` — token ids resolved at startup) per 256 tokens | free (incremental) | (a)/(b) = degenerate loop → hard-stop (clamp cap to `think_count + 32`); (c) high = productive self-verification only if S2 is falling, otherwise churn | model-specific marker list; keep as tunable strings |
| S8 | budget position vs `max_tokens` | scheduler | free | never escalate a rung the answer cannot fit after | |
| S9 | tool outcomes | typed `success/error/timeout/empty` from the Responses built-in-tool loop (`vllm/entrypoints/openai/responses/serving.py`, `context.py`); for plain Chat clients only an *opt-in, untrusted* text heuristic over the last `role: tool` message (traceback / non-zero exit / same call+args repeated) | free | a failed tool step deserves a faster ladder (start rung stays 0, but the escalation threshold drops and the first check fires earlier) | stateless: derived per request from `messages`; clients may also pass `vllm_xargs.effort_bias` |
| S10 | request-context priors (prompt time) | frontend | free | tools declared, code fences, prompt length, `tool_choice` → coding-agent shape → different ladder table | priors only; never a start rung above 0 |
| S11 | scheduler pressure (running seqs, KV usage) | scheduler | free | capacity-aware ladder: at high batch the top rung is withheld (mirrors `num_speculative_tokens_per_batch_size`) | policy, not a per-request signal |
| S12 | wall-clock / TTFT budget from the client (`vllm_xargs.deadline_ms`) | frontend | free | hard cap the top rung so the keep-alive middleware and CF 100 s edge are respected | optional |

Signals **not** worth adding now: hidden-state probes (needs training),
draft-vs-target KL (S6 already carries it), multiple-sample agreement (×N cost).

## 4. Controller (per request, in the scheduler)

State: `rung`, `cap`, `think_count`, `H_fast`, `H_slow`, `margin_ema`,
`acc_ema` (S6), `acc_base`, `loop_flag`, `marker_rate`, `checks_done`,
`escalations`, `bias` (S9/S10), `deadline`.

Per step (after `update_from_output`): update EMAs from the per-request
scalars the worker returned (§6), update S7 incrementally from the new tokens,
recompute `think_count`.

Decision points:

1. **Hard-stop** (any rung): `loop_flag` → `cap = think_count + 32`; the
   holder forces `</think>` within a few steps. Log `stall`.
2. **Escalation check** at `think_count ≥ 0.75·cap` (re-armed once per rung,
   plus a final check at `0.9·cap`), only if `rung < top` allowed by S8/S11/S12:

   ```
   score = w_H·z(H_fast) + w_M·(−z(margin_ema)) + w_T·[H_fast ≥ H_slow]
         + w_A·(acc_base − acc_ema)  # MTP only, 0 when spec off
         + bias
   escalate iff score ≥ θ_rung   (θ grows with rung: cheap to go 0→1, hard 2→3)
   ```

   `z()` = z-score against a fixed per-model calibration table produced in
   Phase 0 (not per batch — keeps decisions deterministic and replayable).
3. **Not escalated** → nothing to do; the existing holder forces the end
   sequence at `cap`.
4. Escalated → `cap = ladder[rung+1]`, send `thinking_budget_update` for that
   request in the next `SchedulerOutput`. Timing margin: the holder decides to
   force when `think_count + spec_len + 1 > budget`; with the check at 75 %
   of ≥1 024 tokens and K ≤ 7 the update lands ≥250 tokens early. One-step
   lag is inherent (scheduler sees step *t*, worker applies at *t+1*).

Rung 0 has one extra rule ("start low, but do not strand an obviously hard
task"): if the model emits `</think>` before 256 think tokens while
`H_fast` is high and `margin_ema` low (or `bias` is high from a tool error),
the **floor** actuator (§5, Phase 5) suppresses the end token once and
injects the continuation phrase, then re-arms at rung 1. Off by default until
Phase 5 proves it does not hurt.

Contract details (from the codex plan, adopted):

- **Ownership/epoch.** Exactly one owner at a time: frontend (creates the
  epoch at rung 0) → `vllm/v1/request.py` `Request` while live (scheduler is
  the only decider) → Responses context between internal tool subturns. A
  Responses built-in-tool subturn may *inherit* the current rung; any new
  external request resets to rung 0. `n > 1` sequences get independent state.
  Worker-side state is scrubbed on batch-slot add/move/remove exactly like
  `ThinkingBudgetStateHolder`.
- **Versioned control + ack.** Each budget update carries
  `(req_id, revision, budget)`; the holder applies only strictly increasing
  revisions and acks in the next `ModelRunnerOutput`. If the old cap already
  forced `</think>` before the ack, record `budget_decision_late` and finish
  normally — thinking cannot be resumed in place. The 75 % check point (the
  "decision reserve") is a benchmark gate: it passes only if the higher
  revision is acked before the force point at target load percentiles.
- **Conflict validation** before rendering: `dynamic` + explicit non-low
  `reasoning_effort`, + static `thinking_token_budget`, or +
  `enable_thinking=false` is rejected; unsupported runner (V2 before P1) fails
  closed via the existing `vllm/config/vllm.py:2126` gate.
- **Missing/NaN/insufficient samples never escalate.** Minimum window sizes,
  dwell and cooldown per rung; one rung per evaluation.

Determinism: all decisions are made in the (single) scheduler process from
CPU scalars, so TP ranks cannot diverge; the worker only applies what it is
told. Every decision is logged with the signal vector so runs are replayable.

## 5. Actuators

- **Cap** — the existing forced end sequence (both runners after the V2 port).
- **Graceful close** — configure `--reasoning-config` with
  `reasoning_end_str` set to Qwen's own budget-forcing string
  (`"\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"`)
  so the forced close is in-distribution instead of a bare `</think>`. The
  holder already forces multi-token sequences; only the parser's
  `reasoning_end_str` used for *detection* must stay `</think>` — needs a
  small split (`force_end_str` vs `reasoning_end_str`) in `ReasoningConfig`.
- **Floor** (Phase 5) — mask `</think>` (`−inf`) while `think_count < floor`
  and, when the model tried to close, bias the first token of a continuation
  phrase (`"Wait, "`) — the s1 "budget forcing" trick. Same holder, one more
  branch.
- **Reporting** — final rung + escalation count per request to the frontend.

## 6. Architecture — recommended: scheduler decides, worker measures and acts

Considered:

- (A) worker-only: extend `ThinkingBudgetStateHolder` to escalate itself.
  Least plumbing, but the holder is per rank (TP4 ⇒ 4 copies deciding from
  float reductions), does not see MTP acceptance or tool context, and does
  not exist in V2. Rejected as the decision site; kept as the actuator.
- (B) **scheduler decides, worker measures/acts** — mirrors patch 0005
  exactly (per-request EMA in `Scheduler`, knob shipped in
  `SchedulerOutput`, runner applies). Rank-agnostic, sees S6/S8/S11, one place
  to log. **Chosen.**
- (C) frontend-only (abort at cap, re-issue with `continue_final_message`):
  loses the KV/GDN state (a re-prefill on a 4x L4 box is seconds), breaks
  streaming, and cannot see per-token signals. Rejected.

Data flow to add:

1. worker → scheduler: `ModelRunnerOutput.effort_signals: dict[str, tuple]`
   (or two `np.ndarray[num_reqs]`: `mean_entropy`, `mean_margin` over the
   accepted rows of that step). **Canonical stage** (identical on V1 and V2
   or the runner is unsupported): after allowed-token/bad-words/logits
   processors and penalties, *before* the thinking-budget force,
   temperature, min-p/top-k/top-p and sampling; entropy normalised by
   `log(vocab)`; under spec decode only committed positions in commit
   order (rejected drafts excluded). Computed in the sampler on the same
   `[rows, vocab]` logits used for sampling, reduced to per-request means with
   `cu_num_logits`, appended to the existing async D2H copy
   (`vllm/v1/worker/gpu/async_utils.py:29-46` in V2; the sampled-ids copy in
   `gpu_model_runner.py` in V1). Only for requests flagged dynamic (mask), so
   the throughput profile with no dynamic requests pays nothing.
2. scheduler → worker: `SchedulerOutput.thinking_budget_updates:
   dict[req_id, int]` (and later `thinking_floor_updates`). V1: applied in
   `InputBatch` to the holder's state entry. V2: written into the new
   per-request `thinking_budget` int32 staged tensor.
3. frontend → engine: `SamplingParams.thinking_token_budget = ladder[0]`,
   `extra_args["dynamic_effort"] = {ladder, theta, bias, deadline}` (validated,
   defaults from `--reasoning-config`).
4. engine → frontend: `EngineCoreOutput.effort = (rung, escalations,
   reasoning_tokens)` on finish → `usage.completion_tokens_details.
   reasoning_tokens` + `x_effort` (or `vllm_x` block) in the response; also
   `finish_reason` stays `stop`/`length` — a stall clamp is not a new
   finish reason.

V2 port of the cap actuator (prerequisite for the latency profile): a
`ThinkingBudgetState` next to `LogitBiasState`: staged int32 tensors
`budget`, `think_count`, `in_think`, `end_progress`; a small Triton/torch
update from `sampled` + `draft_tokens` (the same rows the rejection sampler
already flattens); apply = `logits[row, end_tok] = +1e9` where
`in_think & (think_count + local_pos ≥ budget)`. Because V2's rejection
sampler needs the *whole* verify tail after the forced token to be rejected,
reuse the V1 holder's rules (force at the first over-budget draft position,
then keep forcing the remaining end tokens; rollback on rejection). Add its
CPU-checkable rules to `serve-configs/tests/`.

## 7. API surface

- `reasoning_effort: "dynamic"` (frontend-only literal; rendered as `low` for
  the template, sets budget + `dynamic_effort` extra arg). Non-dynamic values
  are untouched.
- `vllm_xargs`: `dynamic_effort_ladder: [1024, 4096, 16384, 65536]`,
  `dynamic_effort_theta`, `effort_bias: float`, `deadline_ms`,
  `dynamic_effort_floor: bool`.
- Server: `--reasoning-config '{"dynamic_effort": {"ladder": [...],
  "check_at": 0.75, "theta": [...], "weights": {...}, "loop_ngram": 16,
  "loop_repeats": 3, "max_rung_by_batch_size": [[1,8,3],[9,32,2],[33,128,1]],
  "force_end_str": "..."}}'`.
- Telemetry cardinality: labels are fixed enums only (`from`,`to` rung;
  `reason` ∈ uncertainty/mtp/repetition/stall/tool_error/boundary/cap;
  `outcome` ∈ applied/late/rejected/unsupported); numbers go to histograms
  or OTEL span attributes; never request ids, prompt/output/tool text.
- Metrics (`vllm/v1/metrics/`): `vllm:effort_final_rung` histogram,
  `vllm:effort_escalations_total{from,to}`, `vllm:effort_stall_clamps_total`,
  `vllm:reasoning_tokens` histogram; per-request fields on
  `FinishedRequestStats` (`vllm/v1/metrics/stats.py:224`).

## 8. Phases

Each phase ends with a commit under `serve-configs/patches/` (+ README row),
tests in `serve-configs/tests/`, and measurements in the goal run's
`artifacts/`. Measurement harness: `serve-configs/bench_single_stream.py`
(latency), the 128×1k/1k aggregate bench (throughput), and a correctness set
= `scripts/ling3_coding_eval.py`-style coding tasks + a math/reasoning slice
+ a prose slice, each scored pass/fail with the model's own reasoning length
recorded. Success = Pareto: ≥ 95 % of `xhigh` accuracy at ≤ 50 % of its
reasoning tokens on the mixed set, no regression on the code-edit slice.

- **P0a — prompt-lever placement (done, §2b):** tail placement is
  cache-safe and steers ≥ system placement; `dynamic` renders as
  `medium` + the `low` sentence on the last user turn.
- **P0 — telemetry (code done in patch 0009, CPU-verified; corpus run pending).** Add S2/S3 computation +
  D2H (both runners) behind a flag, log per-token `H`, margin, accepted-count,
  think position, marker hits to a JSONL sink for a corpus run at
  `xhigh`/unbounded. Offline: how well do S2-S7 at 1k/4k/16k thinking tokens
  predict "answer would still be correct if cut here"? Fit the calibration
  z-tables and initial `theta`. Also measure the cost of S2/S3 at 96 and 128
  seqs (expect < 0.2 ms/step). Deliverable: `artifacts/effort-p0/*.jsonl`
  + a notebook-free summary in this doc.
- **P1 — cap actuator on V2 (done in patch 0009, CPU-verified).** The
  deployed package already had the actuator; lane B added a torch reference,
  26 CPU tests cross-checked against the V1 holder, versioned updates + acks
  on both runners, and the greedy-path forcing fix. GPU parity check (Triton
  vs torch masks on the same inputs) is pending the GPU window.
- **P1b — shadow policy.** Run the full controller in *decide-and-log*
  mode on real traffic (no enforcement): measure escalation precision/
  recall against offline quality deltas and the decision-reserve ack
  latency before any request is actually capped.
- **P2 — controller (code done in patch 0009, CPU-verified; 50 tests).** Scheduler-side per-request state (§4), ladder,
  hard-stop, `thinking_budget_updates` in `SchedulerOutput`, both runners
  apply. Deterministic replay test from a recorded signal stream. Frontend
  `reasoning_effort: "dynamic"` + `vllm_xargs` + usage/`x_effort` + metrics.
- **P3 — tune on the box.** Sweep `theta`, `check_at`, ladder sizes,
  `max_rung_by_batch_size` on both profiles; report the Pareto vs fixed
  `low`/`medium`/`xhigh`; verify no single-stream regression when no request
  is dynamic. Choose defaults, write them into the two YAMLs.
- **P4 — tool outcomes + priors (S9/S10).** Frontend classifier for the last
  tool turn → `effort_bias`; coding-agent ladder table; verify on an agent
  transcript replay (the compaction/agent traffic that motivated the
  keep-alive work).
- **P5 — floor + graceful close experiments.** `force_end_str` recipe,
  `</think>` suppression + `"Wait, "` continuation at rung 0; keep only if P3's
  metric improves.

Effort estimate: P0 2 days, P1 3-4 days (V2 spec-decode interplay is the
risky part), P2 3 days, P3 2-3 days of box time, P4/P5 2 days each.

## 9. Relation to `docs/dynamic-reasoning.codex.md`

Adopted from it: existing repetition detector as stall evidence, V2 fail-closed
gate, versioned control + ack + `budget_decision_late`, canonical logit stage,
MTP as corroboration only, typed tool outcomes, single-owner epoch/lifecycle,
conflict validation, shadow mode, bounded telemetry cardinality. Not adopted
for this deployment: tail-adapter caution (measured here, §2b — tail is
cache-safe and steers ≥ system on Qwen3.8), V1-first ordering (our latency
profile is V2, so the V2 port is P1), and the prefill-cost estimator /
mid-generation restart machinery (restart is out of scope, §6 option C).

## 10. Risks / open questions

- Forced closes degrade quality on hard tasks; the ladder must be tuned on
  the actual eval sets, not proxies (P0/P3 own this).
- V2 + adaptive draft length + a forced token inside the draft window: the
  fused draft graphs (0005) assume the drafter's rows are consumed unchanged;
  forcing happens on the *target* rows only, so it should compose, but this
  is the P1 test to write first.
- Entropy on fp8 lm_head (0008) is slightly noisier than bf16; calibrate on
  the prod config, not on a bf16 head.
- MTP acceptance is absent on the throughput profile above 32 seqs; the score
  degrades gracefully (`w_A` term is 0) but P3 must tune both profiles.
- Template default is `xhigh`; today's clients that send no `reasoning_effort`
  get the most expensive prompt. Whether `dynamic` should become the server
  default is a product decision, not part of this plan.
- Upstreamability: P1 (V2 thinking budget) and the entropy/margin telemetry
  are model-agnostic and worth PRs; the controller is opinionated and can stay
  venv-local until it has numbers.
