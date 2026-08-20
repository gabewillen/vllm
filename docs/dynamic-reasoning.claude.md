# Dynamic reasoning effort — signals and implementation plan

Status: P0-P2 shipped as patch 0009 and GPU-validated 2026-08-18 (§2c); P6 (§11)
rewrites the controller and P7 (§12) makes the close soft and the rule
evidence-gated - both CPU-tested, GPU validation pending; P3-P5 open. §11.0 is
a standing constraint on every signal added from here on. Target: the two Qwen3.8-27B-FP8 profiles on the
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

### 2c. Measured 2026-08-18: patch 0009 on the live latency profile (P0/P1/P2 GPU validation)

Window: `vllm-qwen38` stopped, patch 0009 applied to the venv, YAML gained
`reasoning-config: '{"dynamic_effort": {...}}'`, unit drop-in
`VLLM_EFFORT_TELEMETRY=/data/effort-telemetry/latency.jsonl`; three restarts
(161-171 s to ready each). All greedy, single request unless noted.

- Triton vs torch reference of the V2 budget kernels: 300 random batches
  (K≤7 drafts, multi-token end sequences, prompt mid-think) → 0 mismatches.
- Static caps on V2 + MTP K=7: `thinking_token_budget` 100 → 99 reasoning
  tokens, 400 → 399; the answer follows.
- Signal path + acks: `dynamic` with `effort_bias=100` escalated 0→1→2,
  7 055 reasoning tokens, `late: false`; effort object on the response;
  `vllm:effort_*` metrics exported.
- Telemetry corpus (8 prompts × xhigh/low, 13 615 decode steps): entropy/logV
  mean 0.043 sd 0.046, margin mean 5.9 sd 4.5 → written into both YAMLs as
  the calibration. Confident grinding (`hard`, H 0.017 / margin 8.9) and
  open-ended uncertainty (`agent`, H 0.066 / margin 3.2) sit ~2 z apart.
- With calibration, `reasoning_effort: dynamic`: fact/edit/algo/debug/aime →
  rung 0, 19-614 tokens, correct; `hard` → rung 0, capped at 1 024 (was
  12 000+ and still wrong at xhigh/low); `agent` → rung 1, 2 901 tokens,
  full answer. `late: false` everywhere; 46-111 tok/s, unchanged.
- Sink bug found and fixed in the window: `in_think` ignored the prompt's
  trailing `<think>`; the tracker is now seeded from the prompt tail.

Not yet measured: accuracy Pareto vs fixed efforts on a real eval set (P3),
`max_rung_by_batch_size` under load, the throughput profile (YAML updated,
takes effect on its next restart), the floor actuator (P5).

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
| S9 | tool outcomes | **typed only**: `success/error/timeout/empty` from the Responses built-in-tool loop (`vllm/entrypoints/openai/responses/serving.py`, `context.py`) | free | a failed tool step deserves a faster ladder (start rung stays 0, but the escalation threshold drops and the first check fires earlier) | the text heuristic over the last `role: tool` message that this row used to allow (traceback / non-zero exit / repeated call) is **dropped**: it is lexical, per-language and per-harness. Plain Chat clients can still pass `vllm_xargs.effort_bias` explicitly — that is the client's statement, not the server guessing |
| ~~S10~~ | ~~request-context priors (prompt time)~~ | ~~frontend~~ | — | ~~tools declared, code fences, prompt length, `tool_choice` → coding-agent shape~~ | **Rejected** (§11.0). Every one of these is a prompt-structure heuristic that means something different per client, language and tokenizer. Nothing in the controller may key on them |
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
- **Soft limit** (default, §12) — from the cap onward the first token of the
  *natural* end marker gets a bias rising to `max_bias` over `ramp_tokens`
  (`bias(t) = max_bias · clamp((t − cap)/ramp, 0, 1)^curve`); the hard force
  moves to `cap + ramp_tokens`. A model that was nearly done closes on its own
  inside the ramp and nothing is forced. Implemented once as arithmetic on the
  think position (`vllm/v1/sample/soft_limit.py`) and applied by both
  actuators, for dynamic *and* static `thinking_token_budget` requests.
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
  "check_at": 0.75, "rule": "length", "uncertainty_min_auc": 0.6,
  "soft_limit": {"enabled": true, "ramp_tokens": 256, "max_bias": 10.0,
  "curve": 1.0}, "loop_ngram": 16, "loop_repeats": 3,
  "max_rung_by_batch_size": [[1,8,3],[9,32,2],[33,128,1]],
  "force_end_str": "...", "quantile_path": "..."}}'`.
- Telemetry cardinality: labels are fixed enums only (`from`,`to` rung;
  `reason` ∈ uncertainty/mtp/repetition/stall/tool_error/boundary/cap;
  `outcome` ∈ applied/late/rejected/unsupported; `kind` ∈
  natural/soft/forced); numbers go to histograms or OTEL span attributes;
  never request ids, prompt/output/tool text.
- Metrics (`vllm/v1/metrics/`): `vllm:effort_final_rung` histogram,
  `vllm:effort_escalations_total{from,to}`, `vllm:effort_stall_clamps_total`,
  `vllm:effort_close_total{kind}` (natural / soft / forced — the direct read
  on whether the soft limit is doing its job), `vllm:reasoning_tokens`
  histogram; per-request fields on `FinishedRequestStats`
  (`vllm/v1/metrics/stats.py:224`).

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
- **P0 — telemetry (done: patch 0009 + 13.6k-step corpus, calibration in the YAMLs; §2c).** Add S2/S3 computation +
  D2H (both runners) behind a flag, log per-token `H`, margin, accepted-count,
  think position, marker hits to a JSONL sink for a corpus run at
  `xhigh`/unbounded. Offline: how well do S2-S7 at 1k/4k/16k thinking tokens
  predict "answer would still be correct if cut here"? Fit the calibration
  z-tables and initial `theta`. Also measure the cost of S2/S3 at 96 and 128
  seqs (expect < 0.2 ms/step). Deliverable: `artifacts/effort-p0/*.jsonl`
  + a notebook-free summary in this doc.
- **P1 — cap actuator on V2 (done; GPU parity 300/300, static caps exact on MTP; §2c).** The
  deployed package already had the actuator; lane B added a torch reference,
  26 CPU tests cross-checked against the V1 holder, versioned updates + acks
  on both runners, and the greedy-path forcing fix. GPU parity check (Triton
  vs torch masks on the same inputs) is pending the GPU window.
- **P1b — shadow policy.** Run the full controller in *decide-and-log*
  mode on real traffic (no enforcement): measure escalation precision/
  recall against offline quality deltas and the decision-reserve ack
  latency before any request is actually capped.
- **P2 — controller (done; GPU-validated end to end incl. escalation and acks; §2c).** Scheduler-side per-request state (§4), ladder,
  hard-stop, `thinking_budget_updates` in `SchedulerOutput`, both runners
  apply. Deterministic replay test from a recorded signal stream. Frontend
  `reasoning_effort: "dynamic"` + `vllm_xargs` + usage/`x_effort` + metrics.
- **P3 — tune on the box.** Sweep `theta`, `check_at`, ladder sizes,
  `max_rung_by_batch_size` on both profiles; report the Pareto vs fixed
  `low`/`medium`/`xhigh`; verify no single-stream regression when no request
  is dynamic. Choose defaults, write them into the two YAMLs.
- **P4 — typed tool outcomes (S9).** Map the Responses built-in-tool loop's
  own typed outcome for the previous subturn onto `effort_bias`. No text
  classifier and no prompt-shape ladder table (§11.0 rules both out); a plain
  Chat client that wants the same effect passes `effort_bias` itself.
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

## 11. P6 — self-normalizing, worker-side controller (2026-08-19)

### 11.0 Standing constraint — model-agnostic signals only

**This governs §11, §12 and everything after them.** Every signal and every
rule in this controller must be a function of *the model's own output
distribution and token stream*, defined identically for any model, tokenizer,
language and quantization. Concretely:

**Allowed.** Logit-derived quantities at the canonical stage (entropy, top1−top2
margin, p(end)); token-stream statistics (n-gram novelty rate, stagnation =
tokens since the last novel n-gram, compressibility / recurrence as loop
evidence); speculative-decode acceptance; termination and length rules
(think position vs cap, `max_tokens` headroom, batch-size rung caps); and
per-model calibration that is **measured** — running quantile sketches, the
AUC gate of §12.2 — rather than typed in by hand.

**Not allowed.** Lexical marker lists (`Wait`, `Hmm`, …); prompt-structure
heuristics (prompt length, code fences, number of declared tools, keyword or
`tool_choice` sniffing); text classifiers over tool output; and per-model magic
numbers that someone fitted by eye. A threshold is acceptable only when it is
expressed in a model-agnostic unit (a percentile, a token count, a logit
delta) *and* a wrong setting is visible as a rate in the metrics.

What this already ruled out on this branch: S10 (request-context priors) is
struck from §3; S9 keeps only the Responses API's own typed tool outcomes and
loses its text heuristic; `backtrack_markers` survives **only** as an
explicitly legacy, weight-0-by-default option, and the churn detector runs on
n-gram novelty instead. The one model-specific string left in the design is
the frontend's rung-0 effort sentence (§2b) and the `force_end_str` transition
(§5) — those are *renderings the server emits*, chosen per deployment in the
YAML, not signals the controller reads.

### 11.1 What P6 answered

P6 answers four complaints about the P2 controller: it was hard-coded (a fixed
`(mean, sd)` calibration table and a `theta` per model and quantization), its
churn evidence was an English word list, it cut hard at the cap even when the
model was 20 tokens from finishing, and its decisions arrived at the worker one
or two steps late. The measurements behind the defaults are in
[`dynamic-reasoning-v3-analysis.md`](dynamic-reasoning-v3-analysis.md); the
short version is that `dynamic` already beat every fixed effort on VulcanBench
v3 (19/23 vs 14/22, 14/21, 8/16, and **zero** tasks lost to a fixed effort), and
that entropy/margin are at chance (AUC 0.41-0.54, length controlled) at telling
which requests need more thinking. P6 is therefore deliberately conservative:
it makes the rule auditable and self-calibrating, and puts the new signal -
p(`</think>`) - where the old ones failed.

### 11.2 What replaced what

| P2 | P6 | why |
|---|---|---|
| `calibration: {entropy: (mean, sd), margin: (mean, sd)}` z-scores | running per-model **quantile sketches** (t-digest, `vllm/v1/core/sched/effort_quantiles.py`), features are **percentile ranks** | a z-score on a right-skewed distribution with a point mass at 0 is not an ordinal statement; a rank is, and it means the same thing on any model/quantization |
| `theta` per rung + five weights (`w_h`, `w_m`, `w_t`, `w_a`) | one tunable per rung: `p_uncertain` | the rule is ordinal and small; a wrong setting shows up directly as an escalation *rate* |
| absolute level only | rank **plus** a within-request baseline over the first `baseline_tokens` think tokens | escalation keys on relative change |
| `backtrack_markers` density (`Wait`, `Hmm`, …) | **n-gram novelty rate** per window + the existing rolling-hash repeat count; markers keep weight 0 | language-agnostic and not model-specific |
| hard cut at the cap | **p(`</think>`) grace window** (superseded by the §12.1 soft limit, which grants the same room unconditionally and biases the close on top; the scheduler now ships `grace_tokens = 0` while the soft limit is on) | a model that is wrapping up gets `grace_tokens` more instead of being cut mid-sentence |
| bare `</think>` forced | `force_end_str` (Qwen's own "Considering the limited time…" transition), detection stays on `reasoning_end_str` | in-distribution close (§5) |
| scheduler decides, worker applies 1-2 steps later (`late`) | **worker evaluates the rule where the cap is applied** (V2); scheduler ships policy only | `late` is 0 by construction |
| ladder `[1024, 4096, 16384, 65536]` | `[1024, 4096, 16384]` | 0 of 1 199 measured requests passed 16 384 think tokens |

Nothing was deleted: `rule: "score"` selects the P2 weighted z-score,
`evaluation: "scheduler"` selects the P2 decision site, and
`backtrack_marker_weight > 0` re-arms the marker list. The V1 runner always
uses the scheduler-side path.

### 11.3 The rule

Signals per request, from the committed rows of the previous step (frozen
tuple, §6 item 1, now four wide): `entropy`, `margin`, `p_end`, `n_rows`.
`p_end` is the softmax probability of the *first token of the reasoning end
sequence*, taken from the same canonical-stage logits as the other two.

```
u        = max(rank(H_fast), 1 - rank(margin_ema))
escalate = u >= p_uncertain[rung]                  # globally uncertain
           and u - u_baseline >= baseline_rise     # and rising for this request
           and not (H_fast < H_slow and p_end rising)   # not converging
           and no loop / novelty churn
           and rank(MTP acceptance) <= acc_veto_rank    # corroboration only
```

`rank()` is a lookup in a monotone quantile grid the scheduler resolves from
its sketches once per step and ships in `SchedulerOutput.effort_policy`; the
scheduler-side and worker-side sites use the same grid and the same predicate
(`vllm/v1/sample/effort_policy.py`), so the decision does not depend on where
it is made. **Cold sketches never escalate**: below `quantile_min_samples` the
policy is `warm=False` and every request stays at rung 0.

This is `rule = "rank"`. It is no longer the default: §12.2 keeps the whole
rule but makes the two uncertainty terms conditional on measured evidence,
because §4 of the analysis says they carry none on this model.

Grace: at the `final_check_at` point, a request that did not escalate and whose
fast p(end) EMA leads the slow one gets `cap += grace_tokens`, once. Flat or
zero p(end) at the final check leaves the rung rule to escalate or close.

### 11.4 Architecture

The §6 recommendation (scheduler decides) is now split:

- **Worker** (`vllm/v1/worker/gpu/sample/effort_escalation.py`, V2 runner):
  per-slot tensors for `rung`, `cap`, `H_ema`, `H_slow`, `margin_ema`,
  `p_end_ema`, baseline, check-point flags, grace, escalations. The rule is
  elementwise torch, evaluated in `Sampler.apply_sampling_params` immediately
  before `ThinkingBudgetState` forces the end sequence, and it writes the cap
  the actuator then uses. Deterministic in its inputs, so all TP ranks agree -
  the same argument that already licenses the budget forcing. Plain torch, so
  it runs on CPU as its own reference and is unit-tested there.
- **Scheduler**: owns the sketches (fed by every dynamic request's step means,
  persisted to `quantile_path`), resolves them into the step policy, keeps the
  token-level loop/novelty detector and ships its vetoes and stall clamps, and
  mirrors the worker's `rung`/`escalations` into the response `effort` object.
  Requests with a client `deadline_ms` need a clock and stay
  scheduler-evaluated.
- Worker → scheduler: `ModelRunnerOutput.effort_reports[req_id] = (rung,
  escalations, grace_tokens, late)` with `late = 0`.

### 11.5 Calibration workflow

`serve-configs/effort_calibrate.py`:

```bash
# warm from live traffic: drive the prompt set against a server whose unit has
# VLLM_EFFORT_TELEMETRY=/data/effort-telemetry/latency.jsonl
python serve-configs/effort_calibrate.py run \
    --base-url http://localhost:8012/v1 --model Qwen3.8-27B

# fold the sink into the file named by dynamic_effort.quantile_path
python serve-configs/effort_calibrate.py build \
    --telemetry /data/effort-telemetry/latency.jsonl \
    --out /data/effort-sketches/qwen38.json --model Qwen3.8-27B

# inspect
python serve-configs/effort_calibrate.py show --sketch /data/effort-sketches/qwen38.json
```

`build` folds only in-think rows, weighted by `n_rows`, and mirrors the
scheduler's per-request acceptance EMA, so a warmed file and a self-warmed
server converge to the same distribution. The server rewrites the file every
`quantile_flush_every` observations, so it also self-maintains.

### 11.6 Defaults

`ladder [1024, 4096, 16384]` · `p_uncertain [0.85, 0.92]` (padded `0.96`) ·
`baseline_tokens 128` · `baseline_rise 0.10` · `grace_tokens 256` ·
`acc_veto_rank 0.85` · `novelty_ngram 8` / `novelty_window 256` /
`novelty_min_rate 0.2` · `backtrack_marker_weight 0.0` ·
`quantile_min_samples 2048` · `evaluation "worker"`.
The reasoning for each is in
[`dynamic-reasoning-v3-analysis.md`](dynamic-reasoning-v3-analysis.md) §5.
`rule` and `grace_tokens` were superseded by §12: the default rule is now
`"length"` and the grace window is folded into the soft-limit ramp.

### 11.7 Not yet measured

p(`</think>`) itself (the column ships with P6, so it is absent from the v3
telemetry), the novelty rate (needs a token stream the sink does not keep), the
on-GPU cost of the escalation tensors, and — new with §12 — what fraction of
force-closed requests the soft-limit ramp converts to a natural close
(`vllm:effort_close_total{kind}` answers it directly) and the AUC the
calibration pass reports on this model. GPU validation of P6/P7 is the next
window. One harness change is worth doing first: VulcanBench discards the
response `effort` object (`TokenUsage` keeps only prompt/completion tokens), so
rung, escalations and `late` are not recoverable from a sweep - persisting
`LLMResponse.raw` would make every question in the analysis directly
measurable.

## 12. P7 — soft-limit close and the evidence-gated rule (2026-08-19)

P7 changes two things and deletes nothing. Both follow §11.0: the new actuator
is arithmetic on the model's own think position and the first token of its own
end marker, and the new rule replaces a hand-asserted premise with a measured
one.

### 12.1 The soft-limit close

The cap was a cliff: at `think_count == cap` the end sequence was forced,
whatever the model was in the middle of. From now on the cap starts a ramp
instead (the idea is lfg.cpp's "reasoning soft-limit sampler"):

```
bias(t) = max_bias * clamp((t - cap) / ramp_tokens, 0, 1) ** curve
```

added to the logit of the **first token of the natural reasoning end
sequence** — the marker the parser detects, not the (possibly graceful,
multi-token) forced close, so a soft close reads like the model's own. The
existing hard force moves to `cap + ramp_tokens`, unchanged in every other
respect: still spec-decode aware, still token-by-token through a multi-token
`force_end_str`, still rolled back when a draft is rejected before the forced
position.

- **Defaults:** `ramp_tokens 256`, `max_bias 10.0` logits, `curve 1.0`
  (linear), `enabled true`. All three are model-agnostic units.
- **Both actuators, one formula.** `vllm/v1/sample/soft_limit.py` holds the
  three lines; the V1 `ThinkingBudgetStateHolder`, the V2 Triton kernel and
  its torch reference each apply them. Under spec decode the bias lands on the
  target rows at their own think positions, exactly like the force does: row
  `p` of a K-token draft window sits at `think_count + p`, so a window can
  straddle the ramp and the force (rows 0-3 biased, rows 4+ forced) — the
  K=7 case is a test.
- **Dynamic and static.** A plain `thinking_token_budget` request gets the
  same ramp, because "close gracefully" is not a property of the ladder. A
  server with no `dynamic_effort` block has no soft limit at all, so a
  deployment that never asked for this keeps exact hard-cap semantics.
- **`close_kind`.** Every request reports how its think block ended:
  `natural` (before the cap, or exactly at it — the bias is 0 there), `soft`
  (inside the ramp: the biased marker won and **nothing was forced**), or
  `forced` (the hard force fired at `cap + ramp_tokens`). It rides on the
  response `effort` object and on `vllm:effort_close_total{kind}`. The v3
  telemetry says 2.9 % of requests were force-closed; this counter is the
  direct measurement of what fraction of those the ramp converts.
- **It composes with the other limits by aiming the force point, not the cap.**
  Two places had a hard guarantee that a ramp would otherwise stretch, so both
  now subtract it: the `max_tokens` headroom (S8) caps a rung at
  `max_tokens - answer_reserve_tokens - ramp`, so the forced close still leaves
  the answer its reserve; and the stall clamp targets
  `think + hard_stop_margin - ramp`, so the loop guard's close lands where it
  always did, with the bias already saturated on the way. A cap cannot go
  negative, so a loop detected inside the first `ramp` think tokens closes at
  `ramp` instead - and the request's own `repetition_detection` stop, which
  none of this weakens, remains the terminal guarantee.
- **The p(end) grace window is retired into this.** The grace granted
  `grace_tokens` *conditionally*, when the fast p(end) EMA led the slow one.
  The ramp grants the same room *unconditionally* and biases the close on top,
  so a p(end)-gated window on top of it would only be the same tokens with an
  extra predicate. Simplification: grace = ramp. The scheduler ships
  `grace_tokens = 0` while `soft_limit` is active; setting
  `soft_limit.enabled = false` brings the old window back.

### 12.2 The default rule: termination/length, uncertainty gated by evidence

`docs/dynamic-reasoning-v3-analysis.md` §4 is unambiguous: with length
controlled, entropy and margin sit at AUC 0.41-0.54 at separating requests that
needed more thinking, in both the absolute level and the within-request rise.
P6 kept escalating on them anyway, just more conservatively. P7 stops asserting
the premise and measures it instead.

**`rule = "length"` (the new default).** At a check point:

```
escalate iff  still in the think block at check_at
          and not looping / churning              (n-gram novelty + recurrence)
          and p(end) not rising                   (not converging)
          and rank(MTP acceptance) <= acc_veto_rank
```

**`use_uncertainty`.** On top of that, and only when the model's calibration
file reports a discriminative AUC of at least `uncertainty_min_auc` (default
**0.60**) for the entropy/margin features on *that* model:

```
          and u >= p_uncertain[rung]  and  u - u_baseline >= baseline_rise
          and H_fast >= H_slow
```

No AUC in the file means no evidence, which means the features stay off — they
are still computed, logged and sketched, they just do not vote. `rule="rank"`
turns them on unconditionally (the P6 behaviour) and `rule="score"` still
selects the pre-P6 weighted z-score; both remain selectable.

Two consequences worth stating plainly. First, with the features off there is
no distribution to warm, so the policy ships `warm=True` immediately and a
fresh deployment escalates from its first request instead of behaving like a
fixed rung-0 cap until the sketches fill. Second, "not converging" reduces to
"p(end) not rising": the entropy trend is itself an uncertainty feature and
leaves with them.

**Measuring the AUC.** `serve-configs/effort_calibrate.py build` now computes
it from the same telemetry sink it folds into the sketches, with the analysis
doc's method and label:

- *positive* = the request needed the higher rung: it passed the rung-0 cap and
  then closed naturally, rather than landing within `cap_slack` of a higher cap;
- *negative* = it closed at or before the rung-0 cap;
- length control: both groups must be at least `2 * window` (default 256) think
  tokens, so a request's first-`window` and last-`window` means are disjoint;
- six features — entropy first / last / rise, margin first / last / drop — each
  scored as a rank statistic (Mann-Whitney, ties at 0.5) **in the direction the
  rule assumes**, so a value under 0.5 means the rule's premise is backwards on
  this model. `uncertainty_auc` is the best of the six.

The block is written into the sketch file next to the digests, is additive (a
file from before this change loads and reports "no evidence"), and survives the
server's periodic reflush. At startup the scheduler logs which features are
active and why, e.g.
`entropy/margin rank features OFF - calibration AUC 0.538 < 0.60`.

### 12.3 What we tried and what carries information

| signal | how it was tested | verdict |
|---|---|---|
| entropy (normalised) | AUC on 1 199 requests, length controlled (v3 §4) | **at chance** (0.41-0.49). Kept as telemetry; gated off by default |
| top1−top2 margin | same | **at chance** (0.50-0.54). Same treatment |
| entropy rise over the request's own baseline | same | **at chance** (0.42). The un-controlled 0.26 was a length artifact |
| margin drop over the baseline | same | **at chance** (0.54) |
| a fixed `(mean, sd)` z-table per model | fitted 2026-08-18, then re-read against the distribution | **wrong shape**: entropy is right-skewed with a point mass at 0, so a z-score is not an ordinal statement. Replaced by quantile sketches |
| backtrack-marker density (`Wait`, `Hmm`, …) | never load-bearing; English-only | **rejected** by §11.0. Weight 0, legacy option only |
| prompt-shape priors (length, code fences, #tools) | — | **rejected** by §11.0 before implementation |
| n-gram novelty rate | three of the four v3 failures show repeated identical probes; not retro-computable (the sink keeps no token stream) | **directionally right, unvalidated.** Language-agnostic, so it stays |
| MTP acceptance | global 0.733, per-request p85 ≈ 0.95 | **corroboration only**, as a veto. Never a standalone vote |
| think position vs cap (termination/length) | 35 of 1 199 requests force-closed; escalation fires on ~7 % | **the load-bearing signal.** It is the one thing that is definitionally about "the model is not done" |
| p(`</think>`) | ships with P6/P7; absent from the v3 telemetry | **the new one.** Definitionally "about to finish", measured from the same logits. Now both the convergence test and, through the ramp, the actuator |

### 12.4 Next

In order. Everything here obeys §11.0.

1. **Offline policy simulator over recorded unbounded-thinking traces.** Record
   greedy traces with no cap and the full per-step signal tuple, then replay any
   stop/escalate policy against them exactly: a policy that only ever *cuts*
   earlier than the recording is answerable from the trace alone, and only a
   policy that changes the token stream needs regeneration — and then only from
   the cut point, reusing the prefix. This turns every remaining question in
   §12.3 from an argument into a measurement, and it is the prerequisite for
   the four items after it.
2. **P(end) hazard.** The ramp uses p(end) as a level; the honest quantity is a
   hazard — P(the block closes within the next *n* tokens | it has not yet).
   Fit it once per model from the same traces, and the check point stops being
   a fraction of the cap and becomes "the hazard says this will not close on
   its own".
3. **Novelty collapse.** The novelty rate as a *trend* rather than a threshold:
   the derivative of distinct-new-n-grams per window is a loop starting, and it
   fires before the current absolute cut-off does.
4. **Stagnation cap.** Tokens since the last novel n-gram, capped directly. The
   simplest possible termination rule that is not a token count, and it is
   defined identically for any tokenizer.
5. **Compressibility loop detection.** Recurrence / compression ratio of the
   think tail as loop evidence, replacing the n-gram + rolling-hash pair with
   one statistic that does not need a window length chosen by hand.
6. **Decision-point margin.** Not the per-token margin (at chance) but the
   margin at the *branch* tokens — the positions where the distribution is
   genuinely multimodal. Selecting those positions is itself a distributional
   test, so it stays inside §11.0.
7. **MTP rank.** Where the target's choice sits in the drafter's ranking, not
   just accept/reject. Free on the latency profile, strictly more information
   than the acceptance ratio.
8. **Typed tool-outcome priors — Responses API only.** The previous subturn's
   typed `error`/`timeout` outcome raises the sensitivity for the next one.
   Only the built-in-tool loop's own typed outcomes; no classifier over tool
   text, and no plain-Chat inference (a Chat client that wants this passes
   `effort_bias`).
9. **Encoder / hidden-state probe (P7-probe).** Last. It needs training data,
   it is per-model by construction, and it should only be reached once 1-7 are
   exhausted — by then the simulator can score it honestly against them.

Explicitly *not* on this list: counterfactual replay by re-running saved
requests with forced closes at each rung. Item 1 subsumes it and is strictly
cheaper — a forced close at rung *r* is a prefix of the unbounded trace, so the
label comes out of the recording instead of a fresh generation.

## 13. P8 — hidden-state effort v3 (the plan; §13.10 records what shipped)

Decision (user, 2026-08-19): the **label-free hidden-state signal is the design
for dynamic-effort v3**. Measurements, code and raw results:
[`effort-hidden-probe.md`](effort-hidden-probe.md) and
`/shared/vllm/work/router-proto/hidden/`. This section is the implementation
plan; nothing here is built yet.

### 13.0 Why this is inside §11.0, not an exception to it

§12.4 item 9 parked "encoder / hidden-state probe" last, because "it needs
training data, it is per-model by construction". The first half turned out to be
wrong. The shipping signal is **retrieval, not a fitted head**: a memory of
pooled prefill states the server itself observed, keyed by cosine, valued by the
reasoning length that request actually spent. Nothing is fitted; the only
constants are percentile ranks of running digests — the same self-calibration
§11.0 already blesses for entropy and margin. The second half is true and
unchanged: it is per-model, exactly as the quantile sketches are.

Checked against §11.0's list: no lexical markers, no prompt-structure heuristics
(and the measurement shows the signal is *not* a prompt-length proxy — rank
partial Spearman 0.572 with length removed), no text classifier over tool
output, no hand-fitted numbers. The one new thing §11.0 did not anticipate is
that the model's own **hidden state** counts as "the model's own output" — it is
the vector `lm_head` consumes, defined identically for any transformer,
tokenizer, language and quantization, and it is free because the prefill happens
anyway.

### 13.1 What the measurement says (headline)

On the 689 natural closes of the shared dataset, leave-one-task-out, out of fold:

| | AUC long-think | within-run | Spearman vs think tokens |
|---|---:|---:|---:|
| kNN-16 over `last_final`, label-free | **0.850** [0.818, 0.877] | **0.762** | **0.685** |
| trained probe (upper bound, not shipped) | 0.866 | 0.778 | 0.657 |
| prompt length (the free control) | 0.729 | 0.581 | 0.494 |
| entropy / margin (what §12.3 measured) | ≈ chance | ≈ chance | ≈ chance |

**Pooling to ship: the last prompt token's final hidden state.** Mean pooling is
far worse (0.622 vs 0.839 at matched k = 8, and 0.665 vs 0.789 on
TwinRouterBench); the layer-32 variant could not be measured on this build
(§1 of the probe doc) and is not required by the plan.

External check, same capture path, no training: on **TwinRouterBench**'s 336
execution-verified agentic-coding steps (Apache-2.0, four tiers) the same
label-free retrieval separates the `high` tier at **AUC 0.789** [0.742, 0.836]
against 0.584 for prompt length and 0.492 for step index; a memory built from
*our* loop and transferred to theirs still scores 0.620 [0.559, 0.678], which is
the argument for filling the memory **online from served traffic** rather than
shipping a fixed one.

**Signals to ship: novelty + kNN over the online memory. Not within-session
surprisal** — as cortext defines it, it is a mild *negative* signal here
(AUC 0.284, i.e. 0.716 inverted) and inverting it just recovers a position
proxy. It stays out of the controller; the memory carries the load.

### 13.2 Files

New:

| file | contents |
|---|---|
| `vllm/v1/worker/gpu/effort_hidden.py` | pooling of the prompt rows (the prototype is `work/router-proto/hidden/hidden_capture.py`; drop the JSONL sink, keep `observe`) |
| `vllm/v1/core/sched/effort_memory.py` | the online memory: ring buffer, cosine kNN, novelty, neighbour spread, persistence |
| `serve-configs/tests/test_effort_memory.py`, `test_effort_two_phase.py` | see §13.7 |

Changed:

| file | change |
|---|---|
| `vllm/config/reasoning.py` | `HiddenEffortConfig` nested in `DynamicEffortConfig` (`enabled`, `memory_size`, `min_entries`, `k`, `temperature`, `q_mid`, `q_high`, `novelty_gate_q`, `spread_gate_q`, `memory_path`, `flush_every`) |
| `vllm/entrypoints/openai/chat_completion/dynamic_effort.py` | stop committing to one sentence: render **one tail token-id list per rung** (server constants) and hand them to the request instead of mutating `messages` |
| `vllm/v1/request.py` | `RequestStatus.WAITING_FOR_EFFORT_DECISION` (the `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` / `WAITING_FOR_REMOTE_KVS` pattern already exists); `Request.effort_tail_variants`, `Request.effort_body_len` |
| `vllm/v1/outputs.py` | `ModelRunnerOutput.effort_prefill_states: dict[str, torch.Tensor] \| None`, next to the existing `effort_signals` (:360) |
| `vllm/v1/worker/gpu/model_runner.py` | pool at the existing call site in `sample_tokens` (after `pcp.maybe_restore_pcp_for_sampling`, before `self.sample`) and put the fp16 vector in the output for requests that are awaiting a decision |
| `vllm/v1/core/sched/scheduler.py` | schedule the body; on the output carrying the vector, decide, append the chosen tail, requeue; insert into the memory at finish |
| `vllm/v1/core/sched/effort_controller.py` | `EffortState.start_rung`; `decide_start_rung(estimate, novelty, spread, sketches, cfg)` |
| `vllm/v1/worker/gpu_model_runner.py` (V1) | not touched — V1 keeps today's behaviour; the hidden path is V2-only, like the rest of the latency profile |

### 13.3 The two-phase prefill decision point

Today `apply_dynamic_effort` appends `cfg.low_effort_sentence` to the last user
turn in the frontend, before tokenization, and the rung is fixed from then on.
The prompt already has the right seam:

```
[ body ................................. ][ tail ]
system + every turn + last user content    effort sentence + <|im_start|>assistant\n<think>\n
identical for every rung                   ~20-40 tokens, one variant per rung
```

1. **Frontend.** `apply_dynamic_effort` sets `request.reasoning_effort =
   cfg.render_effort`, renders the tail once per rung, submits with
   `prompt_token_ids = body` and stashes the variants on the request. Everything
   else it does today (ladder/theta validation, `thinking_token_budget =
   ladder[0]`, `extra_args["dynamic_effort"]`) is unchanged.
2. **Scheduler.** The request enters `WAITING_FOR_EFFORT_DECISION`. It is
   scheduled exactly like a normal prefill; when its body finishes it produces no
   sampled token (`num_tokens == body_len`), so nothing is emitted to the client.
3. **Runner.** At the existing `sample_tokens` call site the pooled
   `last_final` row is already in hand; for requests in that state it is copied
   to host (`[5120]` fp16, 10 KB) and returned in
   `ModelRunnerOutput.effort_prefill_states`. Take the row from the request's
   **logit index** (`input_batch.logits_indices[cu_num_logits_np[i]]`), not from
   the prefill accounting: a 100 %-prefix-cached body schedules no prompt tokens
   and the prototype silently produced no record for 11 such requests.
4. **Scheduler, next step.** `effort_memory.query(vec)` returns
   `(estimate, novelty, spread, n_entries)`; `decide_start_rung` maps it through
   the digests (§13.5); the chosen tail token ids are appended to the request,
   `num_tokens` is bumped, `start_rung` is recorded on `EffortState`, the
   thinking budget is set to `ladder[start_rung]` through the **existing**
   versioned `thinking_budget_updates` path, and the request moves to `WAITING`.
5. The tail prefills in the next step and generation starts.

**Cost.** One extra engine step per dynamic request — order 10–15 ms at TP4 on
the L4 box, a few tenths of a percent of TTFT on a median 13 k-token agent
prompt, a few percent on a short one. The probe itself is a `[M,5120]@[5120]`
matvec: 21 MFLOP at `M = 4096`, i.e. free. Prefix-cache behaviour improves
slightly: the body is byte-identical across rungs, so one body per conversation
is cached instead of one per (conversation, rung).

**Failure modes and fallbacks.** Cold memory (`n_entries < min_entries`) → skip
the split entirely, render rung 0 in the frontend as today. Missing vector
(preemption, PP, V1 runner, a non-last PP rank) → same fallback. The decision
never blocks a step: if it is not ready, the request waits one more step, which
is the same cost as the split itself.

### 13.4 The online memory

`vllm/v1/core/sched/effort_memory.py`, scheduler-side, one per engine core.

Per entry (**inserted at request finish, from the engine's own observation** —
no labels, no offline corpus):

| field | bytes | note |
|---|---:|---|
| pooled `last_final`, fp16, L2-normalised | 10 240 | the key |
| `reasoning_tokens` | 4 | the value; `log1p` at query time |
| `close_kind` (natural / soft / forced) | 1 | **only `natural` entries contribute a value** — soft and forced closes are right-censored, so they are inserted as keys with `value = None` and are counted for novelty but skipped by the kNN average |
| `effort_used` (start rung, final rung, escalation count) | 3 | lets the memory answer "where did the ladder end up", and lets a rung change invalidate stale values |
| session id, monotonic insert index | 16 | same-session exclusion and eviction |
| **total** | **≈10.3 KB** | |

- **Size:** `memory_size = 4096` → **≈42 MB** of host RAM. Measured: 512 entries
  already reach AUC 0.842 vs 0.843 at 2048 and 0.808 at 128, so 4096 is
  comfortable headroom, not a requirement.
- **Eviction:** FIFO ring, with at most `ceil(memory_size / 64)` entries per
  session id, so one long conversation cannot evict the memory. (Pure FIFO is
  adequate at 4096; the per-session cap is cheap insurance.)
- **Query:** cosine against the ring (one fp16 GEMV), top-`k = 16`,
  `softmax(cos / 0.05)` weights over `log1p(reasoning_tokens)` of the entries
  that have a value; `novelty = (1 - max cos)/2`; `spread` = the weighted stdev
  of those neighbour values.
- **Persistence:** written next to `quantile_path` with the same atomic-replace
  flush every `flush_every` inserts, so a restart warms instead of running
  blind. Version the file; drop it on a model, `hidden_size` or ladder change.
- **Cardinality:** the memory itself emits no per-entry telemetry. Metrics are
  aggregate only — `effort_memory_entries`, `effort_memory_hit_rate`,
  `effort_start_rung_total{rung}`, `effort_decision_skipped_total{reason}` —
  four bounded label sets, no request or session ids.

### 13.5 The mapping is asymmetric

TwinRouterBench's finding, adopted: **one under-routed step can kill a
trajectory; over-routing only costs tokens.** So the map raises freely and lowers
only on confidence. All cuts are percentile ranks of running digests, so nothing
is a per-model constant:

```
q = rank(estimate, digest)
q >= q_high (0.60)                                        -> rung 2
q >= q_mid  (0.35)                                        -> rung 1
q <  q_mid AND novelty <= rank 0.6 AND spread <= rank 0.6 -> rung 0
otherwise                                                 -> rung 1     # unsure -> safe
```

Two gates guard the *downward* band only: `novelty` (the memory has nothing
similar, so it cannot be trusted to say "easy") and `spread` (the neighbours
disagree). Measured on the 689 natural closes
(`work/router-proto/hidden/results_dynamicv2/policy.json`):

| starting policy | rung mix (low/med/high) | starts under-provisioned | of those, at rung 0 | p90 think tokens in the low band | mean granted budget |
|---|---|---:|---:|---:|---:|
| today: always rung 0 | 689 / 0 / 0 | **12.3 %** (85) | 85 | **1 471** | 1 024 |
| symmetric tertiles | 230 / 229 / 230 | 0.44 % (3) | 0 | 61 | 7 173 |
| **asymmetric, gated** (0.35 / 0.60) | 194 / 219 / 276 | **0.44 %** (3) | **0** | **59** | 8 153 |
| asymmetric, ungated (0.40 / 0.65) | 276 / 172 / 241 | 0.44 % (3) | **1** | 66 | 7 164 |

Read this carefully. "Starts under-provisioned" means the request would have had
to escalate to finish — today that is 12.3 % of natural closes, and the 90th
percentile of the requests today's policy sends to the 1 024 cap actually wants
1 471 tokens. The router cuts that to 0.44 %, and the requests it does send to
rung 0 want a p90 of **59** tokens. The gate's own contribution is small but
real and in the predicted direction: at the most aggressive downward cut it is
the difference between 0 and 1 under-routed rung-0 starts. On 689 requests with
3 total under-routes that is not a significant difference — the gate is
insurance, priced at ~3–6 % of mean granted budget, and should ship on that
argument, not on this table.

**What this table cannot say:** the token *cost* of the higher starting rungs. A
bigger cap is a ceiling, not an instruction; the thing that actually makes the
model think less is the rung-0 sentence. Only a rerun measures that — §13.8.

### 13.6 What is removed and what is kept

**Removed** (all pre-P6/P7 machinery that the v3 measurement already found
inert, plus what the prefill decision makes redundant):

- `rule = "score"` and its whole surface: `theta`, `w_h`, `w_m`, `w_t`, `w_a`,
  the fixed `calibration` z-table. Already deprecated in §11; P8 deletes them
  (pre-1.0, venv-local, no deployed consumer — the hard-cutover rule applies).
- `backtrack_marker_weight` and `QWEN`-specific marker plumbing. §11.0 already
  reduced it to a weight-0 legacy option; delete it.
- `p_uncertain`, `baseline_rise`, `baseline_tokens` and the entropy/margin rank
  features, **and** the `uncertainty_min_auc` gate that exists only to hold them
  off. §12.3 measured them at chance on this model and the gate has kept them
  inert ever since; the hidden-state signal is what they were a proxy for.
- `grace_tokens` (already subsumed by the soft-limit ramp).

**Kept:**

- The **ladder** and the mid-generation escalation. The prefill decision sets the
  *starting* rung; the live rule still climbs. This is what makes a wrong
  prefill decision recoverable, and it is why the asymmetric map can afford a
  rung-0 band at all.
- The **soft-limit close** (§12.1) and `graceful_force_end` / `force_end_str`.
- The **loop-stall guard** — `loop_ngram`, `loop_repeats`, `loop_window`,
  `hash_window`, and the n-gram novelty churn detector. Explicitly retained:
  it is the only thing that stops a degenerate loop, it is language-agnostic, and
  it is orthogonal to how much thinking was budgeted.
- `check_at` / `final_check_at`, `dwell_tokens`, `cooldown_tokens`,
  `min_samples`, the p(end) convergence test and the MTP acceptance veto.
- **`close_kind` telemetry — kept and promoted.** It stops being only a metric:
  it is the field that decides whether a finished request contributes a *value*
  to the memory (§13.4). Same for `reasoning_tokens` and the rung/escalation
  counters. The per-token entropy/margin JSONL sink stays available behind
  `VLLM_EFFORT_TELEMETRY` for calibration work but is no longer on any decision
  path.
- The **`X-Request-Id` join**: every effort telemetry record must carry the
  client-visible request id. The v3 sink did not, which is why the offline
  analysis could join only 244 of 791 requests; the capture prototype fixed it
  for free.

### 13.7 Tests

| test | what it pins |
|---|---|
| `test_effort_memory.py::test_ring_evicts_fifo_and_caps_one_session` | eviction, per-session cap, `memory_size` respected |
| `…::test_censored_closes_are_keys_not_values` | forced/soft closes count for novelty, never for the kNN average |
| `…::test_query_matches_numpy_reference` | the GEMV kNN against a plain numpy implementation, fp16 tolerance |
| `…::test_cold_memory_returns_none` | `< min_entries` → `None`, and the caller falls back to today's path |
| `…::test_persistence_roundtrip_and_version_mismatch` | atomic flush, warm start, drop on model/dim/ladder change |
| `test_effort_two_phase.py::test_body_prefill_emits_no_token` | a request in `WAITING_FOR_EFFORT_DECISION` produces no client-visible token |
| `…::test_tail_appended_and_budget_set` | chosen tail ids land on the request; `thinking_budget_updates` carries `ladder[start_rung]` with the right version |
| `…::test_decision_unavailable_falls_back_to_rung0` | cold memory / missing vector / V1 runner → today's behaviour, byte-identical prompt |
| `…::test_fully_cached_body_still_yields_a_vector` | the 100 %-prefix-cache-hit case the prototype missed |
| `…::test_asymmetric_map_never_lowers_without_both_gates` | the downward band is unreachable when novelty or spread is above its rank |
| `…::test_prefix_cache_body_shared_across_rungs` | two requests with the same body and different rungs share body blocks |
| `serve-configs/tests/test_effort_hidden_pooling.py` | pooling arithmetic over chunked prefill: sum over chunks == sum over the whole prompt; last row is the last prompt token |
| GPU: `serve-configs/bench_single_stream.py` before/after | the split costs ≤ 1 engine step of TTFT and no decode regression |
| GPU: spec-decode parity | MTP draft/verify path unchanged — the capture must not alter `aux_hidden_states` reaching the speculator |

Regression guard from this measurement: `test_effort_hidden_pooling.py` must
also assert that the runner tolerates a model that ignores an auxiliary-output
request (the layer-32 finding in the probe doc) rather than asserting on the
output shape.

### 13.8 Rollout

1. **Shadow first.** `hidden_effort.enabled = true`, `shadow = true`: the
   decision is computed, logged (`effort_start_rung_shadow_total{rung}`) and
   thrown away; the request runs at rung 0 as today. Confirms the memory warms,
   the split never fires, and the cost is zero. One day of real traffic.
2. **Offline simulator.** §12.4 item 1 (recorded unbounded-thinking traces) now
   has a second consumer: replay the asymmetric map against the traces and get
   the token cost of the higher starting rungs that §13.5 could not measure.
3. **New VulcanBench v3 run set beside `dynamic` and `dynamic-v2`.** Add
   `dynamic-v3` to `VULCANBENCH_ADAPTIVE_EFFORTS`
   (`/shared/VulcanBench/harness/effort.py` already maps unknown adaptive levels
   to a verbatim `reasoning_effort`, so only the map entry
   `"dynamic-v3": "dynamic"` and the column name are needed). Run the full v3
   suite, 23 tasks, same harness settings as the `dynamic-v2` column, so the
   comparison is column-to-column on the existing leaderboard. The bar to clear:
   `dynamic-v2` scored **0.857 mean functional at 1 071 completion tokens per
   request**; `dynamic` scored **0.826 at 581**. v3 has to sit on or above that
   frontier, and the interesting cell is "dynamic-v2 quality at dynamic cost".
4. **Then and only then** enable it on the latency profile YAML, as a new
   numbered patch under `serve-configs/patches/`, with the capture path
   (`VLLM_EFFORT_HIDDEN_CAPTURE`) kept env-gated and off in production.

### 13.9 What this does not settle

- Multimodality is an argument, not a measurement: the v3 traces are text-only.
  The hidden state contains the image tokens by construction, which is the whole
  reason this beats a text encoder on screenshots — but it is untested here.
- The memory's values all come from one engine configuration (`dynamic-v2`).
  A ladder change makes old values stale; `effort_used` is stored so that a
  future version can weight or drop them.
- 23 tasks, 689 labelled requests, one model, one box.

### 13.10 Addendum — what shipping P8 changed (2026-08-19)

P8 is implemented (venv patch `serve-configs/patches/0012`, branch
`qwen3.8-27B-effort-v3`) and benchmarked as the VulcanBench v3 `dynamic-v3`
column. Results, the placement measurement and the full bug list:
[`dynamic-reasoning-v3-results.md`](dynamic-reasoning-v3-results.md).

**What `dynamic` is now.** One effort **level** per request, decided from the
prompt's pooled prefill hidden state *before the model thinks*, rendered as that
level's sentence at the **true tail** of the prompt. That sentence is the whole
actuator. Everything in §1-§12 that acted on the think block is removed from
this path: the rung ladder and its caps, mid-generation escalation, the quantile
sketches and the rank rule, the soft-limit ramp, the forced close, and the
loop-stall clamp. A dynamic request sets no thinking budget at all; the model
ends its own reasoning, bounded only by the client's `max_tokens` and timeouts,
exactly as it is at a fixed effort level. `close_kind` is `natural` or
`client-limit`, and only a natural close contributes a value to the memory.

Three corrections to §13.3, all found on the GPU:

1. **The sentence has to move to the tail, and that is a measurement, not a
   preference.** §2b measured placement on single-turn prompts, where the last
   user message is the end of the prompt. An agent turn ends in a tool result,
   and there the last-user-message placement is *inert*: the `xhigh` wording
   moves reasoning length 1.14x against no sentence, versus 1.23x up / 0.78x
   down for a trailing user message. So §13.3's seam is not where the sentence
   was, and moving it there is what makes the split cover agent traffic at all.
2. **A non-final prefill chunk cannot end wherever it likes.** The served
   profile is hybrid GDN with prefix caching, so vLLM uses the Mamba `align`
   cache mode and widens the attention block to **1648 tokens**; in that mode a
   non-final chunk may only end on a block boundary. The body chunk ended at the
   seam, the aligning split clipped it to zero, and every dynamic request hung
   with the engine spinning on one waiting request. The boundary is now the
   largest multiple of the block at or before the seam.
3. **`split_min_fraction` is the safety net, not the router.** A prompt with no
   boundary covering at least that fraction of itself takes no decision and runs
   at `default_level` with a byte-identical prompt. With a 1648-token block that
   is prompts under about one block.
4. **The vector belongs to a *step*, not to a token counter.** `update_from_output`
   resolved the level for any held request whose `num_computed_tokens` had
   reached the body boundary. Async scheduling schedules step N+1 before it
   processes step N's output, so a body wider than one prefill chunk — with a
   1648-token block, every prompt whose boundary is 3296 or more — already had
   its counter at the boundary when the *earlier* chunk's output arrived. That
   output captured nothing, the decision was spent against it with reason
   `no-vector`, and the real vector, arriving one step later, was dropped on a
   request that was no longer pending. It cost **148 of 479** requests in the
   greedy `dynamic-v3` sweep (31%, held timeouts 0, preemptions 0): every long
   prompt ran at `default_level`, and none of them entered the memory either,
   which is why the memory warmed on short prompts only. The resolve is now
   gated on the step's own `effort_prefill_capture` list — the only thing that
   says the vector came back with *this* output. Regression:
   `test_a_multi_chunk_body_is_decided_by_the_step_that_computed_it`.

Verified on the box, 2026-08-19 (`/tmp/dflash2-arms/V3-fix.log`): 160 real
VulcanBench agent turns replayed at `reasoning_effort: dynamic`, concurrency 12,
cold prefix cache and cold memory. **`no-vector` 0 of 151 decisions, held
timeouts 0**, and the first five traced bodies — 3 296, 11 536, 13 184, 32 960
and 34 608 tokens, every one of them a multi-chunk prefill — all report `vector
present`. On the same server before the fix the same trace only ever reported
`vector present` for a 1 648-token (single-chunk) body, and the 128-entry
histogram read `no-vector: 77` of 242.

§13.5's open question is answered differently than it framed itself: the token
cost of a higher level is now *entirely* the sentence, because there is no cap
to grant. §13.6's deletions are done and then some — what remains is the
worker-side escalation tensors, which are inert (nothing sets `worker_eval`) and
are the next cutover on this branch.
