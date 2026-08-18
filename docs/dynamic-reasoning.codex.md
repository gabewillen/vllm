# Dynamic reasoning implementation plan

## Status and objective

This document proposes a source-grounded implementation plan. It does not claim
runtime correctness, performance, model quality, or production readiness.

Dynamic reasoning is opt-in and starts every **top-level API request** at `LOW`.
It may move monotonically to `MEDIUM` and `HIGH` when bounded evidence indicates
that more reasoning is useful. Evidence comes from entropy, top-logit margin,
entropy trend, MTP acceptance, repetition or stall detection, and typed tool
outcomes. The normal escalation action raises an in-flight thinking-token budget
and does not re-render the prompt or re-run prefill.

An internal tool subturn created by the Responses built-in-tool loop is part of
the same logical request. It may inherit the logical request's current level. A
new external request always creates a new epoch at `LOW`, even if it references
the same conversation or a previous response.

## Goals and non-goals

Goals:

- preserve existing behavior unless dynamic reasoning is explicitly enabled;
- define one deterministic, bounded policy with explicit ownership and cleanup;
- use signals at the point where they are available without transferring full
  logits to the CPU;
- prefer actions that preserve generated tokens and prefix-cache reuse;
- support tool-boundary conditioning only through model-specific contracts;
- provide bounded-cardinality OTEL evidence for quality, cost, and rollback;
- stage implementation so unsupported runners and templates fail closed.

Non-goals:

- choosing production thresholds before offline calibration;
- changing the meaning of a model's chat template or reasoning delimiters;
- globally adjusting speculative decoding's draft-token count;
- restarting generation in the first release;
- treating arbitrary tool text as trusted outcome metadata;
- changing existing hard length, repetition, abort, or safety stops.

## Current source inventory

| Concern | Current producer or consumer | Availability and implication |
| --- | --- | --- |
| Prompt effort | `vllm/entrypoints/openai/chat_completion/protocol.py` and `vllm/entrypoints/openai/responses/protocol.py` | `reasoning_effort` is consumed while rendering/tokenizing. Changing it later is prompt reconditioning, not an in-flight control. |
| Thinking cap | `vllm/sampling_params.py`, `vllm/v1/sample/thinking_budget_state.py` | `thinking_token_budget` is an existing static request cap enforced while sampling. Its state-holder lifecycle is the pattern for dynamic controls. |
| Decision logits | `vllm/v1/sample/sampler.py` and `vllm/v1/worker/gpu/sample/sampler.py` | Logits remain on device. Entropy and margin need a compact reduction at a precisely defined stage. |
| Repetition | `vllm/sampling_params.py`, `vllm/v1/core/sched/utils.py` | `RepetitionDetectionParams` and `check_stop` exist, but repetition evidence currently leads to a terminal stop. Reuse the evidence earlier without weakening that stop. |
| Speculation | `vllm/v1/core/sched/scheduler.py`, `vllm/v1/spec_decode/metrics.py` | `Scheduler.update_from_output` has request-local proposed and accepted counts. Exported statistics are aggregate and may be disabled by `log_stats`; policy input must tap the transient values first. |
| Request lifecycle | `vllm/v1/request.py`, `vllm/v1/core/sched/output.py` | The scheduler request is the natural owner while an engine subrequest is live; scheduler output can carry versioned control updates. |
| Worker output | `vllm/v1/outputs.py` | Add compact request-aligned signal summaries, never full logits. |
| Responses tool loop | `vllm/entrypoints/openai/responses/serving.py`, `vllm/entrypoints/openai/responses/context.py` | The built-in loop creates engine subrequests inside one logical request and is the ownership-transfer boundary for retained state and typed tool outcomes. |
| Prefix cache | `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/kv_cache_manager.py` | Block hashes depend on the parent hash. A stable token prefix preserves prior full-block hashes; a changed tail invalidates blocks from the changed block onward. |
| Prefill cost | `vllm/v1/metrics/stats.py`, `vllm/v1/metrics/loggers.py`, `vllm/entrypoints/openai/responses/context.py` | Computed/cached tokens and prefill time exist, mostly as post-action evidence. A request-local predictor is cheaply derived from them. |
| Tracing | `vllm/config/observability.py`, `vllm/tracing/otel.py`, `vllm/tracing/utils.py`, `vllm/v1/engine/output_processor.py` | Existing OTEL request traces can parent transition spans and carry bounded decision summaries. |
| V2 support | `vllm/config/vllm.py`, `vllm/v1/worker/gpu/states.py` | V2 currently rejects or warns for unsupported thinking budgets. Dynamic reasoning must remain unavailable there until control and slot-cleanup parity are proven. |

## Contract

### Preconditions

Dynamic mode is accepted only when all of the following hold:

1. Engine configuration enables dynamic reasoning and defines ordered budgets
   `low < medium < high <= max_completion_tokens`.
2. The selected runner can enforce a thinking budget and report control
   acknowledgements. Initial rollout therefore targets V1 only.
3. The model's configured reasoning delimiters make thinking-token accounting
   meaningful.
4. Request `reasoning_effort` is absent or `low`; any explicit non-low value is
   rejected as conflicting prompt conditioning.
5. A static request `thinking_token_budget` is absent. Static and dynamic budget
   ownership are mutually exclusive.
6. `enable_thinking=false` is rejected for dynamic mode.
7. Prompt reconditioning is disabled unless a matching tail-adapter capability
   is registered for the exact model/template contract.

The server returns a validation error before engine submission when a required
capability is missing. It never silently approximates unsupported semantics.

### Postconditions

- A new logical request has a fresh epoch, empty evidence windows, budget usage
  zero, and level `LOW` before its first token is scheduled.
- A live engine subrequest has exactly one authoritative policy state and one
  monotonically increasing control revision.
- Every accepted transition raises the level by exactly one step, respects the
  configured maximum, and produces either an acknowledged action or a bounded
  fallback event.
- Completion, abort, and cap termination remove scheduler, worker-slot, and
  serving-context state associated with that epoch.
- Disabled dynamic mode follows existing request rendering and sampling paths.

### Invariants

- `LOW <= current_level <= configured_max_level`; there is no in-request
  de-escalation.
- Only a new top-level request may reset a logical request to `LOW`.
- Internal Responses tool subturns may retain or raise state, but never share it
  with another logical request.
- Request identifiers, batch slots, and control revisions must all match before
  signals or actions mutate state. Stale or out-of-order data is ignored and
  counted.
- Hard token limits and terminal repetition checks always dominate the policy.
- Missing, non-finite, or insufficient signal samples never cause escalation.
- All windows, counters, queues, labels, and per-request work have fixed bounds.
- Prompt reconditioning never occurs under the name of budget escalation.

### Ownership and lifecycle

Use a serializable `DynamicReasoningState`, but only one owner at a time:

1. The API layer creates `logical_epoch` and an explicit initial `LOW` state.
2. On engine admission, ownership transfers to `vllm.v1.request.Request`; the
   scheduler alone evaluates events and commits transitions.
3. Workers produce `ReasoningSignalBatch` and apply versioned
   `ReasoningControlUpdate`; they do not make policy decisions.
4. At an internal tool boundary the finished engine subrequest exports a final
   snapshot, deletes its state, and transfers the snapshot to the Responses
   context. The context incorporates a typed tool outcome and seeds the next
   internal subrequest.
5. At logical completion or abort, the current owner destroys the state.

`(logical_epoch, engine_request_id, sequence_index, control_revision)` is the
correlation key. For `n > 1`, each sequence gets independent windows and state.
Batch-slot reuse must scrub all worker-side state, following the add/move/remove
discipline in `ThinkingBudgetStateHolder`.

A `StreamingUpdate` attached to an already admitted engine request retains that
request's state and epoch, but cannot replace dynamic parameters, lower a budget,
or inject inherited state. An external resumable/continuation API call is still
a new top-level request and resets to `LOW`.

## Configuration and API behavior

Extend `vllm/config/reasoning.py` with an engine-level
`DynamicReasoningConfig`. Proposed fields are:

- enable/kill switch and allowed models or templates;
- ordered per-level thinking budgets and an absolute maximum;
- fixed window sizes, minimum sample counts, thresholds, weights, dwell, and
  cooldown values;
- a decision reserve before each active thinking budget;
- maximum per-step signal work and maximum state bytes per request;
- MTP-only acceptance settings;
- prompt-action mode: `budget_only` by default or `tail_at_boundary`;
- adapter allowlist and maximum estimated re-prefill/discard cost;
- shadow, emit-telemetry, and enforcement modes.

Add `DynamicReasoningParams | None` to `SamplingParams` for the validated,
engine-facing request contract. Add an opt-in `dynamic_reasoning` extension to
Chat and Responses protocols. Request fields may select a maximum level or turn
the feature off, but may not weaken operator caps or override calibrated
thresholds. The API schema must document that dynamic mode always renders the
first turn as `LOW`.

Validation is centralized before prompt rendering. Chat and Responses must use
the same normalizer, conflict rules, and error messages. The Responses context
may pass an internal inherited-state token that cannot be supplied by external
clients. Authentication is by unforgeable in-process object, not a public ID.

## Signals and cost

### Signal contract

| Signal | Status | Production point | Timing and bounded cost |
| --- | --- | --- | --- |
| Token repetition | Available now | Share the detector inputs used by `check_stop` | CPU update over a bounded suffix after committed tokens. Export pattern length/count or threshold proximity before preserving the terminal stop. |
| Stall score | Cheaply derived | Scheduler request token history | Fixed-size rolling n-gram/unique-token progress summary. No unbounded rescans and no semantic inference from text. |
| MTP acceptance | Cheaply derived | `Scheduler.update_from_output` before aggregate/logging gates | Rolling `accepted / proposed` only when the speculative method is MTP and the minimum proposal count is met. It measures draft/target mismatch, not reasoning difficulty. Absence is unknown, not zero. |
| Entropy | Requires instrumentation | GPU sampling path | Fused reduction to one normalized scalar per committed decision position; transfer only compact scalars asynchronously. |
| Top-logit margin | Requires instrumentation | Same reduction as entropy | Top-1 minus top-2 logit scalar; no extra vocabulary transfer. |
| Entropy trend | Cheaply derived after entropy | Scheduler request | Bounded robust slope over the entropy ring buffer. |
| Thinking use/boundary | Requires dynamic transport | Thinking-budget state holder | Report used tokens and a soft-boundary event early enough to change the next sampling step. |
| Tool outcome | Requires typed instrumentation | Responses context at `call_tool` completion | Enum `success`, `error`, `timeout`, or `empty`; never parse arbitrary result text. |
| Prefill/cache cost | Available plus cheaply derived prediction | Prefill stats, Responses usage context, prefix-token comparison | Actual cached/computed tokens and prefill time update a bounded EWMA; prediction compares only current and candidate token arrays. |

### Canonical logit stage

Entropy and margin must describe the model's decision, not the budget forcing
operation. Compute them after deterministic allow/bad-word/custom processors and
penalties, but before:

- the thinking-budget end-token force;
- temperature scaling;
- min-p, top-k, and top-p truncation; and
- random sampling.

Normalize entropy by `log(valid_vocab_size)` so thresholds transfer more safely
across vocabularies. Record the stage as a versioned enum. If either sampler
cannot implement the same stage, that runner remains unsupported rather than
emitting incomparable data.

For speculative decoding, emit summaries only for committed positions, in
commit order. Rejected draft positions must not enter uncertainty windows. MTP
acceptance is separate evidence and must not be inferred from entropy payloads.
When dynamic reasoning is off, skip the reduction entirely.

### Evidence evaluation

Each active request keeps fixed-capacity rings. A pure policy function consumes
an immutable state snapshot, typed event, and immutable configuration, and
returns `NoAction`, `Escalate(next_level, reason)`, or `Terminate(reason)`.

An uncertainty vote requires enough samples and combines normalized entropy,
inverse margin, and positive entropy slope. A configurable number of votes in a
window is required. Low MTP acceptance and stall/repetition each have their own
minimum samples and consecutive-window guards. Because MTP acceptance is also
affected by draft quality and sampling configuration, standalone MTP escalation
is disabled by default; it corroborates an uncertainty or stall vote. A typed
`error` or `timeout` tool outcome may cast one strong vote on the next internal
subturn. `success` does not de-escalate; it only closes the prior tool event.

All numeric defaults are experimental calibration inputs. The first release
must not present them as universal model-quality thresholds.

## State machine

States are `NEW`, `LOW`, `MEDIUM`, `HIGH`, `COMPLETED`, `CAPPED`, and `ABORTED`.
`COMPLETED`, `CAPPED`, and `ABORTED` are terminal.

| Current | Event | Guard | Transition/action |
| --- | --- | --- | --- |
| `NEW` | `TopLevelAccepted` | Always | Clear all state; enter `LOW`; install low budget revision 0. |
| `LOW` or `MEDIUM` | `EvidenceWindowReady` | Minimum samples, dwell/cooldown satisfied, configured vote threshold met | Advance one level; emit higher budget control; start cooldown. |
| `LOW` or `MEDIUM` | `SoftBudgetBoundary` | Evidence is sufficient and next level is allowed | Advance one level before the old cap can force the reasoning end marker. |
| Any active | `ToolOutcome` | Same logical epoch and typed outcome | Update bounded evidence; evaluate at the next internal subturn boundary. |
| Any active | `ControlApplied` | Request and revision match | Mark action acknowledged and continue. |
| Any active | `ControlRejected` or acknowledgement timeout | Request and revision match | Keep last acknowledged budget; record fallback; never re-render automatically. |
| `HIGH` | Any escalation evidence | Maximum reached | Stay `HIGH`; preserve hard cap and record suppression once per window. |
| Any active | Hard token/repetition cap | Existing stop says terminal | Enter `CAPPED`; do not override the stop. |
| Any active | Normal finish | Request matches | Enter `COMPLETED` and delete live state. |
| Any active | Abort/error | Request matches | Enter `ABORTED` and delete live state. |

Only one level may be crossed per evaluation. Monotonicity is the primary
hysteresis; minimum dwell, consecutive-window voting, and cooldown prevent noisy
double transitions. Unknown events and stale correlations are no-ops with a
bounded diagnostic counter.

If the old thinking budget has already forced the reasoning-end token, the
request cannot resume thinking in place. Record `budget_decision_late`; finish
normally or defer new conditioning to a later internal tool subturn. The
decision reserve must therefore cover at least one scheduler/worker round trip,
and its adequacy is a benchmark gate.

## Controls and prefill-aware action selection

### In-flight budget escalation

This is the default and first implementation target. Add a versioned
`ReasoningControlUpdate` to scheduler output with request identity, revision,
level, and absolute thinking budget. Extend the thinking-budget state holder to
apply only increasing, next-revision updates and acknowledge them in compact
worker output. The sampler continues from existing KV and generated tokens: no
prompt change, no restart, and no re-prefill.

Scheduler state remains authoritative. A missing acknowledgement by a bounded
deadline leaves the last acknowledged budget in force. It does not trigger a
speculative retry, prompt rewrite, or unbounded control queue.

### Prompt conditioning is a different action

Changing `reasoning_effort` changes rendered model input. Mid-generation that
normally means discarding generated suffix tokens, rendering again, and running
prefill for any uncached prefix/tail. It may also reduce prefix-cache reuse.
Therefore:

- mid-generation prompt reconditioning is disabled by default and absent from
  the initial rollout;
- budget escalation never mutates prompt conditioning;
- a future mid-generation mode requires a separate explicit gate, restart
  budget, discarded-token accounting, and quality/cost evidence;
- at a tool/turn boundary, where a render is already required, conditioning may
  be considered without discarding an active generated suffix.

### Tail-control prompt adapters

A `TailEffortAdapter` is registered against an exact model and template
capability. When its contract explicitly supports the semantics, it places an
effort directive immediately before the assistant/generation suffix while
leaving the tokenized conversation prefix stable. The adapter must return the
new tokens plus the asserted longest-common-prefix length, and runtime verifies
that assertion before use.

Effort semantics encoded in a system position are not assumed movable to the
tail. In particular, Harmony or any other template-specific system effort
directive is not equivalent merely because a similar string can be inserted
before generation. Models with no proven tail contract remain budget-only;
models lacking budget support fail dynamic mode closed or use existing static
behavior when dynamic mode is off.

During Responses tool loops, re-render only the changed tail and rely on normal
prefix-cache lookup for the stable full blocks. Adapter or cache misses do not
change correctness: they change the estimated cost and may suppress the prompt
action.

### Prefill cost estimator

For current prompt tokens `P` and candidate tokens `Q`, compute their token
longest common prefix `lcp`. With cache block size `B`:

```text
reusable_full_block_tokens = floor(lcp / B) * B
uncached_tail_tokens = len(Q) - reusable_full_block_tokens
discarded_generated_tokens = generated tokens abandoned by a restart
estimated_prefill_ms = bounded_EWMA_ms_per_computed_token * uncached_tail_tokens
```

The estimate is conservative: actual cache lookup is authoritative, and the
cache manager's last-token/block recomputation constraint still applies. Feed
back actual cached tokens, computed tokens, prefill latency, and cache hit/miss
after every subturn. Clamp all estimates and label estimate versions.

The action selector first asks whether an in-flight budget increase is possible.
If yes, choose it. At a natural turn boundary, a validated tail adapter may be
chosen only when its predicted conditioning benefit clears configured ceilings
for uncached tail tokens and prefill latency. A mid-generation candidate also
prices discarded tokens and is rejected by default.

## Producer-to-consumer flow

1. Protocol validation creates a fresh logical epoch, renders initial effort as
   `LOW`, and constructs validated dynamic sampling parameters.
2. Engine admission constructs scheduler-owned state and sends the initial
   budget control to the sampler state holder.
3. The GPU sampler reduces canonical decision logits to compact entropy/margin
   records; the worker returns committed-position records plus control acks.
4. `Scheduler.update_from_output` appends uncertainty records, adds request-local
   MTP acceptance, updates repetition/stall evidence, and emits typed events.
5. The pure policy evaluates at most once per configured token/window boundary.
6. A committed escalation updates scheduler state and sends the next versioned
   budget control. Telemetry records the decision separately from its ack.
7. On an internal tool boundary, the scheduler's final snapshot transfers to
   the Responses context. Typed tool outcome and prefill estimate inform the
   next subturn; the new engine request receives the inherited level explicitly.
8. Completion or abort removes all epoch state and emits final bounded summary.

## Concrete implementation map

| Path | Planned change |
| --- | --- |
| `vllm/config/reasoning.py` | Define engine caps, validation, kill switch, rollout mode, and adapter registry contract. |
| `vllm/sampling_params.py` | Add immutable validated dynamic parameters; preserve existing static and repetition types. |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | Add opt-in schema, shared conflict validation, and unconditional first-turn low rendering. |
| `vllm/entrypoints/openai/responses/protocol.py` | Add the same public contract and keep internal inheritance non-public. |
| `vllm/entrypoints/openai/responses/context.py` | Own state only between internal subturns; normalize tool outcomes and track actual/predicted prefill cost. |
| `vllm/entrypoints/openai/responses/serving.py` | Transfer state around `_generate_with_builtin_tools`; render a validated changed tail at tool boundaries. |
| `vllm/v1/request.py` | Store authoritative live state and fixed-size evidence buffers. |
| `vllm/v1/core/sched/output.py` | Carry versioned control updates and bounded signal/ack correlation. |
| `vllm/v1/core/sched/scheduler.py` | Consume signals, tap transient per-request acceptance, evaluate pure policy, transfer/finalize state. |
| `vllm/v1/core/sched/utils.py` | Expose bounded repetition evidence without changing terminal `check_stop` behavior. |
| `vllm/v1/sample/sampler.py` | Implement canonical V1 signal reduction hook and dynamic control application. |
| `vllm/v1/sample/thinking_budget_state.py` | Accept monotonic control revisions, report used thinking tokens/soft boundary, and scrub lifecycle state. |
| `vllm/v1/outputs.py` | Add compact request-aligned signal and acknowledgement structures. |
| `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/kv_cache_manager.py` | Reuse existing hash/lookup behavior; expose no policy mutation, only cost evidence needed at integration boundaries. |
| `vllm/v1/metrics/stats.py`, `vllm/v1/metrics/loggers.py` | Add bounded aggregate cost/transition metrics; never gate policy inputs on logging. |
| `vllm/tracing/utils.py`, `vllm/v1/engine/output_processor.py` | Add transition/action/final-summary span attributes and child events. |
| `vllm/config/vllm.py`, `vllm/v1/worker/gpu/sample/sampler.py`, `vllm/v1/worker/gpu/states.py` | Later V2 parity; keep validation fail-closed until budget, signal, and reused-slot cleanup tests pass. |

Prefer focused modules for the policy, signal types, and prompt-adapter interface
rather than growing the scheduler or protocol files with parallel conditionals.
All integration paths call the same validator and pure transition function.

## Observability and cardinality

OTEL is the authoritative per-request diagnostic surface. Add one child span or
span event per transition/action, not per token. Final request spans summarize
initial/final level, transition count, terminal reason, and aggregate cost.

Suggested metrics:

- transition and suppressed-transition counters;
- control sent/ack/rejected/late counters;
- signal invalid/stale/insufficient counters;
- decision latency and tokens-to-transition histograms;
- cached, computed, uncached-tail, and discarded-token histograms;
- predicted and actual prefill-latency histograms;
- tool-outcome-to-next-level counters.

| Dimension | Allowed values/cardinality rule |
| --- | --- |
| `from_level`, `to_level` | Fixed enums: low, medium, high, terminal. |
| `reason` | Fixed enum: uncertainty, mtp_acceptance, repetition, stall, tool_error, budget_boundary, cap. |
| `action` | Fixed enum: none, budget, tail_boundary, restart, fallback. |
| `outcome` | Fixed enum: applied, rejected, late, unsupported, completed, aborted. |
| `runner` and `signal_stage` | Fixed implementation enums, not arbitrary class names. |
| Numeric evidence/cost | Histogram values or span attributes, never metric labels. |
| Request/model/template/tool identity | Never a new metric dimension. Do not record prompts, generated text, tool names, arguments, or outputs. |

Existing trace context correlates events. Request IDs may remain where existing
tracing already uses them, but this feature must not add them as metric labels.
High-volume numeric samples are summarized in bounded windows; no per-token spans.

## Verification plan

### Unit and property tests

Add focused tests under `tests/v1/core` and `tests/v1/sample` for:

- every top-level request enters `LOW` with empty state, including reused request
  IDs/batch slots, aborts, and sequential requests for the same conversation;
- internal tool subturn inheritance only with the matching unforgeable epoch;
- transition-table coverage, one-level steps, monotonicity, caps, dwell,
  cooldown, insufficient/NaN data, stale events, and deterministic replay;
- fixed memory/work bounds for long generations;
- dynamic/static conflict validation and fail-closed unsupported runners;
- canonical entropy/margin stage, normalized entropy, and no feature-off work;
- speculative committed-position ordering and rejected-position exclusion;
- MTP-only acceptance, zero-proposal unknown handling, and logging disabled;
- repetition evidence before the unchanged terminal repetition stop;
- increasing control revisions, lost/late/duplicate acks, and old-budget force;
- worker state cleanup on add, move, remove, cancellation, and slot reuse.
- streaming updates cannot replace the epoch, state, or dynamic policy;
- OTEL/metric enums stay within their declared sets, work with `log_stats` off,
  and never contain prompt, output, tool argument, or tool result text.

Property tests generate event streams and assert that levels never decrease,
terminal states do not transition, revisions strictly increase, and two epochs
cannot influence each other.

### API and tool-loop integration tests

Under `tests/entrypoints/openai`, cover Chat and Responses parity, streaming,
disconnect/abort, `n > 1`, explicit conflict errors, static feature-off behavior,
and built-in tool outcomes `success/error/timeout/empty`. Verify that a new
external request resets to `LOW`, while subturns created inside one Responses
request may retain `MEDIUM` or `HIGH`.

Test tail adapters with golden token arrays rather than string-prefix assertions.
Cases include supported templates, unsupported templates, a template whose
effort semantics must remain in the system position, cache miss, and adapter
LCP assertion failure. Unsupported cases must use budget-only behavior or reject
dynamic mode according to declared capability.

### Offline evaluations

Replay representative reasoning, coding, tool-use, and adversarial repetition
datasets with fixed model/build/seed where supported. Compare static low,
static medium/high, shadow policy, and enforced policy on task quality, reasoning
tokens, total tokens, latency, tool success, escalation precision/recall against
quality deltas, and cap/abort rates. Stratify by model, prompt length, speculative
mode, cache status, and tool-loop depth. Thresholds graduate only when held-out
results meet predeclared quality and cost bounds.

### Benchmarks

Measure feature-off overhead and per-token signal overhead at concurrency and
batch-size sweeps. Benchmark low-to-medium-to-high transitions with short and
long prompts, MTP on/off, prefix cache warm/cold, cache miss, and tool loops.

Prefill benchmarks must report cached tokens, computed/uncached-tail tokens,
discarded generated tokens, TTFT, prefill latency, decode latency, and end-to-end
latency. Compare:

1. in-flight budget escalation with no re-prefill;
2. supported tail adaptation at a natural tool boundary;
3. long-prompt boundary re-render with prefix-cache reuse;
4. cold-cache re-render;
5. unsupported-template fallback; and
6. an experimental gated mid-generation restart, if ever implemented.

The budget decision reserve passes only if the higher revision is acknowledged
before the prior force point at target load percentiles. Tail adapters pass only
if token LCP/cache evidence matches their stable-prefix claim.

## Phased delivery and rollout

### Phase 0: contracts and shadow evidence

Land shared validation, types, pure HSM, lifecycle cleanup, compact V1 signals,
OTEL, and shadow decisions. Do not enforce transitions. Establish feature-off
overhead and signal-quality baselines.

### Phase 1: V1 budget-only enforcement

Enable versioned in-flight budget increases for allowlisted models. Keep prompt
conditioning fixed at initial `LOW`; use uncertainty and soft-budget evidence.
Ship kill switch and static fallback.

### Phase 2: scheduler and tool evidence

Add MTP acceptance, early repetition/stall evidence, typed Responses tool
outcomes, state transfer across internal subturns, and prefill cost feedback.

### Phase 3: V2 parity

Implement the same canonical logit stage, dynamic budget controls, correlation,
and slot cleanup in V2. Remove fail-closed validation only after parity tests and
benchmarks pass.

### Phase 4: allowlisted boundary tail adapters

Enable model/template-specific tail conditioning only at natural tool/turn
boundaries. Require golden token tests, verified LCP, prefix-cache benchmarks,
and budget-only fallback.

### Phase 5: optional restart experiment

Consider mid-generation reconditioning only as a separate experimental action
after offline evidence shows benefit beyond budget escalation. It stays off by
default and has strict prefill, discarded-token, and latency ceilings.

Each phase progresses through dark telemetry, offline replay, shadow traffic,
small allowlisted canary, and bounded expansion. Roll back on quality regression,
feature-off overhead, control lateness, state leakage, error rate, prefill cost,
or telemetry-cardinality breach. The engine kill switch restores existing static
behavior for unaffected requests and rejects new dynamic opt-ins explicitly;
it never silently converts an opted-in request to different semantics.

## Open decisions

- Per-model budgets, window lengths, weights, thresholds, dwell, cooldown, and
  decision reserve require calibration.
- Define which reasoning parsers can reliably report thinking-token use and
  where parser capability is declared.
- Decide whether public request parameters expose only `max_level` or also a
  bounded cost preference.
- Specify the exact device reduction implementation and its numerical tolerance
  across V1/V2 backends.
- Decide whether tool `empty` is neutral or strong evidence per tool contract;
  it must remain typed and configurable, never inferred from text.
- Define the benefit estimator required before any boundary prompt action; cost
  prediction alone cannot prove a quality benefit.
- Choose retention and sampling policy for OTEL decision evidence without adding
  unbounded attributes.

## Documentation definition of done

The implementation proposal is complete when its future code change can prove:

- unconditional `LOW` initialization and cross-request isolation;
- single-owner lifecycle, correlation, state cleanup, and deterministic HSM;
- all six requested evidence families with explicit availability/timing/cost;
- separate budget and prompt-conditioning actions with safe fallbacks;
- prefix-cache-aware tail behavior and measured prefill/discard cost;
- V1 feature-off compatibility and V2 fail-closed behavior until parity;
- bounded OTEL cardinality and no sensitive text in new telemetry;
- unit, integration, property, offline-eval, benchmark, canary, and rollback
  evidence for the claimed runtime scope.

This plan itself supplies documentation-plan-completeness only. Runtime claims
remain unproven until the phased implementation and evidence above exist.
