# `reasoning_effort: dynamic` on VulcanBench v3 — what the run says, and what P6 changes

Data, all measured 2026-08-18/19 on the 4x L4 box, latency profile
(Qwen3.8-27B-FP8, V2 runner, MTP K=7):

- `/shared/vulcan-runs/v3/{dynamic,low,medium,extra-high}/<run>/` — VulcanBench
  v3, 23 tasks (`/shared/VulcanBench/tasks/v3`), `summary.json` + `trace.jsonl`
  + `final.patch` per run.
- `/data/effort-telemetry/latency.jsonl` — 222 582 per-request-per-step records
  (`req_id, step, num_output_tokens, entropy, margin, n_rows, num_draft_tokens,
  num_accepted, in_think`), 1 199 distinct requests, written by patch 0009's
  sink.

Everything below uses only runs that have a `summary.json`. This is the
evidence that set the P6 defaults in §5; §4 is the honest part, and it is not
flattering to the signal the pre-P6 controller escalates on.

## 1. Completeness

The sweep was still filling in when this was written.

| profile | run dirs | task dirs | with `summary.json` | missing |
|---|---:|---:|---:|---|
| dynamic | 24 | 23 | **23** | — (the 24th dir is `suite-aaad5f92`, suite metadata, not a task) |
| low | 23 | 23 | 22 | `itertools-strip-prefix` (stale) |
| medium | 23 | 23 | 21 | `hono-client-header-merge`, `pennylane-trotter-fragmented` (stale) |
| extra-high | 21 | 21 | 16 | `chi-readfrom`, `jiff-strftime`, `semver-inc-dotted`, `semver-xrange`, `zod-invert-codec` (in flight) |

**13 of 23 tasks have all four efforts.** `dynamic` is the only complete
column. Cross-profile pass rates are therefore over different task subsets —
`extra-high` in particular is missing 7 tasks and its remainder skews hard.

## 2. Outcome

| profile | pass | rate | mean tokens | median tokens | mean completion | mean steps | mean duration (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dynamic | 19/23 | **0.826** | 522 595 | 347 110 | 17 875 | 135.6 | 2 640 |
| low | 14/22 | 0.636 | 348 502 | 249 128 | 17 634 | 114.7 | 2 289 |
| medium | 14/21 | 0.667 | 319 094 | 221 022 | 31 592 | 107.1 | 2 639 |
| extra-high | 8/16 | 0.500 | 464 126 | 509 825 | 36 579 | 135.7 | 3 836 |

Per task (`P`/`F` + total tokens; `-` = no `summary.json`):

| task | dynamic | low | medium | extra-high |
|---|---|---|---|---|
| aiohttp-upgrade-deferred | F 1 382 451 | F 284 318 | F 238 063 | F 689 653 |
| chi-readfrom-tee-doublecount | P 119 986 | P 173 341 | P 120 343 | - |
| cobra-noduplicateargs | P 47 668 | P 24 757 | P 37 574 | - |
| flask-teardown-robust | **P 481 401** | F 572 792 | F 245 364 | F 739 124 |
| hono-client-header-merge | P 202 334 | F 20 120 | - | P 539 256 |
| hono-request-bytes | P 153 137 | P 340 171 | P 129 186 | P 690 959 |
| itertools-strip-prefix | **P 1 772 583** | - | F 1 218 691 | F 275 739 |
| jiff-date-day-lt1 | P 381 548 | P 368 496 | P 189 384 | P 480 394 |
| jiff-signdur-panic | P 219 859 | P 228 578 | P 194 469 | P 254 773 |
| jiff-strftime-negpad | P 604 102 | F 512 829 | P 1 276 716 | - |
| more-itertools-interleave-empty | P 34 702 | P 45 401 | P 38 038 | P 86 019 |
| networkx-leiden-communities | F 1 527 251 | F 97 135 | F 270 550 | F 552 515 |
| packaging-range-prerelease-policy | **P 169 168** | P 424 728 | F 297 058 | F 123 084 |
| pennylane-trotter-fragmented | F 1 042 300 | F 109 831 | - | F 36 652 |
| pflag-uintslice-hex | P 40 065 | P 181 602 | P 52 672 | - |
| semver-inc-dotted-prerelease | P 127 393 | P 262 956 | F 35 689 | - |
| semver-truncate | P 347 110 | P 235 299 | P 89 253 | P 747 312 |
| semver-xrange-order | P 290 378 | P 364 741 | P 360 602 | - |
| sqlglot-canonicalize-internal-names | F 1 206 627 | F 198 774 | F 368 985 | F 86 113 |
| sqlglot-iso8601-nanos | P 533 250 | P 1 010 439 | P 221 022 | P 842 576 |
| sqlglot-qualify-lateral-star | P 704 791 | P 883 195 | P 374 989 | F 949 512 |
| zod-invert-codec | P 507 684 | F 1 193 816 | P 856 058 | - |
| zod-proto-catchall | P 123 888 | P 133 719 | P 86 268 | P 332 339 |

**There is no task that `dynamic` failed and a fixed effort solved.** The
reverse happens 8 times. That is the headline, and it constrains how much P6
should be allowed to change.

## 3. Why the four dynamic failures failed — not the controller

`aiohttp-upgrade-deferred`, `networkx-leiden-communities`,
`pennylane-trotter-fragmented`, `sqlglot-canonicalize-internal-names`. Every
one of them ends with `budget_exceeded` at the 7 200 s wall clock, never at
`max_steps` (300; the highest was 236). The verifier reports
`{"functional": 0.0, "error": "run budget exceeded before verification"}` —
**the hidden tests never ran in any of the four**, and two of them
(`pennylane`, `sqlglot-canonicalize`) had an empty `final.patch` at the moment
the clock expired.

| task | steps/max | duration (s) | llm req/resp | tool calls | repeated calls | max completion |
|---|---|---:|---|---:|---:|---:|
| aiohttp-upgrade-deferred | 236/300 | 7 205 | 56/55 | 60 | 0 | 6 724 |
| networkx-leiden-communities | 197/300 | 7 360 | 46/46 | 50 | 2 | 26 272 |
| pennylane-trotter-fragmented | 175/300 | 7 402 | 38/38 | 47 | 5 | 6 534 |
| sqlglot-canonicalize-internal-names | 209/300 | 7 248 | 47/47 | 55 | 3 | 16 633 |

The secondary pattern is a late drift into re-issuing near-identical probes.
The model says so itself — pennylane step 172: *"I've been repeating the same
command. Let me stop and carefully work out the global phase…"*, 25 s before
`budget_exceeded`; networkx step 194: *"The CPM values are way too high. Let me
reconsider the CPM formula…"*, then `budget_exceeded`.

Classification, per the four: **(iv) ran out of time**, unanimously, with
**(ii) tool-loop drift** contributing in three of four. Not (i) force-closed at
a cap: individual completions run to 26 272 / 16 633 / 10 212 tokens, far past
any rung. Not (iii) wrong-edit: no test ever ran, so it cannot be assessed.

**No P6 rule would have changed these four outcomes.** The cap was never the
binding constraint; the wall clock was. The one P6 mechanism that touches this
shape at all is the churn detector (§4 of the design), and only indirectly: the
loop clamp shortens a thinking phase that is going in circles, it does not stop
the agent loop from re-issuing the same shell command. Fixing this class needs
harness-side work (a repeated-tool-call brake, a time-aware budget), not the
effort controller.

## 4. What the telemetry says about the signals P2 escalates on

1 199 requests, 136 101 in-think steps. Think tokens per request = Σ`n_rows`
over in-think steps.

**Distribution is bimodal.** Half of all requests think under 32 tokens (tool
call preamble); the 90th percentile is ~1 000.

| pct | 1 | 5 | 10 | 25 | 50 | 75 | 90 | 95 | 99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| think tokens | 0 | 7 | 10 | 16 | 31 | 140 | 993 | 1 495 | 3 810 | 16 383 |

Buckets: `[0,1) 16 · [1,32) 585 · [32,64) 190 · [64,128) 100 · [128,256) 60 ·
[256,512) 62 · [512,1024) 92 · [1024,2048) 56 · [2048,4096) 31 · [4096,8192) 5
· [8192,16384) 2 · [16384,∞) 0`.

**Cap landings and escalations.** Think counts cluster 1–3 tokens under a rung
cap, which is the fingerprint of a hard forced close at MTP granularity:

| cap | exactly at | within 8 at-or-below | strictly above (⇒ escalated) |
|---:|---:|---:|---:|
| 1 024 | 4 | 28 | **88 (7.3 %)** |
| 4 096 | 2 | 6 | 5 (0.4 %) |
| 16 384 | 0 | 1 | **0** |

So: escalation fires on ~7 % of requests, the second rung on 0.4 %, and the
65 536 rung was never entered. **35 requests (2.9 %) were force-closed at a
cap** — that is the whole population P6's grace window can help.

**Calibration (entropy is normalised by log V; margin in logit units):**

| | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| entropy, all in-think steps | 0.0000 | 0.0000 | 0.0001 | 0.0077 | 0.0389 | 0.0790 | 0.1232 | 0.1516 | 0.2080 |
| margin, all in-think steps | 0.000 | 0.250 | 0.625 | 1.500 | 3.792 | 7.708 | 11.500 | 13.083 | 15.083 |
| entropy, last 25 % of think | 0.0000 | 0.0000 | 0.0000 | 0.0028 | 0.0295 | 0.0673 | 0.1083 | 0.1348 | 0.1929 |
| margin, last 25 % of think | 0.125 | 0.375 | 0.750 | 1.875 | 4.708 | 8.771 | 12.125 | 13.500 | 15.500 |

Two things fall out of this table:

1. **Entropy falls and margin rises toward the end of a think phase.** The
   model gets *more* confident as it wraps up. The P2 premise ("sustained high
   entropy near the cap = still working") is at best weakly supported.
2. The pre-P6 calibration in the YAMLs (`entropy` mean 0.043 sd 0.046, `margin`
   mean 5.9 sd 4.5) is a *mean/sd* summary of a distribution whose entropy arm
   is strongly right-skewed with a large point mass at 0. A z-score on that is
   not a meaningful ordinal statement, which is the core P6 argument for ranks.

**The uncomfortable result.** Does uncertainty rise over a request's own
baseline separate requests that needed more thinking? Comparing first-128 vs
last-128 think tokens, long (>1024, n=88) vs medium (256–1024, n=160) — both
groups long enough that the two windows are disjoint:

| | long | medium |
|---|---:|---:|
| entropy first-128 median | 0.0584 | 0.0594 |
| entropy last-128 median | 0.0346 | 0.0428 |
| margin first-128 median | 4.437 | 4.063 |
| margin last-128 median | 5.685 | 5.291 |
| fraction with last-128 entropy > first-128 | 0.227 | 0.275 |
| fraction with last-128 margin < first-128 | 0.284 | 0.294 |

| discriminator | AUC (P(long > medium)) |
|---|---:|
| last-128 entropy | 0.407 |
| first-128 entropy | 0.494 |
| entropy *rise* (last − first) | 0.417 |
| last-128 margin | 0.534 |
| first-128 margin | 0.498 |
| margin *drop* (last − first) | 0.538 |

**Every AUC is at chance.** With length controlled, neither the absolute level
nor the within-request rise separates. The apparently strong numbers you get
without controlling for length (entropy-rise AUC 0.264, margin AUC 0.873) are a
length artifact: short requests' "first-128" and "last-128" windows are the
same steps.

Same story for cap-landing vs natural close (both ≥256 think tokens):

| | cap-landing (n=37) | natural (n=211) |
|---|---:|---:|
| entropy delta (last−first) median | −0.0141 | −0.0169 |
| fraction with entropy rise | 0.162 | 0.275 |
| margin delta median | +1.000 | +0.993 |
| fraction with margin drop | 0.270 | 0.294 |

Indistinguishable, and where they differ it is in the *opposite* direction from
the hypothesis. **The entropy/margin telemetry as currently computed does not
carry a local "still working / about to finish" signal on this model.**

MTP acceptance, for the corroboration rank: global Σaccepted/Σdrafted =
421 287/574 510 = **0.733**; per-request p10 0.635, p25 0.777, p50 0.854,
p75 0.922, p90 0.964.

## 5. What P6 changes, and the defaults this evidence sets

| P6 change | verdict from this data |
|---|---|
| **Rank-based, self-normalizing signals** | Right for the wrong-looking reason. It does not make entropy/margin more predictive — nothing can, they are at chance here — but it removes the fixed `(mean, sd)` table that has to be re-fitted per model/quantization and makes "top 15 % of uncertainty" mean the same thing on every deployment. It also makes the threshold auditable: `p_uncertain` is a percentile, so a wrong setting is visible as an escalation *rate*. |
| **Within-request baseline** | Keep, but as a *brake*, not a discriminator. Requiring a rank rise over the request's own first-128 tokens can only reduce escalations; given §4 that is the safe direction. |
| **p(`</think>`) + grace window** | The one genuinely new signal. Everything in §4 says the existing signals cannot tell when the model is about to finish — p(end) is measured directly from the same logits and is definitionally that quantity. It is also the mechanism that actually addresses the 35 force-closed requests, by turning a hard cut into a soft one exactly when the model is closing. **This is where P6 earns its keep.** No claim can be made about its magnitude from this run: p(end) is not in the v3 telemetry (the column ships with P6). |
| **Language-agnostic novelty churn** | Directionally right (three of four failures show repeated identical probes) but unvalidated here: the telemetry has no token stream, so the novelty rate cannot be computed retrospectively. Marker weight 0 costs nothing since the markers were never load-bearing. |
| **Worker-side decision (kills `late`)** | Correct but low-value on this run: `late: false` everywhere in the §2c window, and the 4 failures were wall-clock-bound. Its value is at load, where the scheduler→worker lag grows. |
| **Graceful `force_end_str`** | Untested here. It only changes what the 2.9 % force-closed requests read like at the cut; the argument for it is in-distribution text, not this data. |

Defaults chosen from the above (all in `DynamicEffortConfig`):

| setting | value | why |
|---|---|---|
| `ladder` | `[1024, 4096, 16384]` | 0 of 1 199 requests passed 16 384 think tokens and 5 passed 4 096. The old fourth rung (65 536) was dead weight; it stays configurable. |
| `p_uncertain` | `[0.85, 0.92]` (padded `0.96`) | Deliberately conservative. §4 shows the uncertainty features at chance, so escalating on them buys latency without evidence of accuracy; 0.85 keeps the current ~7 % escalation rate roughly where it is rather than raising it. |
| `baseline_rise` | `0.10` | The rank must rise 10 percentile points over the request's own baseline. Another brake, same reasoning. |
| `grace_tokens` | `256` | A model whose p(end) is rising needs a sentence, not a paragraph: half of all requests think under 32 tokens in total, so a closing tail is small. 256 is 25 % of the rung-0 cap — enough to finish, cheap enough to grant unconditionally once. |
| `acc_veto_rank` | `0.85` | Per-request acceptance p85 ≈ 0.95: only requests whose drafter predicts nearly everything (boilerplate) get the veto. |
| `quantile_min_samples` | `2048` | 136 k in-think steps in one sweep; the sketches warm within seconds of real traffic, and stay warm across restarts through `quantile_path`. |
| `backtrack_marker_weight` | `0.0` | Never load-bearing, English-only, model-specific. |

## 6. What this data cannot answer

- **Reasoning tokens, rung, escalations, `late` per VulcanBench request.** The
  harness discards them: `harness/agent/providers.py` `TokenUsage` has exactly
  `{prompt_tokens, completion_tokens}` and `_parse_chat_completions_response`
  drops the rest; `harness/agent/loop.py` writes only `{content, tool_calls,
  usage}` and never `LLMResponse.raw`. The `effort` object in `llm_request` is
  the harness's own request-side metadata (`{"requested":"dynamic",…}`),
  identical on every request, not the server's response. Verified over all
  2 397 `llm_response` records. Recovering this needs a harness change
  (persist `raw`, or widen `TokenUsage`).
- **Telemetry ↔ VulcanBench attribution.** No shared key: telemetry has no
  timestamp and the trace keeps only `chatcmpl-tool-*` ids. Token-count
  matching gives 1 042/1 199 multiset hits but only 165 both-side-unique
  candidates, and 59 % of telemetry requests are under 200 output tokens where
  counts collide. The telemetry file also starts 3.4 h before the sweep and
  ends 19–63 min before the later profiles finish. §4 is therefore global and
  unattributed.
- **Whether the four failures would pass with more wall clock.** No
  verification ran; two had an empty patch.
- **Whether the 88 escalating requests escalated once or more.** Only the
  resulting token counts are in the telemetry, not the decisions.

The first bullet is the one worth fixing before the next sweep: with the
`effort` object persisted, every question in §3 and §5 becomes directly
measurable instead of inferred from round numbers.
