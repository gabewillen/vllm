# P8 on the box: the effort level, decided before thinking

Status: **implemented and benchmarked 2026-08-19** on the 4x L4 trial arm
(`:8013`, `/shared/vllm/.venv-effort-v3`, branch `qwen3.8-27B-effort-v3`, venv
patch `serve-configs/patches/0012`). Design:
[`dynamic-reasoning.claude.md`](dynamic-reasoning.claude.md) §13. The
measurement the signal rests on: [`effort-hidden-probe.md`](effort-hidden-probe.md).

What shipped is narrower and stricter than §13 as written:

> `reasoning_effort: "dynamic"` chooses **one effort level per request, before
> the model thinks**, from the prompt's own pooled prefill hidden state, and
> renders that level's **sentence at the true tail of the prompt**. Nothing else
> touches the think block — no cap, no ramp, no forced close, no escalation, no
> stall detector. The model ends its own reasoning, bounded only by the client's
> `max_tokens` and timeouts, exactly as it is at a fixed effort level.

This file records what that took, the placement measurement it rests on, the
`dynamic-v3` VulcanBench column, and what is still open.

## 1. Headline

Greedy, cold memory, 22 of 23 tasks (one killed as a runaway, §5):

| column | solved | mean functional | completion tokens | per request | reasoning p90 / p99 / max |
|---|---:|---:|---:|---:|---:|
| `dynamic` (v1, greedy, c=5) | 19/23 | 0.826 | 411 118 | — | — |
| `dynamic-v2` (greedy, c=5) | 19/23 | **0.857** | 846 811 | 1 071 | 1 303 / 5 865 / 16 663 |
| `dynamic-v3` (greedy, c=12) | 17/22 | 0.773 | 753 027 | 1 572 | 2 092 / 19 355 / **40 849** |

**The router works and the uncapped greedy configuration does not.** The
hidden-state level separates the traffic as designed — level 2 requests think
3.7x as long as level 0 at the mean and 7x at p90 (§5) — but removing the cap
under **greedy** decoding lets a think block run away: the median request thinks
*less* than under `dynamic-v2` (28 vs 68 tokens) while the tail goes 2.5x
further, and one task (`oss-chi-readfrom-tee-doublecount`) spent **254 066**
completion tokens to solve what `dynamic` and `dynamic-v2` solved with 3 214 and
2 971. That is the failure mode Qwen's model card names for greedy decoding, and
with no cap nothing bounds it. §5b reruns the same build with the model's
recommended thinking-mode sampling.

## 2. Where the effort sentence has to go (measured 2026-08-19)

§2b measured sentence placement on **single-turn** prompts, where the last user
message *is* the end of the prompt. An agent turn is not like that: the last
message is a tool result, so appending to the last user message buries the
sentence thousands of tokens above the generation point. Nothing in the design
had ever measured that case, so this run measured it before building on it.

Harness: `work/../scratchpad/effort-placement/agent_grid2.py` — six problems
known to make this model deliberate, each delivered as the **content of the last
tool result** in a system / user / assistant-tool-call / tool-result
conversation, greedy, one sample each, `max_tokens=20000`, on the `:8013` arm.
Four placements, all prefix-cache-safe:

| placement | where the sentence goes |
|---|---|
| `last_user` | appended to the last **user** message (what patch 0009 ships) |
| `tool_tail` | appended to the last message whatever its role (inside the tool result) |
| `user_tail` | a trailing user message carrying only the sentence |
| `none` | no sentence (the chat template's own `medium`) |

Reasoning tokens per prompt:

| prompt | none | last_user low | tool_tail low | user_tail low | last_user high | tool_tail high | user_tail high |
|---|---:|---:|---:|---:|---:|---:|---:|
| aime | 567 | 663 | 403 | 543 | 351 | 918 | 391 |
| algo | 64 | 120 | 126 | 113 | 973 | 1123 | 783 |
| count | 422 | 364 | 204 | 235 | 214 | 387 | 434 |
| debug | 63 | 160 | 254 | 217 | 3446 | **9680** | 1864 |
| puzzle | 286 | 333 | 315 | 373 | 349 | 370 | 399 |
| sched | 293 | 250 | 355 | 179 | 312 | 294 | 314 |

Medians, and the ratios that matter:

| placement | high / low | high / none (all 6) | low / none (the 4 that already deliberate) |
|---|---:|---:|---:|
| `last_user` | 1.15x | 1.14x | 1.01x |
| `tool_tail` | 2.09x | 1.46x | 0.91x |
| `user_tail` | 1.80x | 1.23x | **0.78x** |

**Read: at the placement the deployment ships, the sentence is inert on agent
traffic.** `last_user` moves reasoning length by 1.01–1.15x in either direction
— noise. Both tail placements are honoured in both directions. `tool_tail` is
the strongest upward (1.46x vs no sentence, driven by `debug` at 9680 tokens);
`user_tail` is the strongest downward (0.78x) and is semantically clean — the
instruction is not text pasted inside a tool result — so **`user_tail` is what
shipped**. Every configuration solved 5 of 6 and every one finished on `stop`;
no placement broke the agent loop.

Caveats: six prompts, one greedy sample each, one model. The direction is
unambiguous; the magnitudes are not tight.

## 3. What §13 said and what shipped

### 3.1 The seam is at the tail now, by construction

§13.3 assumed the prompt splits at the effort sentence with the body being
"system + every turn + last user content". With the sentence moved to a trailing
user message, the body is *everything the model reads except the sentence*, for
any conversation shape. That is what makes the two-phase form apply to ordinary
agent turns rather than only to single-turn prompts.

### 3.2 The KV block is 1648 tokens wide

The served profile is hybrid GDN with prefix caching, so vLLM picks the Mamba
`align` cache mode and widens the attention block so its page covers the mamba
page:

```
Setting attention block size to 1648 tokens to ensure that attention page size
is >= mamba page size
```

In `align` mode a **non-final prefill chunk may only end on a block boundary**.
The body chunk ended at the frontend's seam, an arbitrary token, so the aligning
split clipped it to zero and the request was never admitted: the engine spun at
`num_requests_waiting 1, reason capacity` and every `dynamic` request hung. That
was the first thing this run found and it is a hard blocker for §13.3 as
written.

The boundary is now the largest multiple of the block at or before the seam. The
first attempt also subtracted one block (mirroring the eagle-pruned "last
cacheable position"), which cost a further 1648 tokens for nothing: an
intermediate chunk that stops at that position is simply followed by one that
runs on to the boundary. Dropping the backoff moves a 4 100-token prompt's body
from 1 648 to 3 296 tokens. `hidden_effort.split_min_fraction` (0.5) is the
remaining safety net: a prompt with no boundary covering at least that much of
itself takes no decision and runs at `default_level`, with a byte-identical
prompt.

### 3.3 Everything that touched the think block is gone

Removed from the dynamic path: the rung ladder and per-rung caps, mid-generation
escalation, the running quantile sketches and the rank rule, the soft-limit
close and its ramp, the graceful forced close, and the loop-stall clamp. A
dynamic request sets **no thinking budget at all**, so the cap actuator is never
armed for it, and `close_kind` reduces to `natural` (the model closed its own
think block) or `client-limit` (`max_tokens`, a timeout or an abort ended it
first — a right-censored entry the memory will not value).

Deleted with them: `vllm/v1/sample/effort_policy.py`,
`vllm/v1/worker/gpu/sample/effort_escalation.py` (see §7),
`serve-configs/effort_calibrate.py`, and their tests. The static
`thinking_token_budget` feature and its soft-limit close are untouched — they
are a separate, explicitly requested capability.

## 4. Method

- Server: the `B_mtp.yaml` latency profile (Qwen3.8-27B-FP8, TP4, MTP K=7
  adaptive, V2 runner, `TRITON_ATTN`, `NCCL_P2P_LEVEL=SYS`, fp8 lm_head, fp8 KV)
  with `hidden_effort` on and nothing else changed:
  `{"dynamic_effort": {"hidden_effort": {"enabled": true, "memory_path": ..., "min_entries": 128, "flush_every": 128}}}`.
  Production (`vllm-qwen38`, `:8012`) stayed stopped throughout.
- Harness: `vulcanbench run --suite v3 --model qwen:Qwen3.8-27B --effort dynamic-v3`,
  greedy, `--no-judges`, `--override-budgets`, `--timeout 7200`,
  `--max-steps 300` — same as the other v3 columns **except**
  `--max-concurrency 12` (the `dynamic` and `dynamic-v2` columns ran at 5).
  Per-task wall clock is therefore **not** comparable to those columns; solved
  counts, tokens and level mixes are.
- **Cold start is part of the measurement.** The server started with an empty
  memory and was not warmed with the bench tasks; the first ~128 finished
  requests got `default_level` (the `low` sentence, which is what `dynamic`
  rendered before v3) while filling it.

## 5. The `dynamic-v3` column (greedy)

23 tasks launched, **22 completed**. `oss-aiohttp-upgrade-deferred` was killed
while still running after 2h10m: 34 agent steps, 94 077 reasoning tokens
accumulated, largest single think block 16 502 — a runaway, so it is recorded as
**killed while running** rather than as a failure, and its trace-only directory
is under `_discarded/dynamic-v3-greedy-killed/`. `dynamic` scored 0.00 on that
task and `dynamic-v2` scored 1.00.

| | `dynamic` | `dynamic-v2` | `dynamic-v3` |
|---|---:|---:|---:|
| solved | 19/23 | 19/23 | 17/22 |
| mean functional | 0.826 | 0.857 | 0.773 |
| completion tokens | 411 118 | 846 811 | 753 027 |
| per request | — | 1 071 | 1 572 |
| requests | — | 791 | 479 |
| prompt tokens | 11 608 558 | 16 681 500 | 5 061 297 |

Per task (functional / completion tokens / steps) is in §5c.

**Levels chosen.** 479 requests: level 0 (the `low` sentence) 338, level 1
(template `medium`, no sentence) 33, level 2 (the `xhigh` sentence) 108. **167**
of them were decided by the memory; the rest ran at the default level because
the memory was still cold (the first ~128 finished requests, by design), because
the prompt had no usable seam (33 — under one KV block), or because no vector
reached the scheduler (148, §7).

Reasoning tokens by chosen level:

| level | n | mean | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| 0 (`low`) | 338 | 587 | 25 | 711 | 25 757 |
| 1 (`medium`) | 33 | 302 | 41 | 665 | 4 649 |
| 2 (`xhigh`) | 108 | **2 183** | 86 | **5 308** | 40 849 |

That separation is the signal doing its job: the level-2 band thinks 3.7x longer
at the mean and 7.5x at p90 than the level-0 band, on the same server, with the
level chosen before either of them produced a token.

## 5a. Reasoning length without a cap

Per-request reasoning-token distribution, from the run traces (the `dynamic`
column's traces predate the field):

| run set | n | mean | p50 | p90 | p99 | max | >4k | >8k | >16k | >32k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dynamic-v2` (capped 1k/4k/16k + soft limit) | 791 | 546 | 68 | 1 303 | 5 865 | 16 663 | 13 | 2 | 1 | 0 |
| `dynamic-v3` (no cap, greedy) | 479 | 927 | **28** | 2 092 | 19 355 | **40 849** | 28 | 13 | 6 | 1 |

Read both halves. The **median falls** — most requests think less once the level
is chosen for them — and the **tail triples**: p99 goes 5 865 -> 19 355 and the
maximum is no longer the 16 384 cap but 40 849 tokens. `dynamic-v2`'s maximum
*is* its cap; `dynamic-v3` has nothing to stop it.

Qwen's model card warns that **greedy decoding can cause endless repetition**,
and with no cap a looping think block is bounded only by `max_tokens` and the
task timeout. That is the most likely explanation for both the runaway that was
killed and the 254 066-token solve of `oss-chi-readfrom-tee-doublecount`. So
`dynamic-v3` is measured twice on the same build: once greedy (above, like every
other v3 column) and once with the model's recommended thinking-mode sampling
(temperature 1.0, top_p 0.95, top_k 20, min_p 0, fixed seed) below.

## 5c. Per task (greedy)

functional score / completion tokens / agent steps.

| task | `dynamic` | `dynamic-v2` | `dynamic-v3` |
|---|---|---|---|
| `oss-aiohttp-upgrade-deferred` | 0.00 / 52,102 / 236 | 1.00 / 135,366 / 472 | — (killed, runaway) |
| `oss-chi-readfrom-tee-doublecount` | 1.00 / 3,214 / 70 | 1.00 / 2,971 / 58 | 1.00 / 254,066 / 78 |
| `oss-cobra-noduplicateargs` | 1.00 / 1,405 / 60 | 1.00 / 1,793 / 52 | 1.00 / 1,626 / 60 |
| `oss-flask-teardown-robust` | 1.00 / 15,480 / 162 | 1.00 / 7,601 / 98 | 0.00 / 20,080 / 48 |
| `oss-hono-client-header-merge` | 1.00 / 10,337 / 80 | 1.00 / 10,558 / 72 | 1.00 / 22,142 / 124 |
| `oss-hono-request-bytes` | 1.00 / 2,460 / 78 | 1.00 / 3,458 / 96 | 1.00 / 17,563 / 148 |
| `oss-itertools-strip-prefix` | 1.00 / 21,584 / 296 | 0.00 / 28,484 / 300 | 0.00 / 52,567 / 88 |
| `oss-jiff-date-day-lt1` | 1.00 / 4,751 / 136 | 1.00 / 2,219 / 60 | 1.00 / 3,053 / 76 |
| `oss-jiff-signdur-panic` | 1.00 / 4,469 / 112 | 1.00 / 2,598 / 80 | 1.00 / 2,365 / 64 |
| `oss-jiff-strftime-negpad` | 1.00 / 18,701 / 172 | 1.00 / 55,986 / 300 | 1.00 / 90,216 / 134 |
| `oss-more-itertools-interleave-empty` | 1.00 / 1,486 / 50 | 1.00 / 1,554 / 54 | 1.00 / 1,615 / 54 |
| `oss-networkx-leiden-communities` | 0.00 / 63,179 / 197 | 0.50 / 94,345 / 288 | 0.00 / 9,155 / 40 |
| `oss-packaging-range-prerelease-policy` | 1.00 / 7,731 / 102 | 1.00 / 8,886 / 128 | 1.00 / 5,660 / 108 |
| `oss-pennylane-trotter-fragmented` | 0.00 / 61,910 / 175 | 0.20 / 124,283 / 322 | 0.00 / 45,860 / 54 |
| `oss-pflag-uintslice-hex` | 1.00 / 1,851 / 46 | 1.00 / 2,516 / 54 | 1.00 / 2,337 / 58 |
| `oss-semver-inc-dotted-prerelease` | 1.00 / 16,768 / 78 | 1.00 / 14,716 / 108 | 1.00 / 30,493 / 74 |
| `oss-semver-truncate` | 1.00 / 6,049 / 108 | 1.00 / 2,662 / 90 | 1.00 / 3,199 / 76 |
| `oss-semver-xrange-order` | 1.00 / 9,949 / 108 | 1.00 / 6,857 / 98 | 1.00 / 48,037 / 138 |
| `oss-sqlglot-canonicalize-internal-names` | 0.00 / 66,729 / 209 | 0.00 / 298,334 / 113 | 0.00 / 80,157 / 92 |
| `oss-sqlglot-iso8601-nanos` | 1.00 / 13,601 / 202 | 1.00 / 8,335 / 138 | 1.00 / 27,515 / 240 |
| `oss-sqlglot-qualify-lateral-star` | 1.00 / 15,128 / 188 | 1.00 / 17,883 / 190 | 1.00 / 24,572 / 218 |
| `oss-zod-invert-codec` | 1.00 / 9,301 / 160 | 1.00 / 13,384 / 230 | 1.00 / 7,679 / 186 |
| `oss-zod-proto-catchall` | 1.00 / 2,933 / 94 | 1.00 / 2,022 / 62 | 1.00 / 3,070 / 94 |

`oss-chi-readfrom-tee-doublecount` is the runaway that still solved:
254 066 completion tokens against 3 214 and 2 971 for the same task in the
two earlier columns.

## 5b. The `dynamic-v3-sampled` column

_(filled in from the run)_

## 6. Bugs found and fixed on the way

| symptom | cause | fix |
|---|---|---|
| every `dynamic` request hangs; engine spins with one request `waiting/capacity` | the body chunk ended mid-block, and Mamba `align` clips a non-final chunk to a block boundary — to zero | `_effort_body_boundary` (commit `f621e16c5a`) |
| the split never engages on agent traffic | the sentence sat on the last *user* message, ~20 % into the prompt | tail placement + the §2 grid (commit `38d474cd53`) |
| the boundary is a further 1648 tokens early for nothing | the eagle "last cacheable position" backoff was copied into the boundary rule, where it is not a constraint | dropped (commit `38d474cd53`) |
| engine dies at request finish | a periodic log still read the pre-rename counter attribute | renamed with the rest |

## 7. What this run does not settle

- **The worker-side escalation tensors** (`effort_escalation.py`,
  `effort_policy.py` is deleted but the worker module stays) are still in the
  tree, permanently unarmed because nothing sets `worker_eval`. Cutting them out
  touches `sampler.py`, `rejection_sampler.py`, `async_utils.py` and
  `model_runner.py` and was not worth doing between a benchmark and its rerun.
  It is the next thing on this branch.
- **The 148 `no-vector` requests were a bug, and it is fixed.** A body wider
  than one prefill chunk (boundary 3296 or more, i.e. every long prompt) had its
  decision consumed by the *previous* chunk's output, which carries no vector,
  because async scheduling had already advanced the token counter past the body
  boundary before that output was processed. So the column above is the router
  running on short prompts only; the long half of the traffic never took a
  decision and never entered the memory. Root cause and fix:
  `dynamic-reasoning.claude.md` §13.10 item 4. **The `dynamic-v3` numbers here
  must be re-measured on the fixed build before they mean anything about long
  prompts.**
- **The placement grid is six prompts, one sample each.** It is enough to
  choose a placement, not to quote a coefficient.
- **`split_min_fraction` still excludes short prompts.** With a 1648-token
  block, a prompt under one block gets no decision at all. On the v3 suite that
  is the opening turn of a task.
- Multimodality: the traces are text-only, as in the probe doc.
- One model, one box, 23 tasks.
