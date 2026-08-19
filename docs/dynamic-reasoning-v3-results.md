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

_(filled in from the run; see §5)_

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

## 5. The `dynamic-v3` column

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
- **The placement grid is six prompts, one sample each.** It is enough to
  choose a placement, not to quote a coefficient.
- **`split_min_fraction` still excludes short prompts.** With a 1648-token
  block, a prompt under one block gets no decision at all. On the v3 suite that
  is the opening turn of a task.
- Multimodality: the traces are text-only, as in the probe doc.
- One model, one box, 23 tasks.
