# P8 on the box: what the hidden-state start rung actually does

Status: **implemented and benchmarked 2026-08-19** on the 4x L4 trial arm
(`:8013`, `/shared/vllm/.venv-effort-v3`, branch `qwen3.8-27B-effort-v3`,
venv patch `serve-configs/patches/0012`). Design:
[`dynamic-reasoning.claude.md`](dynamic-reasoning.claude.md) §13. Measurement the
design rests on: [`effort-hidden-probe.md`](effort-hidden-probe.md).

This file records three things: what §13 turned into once it met the real
serving profile, the VulcanBench v3 `dynamic-v3` column, and what is still open.

## 1. Headline

_(filled in from the run; see §4)_

## 2. What §13 said and what shipped

§13.3 assumed the prompt splits cleanly at the effort sentence:

```
[ body ................................. ][ tail ]
system + every turn + last user content    effort sentence + <|im_start|>assistant\n<think>\n
```

Two properties of this deployment break that seam, and both were only visible on
the GPU. They are the substance of this run.

### 2.1 The seam is not at the tail of an agent prompt

`apply_dynamic_effort` appends the effort sentence to the last **user** message
(§2b measured that placement, and patch 0009 shipped it). In a VulcanBench agent
turn the last message is a *tool result*, so the seam lands early: measured on
the live run, the first turn of a task has its seam at ~96 % of the prompt, but
every later turn puts it at a fraction of a prompt that grows to tens of
thousands of tokens. The body the decision would read is then a small prefix of
what the model actually reads — and the last few thousand tokens of an agent
prompt (the diff, the failing test, the tool output) are exactly the part that
says how hard the step is.

Moving the sentence to the true tail was rejected: it would put an instruction
sentence inside a tool-result message, and it would change the rung-0 rendering,
which is the thing `dynamic-v2` is the control for.

### 2.2 The KV block is 1648 tokens wide

The served profile is hybrid GDN with prefix caching, so vLLM picks the Mamba
`align` cache mode and widens the attention block so its page covers the mamba
page:

```
Setting attention block size to 1648 tokens to ensure that attention page size
is >= mamba page size
```

In `align` mode a **non-final prefill chunk may only end on a block boundary
whose SSM state is cacheable** (`Scheduler._mamba_block_aligned_split`). The
body chunk ended at the frontend's seam, an arbitrary token, so the aligning
split clipped it to zero and the request was never admitted: the engine spun at
`num_requests_waiting 1, reason capacity` with no forward progress, and every
`reasoning_effort: "dynamic"` request hung. That is the first thing this run
found, and it is a hard blocker for the §13.3 mechanism as written.

The fix (`_effort_body_boundary`) moves the boundary to the last cacheable block
boundary at or before the seam — one further block back, because an
eagle-family drafter prunes the last matching block. With a 1648-token block
that is up to 3 296 tokens before the seam, and a prompt shorter than two blocks
has no usable boundary at all.

### 2.3 So the shipped mechanism has two forms

`hidden_effort.split_min_fraction` (default 0.75) decides between them per
request:

| | two-phase (§13.3) | cap-only (this run) |
|---|---|---|
| when | the body still covers ≥ `split_min_fraction` of the prompt | otherwise |
| prompt | body prefills alone, chosen rung's tail appended after the decision | untouched, prefills in one go |
| vector | last row of the body | last row of the **whole prompt** — the `last_final` the probe measured |
| decision sets | the rung's prompt sentence **and** the starting cap | the starting cap only |
| cost | one extra engine step | none |

Both forms share everything else: the same online memory, the same kNN, the same
asymmetric map, the same fallbacks. `hidden_effort.enabled=false`, a cold
memory, a missing vector, `shadow=true`, the V1 runner, or a rung whose cap the
`max_tokens` headroom cannot honour all render rung 0 with a byte-identical
prompt.

On VulcanBench v3 **every** request took the cap-only form, so this column
measures the hidden-state signal and the asymmetric map without the prompt-tail
half of §13.5. That is a strictly cleaner comparison against `dynamic-v2` — the
prompt is byte-identical — but it does not answer §13.5's open question about
the token cost of the higher starting rungs' *sentences*, because no sentence
changed.

## 3. Method

- Server: the `B_mtp.yaml` latency profile (Qwen3.8-27B-FP8, TP4, MTP K=7
  adaptive, V2 runner, `TRITON_ATTN`, `NCCL_P2P_LEVEL=SYS`, fp8 lm_head, fp8 KV)
  with `hidden_effort` on and nothing else changed:
  `{"dynamic_effort": {"hidden_effort": {"enabled": true, "memory_path": ..., "min_entries": 128, "flush_every": 128}}}`.
  Production (`vllm-qwen38`, `:8012`) stayed stopped throughout.
- Harness: `vulcanbench run --suite v3 --model qwen:Qwen3.8-27B --effort dynamic-v3`,
  greedy sampling, `--no-judges`, `--override-budgets`, `--timeout 7200`,
  `--max-steps 300`, same as the other v3 columns **except**
  `--max-concurrency 12` (the `dynamic` and `dynamic-v2` columns ran at 5).
  Per-task wall clock is therefore **not** comparable to those columns; solved
  counts, tokens and rung mixes are.
- **Cold start is part of the measurement.** The memory was not warmed with the
  bench tasks: the server started with an empty ring (3 synthetic entries from
  the smoke test, far below the 128-entry `min_entries` threshold, so no
  decision could fire), and the first ~128 finished requests of the run got
  today's rung-0 behaviour by construction while filling it.

## 4. The `dynamic-v3` column

_(filled in from the run)_

## 5. Bugs found and fixed on the way

| symptom | cause | fix |
|---|---|---|
| every `dynamic` request hangs; engine spins with one request `waiting/capacity` | the body chunk ended mid-block, and Mamba `align` clips a non-final chunk to a block boundary — to zero | `_effort_body_boundary` (commit `f621e16c5a`) |
| decision never fires; memory stays empty on agent traffic | the seam sits ~20 % into an agent prompt, and the 1648-token block quantises the boundary below it | `split_min_fraction` + the cap-only form (commit `17d46c0c23`) |
| `start_rung` absent from the response | `EffortInfo` did not carry it | added; it is also `vllm:effort_start_rung` |

## 6. What this run does not settle

- **The prompt-sentence half of §13.5.** No request took the two-phase form, so
  the token cost of a higher starting *sentence* is still unmeasured. Getting it
  on this profile needs either the effort sentence at the true prompt tail or a
  narrower attention block, and both change the rung-0 rendering that
  `dynamic-v2` is the control for.
- **The §13.6 deletions.** `rule="score"` and its `theta`/`w_*`/`calibration`
  surface, the backtrack markers, `p_uncertain`/`baseline_*`/
  `uncertainty_min_auc` and `grace_tokens` are all still present. Every one of
  them is inert at the shipped defaults (the AUC gate has kept the entropy and
  margin features off since P7), and deleting them churns the worker-side torch
  rule that this benchmark depends on, so the cutover was deferred rather than
  landed in the same change as the measurement. It is the next thing to do on
  this branch.
- Multimodality: the traces are text-only, as in the probe doc.
- One model, one box, 23 tasks.
