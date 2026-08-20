# Effort router prototype — can a small encoder pick the reasoning effort before thinking starts?

Status: **offline CPU prototype, 2026-08-19.** After the scope change on
2026-08-19 this track is the **external-encoder comparison baseline** for the
hidden-state (Qwen prefill activations) design; the shared dataset in §1 is what
both tracks consume. No GPU, no server, no VulcanBench
process was touched. Nothing was committed to a vLLM repo and the VulcanBench
repo is unchanged. All code, data and raw results are under
`/shared/vllm/work/router-proto/`.

This is the P4 question in
[`docs/dynamic-reasoning.claude.md`](dynamic-reasoning.claude.md) §8 — "frontend
classifier for the last tool turn → `effort_bias`" — asked properly: **before
`<think>` opens, from the request alone, can we tell how hard the next step is?**
The user's constraint stands: general, model-agnostic signals only, no
hand-written lexical rules. Everything below is either a zero-shot semantic
classifier, a learned head on a frozen encoder, or a label-free geometric
statistic over embeddings. No keyword lists were written.

> Note on the brief: the task pointed at `docs/dynamic-reasoning.claude.md`
> §11–12. That document ends at §10; there are no §11–12. The constraints used
> here are the ones actually in the file (§3 signal inventory, §4 controller,
> §8 P4, §10 risks) plus the standing "no ad-hoc heuristics" rule.

## TL;DR

1. **The zero-shot LFM2.5-350M-Prompt-Router does not work for this.** Spearman
   **−0.157** against reasoning tokens at natural close, AUC **0.437**
   [0.387, 0.480] for "long think", task-level AUC **0.22** for "hard vs easy" —
   consistently *inverted*, not merely at chance. The sign of its answer flips
   with how the lanes are worded. And it costs **988 ms** at a 512-token tail and
   **110 s** at 32k tokens, against a ~215 ms budget.
2. **Free prompt length is the strongest single signal in the whole study**:
   AUC **0.729**, Spearman **0.494**. The in-distribution ceiling for *any*
   prompt-side predictor here is ~0.75, so there is almost no headroom to win.
3. **Every sub-100M candidate is within noise of that control**, and of each
   other. If we shipped one it would be **AIST-87M** (AUC 0.713, within-run
   0.593, 398 ms at `CORTEXT_AIST_THREADS=8`, the only sub-100M model with a real
   shared text+image space) — but "the one we'd ship" is not "worth shipping".
4. **The embedding is the least useful part of every candidate.** A 1280-d AIST
   vector into a logistic head scores **0.441**; four label-free scalars derived
   from the same vectors score **0.593**. The best label-free signals are
   `knn_effort` (kNN over past runs, 0.637 raw / 0.593 within-run) and
   `novelty_loto` over hashed n-grams (**0.632** within-run, the best number
   here). Surprisal is flat (0.472).
5. **The simulated policy is strictly worse on both axes**: 1.93× the reasoning
   tokens of `dynamic-v2` and 3.97× the `dynamic` column, leaning hardest on
   `extra-high` for a quarter of requests, which is the *lowest*-scoring effort
   column in the matrix.
6. **The 230M fine-tune was not run** (scope cut; spec and cost in §10). The
   blocker is labels, not compute: 689 think-length examples over 23 tasks. One
   harness one-liner — record the effort object in `llm_response` for the other
   four columns — gives a 4.7× larger set and the only way to test whether a
   signal survives a change of effort rendering.

## 0. What was built, and where

| path | what |
|---|---|
| `work/router-proto/scripts/extract_requests.py` | reconstructs the exact prompt for every `llm_request` in the v3 traces |
| `work/router-proto/scripts/make_dataset.py` | writes the shared dataset (below) |
| `work/router-proto/scripts/bench_latency.py` | CPU latency of every candidate encoder |
| `work/router-proto/scripts/score_lanes.py` | batched zero-shot lane scoring with the LFM2.5 router |
| `work/router-proto/scripts/embed_aist.py` | AIST-87M embeddings via `cortext` |
| `work/router-proto/scripts/baselines.py` | frozen-encoder embeddings + GroupKFold heads |
| `work/router-proto/scripts/probes.py` | label-free surprisal / novelty / kNN + trained heads |
| `work/router-proto/scripts/analyze.py` | the (a)/(b)/(c) validation and the controls |
| `work/router-proto/scripts/simulate_policy.py` | the simulated Pareto point |
| `work/router-proto/scripts/aist_image_probe.py` | AIST image path — shape and latency only |
| `work/router-proto/results/` | every number below, as JSON/CSV |
| `work/router-proto/dataset/` | the shared dataset + README (see §1) |

Environment: `/shared/vllm/.venv-effort-v2` (Python 3.12, torch 2.13.0,
transformers 5.15.0) with `scikit-learn`, `scipy`, `model2vec` and `cortext`
added through `uv pip`. `HF_HOME=/data/huggingface`. Nothing was installed with
system `python3` or bare `pip`.

**Box.** AMD Ryzen Threadripper PRO 3945WX — **12 physical cores**, 20 logical
CPUs visible inside the LXC, 120 GB RAM (98 GB available), AVX2 but **no
AVX-512, no VNNI, no AMX**. The 4×L4 vLLM server was serving throughout
(`.venv-dflash2`, ~2 cores of steady load, load average ≈ 2.3/20). Every CPU
number below is therefore a *loaded-box* number; treat them as upper bounds by
maybe 10–20%, not as clean-room figures.

## 1. Reconstructing what the model actually saw

`harness/agent/loop.py` builds the message list deterministically:

```
messages[0] = system : SYSTEM_PROMPT (+ a repo-scale sentence for medium/large repos)
messages[1] = user   : "# Issue\n\n" + tasks/v3/<task_id>/issue.md
per step     : assistant(content, tool_calls), then one tool message per call whose
               content is json.dumps({"result":…, "error":…}) cut at 8000 chars
```

Replaying that against the traces reproduces the conversation exactly. The
harness records `llm_request.data.messages` (`len(messages)` at call time), which
gives a free end-to-end check: **3258 / 3258 requests match, 0 mismatches.**

The reconstruction, the targets and the split are published for other tracks at
**`/shared/vllm/work/router-proto/dataset/requests.jsonl`** (3258 lines, 334 MB)
with `dataset/README.md`, `dataset/tools.json` and `dataset/dataset_meta.json`.
Written **2026-08-19 12:58 UTC**; the path is stable and the file is regenerated
in place by rerunning `scripts/make_dataset.py`.

### What the traces do and do not carry

| column | runs | requests | per-request reasoning tokens |
|---|---:|---:|---|
| `low` | 23 | 582 | no |
| `medium` | 22 | 521 | no |
| `extra-high` | 22 | 657 | no |
| `dynamic` | 23 | 707 | no |
| `dynamic-v2` | 23 | 791 | **yes** — `{rung, escalations, reasoning_tokens, late, close_kind}` |

Only `dynamic-v2` carries think lengths. The other columns record neither
`reasoning_tokens` nor `reasoning_content` (`llm_response.content` is the visible
answer only), so the "reasoning_content length elsewhere" fallback in the brief
**does not exist in this data**. Every think-length number below is therefore on
`dynamic-v2`: 791 requests, of which **689 closed naturally**, 97 forced, 5 soft.
Natural-close reasoning tokens are heavily skewed: p25 = 22, p50 = 48, p75 = 233,
p90 = 1482, max = 10 731, mean = 433.

Prompt sizes are large: p50 = 13 043 Qwen tokens, p90 = 31 337, p99 = 64 801,
max = 91 274 (p50 = 38 327 characters). This matters for every candidate below.

### `/data/effort-telemetry/latency.jsonl` is not joinable

529 594 lines, 2 759 distinct `req_id`s, fields
`{req_id, step, num_output_tokens, entropy, margin, p_end, n_rows,
num_draft_tokens, num_accepted, in_think}`. There is **no timestamp, no task id,
no run id, no step-of-conversation, no effort/rung/close_kind field**, and the
`req_id` (`chatcmpl-<hex>-<hex>`) never appears in the traces — the trace's tool
ids (`chatcmpl-tool-<hex>`) are a different namespace. The only possible join is
a fuzzy match on total completion tokens, and it is unique for **244 of 791**
dynamic-v2 requests (31%). Conclusion: **not reliably linkable**; do not build on
it. (Its max `num_output_tokens` = 235 349 does equal the max dynamic-v2
completion count, so the file does cover these runs — it just cannot say which
row is which.) `in_think` would let one recompute reasoning tokens per request,
which the traces already give for dynamic-v2.

## 2. The candidates

Two constraints shaped the field. First, the user's size bar: **anything ≥230M is
expensive for this job; the target is <100M parameters.** Second, requests in this
loop can carry images, so a shippable classifier must embed **text and image**.

| candidate | params | modality | context it can actually see | how it is used |
|---|---:|---|---|---|
| LFM2.5-Encoder-350M-Prompt-Router | 355M | text | 8 192 tokens (trained; RoPE allows 128k) | zero-shot `route(prompt, lanes)` |
| LFM2.5-Encoder-230M | 230M | text | 8 192 tokens | frozen body + mean pool + linear head (the cookbook recipe's head, trained on CPU) |
| AIST-87M (cortext.cpp, GGUF q8_0) | 87M | **text + image + audio, one shared space** | ~2 000 characters (≈512 tokens), truncated from the head | `EmbedText` + linear/MLP head |
| all-MiniLM-L6-v2 | 22.7M | text | 512 tokens | mean pool + head |
| bge-small-en-v1.5 | 33.4M | text | 512 tokens | CLS pool + head |
| e5-small-v2 | 33.4M | text | 512 tokens | mean pool + head |
| potion-base-8M (model2vec) | 7.6M | text | unlimited (static bag of embeddings) | mean of token vectors + head |
| potion-retrieval-32M | 32.8M | text | unlimited | as above |
| prompt length / step index | 0 | — | — | the free control the server already has |

**Context limits, and the truncation policy this forces.** The LFM2.5 encoders are
trained to 8 192 tokens (model card); `max_position_embeddings` is 128 000 but that
is the RoPE range, not the trained window. The BERT-family baselines stop at 512.
AIST-87M saturates at roughly 2 000 characters and — measured — keeps the **head**:
`cos(e(x), e(x + junk)) = 0.96` while `cos(e(x), e(junk + x)) = cos(e(x), e(junk)) = 0.83`.
Since the median agent request is 13 043 tokens, **every candidate must truncate**,
and the caller must apply the tail window itself (`truncation_side="left"` for the
HF tokenizers, explicit `text[-N:]` for AIST).

Two windows were compared: `tail` (last N tokens) and `head_tail` (first N/4
tokens — system prompt + task issue — then the last 3N/4). Latency is identical
for both at equal token count, so only the accuracy comparison is meaningful; see
§3.

### Making the size comparison fair: what "multimodal" costs

The v3 traces are text-only, so nothing below *measures* image accuracy. But the
production loop can carry screenshots, and a text-only encoder cannot be made
multimodal by bolting a vision tower onto it — the two towers have to have been
trained into a **shared** embedding space, otherwise their vectors are not
comparable and the head sees two unrelated feature blocks.

| shippable multimodal option | text side | image side | total params |
|---|---|---|---:|
| AIST-87M | included | included | **87M** (one model, one space) |
| CLIP ViT-B/32 | its own text tower ≈63M | ≈88M | ≈151M |
| SigLIP-base-patch16-224 | ≈110M | ≈93M | ≈203M |
| MiniLM-L6 + a separate vision encoder | 22.7M | — | not a shared space; the MiniLM text vectors would have to be dropped and replaced by the CLIP/SigLIP text tower |

So the honest sub-100M multimodal field is AIST-87M alone; MiniLM/bge/e5/potion
are sub-100M **only as text-only** models, and the moment images matter they are
replaced (not augmented) by a ≥151M CLIP/SigLIP pair. That is why AIST-87M is
treated as the preferred candidate if it lands within noise of the best.

## 3. Method: targets, split, metrics

All tracks share `dataset/README.md`'s protocol.

**Targets** (all derived from the traces; no hand-labelling):

- `reasoning_tokens` at **natural close** — how long Qwen chose to think when
  nothing forced it to stop. 689 requests (dynamic-v2).
- `long_think` — `reasoning_tokens ≥ 233` (the 75th percentile). 172 positives.
- `think3` — tertiles of the same, cuts at 26 and 137.8 tokens.
- `trouble` — this step's tool calls returned a harness error or a non-zero exit
  code. 277 positives over all 3258 requests. This is the (c) question: does the
  signal rise on the step *before* things go wrong?

**Split.** GroupKFold over `task_id`, 5 folds, out-of-fold predictions only. Never
group by run: the same task appears in five effort columns with near-identical
early prompts, so a random split leaks almost perfectly.

**Metrics.** Rank AUC with 1000-sample bootstrap CIs, and Spearman against raw
reasoning tokens. Everything is reported twice: raw, and after rank-normalising
the signal **within each run** — the second number answers "does this pick the
hard steps *of this conversation*", which is what a per-request policy needs.

**The bar.** The brief set the bar at "better than the entropy/margin signals we
already found at chance (AUC ≈ 0.5)". That bar is far too low. The free controls
the API server already has at prefill time do much better than chance:

| control | Spearman vs think tokens | AUC long-think | AUC within-run | AUC trouble |
|---|---:|---:|---:|---:|
| prompt length (tokens) | **0.494** | **0.729** [0.691, 0.769] | 0.581 [0.537, 0.628] | 0.479 |
| step index | 0.467 | 0.703 | 0.581 | 0.485 |
| message count | 0.477 | 0.708 | 0.581 | 0.487 |

(The three within-run numbers are identical because inside one run all three are
the same ranking.) So the real bar is **AUC 0.73 raw / 0.58 within-run for
think length, and ~0.50 for trouble.** Anything that does not clear prompt length
is not worth a millisecond of the request path.

Task tiers used for question (b) are derived from the observed score matrix, not
from a hand-written list: **11 solved-by-all**, **9 effort-gated** (low failed,
some higher effort solved), **3 unsolved-everywhere**.

### An upper bound worth knowing first

Before asking whether any particular encoder works, it is worth asking how much
signal is there at all. Fitting the same logistic head on hashed word n-grams of
the prompt tail with a **random** 5-fold split — i.e. deliberately letting the
same task appear in train and test, which is leakage — tops out at
**AUC ≈ 0.75** for "long think", against 0.729 for free prompt length with a
proper task-grouped split.

So the ceiling for *any* prompt-side predictor of how long Qwen will think on this
loop is around 0.75, and the free control is already at 0.73. There is very little
headroom for a learned router to claim, and any claim above ~0.75 should be
treated as leakage until proven otherwise.

### AIST-87M image path (shape and latency only)

`cortext.Cortext().embed_image(rgb_bytes, w, h, 3)` returns a **1280-d** vector —
the same dimension as `embed_text`, i.e. one shared space, which is the whole
point of the candidate. A 224×224 RGB frame costs **436 ms p50 / 508 ms p95** at
one thread on the loaded box (`results/aist_image_smoke.json`). The v3 traces
contain no images, so this validates the mechanism and its cost, **not** its
accuracy on agent screenshots. Any claim that the router handles screenshots well
needs a trace set that actually contains them.

## 4. Zero-shot lanes

### Design

Four lane sets were written, all describing *the step*, never keywords:

| id | lanes |
|---|---|
| `A_step3` | trivial mechanical step / moderate routine step / deep multi-step reasoning step |
| `B_think3` | needs no deliberation / needs a little deliberation / needs extended deliberation |
| `C_step4` | trivial mechanical / lookup and exploration / moderate implementation / hard debugging after a failure |
| `D_bare3` | `Trivial tool step` / `Routine code edit` / `Hard multi-step debugging` (the bare-label style the model card demonstrates) |

`model.route()` prepends `Categories:\n- <lane>\n…\n\nText:\n` and mean-pools the
**text** token states into one vector, mean-pools each lane's own tokens into a
second, and softmaxes their cosine similarities. Two consequences matter here:

1. Scores are **relative** — they always sum to 1 across lanes, so "everything is
   hard" and "everything is easy" are indistinguishable; only the ordering is
   information.
2. The text representation is a **mean over every text token**. Our inputs are
   agentic conversations whose bulk is JSON tool output; a 512-token tail of
   `{"result": {...}, "error": null}` mean-pools to nearly the same vector at
   step 3 and step 40 of the same run. This is the mechanism to keep in mind when
   reading the numbers below.

`route()` was reimplemented as a batched call for the sweep (identical maths,
right padding + attention mask); it agrees with the model's own `route()` to
**3.0e-08** max absolute error (`scripts/score_lanes.py --verify-batching`).

## 5. Label-free signals from cortext.cpp

Three statistics that need **no labels at all**, computed over each candidate's
embeddings of the same reconstructed requests, re-implemented in numpy from
`cortext.cpp` (`include/cortext/operations/embedding_prediction_error.hpp`,
`docs/paper/_manuscript/index.md` §3.1.4 and the novelty section):

```
prediction_error_t = 1 - max(0, cos(x_t, x_pred_ema))
surprisal_t        = sigmoid((prediction_error_t - err_ref(S)) * k(S,T))
x_pred_ema        <- l2_normalize((1 - beta) * x_pred_ema + beta * x_t),  beta = 0.25

novelty_recent_t   = clamp((1 - max cos to the earlier steps of THIS run) / 2, 0, 1)
novelty_loto_t     = clamp((1 - max cos to requests from OTHER tasks) / 2, 0, 1)
knn_effort_t       = mean reasoning-tokens-at-natural-close of the k=16 nearest
                     leave-one-task-out neighbours
```

`sigmoid` is monotone, so AUC and Spearman are invariant to `err_ref` and `k` —
the inner prediction error is what is scored, and the cortext constants cannot
change the ranking metrics. `novelty_recent` is cortext's own definition (recent
context = this session's earlier steps); `novelty_loto` and `knn_effort` use a
memory built strictly from other tasks, so nothing leaks across the split.

`knn_effort` is the "embedding-kNN over past runs" baseline: it is the only one
of the three that uses labels at all, and only through retrieval, not fitting.

## 6. Simulated policy — method and its limits

The simulation asks: *if the lane had chosen the effort sentence per request,
what would the token spend have been?* It is a **lookup** simulation, not a rerun:

- lane 0 → `low`, middle lane(s) → `medium`, top lane → `extra-high`;
- for each request in the reference column, look up the observed **mean
  completion tokens per request** for that `(task, effort)` run;
- sum over the run.

Observed per-column totals (the lookup table, `results/effort_lookup_table.json`):

| column | runs | requests | completion tokens | mean/request | mean functional |
|---|---:|---:|---:|---:|---:|
| `low` | 23 | 583 | 643 268 | 1 103 | 0.609 |
| `medium` | 22 | 501 | 792 181 | 1 581 | 0.636 |
| `extra-high` | 22 | 645 | 788 635 | 1 223 | 0.545 |
| `dynamic` | 23 | 708 | 411 118 | **581** | 0.826 |
| `dynamic-v2` | 23 | 791 | 846 811 | 1 071 | **0.857** |

Two things are already visible without any router: **`extra-high` is both the
most expensive per task and the *worst* scoring column (0.545)**, and today's
`dynamic` is the cheapest column by a factor of two while scoring 0.83. That is
the Pareto frontier a router has to beat, and it is not an easy one.

The simulation ignores that changing the effort changes the trajectory — step
count, which files get read, whether the task is solved at all. Treat every number
in §9 as an order-of-magnitude sketch, clearly labelled as simulation.

## 7. Results

### 7a. The zero-shot router is at chance — and slightly inverted

All 3153 unique prompts scored against `A_step3` with a 512-token tail window
(`results/lane_scores_w512.jsonl`, 3814 s of CPU). The scores are not degenerate:
lane means 0.298 / 0.342 / 0.359, per-lane std 0.11 / 0.10 / 0.15, argmax split
428 / 1770 / 955.

| signal | Spearman vs think tokens | AUC long-think | AUC within-run | AUC trouble |
|---|---:|---:|---:|---:|
| `A_step3` expected lane | **−0.134** | **0.441** [0.393, 0.484] | 0.480 | 0.483 [0.445, 0.517] |
| `A_step3` P(deepest lane) | **−0.157** | **0.437** [0.387, 0.480] | 0.500 | 0.495 [0.459, 0.529] |
| prompt length (control) | 0.494 | 0.729 [0.690, 0.767] | 0.581 | 0.479 |

The router is not merely uninformative — it is **mildly anti-correlated** with how
long Qwen actually thought. Both AUC confidence intervals exclude 0.5 on the wrong
side.

**(b) per task.** The same inversion, larger:

| tier | tasks | mean P(deep) | deep-lane fraction |
|---|---:|---:|---:|
| solved by every effort | 11 | 0.398 | **0.414** |
| effort-gated + unsolved | 12 | 0.350 | **0.271** |

Task-level AUC for "hard vs easy" is **0.22** — i.e. it ranks the hard tasks as
*easier* about four times out of five. The tasks it calls deepest are
`jiff-date-day-lt1` (0.66), `chi-readfrom-tee-doublecount` (0.57),
`jiff-signdur-panic` (0.53) — all solved by every effort level. The ones it calls
shallowest are `sqlglot-canonicalize-internal-names` (0.06, unsolved by
everything), `sqlglot-qualify-lateral-star` (0.16) and
`aiohttp-upgrade-deferred` (0.19) — all effort-gated or unsolved. Prompt length,
by contrast, separates those tiers at AUC 0.89.

The mechanism is visible in the data: the hard tasks live in large repos
(sqlglot, aiohttp, semver), so their 512-token tails are dominated by long code
and JSON dumps, which mean-pool towards "trivial mechanical step"; the easy tasks
are small repos whose tails read more like prose. The router is classifying
**surface register**, not difficulty.

**(c) within task.** AUC 0.483 / 0.495 for "this step's tool calls will error or
exit non-zero", within-run 0.495–0.498. Flat chance. Feeding the 3-d lane vector
to a trained head does not rescue it: `logreg(lane scores) → think3` reaches
AUC 0.558, and its **random-split ceiling is only 0.570** — the lane vector
carries almost no information about think length even when leakage is allowed.

### 7b. The comparison table

Each cell is the **best held-out variant of that candidate for that metric**
(GroupKFold by task, out-of-fold predictions, bootstrap CI over requests); the
variants are not necessarily the same across columns, and the full per-variant
grid is in `results/probe_summary.csv`. `lf` =
the cortext-style label-free block (surprisal, novelty-recent, novelty-LOTO,
kNN-16 effort); `meta` = prompt length, step index, message count, tool-call
count — signals the server already has for free.

| candidate | params | modality | CPU p50 (its own window) | best Spearman | best AUC long-think | AUC within-run | AUC trouble |
|---|---:|---|---:|---:|---:|---:|---:|
| **prompt length only** (control) | **0** | — | ~0 ms | **0.494** | **0.729** [0.690, 0.767] | 0.581 [0.537, 0.628] | 0.479 |
| hashed n-grams + logreg (control) | 0 | text | **0.4 ms** @2k chars | 0.473 | 0.710 [0.668, 0.750] | **0.632** [0.587, 0.678] | 0.566 |
| potion-base-8M | 7.6M | text | **1.2 ms** @2k, 1.8 ms @32k tok | 0.435 | 0.698 [0.659, 0.738] | 0.618 [0.571, 0.665] | 0.587 |
| potion-retrieval-32M | 32.8M | text | 0.9 ms @2k, 1.4 ms @32k tok | 0.440 | 0.698 [0.659, 0.736] | 0.604 [0.557, 0.647] | 0.578 |
| all-MiniLM-L6-v2 | 22.7M | text | 37 ms @512 (8 thr) | 0.457 | 0.704 [0.663, 0.743] | 0.631 [0.583, 0.675] | 0.563 |
| bge-small-en-v1.5 | 33.4M | text | 65 ms @512 (8 thr) | 0.457 | 0.712 [0.674, 0.751] | 0.580 [0.535, 0.622] | 0.527 |
| e5-small-v2 | 33.4M | text | 89 ms @512 (8 thr) | **0.478** | **0.724** [0.687, 0.762] | 0.627 [0.583, 0.673] | 0.560 |
| **AIST-87M** (GGUF q8_0) | **87M** | **text+image+audio** | **398 ms** @8 thr, 324 @12 | 0.454 | 0.713 [0.672, 0.753] | 0.593 [0.543, 0.643] | 0.581 |
| LFM2.5-350M-Prompt-Router, zero-shot | 355M | text | 988 ms @512 tok, 110 s @32k | **−0.157** | **0.437** [0.387, 0.480] | 0.500 | 0.495 |
| LFM2.5-350M lane vector + trained head | 355M | text | as above | 0.445 | 0.690 [0.650, 0.729] | 0.555 [0.507, 0.598] | 0.554 |
| LFM2.5-Encoder-230M frozen + head | 230M | text | 343 ms @512 tok, 85 s @32k | not run (see §10) | — | — | — |

Read it this way:

- **Every confidence interval overlaps the free prompt-length control.** Nothing
  here is significantly better than a number the API server already has.
- On raw AUC nothing beats prompt length (0.729). The best encoder, e5-small, ties
  it at 0.724 — and only by having prompt length *inside* its feature block.
- Within-run — the number a per-request policy actually needs — hashed n-grams
  (0.632), MiniLM (0.631) and e5-small (0.627) edge past the control's 0.581, but
  their CIs all contain it. AIST-87M's 0.593 is in the same cloud.
- **AIST-87M is within noise of the best**, at 87M and with one shared text/image
  space. On the size-and-modality axis it is the candidate to ship — if we were
  shipping one.
- The zero-shot 350M router is the only candidate that is *worse than chance*.
- The 8M static-embedding model (potion) is within noise of the 33M transformers
  at **1/30th the latency**, which says the encoder's semantics contribute nearly
  nothing here beyond surface statistics.

### 7c. Where the signal actually lives

Splitting the feature blocks apart, on the same folds:

| feature block | Spearman | AUC long-think | AUC within-run |
|---|---:|---:|---:|
| `meta` only (length, step, messages, tool calls) | 0.438 | 0.684 | 0.554 |
| `lf` only (surprisal + novelty + kNN, 4 numbers) — MiniLM | 0.306 | 0.683 | **0.631** |
| `lf` only — e5-small | 0.348 | 0.701 | 0.627 |
| `emb` only — AIST-87M (1280-d) | −0.032 | 0.441 | 0.496 |
| `emb` only — MiniLM (384-d) | 0.165 | 0.567 | 0.570 |
| `emb + meta` — AIST-87M | 0.454 | 0.713 | 0.580 |

The **raw embedding is the least useful part of every candidate.** What carries
signal is (i) the free structural features and (ii) the four label-free geometric
statistics computed *from* the embeddings. A 1280-d AIST vector fed straight to a
logistic head scores 0.441 — worse than chance — while four scalars derived from
the same vectors score 0.593–0.631 within-run.

### 7d. The label-free cortext signals, on their own

| signal | encoder | Spearman | AUC long-think | AUC within-run | AUC trouble |
|---|---|---:|---:|---:|---:|
| `knn16_effort` | AIST-87M | 0.179 | 0.637 [0.593, 0.681] | 0.593 [0.543, 0.643] | 0.527 |
| `knn16_effort` | MiniLM | 0.145 | 0.582 [0.534, 0.629] | 0.582 [0.535, 0.628] | 0.467 |
| `knn16_effort` | e5-small | 0.188 | 0.601 [0.552, 0.647] | 0.575 [0.527, 0.625] | 0.452 |
| `novelty_loto` | hashed n-grams | 0.336 | 0.701 [0.655, 0.746] | **0.632** [0.587, 0.678] | 0.494 |
| `novelty_loto` | AIST-87M | 0.163 | 0.612 [0.563, 0.660] | 0.556 [0.507, 0.606] | 0.498 |
| `surprisal_ema` | AIST-87M | 0.030 | 0.472 [0.421, 0.522] | 0.485 [0.436, 0.534] | 0.499 |
| `surprisal_ema` | potion-8M | 0.150 | 0.558 | 0.547 | 0.464 |
| `novelty_recent` | potion-8M | 0.071 | 0.552 | 0.540 | 0.441 |

- **`knn_effort` — embedding-kNN over past runs — is the strongest label-free
  signal**, exactly as expected, and it needs no fitting at all: 0.59–0.60 AUC
  within-run from a 16-nearest-neighbour lookup against other tasks' requests.
- **Surprisal is the weakest.** Over AIST embeddings it is dead flat (0.472/0.485).
  A step that "breaks the trajectory" of the embedding stream is not a step Qwen
  thinks longer about — at least not when the embedding is a mean pool of a
  512-token tail dominated by tool output.
- `novelty_loto` over hashed n-grams is the single best within-run number in the
  whole study (0.632), which is a slightly deflating result: "how lexically unlike
  other tasks' requests is this one" beats every neural encoder.
- None of them helps with `trouble` (0.44–0.53, i.e. chance).

### 7e. Lane-phrasing sensitivity

All four phrasings on a fixed 250-prompt random subsample (259 requests; CIs are
wide at this size, `results/analysis_phrasing.json`):

| lane set | Spearman (expected lane) | AUC long-think | AUC within-run |
|---|---:|---:|---:|
| `A_step3` (describes the step) | +0.09 | 0.529 | 0.592 |
| `B_think3` (describes how much deliberation) | **+0.19** | **0.612** | 0.604 |
| `C_step4` (four lanes, adds a "just failed" lane) | −0.10 | **0.363** | 0.539 |
| `D_bare3` (bare labels) | −0.10 | 0.454 | 0.610 |

**The sign of the correlation flips with the wording.** `B_think3` — the only
phrasing that talks about *deliberation* rather than about the *step* — is the
only one that points the right way; `C_step4`, which adds an explicit "a test just
failed, work out the cause" lane, points the most wrongly. That is the opposite of
a robust zero-shot classifier: it means the output is dominated by lexical
similarity between the lane text and the prompt's surface register, not by any
judgement about the task.

Not run (cut when the scope was reduced): the head-tail vs tail window comparison
and the 2048/8192-token window sweep. The full-set numbers therefore stand at a
512-token tail window only, and it remains formally open whether a longer window
would rescue the router. Two things argue it would not: the sign of the effect is
wrong rather than merely small, and a 4096-token window already costs 6.3 s of
CPU per request (§8), which rules it out of the request path regardless.

## 8. CPU latency (deliverable 1)

Full grid in `results/latency_summary.csv` / `.md`. Every row is the **deployable
path** for that candidate (tokenize + pool + forward), 4 real VulcanBench prompts
cut to length, ≥3 samples per cell, on the loaded box described in §0.

| model | params | threads | prompt tokens | tokens actually seen | p50 ms | p95 ms | RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| potion-base-8M | 7.6M | 1 | 512 → 32768 | all of them | **1.0 – 1.3** | 1.1 – 1.6 | 680 |
| all-MiniLM-L6-v2 | 22.7M | 1 / 4 / 8 / 20 | any | 512 (hard cap) | 165 / 64 / **37** / 712 | 172 / 66 / 40 / 790 | 975 |
| bge-small-en-v1.5 | 33.4M | 1 / 4 / 8 / 20 | any | 512 (hard cap) | 328 / 125 / **65** / 1563 | 334 / 131 / 82 / 2472 | 1017 |
| e5-small-v2 | 33.4M | 1 / 4 / 8 / 20 | any | 512 (hard cap) | 371 / 158 / **89** / 1518 | 403 / 267 / 109 / 1982 | 1016 |
| AIST-87M q8_0 | 87M | torch 1 / 4 / 8 (no effect) | any | ~512 (~2 000 chars) | 742 / 717 / 727 | 823 / 745 / 764 | 1764 |
| " (`CORTEXT_AIST_THREADS` 1 / 2 / 4 / 8 / 12) | | | | | 2617 / 1344 / 702 / **398** / **324** | 2623 / 1352 / 720 / 444 / 356 | 1764 |
| LFM2.5-Encoder-230M | 229.7M | 8 | 512 | 512 | 343 | 365 | 2086 |
| " | | 8 | 1024 | 1024 | 807 | 846 | 2090 |
| " | | 8 | 4096 | 4096 | 4 218 | 4 895 | 2334 |
| " | | 8 | 16384 | 16384 | 30 891 | 33 412 | 4160 |
| " | | 8 | 32768 | 32768 | **84 812** | 84 839 | 10 370 |
| LFM2.5-350M-Prompt-Router | 355M | 1 | 512 / 1024 / 4096 | same | 4 142 / 8 154 / 36 339 | 4 163 / 8 175 / 36 399 | 2696 |
| " | | 8 | 512 | 512 | 988 | 1 082 | 2380 |
| " | | 8 | 1024 | 1024 | 1 918 | 2 048 | 2387 |
| " | | 8 | 4096 | 4096 | 8 091 | 8 191 | 2687 |
| " | | 8 | 16384 | 16384 | 45 171 | 45 731 | 4488 |
| " | | 8 | 32768 | 32768 | **110 409** | 117 974 | 10 706 |
| 350M router, dynamic int8 | 67M eff. | 8 | 512 / 1024 / 4096 | same | 542 / 1 317 / 5 705 | 556 / 1 542 / 5 764 | 3658 |

**The 1k/4k/16k/32k grid the brief asked for only exists for the two LFM2.5
encoders and potion.** Everything else has a hard window: 512 tokens for the
BERT-family baselines, ~512 for AIST (it keeps the head and saturates at ~2 000
characters), unlimited for potion (a static bag of embeddings).

Read-outs:

1. **The 350M router is 110 seconds at 32k tokens.** At the p50 agent prompt of
   13k tokens it is already ~35 s. The whole 4×L4 prefill for that request is
   ~4.3 s. This is not a latency budget problem, it is three orders of magnitude.
   Even at a 512-token tail it costs ~1 s. It cannot go in the request path.
2. **20 torch threads is 10–20× *slower* than 8** on every transformer here
   (MiniLM 37 → 712 ms). The box has 12 physical cores and 20 logical CPUs, and
   the vLLM server is using some of them; setting torch threads to `nproc` is
   actively harmful. Any deployment must pin the router to ~8 threads or fewer.
3. **Memory is a real constraint at long context.** The 350M router needs
   10.7 GB RSS at 32k tokens and the 230M needs 10.4 GB — quadratic attention on
   an 8k-trained model being pushed to 32k. A router in the API-server process
   would have to be windowed for memory alone.
4. **Dynamic int8 buys 1.8× and costs 1.3 GB more RSS** (542 vs 988 ms at 512
   tokens). Not enough to change any conclusion. This box is Zen 2: AVX2 only, no
   AVX-512/VNNI/AMX, so int8 and bf16 have little hardware to exploit.
5. **AIST-87M ignores torch threads entirely** — it runs its own GGML backend
   whose thread count comes from `CORTEXT_AIST_THREADS`
   (`src/models/aist_gguf_encoder.cpp:4088`), defaulting to 4. Set it properly and
   it scales cleanly: 2 617 ms at 1 thread, 702 at 4, **398 at 8, 324 at 12**.
   That is the number to quote for AIST, not the 727 ms you get by leaving the
   variable unset. Its image path is ~400 ms p50 at every resolution from 224×224
   to 1920×1080 (fixed preprocessing), output dim 1280 in both modalities — one
   shared space, which is the whole point of the candidate.
6. **potion-base-8M is 1.2 ms at 32 768 tokens** — length-independent, because it
   is an embedding-table lookup plus a mean. It is 600× cheaper than MiniLM and
   within noise of it on every accuracy metric in §7b.

### Truncation policy

The LFM2.5 encoders are trained to **8 192 tokens** (model card), so the p90 agent
request (31 337 tokens) has to be cut even ignoring latency. The window policy
comparison the brief asked for (last-N vs task-statement-plus-last-observation)
was **not completed** — it was cut with the rest of the sensitivity sweep when the
scope was reduced. What is measured: latency depends only on the token count, not
on which tokens, so the two policies cost the same; the full-set accuracy numbers
in §7 are all last-512-tokens.

## 9. Simulated policy — result (clearly labelled: simulation)

Mapping `A_step3` lanes to `low` / `medium` / `extra-high` and looking up each
task's observed mean completion tokens per request at that effort
(`results/policy_simulation_dv2.json`):

- picks over the 791 dynamic-v2 requests: **99 low, 440 medium, 196 extra-high**
  (56 requests had no lookup cell because the task's column is missing);
- simulated spend **1 631 888** completion tokens vs **846 811** observed on
  dynamic-v2 → **1.93× more expensive**, and **3.97×** the `dynamic` column's
  411 118;
- with no accuracy gain to show for it: a quarter of all requests (196 of 735
  with a lookup cell) are routed to `extra-high`, which is the **lowest-scoring**
  column in the whole matrix (mean functional 0.545 vs 0.857 for dynamic-v2) and
  the second most expensive, and only 99 get `low`.

So the simulated Pareto point is strictly worse on both axes than what is already
deployed. Caveats in §6 apply — this is a lookup, not a rerun — but the direction
is not marginal.

## 10. Recommendation

**Zero-shot LFM2.5-Encoder-350M-Prompt-Router: not usable, and not worth a
fine-tune of its own.** Two independent reasons, either of which is disqualifying:

- *Accuracy.* It is worse than chance at the thing we need. Spearman −0.157
  against reasoning tokens at natural close, AUC 0.437 [0.387, 0.480] for
  long-think, task-level AUC 0.22 for "hard vs easy". The sign of its answer
  flips with lane wording (§7e), which is the signature of a lexical
  similarity match, not a judgement. It is doing what it was built for — routing
  a short natural-language prompt to a topic — and an agentic step is neither
  short nor a topic.
- *Latency.* 988 ms at a 512-token tail, 8 s at 4k, **110 s at 32k**, against a
  ~215 ms budget (5% of the ~4.3 s prefill for a p50 13k-token request). Even
  int8 only buys 1.8×.

**The sub-100M field: all within noise of each other, and of doing nothing.**
Every candidate's CI overlaps the free prompt-length control on both the raw
(0.729) and the within-run (0.581) metric. If we shipped one it would be
**AIST-87M** — it is within noise of the best (AUC 0.713 [0.672, 0.753],
within-run 0.593 [0.543, 0.643]), it is the only sub-100M candidate with a real
shared text+image space (§2), and at `CORTEXT_AIST_THREADS=8` it is 398 ms, the
only one of the ≥33M models whose cost is even the right order of magnitude. But
"the one we would ship" is not the same as "worth shipping", and on this evidence
it is not.

**What actually predicts think length here is free.** Prompt length alone gets
AUC 0.729 and Spearman 0.494 — better than any encoder in the study on the raw
metric. The random-split ceiling for any prompt-side predictor is ~0.75. There is
almost no headroom, and none of it is in the embedding: a 1280-d AIST vector fed
to a logistic head scores **0.441**, worse than chance, while four scalars derived
from those same vectors score 0.593.

**Fine-tune?** For the 350M router, no — a fine-tune would discard the routing head
that is the only reason to pick that checkpoint, at which point it is just a slow
355M encoder. For the 230M, the spec is written up in
`results/FINETUNE_SPEC.md` (cookbook recipe: frozen-MLM head dropped, mean pool +
linear classifier, lr 2e-5, 3 epochs, max_length 512; ~1–2 min on one L4 for the
689 labels, ~35 min on this CPU) — but **it was not run**, and the evidence says
not to bother yet:

- the label set is 689 think-length examples over **23 tasks**, and the split
  that matters is by task, so the effective n is 23;
- the frozen-encoder version of exactly that head, on every encoder tried, lands
  at or below the free control;
- the fix that would change this is not a bigger model, it is **more labels**:
  rerun the `low` / `medium` / `extra-high` / `dynamic` columns with the effort
  telemetry recorded in the harness's `llm_response` event (the server already
  emits it), which is a ~4.7× larger label set and, crucially, the only way to
  see whether a signal survives a *change of effort rendering* — which is exactly
  what a router does.

**One thing worth keeping regardless of the router decision.** `knn_effort` — mean
reasoning tokens of the 16 nearest past requests, leave-one-task-out — is
label-free at inference, needs no training, costs a dot product against a small
index, and reaches AUC 0.637 / within-run 0.593 on AIST embeddings. Over hashed
n-grams, `novelty_loto` reaches within-run **0.632**, the best number in the study.
Both are cheap enough to be free and are the natural fallbacks if the hidden-state
track wants a comparison point that does not touch the model.

### Integration sketch (if a router is ever wired in)

`results/INTEGRATION_SKETCH.md` has the detail. Summary:

- **Where.** `vllm/entrypoints/openai/chat_completion/serving.py` already calls
  `apply_dynamic_effort(request, self._dynamic_effort_config())` when
  `reasoning_effort == "dynamic"`; that function today appends **one fixed** `low`
  sentence via `append_to_last_user_message`. A router replaces that constant and
  nothing else — no scheduler, ladder or actuator change, no new request field.
- **Prefix-cache safety.** Only tail placement is safe. Measured in
  `dynamic-reasoning.claude.md` §2b: with the sentence in the system message a
  turn that changes effort gets **0 prefix-cache hits** (full re-prefill, 3–14 s
  at 9–37k tokens, 130–150 s at ~200k); with tail placement it gets 8240/10096
  hits, i.e. free. The router must also be **deterministic** for a given prompt,
  or two identical requests render different tails and the last block misses.
- **Budget.** ≤215 ms (5% of a p50 prefill), off the async event loop in a thread
  pool, torch threads pinned to ≤8, fail-open to today's fixed rendering, never
  run when the client set an explicit effort.

## 11. Limitations

1. **23 tasks.** The split is grouped by task, so the effective sample size for
   generalisation is 23, not 3258. The bootstrap CIs are over requests and
   therefore understate the true uncertainty across tasks.
2. **Think-length labels exist for one effort column only** (`dynamic-v2`, 689
   natural closes). We cannot check whether a signal that predicts think length
   under one effort rendering still does under another — which is what a router
   changes.
3. **The target is what the model did, not what it should have done.** A router
   that predicted reasoning tokens perfectly would reproduce today's behaviour,
   not improve the accuracy/token Pareto. The outcome-linked targets available are
   coarse (per-task functional, per-step tool trouble).
4. **The policy simulation is a lookup, not a rerun** (§6).
5. **Text only.** The traces carry no images; AIST's image path is validated for
   shape and latency, not accuracy.
6. **Loaded box.** The vLLM server was serving throughout. The sub-100M sweep ran
   at load ≈ 2/20; the AIST and LFM sweeps ran at load 9–16 as the embedding jobs
   drained. Two `router350` 512-token cells measured 705 ms and 988 ms in
   different passes — read the transformer p50s as ±30%. None of the conclusions
   turn on a factor smaller than 3.
7. **Cut when the scope was reduced:** the window-policy comparison
   (tail vs task-statement+tail), the 2048/8192-token window sweep, the frozen
   230M head, and the 230M fine-tune. Scripts for all four are in place and
   documented in `work/router-proto/README.md`.
8. **Nothing was served.** Closing the loop needs a VulcanBench column with a
   router in the path, which needs the prod window.

## 12. Artefacts

| path | contents |
|---|---|
| `/shared/vllm/work/router-proto/dataset/requests.jsonl` | shared dataset, 3258 requests, written 2026-08-19 12:58 UTC |
| `/shared/vllm/work/router-proto/dataset/README.md` | field-by-field spec, targets, split rule, metric protocol |
| `results/latency_summary.{csv,md}` | every latency cell |
| `results/probe_summary.{csv,md}` | every probe/signal with CIs |
| `results/analysis_lanes.json` | (a)/(b)/(c) for the zero-shot router + controls |
| `results/analysis_phrasing.json` | the four lane phrasings on the 250-prompt subsample |
| `results/policy_simulation_dv2.json` | the simulated Pareto point |
| `results/lane_scores_w512.jsonl` | zero-shot scores for all 3153 unique prompts |
| `results/emb_*.npy` | cached embeddings per encoder |
| `results/{FINETUNE_SPEC,INTEGRATION_SKETCH,LIMITATIONS}.md` | working notes folded into §8–§11 |
| `work/router-proto/README.md` | exact commands to reproduce everything |
