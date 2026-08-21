# Survey: reusable pre-generation reasoning-effort classifiers

**Date:** 2026-08-19 · **Scope:** things we could put in front of the vLLM server
(Qwen3.8-27B, TP4 L4 box, agentic coding loop) to choose a `reasoning_effort`
rung (`low`/`medium`/`high`/`xhigh`) **before** generation, on **CPU**, at low
latency. Companion to [`dynamic-reasoning.claude.md`](dynamic-reasoning.claude.md)
— this document covers only signal **S10** (request-context priors at prompt
time). Everything else in that plan is in-flight (S1–S8) and unaffected.

Every model ID, PR number, config value and licence below was fetched directly
from a model card, `config.json`, chat template, GitHub API or arXiv page on
2026-08-19. Anything sourced from a search snippet or not independently
confirmed is marked **unverified**. Where a project is a marketing claim with no
downloadable artifact, it says so.

---

## 0. The decision we actually need to make

| requirement | value | why it eliminates most of the field |
|---|---|---|
| output | **4 ordered rungs**, not binary | almost every published router is `think` / `no_think` |
| input | prompt only, **pre-generation** | rules out anything needing rollouts or target-model hidden states |
| prompt shape | agentic coding **steps**: 4k–100k tokens of code, diffs, tool output, multi-turn history | rules out every 512-token BERT unless we pre-reduce the prompt |
| modality | **text + images** — screenshots and computer-use turns, not text only | rules out every text-only classifier as a complete solution (§8) |
| host | CPU, alongside the GPU server; budget ≪ TTFT | the one comparable system measures **4,918 ms** on CPU (§4) |
| cost of being wrong | asymmetric: under-shooting a hard step wastes an agent turn; over-shooting costs tokens | argues for a calibrated *score*, not an argmax label |
| model-agnostic | must survive Qwen3.8-27B → next model | rules out anything keyed to one target model's embeddings |

Two project constraints override everything in this survey:

1. **Standing rule: the effort controller uses general, model-agnostic
   algorithmic signals only, and those must be exhausted before learned probes.**
   Essentially every artifact surveyed here *is* a learned probe. The honest
   framing is that any of them would be a **turn-1 prior feeding the existing
   controller**, never a replacement for it — and that the training-free option
   in shortlist #1 should be tried first on principle, not just on merit.
2. **The in-flight controller already escalates** from entropy / margin / MTP
   acceptance. A prompt-time classifier is only worth shipping if it beats
   *"start at rung 0 and let the escalator work"*. Almost nothing in the
   literature is evaluated against a baseline that strong.

### Qwen3.8-27B constrains the output space to three strings

Verified from `https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja`:

```jinja
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('Unexpected reasoning effort ... Supported types are xhigh (default), medium, and low.') }}
```

`xhigh` and `low` inject an instruction sentence; **`medium` injects nothing**.
Any other value **hard-errors** — there is no `auto`. So a classifier can emit at
most three distinct *prompt* states; a genuine 4-rung ladder has to come from
`thinking_token_budget`, which is a **sampling** parameter.

That distinction matters more than it looks. Effort strings are rendered into the
prompt, so **varying effort per request costs prefix-cache hits** — Anthropic
states this outright for their own API (*"Treat any thinking or effort change as
starting the cache over"*), and Qwen3.8 has the same property.
`thinking_token_budget` does not, because it never touches the tokens. **On a box
that is already HBM- and NCCL-bound, the budget should be the primary actuator
and the prompt string the secondary one** — which is what patch 0009 already
does by putting the effort sentence in the last user turn (plan §2b).

---

## 1. Ranked shortlist — what to try next

*Six entries, not five — TwinRouterBench earned a slot. Text-side ranking; the multimodal side is ranked separately in **§8a**, where the
recommendation is **AIST-87M** (87 M for text + image + audio in one shared 1280-d
space, matryoshka down to 32 dims, Apache-2.0 engine, **already checked out at
`/shared/cortext.cpp`**) — with §8c's caveat that a 224–384 px vision tower may
well contribute nothing on screenshots, which the probe in step 4 below settles
cheaply.*


### 1. DART — **training-free, model-agnostic, working MIT code; try this before any classifier**
["DART", arXiv 2606.23181](https://arxiv.org/abs/2606.23181) (2026-06-22) ·
[github.com/js-lee-AI/DART](https://github.com/js-lee-AI/DART) — **MIT, 17 ★,
pushed 2026-07-05, real working code** (not a stub).

Sample K=2 **no-think** drafts; if they agree under a pluggable equivalence
function, accept directly; on disagreement, escalate through staged thinking
budgets. No training, no labels, no separate model, nothing keyed to Qwen. Its
reference implementation (`dart/sc_budget_router.py`) drives **exactly the
primitives we already have**:

```python
draft = self.model.generate(prompt=question, enable_thinking=False, ...)
think_resp = self.model.generate(prompt=question, enable_thinking=True,
                                 thinking_budget=budget, ...)  # budget_stages=[1024, 2048]
```

Reported: matches or beats always-think on **13 of 14** model-benchmark pairs at
15–69 % fewer thinking tokens. **Qwen3-32B HumanEval 72.6 → 95.1 (+22.5) at −63 %
thinking tokens** — the only coding result in this survey where a routing method
*improves* accuracy while cutting tokens. Only regression is Qwen3-8B
OlympiadBench (−1.7).

**This is the one option that satisfies the standing rule** — it is an
algorithmic signal (self-consistency of cheap drafts), not a learned probe, and
it transfers to any model with a thinking switch.

**Honest caveat, now quantified:** it is a **multi-call cascade costing ~2.3×
wall-clock**. It trades latency and prefill for token savings, and our latency profile is 66/65/105/160 tok/s
with a hard CF 100 s edge. Two extra no-think drafts per step on a 27B TP4 model
is not free, and the paper's savings are measured in *tokens*, not in
*end-to-end latency on a saturated box*. Evaluate on the throughput profile
(1170/912/615 @ c128/64/32) where spare capacity exists, before the latency one.

Read alongside [HRBench, arXiv 2605.28398](https://arxiv.org/abs/2605.28398)
([github](https://github.com/usail-hkust/HRBench), 9 ★), the survey that
benchmarks external routing vs prompt-based vs speculative across training
regimes: external routing lands at a **consistent 18–21 % token saving with
preserved accuracy**. That is the realistic size of the prize — useful, not
transformative. Whether shipped checkpoints exist is **unverified**.

### 2. TwinRouterBench — **the only 4-tier, execution-verified, agentic-coding-step label set that exists**
["TwinRouterBench", arXiv 2605.18859](https://arxiv.org/html/2605.18859v1) ·
[github.com/CommonstackAI/TwinRouterBench](https://github.com/CommonstackAI/TwinRouterBench)
— **Apache-2.0, real code, pushed 2026-08-18.**

Step-level routing with **four tiers — `low` / `mid` / `mid_high` / `high` — the
same rung count we need**, on **970 execution-verified rows**, of which
**336 are SWE-bench Verified steps** (high 168 / low 94 / mid_high 41 / mid 33),
plus BFCL 248 (almost all `low`), mtRAG 193, QMSum 145, PinchBench 48, and 100
held-out SWE-bench Verified instances for dynamic evaluation.

**This is the single most valuable artifact in the survey for us.** Every other
labelled set is either binary, non-agentic, math-only, or derived from trace
length. This one is per-step, four-level, execution-verified, on our task family,
and permissively licensed.

**Its findings are also the strongest argument for building this carefully:**
- **Claude Opus 4.6 prompted as a router predicted `high` on only 7 of 147
  verified-high SWE steps, and failed all 40 trajectories.** A frontier LLM asked
  to judge effort is *not* a usable oracle — which also means LLM-generated
  labels are not a shortcut.
- Rule-based routers scored **4.2–62.5 %** on high steps.
- **One under-routed critical step fails the entire trajectory.** This is the
  asymmetry from §0 measured on real data, and it says: bias the `high`/`xhigh`
  rungs conservatively, and never let a classifier *lower* a rung on its own.
- A trained logistic router achieved **−53.1 % cost at matched resolution** —
  the one genuinely encouraging number for a learned router on agentic coding.

**Caveats:** 970 rows is small, and outside SWE-bench the distribution is
overwhelmingly `low`. Use it as the **evaluation set and the label schema**, and
generate volume ourselves with the ARES recipe (§7a).

### 3. `ilya-kolchinsky/PromptComplexityEstimator` — **the best-targeted classifier; run it as an offline probe today**
[HF](https://huggingface.co/ilya-kolchinsky/PromptComplexityEstimator) ·
[code](https://github.com/ilya-kolchinsky/ComplexityEstimator) ·
**Apache-2.0**, DeBERTa-v3-base, **184,128,769 params**, 512-token cap, prompt-only.
Output is a **single continuous scalar in [0,1]** (mean-pool → LayerNorm →
Linear→ReLU→Linear→Sigmoid), which maps onto a ladder without inventing buckets.
Held-out **MAE 0.0855, Spearman ρ 0.735**.

**Why it beats the NVIDIA classifier on fit:** its training target is
*cross-model item difficulty* — `BatsResearch/Cross-Difficulty`,
`furonghuang-lab/Easy2Hard-Bench`, `hendrycks_math`, `ai2_arc`, `race`, `anli` —
i.e. how hard the item is for *many* models, not which of one frozen model pair
wins. That is the closest published thing to model-agnostic difficulty, and its
card names *"adaptive compute allocation"* as an intended use.

**Skepticism:** **79 downloads/30 d, 1 like** — essentially unproven in the
field, and its card explicitly disclaims *"guaranteed difficulty estimation for
a specific target model."* At 184 M it is over the ≤100 M deployment budget.
So use it as an **offline probe and auto-labeller**, not as the deployed router:
score ~2k logged agentic steps, correlate against the think-token count the
server actually recorded, read the Spearman ρ. **If ρ is under ~0.3 on our
traffic, prompt-only effort prediction is dead and we stop.** Zero training cost.

Pair it with [`nvidia/prompt-task-and-complexity-classifier`](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier)
as a second opinion (240× the adoption) — read its dedicated **binary `reasoning`
head** directly rather than its `prompt_complexity_score`, whose formula puts a
0.35 weight on *creativity*, the wrong quantity for a reasoning ladder.

### 4. Static embeddings (`minishlab/potion-base-32M` / `potion-base-8M`) + a linear head — **the only candidate whose latency is safe at 8k+**
MIT, 32.3 M / 7.6 M params (verified). A static embedding is a token-embedding
lookup plus pooling: **no context-length limit, no T² attention term**, O(tokens)
with a tiny constant, so a 100k-token agentic step costs milliseconds. SwiftEmbed
([arXiv 2510.24793](https://arxiv.org/html/2510.24793)) serves `potion-base-8M`
at **1.12 ms p50**, 50k req/s. `potion-base-32M` reaches ~94.7 % of
`all-MiniLM-L6-v2` MTEB quality, and [Model2Vec](https://github.com/MinishLab/model2vec)
(MIT, 2.2k ★, pushed 2026-08-18) has shipped classifier training since 2025-12-02.

Given that the one comparable production system measures a **4,918 ms CPU
baseline** for a single routing decision (§4), this is not a compromise choice —
it is the only entry whose cost model survives an 8k-token agentic prompt on CPU.
It is simultaneously the **baseline every learned option must beat**: if a
ridge/logistic head on potion embeddings lands within a couple of points of a
fine-tuned encoder, ship the linear probe and stop.

### 5. Two sub-100 M off-the-shelf artifacts nobody has validated — **an hour each to check**
Both are on-target and under budget, and both are essentially unproven. Checking
them costs an afternoon and could skip the training exercise entirely.

- [`tripathyShaswata/ThinkingBudgetRouter`](https://huggingface.co/tripathyShaswata/ThinkingBudgetRouter)
  — **DistilBERT, 67 M, Apache-2.0.** Labels are literally `no_thinking` (0) /
  `brief_thinking` (512) / `deep_thinking` (8192), and the card documents a Qwen3
  `thinking_budget` integration. **Exactly our problem statement at exactly our
  parameter budget.** But: **zero eval numbers, no training data disclosed,
  4 downloads.** Treat every claim as unverified.
- [`agentlans/bge-small-en-v1.5-prompt-difficulty`](https://huggingface.co/agentlans/bge-small-en-v1.5-prompt-difficulty)
  — **33.4 M, MIT, continuous regression.** Smallest transformer option;
  single-digit ms on CPU at 512 tokens. Card is auto-generated boilerplate whose
  only metric is `MSE 1.396` on an undocumented scale.

Also worth 20 minutes as a free first-stage gate:
[`veritiana-ai/prompt-task-complexity-classifier`](https://huggingface.co/veritiana-ai/prompt-task-complexity-classifier)
— a **75 KB ONNX logistic regression over 1,544 hashed features**,
sub-millisecond, Apache-2.0. Its own card flags its 86.8 % as *"internal
weak-label evaluation, not independently established"*, and it trained in 13
seconds. Useless as the decision; useful as a brutally honest bag-of-features
baseline.

If a *decision* rather than a scalar is wanted and 142 M is acceptable,
[`appriai/gen-router-t1`](https://huggingface.co/appriai/gen-router-t1)
(DeBERTa-v3-small, Apache-2.0, `fast`/`balanced`/`frontier`) is the **only
artifact in this survey that publishes real CPU latency alongside quantified
error asymmetry**: ONNX CPU **p50 77.8 ms / p95 360 ms / p99 488 ms**, accuracy
89.0 % on n=13,947, ECE 0.053, critical under-provision 3.6 % vs over-provision
0.17 %. Watch the p99 — on a serving path the tail is what bites.

### 6. Design prior art to steal, not services to adopt: vLLM Semantic Router + R2-Router
[github.com/vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
— **Apache-2.0, 5.2k ★, pushed 2026-08-19**, releases v0.1 "Iris" → v0.2
"Athena" → v0.3 "Themis", with Helm charts, GHCR images, PyPI `vllm-sr` and
crates.io `candle-semantic-router`. Better packaged than anything else here. Its
paper ([arXiv 2510.08731](https://arxiv.org/abs/2510.08731)) benchmarks on
**NVIDIA L4 with Qwen3-30B-A3B at TP=4 — essentially our hardware** — for
+10.24 pp MMLU-Pro (58.57 % vs 48.33 %), −47.1 % latency, −48.5 % tokens.

**It is the only shipping system anywhere that actually chooses effort**, and its
config schema is the most transferable artifact in this survey. From
`config/config.yaml`:

```yaml
providers:
  defaults:
    default_reasoning_effort: medium
    reasoning_families:
      qwen3:   { type: chat_template_kwargs,       parameter: enable_thinking }
      gpt:     { type: reasoning_effort,           parameter: reasoning.effort }
      mistral: { type: top_level_reasoning_effort, parameter: reasoning_effort }
```

plus per-decision `use_reasoning` / `reasoning_effort: low|medium|high`, a
`complexity` signal family (`needs_reasoning`, threshold 0.75, scored by
embedding similarity against hard/easy anchor phrases), a `weighted_sum`
`request_difficulty` score, `threshold_bands` with `sigmoid_distance` calibration
producing named bands (`support_fast`/`support_balanced`/`support_escalated`),
and an `x-vsr-selected-reasoning` attribution header. Copy that shape.

**Do not adopt the service.** Four verified reasons:
- **CPU latency.** The follow-up paper
  [arXiv 2603.12646, *"98× Faster LLM Routing Without a Dedicated GPU"*](https://arxiv.org/abs/2603.12646)
  states a **4,918 ms baseline**, optimised to 127 → 62 → 50 ms, and 108 ms at
  16k tokens. Read the title precisely: *"without a dedicated GPU"* means the
  router **co-locates on the serving GPU** at <800 MB. **The fast path is not CPU.**
  The repo's own `bench/cpu-vs-gpu/README.md` declines to publish headline
  numbers, and the committed `perf/testdata/baselines/classification.json` has an
  **empty** benchmarks object.
- **Size.** Its 307 M mmBERT-32K classifier is 3× our budget.
- **It cannot bound thinking.** A code search for `thinking_token_budget` in that
  repo returns **no hits** — it only manipulates `enable_thinking` /
  `reasoning_effort`, so it cannot control a model whose template ignores effort.
- **Its multi-level effort is *config*, not a prediction.** The learned part is a
  **14-domain intent classifier**; `reasoning_effort` is then a **static
  per-category value written in `config.yaml`**. It classifies *topic* and looks
  up an effort, which is not the same thing as predicting effort. Replacing that
  head with a learned 4-way effort head is exactly the work we would be doing.

If we want only the classifier, it exposes a standalone REST API on :8080
(`POST /api/v1/classify/intent` → `{category, confidence, processing_time_ms,
probabilities, recommended_model, routing_decision}`) with no Envoy required.

Also read [`JiaqiXue/R2-Router-RouterArena`](https://huggingface.co/JiaqiXue/R2-Router-RouterArena)
(Apache-2.0, sklearn KNN over Qwen3-0.6B embeddings, ~3.3 MB, **0 downloads**,
#1 on RouterArena at 71.23 %): the only published artifact that predicts quality
**per (model, token-budget) pair** with a cost/accuracy λ knob, and its reference
deployment already assumes a vLLM pooling server. Conceptually the closest thing
to what we are building.

### Honourable mention: the ARES label recipe (its code does not exist)
["Ares: Adaptive Reasoning Effort Selection for Efficient LLM Agents",
arXiv 2603.07915](https://arxiv.org/abs/2603.07915) (UCSB, 2026-03-09). Per-step
effort selection for multi-step agents; a Qwen3-1.7B router predicts *the lowest
reasoning level that still completes the step*; −52.7 % reasoning tokens on
TAU-Bench, BrowseComp-Plus, WebArena.

**Verified 2026-08-19: [github.com/UCSB-NLP-Chang/Ares](https://github.com/UCSB-NLP-Chang/Ares)
contains one file — a 104-byte `README.md`.** 1 ★, no licence, created and last
pushed 2026-03-04. No code, no weights. What is reusable is the **label
definition**: *don't label difficulty, label the minimum effort rung that still
succeeded.* That turns our own agent loop into a self-labelling data source and
is strictly better than regressing on think-token count, which is contaminated by
sampling noise (§2).

### Recommendation vs. the LFM2.5 prototype
The LFM2.5 prompt-router is, on adoption, the **most-used genuine prompt router
in this survey (4,391 downloads/30 d, 56 likes)** — far more traction than
anything else on-target. Keep it as the zero-training control. But **do not
invest further in the 230M fine-tune until the ρ probe in #2 comes back
positive**, and weigh three facts:

- **Licence.** LFM Open License v1.0 permits free commercial use only below
  **$10 M annual revenue**; above that the commercial right terminates
  ([liquid.ai/lfm-license](https://www.liquid.ai/lfm-license),
  [docs.liquid.ai/lfm/help/model-license](https://docs.liquid.ai/lfm/help/model-license)).
  PromptComplexityEstimator, gen-router-t1, ThinkingBudgetRouter (Apache-2.0),
  potion, bge-small-prompt-difficulty, mmBERT, DeBERTa-v3 (MIT) have no such
  clause. For something sitting in the serving path this matters.
- **Size and evidence.** 355,008,768 params (verified) — 5–11× the sub-100 M
  candidates. Liquid's paper is titled *"LFM2.5-Encoders: Fast at Long Context,
  Even on CPU"* and they host a CPU-only demo Space, but **no per-prompt ms
  figure is published anywhere**, and the prompt-router card publishes no
  training data and no benchmark results.
- **The zero-shot framing is wrong for this task.** It scores a prompt against
  free-text lane *semantics*; it has no built-in notion of difficulty beyond what
  the lane text says. Effort is not a topic — "fix this typo" and "fix this race
  condition" are both the Coding lane. You *can* write lanes like "requires long
  multi-step reasoning" vs "simple factual lookup", and that is the fair test to
  run — but expect the fine-tune, not the zero-shot lane, to be the version that
  works.

**Concrete next experiment.** Note the ordering: step 1 can kill the project, so
it comes before any modelling.

1. **Measure the noise floor on our own traffic first.** Replay ~500 logged
   agentic steps at K=8 samples per rung and compute within-prompt σ against
   between-prompt σ. Per §2b, a predictor cannot beat the sampling noise, and the
   one paper that measured both found its best predictor **already at the floor**
   (MAE 30.35 vs noise radius 32.94). If within-prompt σ dominates here, stop and
   retarget to a distributional/quantile head rather than a rung label.
2. Dump ~2k logged steps with prompts, realised think-token counts **and
   outcomes** from the `reasoning_effort: "dynamic"` telemetry sink (patch 0009).
   **Do not train on think-token count** — §2c: length is anti-correlated with
   correctness at −0.72, so that target trains a "where will it flail" predictor
   and then rewards flailing with more budget.
3. Score each step with seven text predictors — (a) prompt token length,
   (b) **potion-32M embedding + kNN**, (c) potion-32M + ridge,
   (d) `PromptComplexityEstimator` scalar, (e) NVIDIA `reasoning` head,
   (f) `ThinkingBudgetRouter` 3-way, (g) LFM2.5-350M zero-shot lanes written as
   effort descriptions — and report Spearman ρ and 4-bucket accuracy against
   **both** a predict-the-median constant baseline **and** (b). Per Routing
   Plateau, kNN is the bar: across 21 routers and 6 encoders no learned router
   beat it by more than 0.22 pp. Never evaluate against random.
   Also fit an **XGBoost over ~16 lexical/structural features** — that is what
   beat prompt-length by 4.6 pts in arXiv 2604.14853, and it costs seconds.
4. **Fit each head twice on the multimodal subset**: text features alone, and
   `[text ; AIST image embedding]`. If the image features add nothing (§8c argues
   they will not, because 224–384 px destroys all legible screenshot text), drop
   the vision tower and keep a text-only classifier that ignores attachments.
   One extra column in an experiment already being run.
5. **Time every candidate on a real 8k-token step on this box**, including
   AIST-87M. Per §4, latency will probably eliminate more candidates than accuracy
   does — the one comparable production system measures **4,918 ms on CPU**.
6. **In parallel and independently of all of the above**, run DART on the
   throughput profile. It needs no classifier, no labels and no training, and if
   its Qwen3-32B HumanEval result (72.6 → 95.1 at −63 % thinking) reproduces on
   Qwen3.8-27B it makes the classifier question moot for coding.
7. **Report every candidate on TwinRouterBench's held-out SWE-bench-Verified
   step rows** (§1.2) — the only external, four-tier, execution-verified agentic
   yardstick that exists. Its published reference points: rule routers 4.2–62.5 %
   on high steps, a trained logistic router −53.1 % cost at matched resolution,
   and Claude Opus 4.6 as a prompted router at 7/147.
8. Only if a classifier clears **both** baselines by a real margin, fine-tune the
   winning backbone on **ARES-style minimum-successful-rung labels** (§7a),
   pretraining on `nvidia/Nemotron-SFT-OpenCode-v1` `complexity_level` and
   domain-adapting on `nvidia/Open-SWE-Traces` (§7g).
9. Ship whatever wins as an `effort_bias` producer, not as a rung selector — the
   seam already exists (§6) and it fails safe. **Make the bias asymmetric**: let
   it raise effort freely and lower it only with high confidence. TwinRouterBench
   measured that one under-routed step kills a trajectory, while over-routing only
   costs tokens.

**Calibrate expectations before starting.** Even a *perfect* prompt-only
difficulty oracle — the gold human SWE-bench Verified label — reaches only
**Kendall τ = 0.32** against real agentic token spend (§2d), and the best measured
external router in a head-to-head study came **last**, at −13 % tokens for −3.5 pp
accuracy (HRBench). The realistic prize is **13–24 % of thinking tokens**, not 50 %.

---

## 2. Is this even predictable? (read this before building)

The literature that actually measures this lives in the LLM **serving-scheduler**
world, not the reasoning world, and it is sobering. The single most important
number is the last one in the first table.

### 2a. How well prompt-only predictors actually do

| system | predictor | input | result |
|---|---|---|---|
| **ELIS** ([arXiv 2505.09142](https://arxiv.org/html/2505.09142v1)) | **BGE-base-en-v1.5 (frozen) + 8 linear layers on CLS** — our exact size class | prompt | **LMSYS: R² = 0.48, MAE 71.5 tok.** (Its MAE 19.9 / R² 0.852 figure is on a narrow 13-model benchmark set and does not generalise.) |
| **Predictive Scheduling** ([arXiv 2602.01237](https://arxiv.org/html/2602.01237)) | LoRA head on R1-Distill-Qwen-1.5B | **prompt only** | **Pearson r = 0.444** |
| same paper | 2-layer MLP on **layer-16 hidden states** | hidden states | **r = 0.742** — reading the served model's own state is worth **≈ +0.3 Pearson** |
| **Learning-to-Rank** ([arXiv 2408.15792](https://ar5iv.labs.arxiv.org/html/2408.15792), NeurIPS 2024) | OPT-125M/350M + ListMLE | prompt | **Kendall τ 0.54 (ShareGPT) / 0.62 (LMSYS)**, collapsing to **0.45 / 0.40 cross-dataset**. The paper's own words: *"predicting the exact generation length of each request is infeasible."* Overhead <2 % of serving time. |
| **ThinkSwitcher** ([arXiv 2505.14183](https://arxiv.org/html/2505.14183)) | **ModernBERT-base** think/no-think classifier | prompt | **60.7 % acc at 6,021 tokens** — *worse on both axes* than its own regression head on the LLM's representation (62.8 % / 5,405) |
| **TIE** ([arXiv 2604.00499](https://arxiv.org/html/2604.00499v2)) | DeBERTa-v3-base multi-pool → (μ,σ) of a log-t | prompt | **R² 0.82 (μ), 0.76 (σ)** — the best published, on curated benchmark prompts |
| **PromptComplexityEstimator** | DeBERTa-v3-base → scalar | prompt | Spearman **ρ 0.735** on curated benchmark items |
| **Codeforces difficulty from statement text** ([arXiv 2310.05791](https://arxiv.org/html/2310.05791v2), 7,976 problems) | BERT | problem text | **54.2 % acc / 53.1 % F1**; adding *code* lifts it to 70.5 % — the signal is in the solution, not the statement |
| **LLM Router w/ prefill activations** ([arXiv 2603.20895](https://arxiv.org/abs/2603.20895)) | MLP on prefill-activation PCA vs best text-only (DeBERTa-LoRA / kNN / MF / GraphRouter) | both | **AUC 0.856 (activations) vs 0.804 (best prompt-only)** — the cost of staying prompt-only, quantified: **≈ −0.05 AUC** |
| **PredictaBoard** ([arXiv 2502.14445](https://arxiv.org/abs/2502.14445), ACL'25; [code](https://github.com/Kinds-of-Intelligence-CFI/PredictaBoard), GPL-3.0, **no trained assessors released**) | embeddings + LogReg/XGBoost over **41 LLMs** | prompt | **AUROC ≈ 0.70 on MMLU-Pro, degrading OOD on BBH** |
| **Adaptive TTC via Constrained Policy Optimization** ([arXiv 2604.14853](https://arxiv.org/abs/2604.14853)) | **XGBoost (100 trees, depth 5) on 16 lexical/structural features** — CPU, seconds to train | prompt | Oracle **.586** / GBM **.575** / **Random .534** / **prompt-length heuristic .529** / Fixed .517. A tree over hand-counted surface features beats prompt-length by **4.6 pts** and random by **4.1** — that is the realistic size of the prompt-only signal. |
| **Routing Plateau** ([arXiv 2606.07587](https://arxiv.org/abs/2606.07587)) | **21 routers × 6 encoders** (MiniLM, MPNet, BGE, ModernBERT-base/large, Qwen2.5-0.5B) | prompt | **kNN is top-2 on all five benchmarks; the top-5 routers differ by 0.22 pp.** The 10–30 pp oracle gap is concentrated in the 11–35 % hardest queries. Conclusion: *"static query embeddings insufficient"* — and **no learned router beats kNN over embeddings.** |

**Read across:** prompt-only encoders land at **r ≈ 0.44–0.74 / R² ≈ 0.2–0.5 /
~55–61 % multi-class accuracy**. Ranking is materially easier than regression but
**generalises badly across distributions** (τ 0.62 → 0.40).

### 2b. The variance floor — how much is even predictable

| source | measurement |
|---|---|
| **ProD / Robust Length Prediction** ([arXiv 2604.07931](https://arxiv.org/html/2604.07931v1)) | **16 generations per prompt**, T=0.8. Median-centred noise radius: Qwen-2.5-7B Math 27.8 / **Coding 21.7** / LongSeq 42.9 / Chat 35.3 tok; Llama-3-8B 16.1 / **23.0** / 38.0 / 33.4. **Normalised noise = 11.5–18.2 % of the per-prompt median**; max/median up to **4.03×**. |
| **the killer comparison, same paper** | Best predictor MAE on Qwen/Math = **30.35** vs the irreducible noise radius **32.94**. **The predictor is already at the noise floor.** Also: training *without* repeated sampling degrades MAE 60.74 → 80.93 — **your training set needs N > 1 samples per prompt.** |
| **Beyond Prediction** (ICML 2026, [arXiv 2606.18431](https://yl3469.github.io/uniboost-icml26/)) | Same prompt, same model, 20 runs: length swings **> 2×**, **CV up to 0.47**. On reasoning workloads *"a single long-thinking request can be 10–100× longer than the median."* Workloads include **BigCodeBench**. |
| **CASTILLO** ([`danfperam/castillo`](https://huggingface.co/datasets/danfperam/castillo), [arXiv 2505.16881](https://arxiv.org/abs/2505.16881)) | **280,000 rows**, 13 open LLMs × 7 corpora (incl. **MBPP**), **10 samples per ⟨prompt, model⟩**, CC-BY-4.0, columns `output_sizes` / `output_mean` / `output_std` / `output_percentiles`. **CV of response length up to 45 %.** The ready-made dataset for quantifying our noise floor. |
| **TIE** ([arXiv 2604.00499](https://arxiv.org/html/2604.00499v2)) | CoV > 1.0 in **78.6 %** of cases; skewness avg **3.10**; top 10 % of lengths = **35.7 %** of all tokens |
| **Token Budget Saturation** ([arXiv 2607.21433](https://arxiv.org/html/2607.21433v1)) | GSM8K/MATH-500 saturate at **256 thinking tokens**. AIME is **bimodal**: 56.5 % converge (96.5 % acc, ~4,100 tok), 43.5 % never terminate within 10,000 (11.5 % acc). **Problem difficulty explains r² ≈ 0.186 of convergence — under 19 %.** Activations at 50 generated tokens give only **AUC 0.615**; the authors state they cannot predict convergence from the prompt alone. |
| reasoning vs non-reasoning | output-length variance up to **2.91×** higher; joint-logprob variance up to **10.17×** higher | [arXiv 2510.05095](https://arxiv.org/pdf/2510.05095) — **snippet-sourced, verify before quoting externally** |

### 2c. Trace length is a *contaminated* label

On MATH, R1-Distill averages **2,907 tokens when correct vs 6,521 when wrong**,
Pearson **−0.72** between length and correctness
([arXiv 2505.00127](https://arxiv.org/html/2505.00127v1)). **Regress on raw trace
length and you train a "where will the model flail" predictor — then hand exactly
those cases more budget.** This alone disqualifies the obvious first experiment.

### 2d. Agentic coding specifically — the most damaging evidence

["How Do Coding Agents Spend Your Money?", arXiv 2604.22750](https://longjubai.github.io/agent_token_consumption/)
(Microsoft Research / Stanford Digital Economy Lab; data
[`loong0814/openhands_trajectories`](https://huggingface.co/datasets/loong0814/openhands_trajectories),
code [LongjuBai/agent_token_consumption_analysis](https://github.com/LongjuBai/agent_token_consumption_analysis)).
Eight frontier LLMs on SWE-bench, 500 tasks, 230 shared successes / 100 shared
failures analysed:

- Agentic coding burns **~3,500×** the tokens of single-round reasoning.
- Per task: **reasoning ~19.5k tokens**, tool outputs ~9.0k, input messages ~4.5k
  — reasoning dominates, so it *is* the right thing to control.
- **Run-to-run variance on the same task: ~2× typical, up to 30× worst case.**
- **Higher token spend does not buy accuracy** — accuracy peaks at intermediate
  cost and degrades above it.
- Models **cannot predict their own cost**: Pearson r ∈ [0.05, 0.39], systematic
  underestimation.
- **The number that caps this whole project: Kendall τ = 0.32 between
  human-rated difficulty and actual token consumption.** 6.7 % of `<15 min`
  tasks exceed the mean consumption of hard tasks; 11.1 % of hard tasks come in
  below the mean of easy ones.

**Even a perfect prompt-only difficulty oracle — the gold human SWE-bench
Verified label — only reaches τ ≈ 0.32 against real token spend on agentic
coding.** A sub-100 M encoder predicting that label lands below it.

**A second, independent agentic result points the same way.**
["Predicting Task Difficulty Without Rollouts", arXiv 2608.05797](https://arxiv.org/abs/2608.05797)
(Aug 2026, **415k agent outcomes**): ridge over task-description features gets
Spearman **ρ = 0.399 in-distribution, collapsing to 0.225 leave-one-benchmark-out**.
A context-length baseline gets 0.086 — but a **benchmark-identity baseline gets
ρ = 0.519 and beats the learned model outright.** Knowing *which benchmark a task
came from* predicts difficulty better than reading the task. For us that is a
warning that a classifier may end up learning "which repo / which tool is this"
rather than "how hard is this step".

Related, and directly actionable:
["Prompt-Induced Waste", arXiv 2608.01347](https://arxiv.org/abs/2608.01347)
(Aug 2026, **4,644 coding-agent runs**) finds prompt *wording* shifts reasoning
length by **2.4–7.4×** with no gain in success. Wording is therefore a real
predictive feature — and a real confound.

And the routing-quality ceiling from **HRBench**
([arXiv 2605.28398](https://arxiv.org/html/2605.28398v1)) on Qwen3.5-9B:
prompt-tuning 47.6 % acc / −24 % tokens; **external routing 44.1 % / −13 %**;
speculative 45.8 % / +21 %. **External routers came last.** Its Appendix C
documents the failure mode to design around: on GPQA the router correctly flagged
**93 %** of items as difficult and the high-effort responses **still failed** —
correct routing does not rescue capability limits.

### 2e. Design consequences

- **Do not regress on trace length.** It is 12–47 % sampling noise, anti-correlated
  with correctness (−0.72), and correlates only τ = 0.32 with real agentic spend
  even from a gold human label.
- **Label the minimum rung that still succeeded** (ARES/CogRouter recipe, §7).
- **Predict a distribution or a rank, not a value.** TIE's (μ,σ) shape is right;
  ranking gets τ 0.54–0.62 in-distribution but budget for the collapse to ≈ 0.40.
- **Never let the classifier pick a high start rung alone.** Use it to lower the
  escalation threshold θ (the `bias` term already in plan §4), so it degrades to
  today's behaviour when wrong.
- **Evaluate against constant-median *and* embedding-kNN**, never random. Across
  21 routers and 6 encoders, kNN was top-2 everywhere and the top five were within
  0.22 pp (Routing Plateau). If a fine-tune cannot beat kNN over potion
  embeddings, there is nothing to ship.
- **Bias the high rungs conservatively.** TwinRouterBench measured that a single
  under-routed step fails a whole trajectory, while over-routing merely costs
  tokens.
- **Sample N > 1 per prompt in training data**, or MAE degrades ~33 % (ProD).
- **Prefer actuating on `thinking_token_budget` over the effort string**, to keep
  the prefix cache (§0).
- The honest prize is **~13–24 % tokens**, not 50 %.

---

## 3. Full table

Legend — **Weights**: ✅ downloadable / ⚠️ partial / ❌ none. **Levels**: B = binary,
G = graded/continuous, N = n-way. **Pre-gen**: ✅ = prompt only, ❌ = needs
target-model generation or hidden states.

### 3a. Off-the-shelf classifiers with public weights

| artifact | predicts | size / arch | ctx | pre-gen | levels | licence | 30 d DL | notes |
|---|---|---|---|---|---|---|---|---|
| [`ilya-kolchinsky/PromptComplexityEstimator`](https://huggingface.co/ilya-kolchinsky/PromptComplexityEstimator) | scalar difficulty [0,1] | **184.1 M** DeBERTa-v3-base | 512 | ✅ | **G** | **Apache-2.0** | 79 | MAE 0.0855, ρ 0.735. Trained on *cross-model item difficulty* — best-targeted objective in the survey. Unproven in the field. |
| [`nvidia/prompt-task-and-complexity-classifier`](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) | 12-way task + 6 complexity dims + composite | **183.9 M** DeBERTa-v3-base, 8 heads | 512 | ✅ | **G** + B (`reasoning`) | NVIDIA OML (**commercial OK**) | 18,184 | Only 4,024 training prompts, **185 of them code**. CPU via [`preflight/…-ONNX`](https://huggingface.co/preflight/prompt-task-and-complexity-classifier-ONNX) fp16 352 MiB (validated, max Δ 4e-4, zero label flips) or [`botirk/…`](https://huggingface.co/botirk/tiny-prompt-task-complexity-classifier) INT8 187 MB (⚠️ apache-2.0 tag contradicts upstream OML). |
| [`appriai/gen-router-t1`](https://huggingface.co/appriai/gen-router-t1) | fast / balanced / frontier | **141.9 M** DeBERTa-v3-small | 320 | ✅ | **N=3** | **Apache-2.0** | 19 | **Only artifact with published CPU latency + error asymmetry**: ONNX CPU p50 77.8 / p95 360 / p99 488 ms; acc 89.0 % (n=13,947), ECE 0.053; under-provision 3.6 % vs over-provision 0.17 %. |
| [`tripathyShaswata/ThinkingBudgetRouter`](https://huggingface.co/tripathyShaswata/ThinkingBudgetRouter) | `no_thinking`/`brief`(512)/`deep`(8192) | **67 M** DistilBERT | 512 | ✅ | **N=3** | **Apache-2.0** | **4** | Exactly our label set at exactly our budget. **Zero eval numbers, no training data disclosed.** Unverified. |
| [`saytes/SoT_DistilBERT`](https://huggingface.co/saytes/SoT_DistilBERT) (Sketch-of-Thought, [arXiv 2503.05179](https://arxiv.org/abs/2503.05179), EMNLP'25) | 3 reasoning **paradigms** (not effort) | **67 M** DistilBERT | 512 | ✅ | N=3 | **MIT** | 509 | Real weights, real paper, ~14.2 k training samples from 13 datasets, up to **84 % token cut**. Predicts *which* reasoning style, not *how much* — but it is proof a 67 M prompt-only router does useful routing work, and the code ([SimonAytes/SoT](https://github.com/SimonAytes/SoT), MIT) is a clean template. |
| [`agentlans/bge-small-en-v1.5-prompt-difficulty`](https://huggingface.co/agentlans/bge-small-en-v1.5-prompt-difficulty) | continuous difficulty | **33.4 M** BGE-small | 512 | ✅ | **G** | **MIT** | 12 | Smallest transformer option. Auto-generated card; only metric MSE 1.396 on an undocumented scale. Largely unverified. |
| [`veritiana-ai/prompt-task-complexity-classifier`](https://huggingface.co/veritiana-ai/prompt-task-complexity-classifier) | task ×9 + low/med/high | **75 KB ONNX** logreg over 1,544 hashed features | ∞ | ✅ | N=3 | Apache-2.0 | 13 | Sub-millisecond. Card itself flags 86.8 % as *"internal weak-label evaluation, not independently established"*; 13 s of training. Honest bag-of-features baseline. |
| [`anasnassar/llm-query-complexity-classifier`](https://huggingface.co/anasnassar/llm-query-complexity-classifier) | LOW/MED/HIGH **reasoning depth** | 149.6 M ModernBERT-base | **128** | ✅ | N=3 | Apache-2.0 | 514 | **32 ms p50 CPU published**, but only **64.2 % acc / 0.640 macro-F1** — refreshingly honest card. 128-token cap is fatal here. |
| [`prvn-ramesh/query-classifier-onnx`](https://huggingface.co/prvn-ramesh/query-classifier-onnx) | low/medium/hard | ModernBERT-large INT8 ONNX, 397 MB | 8192 | ✅ | N=3 | Apache-2.0 | 114 | p50 67.7 / p90 134.7 / p99 161.7 ms, 12.3 q/s published. Zero hard→low errors. Trained on **1,800 queries**, tested on 301. |
| [`RowRed/ComplexityRouter`](https://huggingface.co/RowRed/ComplexityRouter) | Trivial/Simple/Moderate/Complex | 184 M DeBERTa-v3-base | 512 | ✅ | **N=4** | Apache-2.0 | 43 | 93.0 % *adjacent* accuracy. 4,400 prompts, labels generated by Qwen3.5-4B. Author calls it a "first attempt". |
| [`JiaqiXue/R2-Router-RouterArena`](https://huggingface.co/JiaqiXue/R2-Router-RouterArena) | quality per **(model, token-budget)** | sklearn KNN over Qwen3-0.6B embeddings, ~3.3 MB | — | ✅ | **G** + λ knob | Apache-2.0 | **0** | Closest published prior art to what we're building. #1 on RouterArena (71.23 %). Needs an embedding server. |
| [`massaindustries/modernbert-capability-classifier`](https://huggingface.co/massaindustries/modernbert-capability-classifier) | 6 sigmoid scores incl. `math_reasoning`, `planning_agentic`, `coding` | ModernBERT-base | 8192 | ✅ | G ×6 | Apache-2.0 | 283 | Not difficulty — a soft **capability feature vector**. `math_reasoning` Pearson 0.919. Good *input* to a policy. |
| [`LiquidAI/LFM2.5-Encoder-350M-Prompt-Router`](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router) | zero-shot score vs free-text lanes | **355.0 M** `Lfm2BidirForSequenceRouting` | 128k pos | ✅ | N (lanes) | **LFM Open v1.0 ($10 M cap)** | **4,391** | Most-adopted genuine prompt router here. No latency, no training data, no benchmarks published. Lanes are *topics*, not effort. ONNX q4 449 MB / int8 357 MB via `kucukkanat/…`. |
| [`AmirMohseni/reasoning-router-0.6b`](https://huggingface.co/AmirMohseni/reasoning-router-0.6b) | `think` / `no_think` | Qwen3-0.6B seq-cls, 1.19 GB | — | ✅ | **B** | Apache-2.0 | 24 | Card admits **"No Coding Tasks"** in training. Its own worked example misclassifies "sum of first 100 primes" as `no_think` (p=0.62). AIME 2025 w/ Qwen3-8B: mmBERT variant **0.2267 vs 0.2400 always-think** while routing 93.3 % to thinking. |
| [`katanemo/Arch-Router-1.5B`](https://huggingface.co/katanemo/Arch-Router-1.5B) | preference-aligned route (domain/action) | **1.54 B** Qwen2.5 decoder | — | ✅ | N (routes) | ⚠️ **Katanemo/DigitalOcean CL — commercial use needs a separate grant** | 1,896 | Not difficulty. Must *generate*. GGUF Q4_K_M 986 MB → hundreds of ms on CPU. Gateway repo renamed `katanemo/plano` (Apache-2.0, 7.0k ★). |
| [`routellm/bert_gpt4_augmented`](https://huggingface.co/routellm/bert_gpt4_augmented) | strong-vs-weak win prob | 278 M XLM-R, 1.11 GB | 512 | ✅ | B (thresholdable G) | Apache-2.0 | 10,712 | The only self-contained RouteLLM router. **No model card on any `routellm/*` repo** — label semantics only from reading `routers.py`. |

**Sub-100 M backbones to fine-tune** (no task head, add your own):

| backbone | params | ctx | licence | note |
|---|---|---|---|---|
| [`jinaai/jina-embeddings-v2-small-en`](https://huggingface.co/jinaai/jina-embeddings-v2-small-en) | **32.7 M** | **8192** | **Apache-2.0** | The sub-100 M + long-context cell. 1.05 M DL/30 d. ⚠️ **global** attention → T² (§4). |
| [`minishlab/potion-base-32M`](https://huggingface.co/minishlab/potion-base-32M) / [`-8M`](https://huggingface.co/minishlab/potion-base-8M) | **32.3 M / 7.6 M** | **∞** | **MIT** | Static lookup. No T² term. 1.12 ms p50 served. |
| [`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small) | ~140 M total / **~42 M non-emb** | **8192** | **MIT** | ModernBERT arch (22 L, h=384, vocab 256k). Local/global attention. |
| [`microsoft/deberta-v3-xsmall`](https://huggingface.co/microsoft/deberta-v3-xsmall) / [`-small`](https://huggingface.co/microsoft/deberta-v3-small) | 22 M + 48 M emb = **70 M** / 44 M backbone | 512 | MIT | Best quality/param, but 512 ctx. |
| `BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`, `thenlper/gte-small` | ~33.4 M | 512 | MIT | Interchangeable. |
| `sentence-transformers/all-MiniLM-L6-v2` | 22 M | 256–512 | Apache-2.0 | The classic cheap baseline. |
| [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) | 149.7 M | 8192 | Apache-2.0 | Over budget; the reference point. |

### 3b. Systems and methods

| system | routes what | predictor | latency | code/weights | licence | verdict |
|---|---|---|---|---|---|---|
| [DART](https://github.com/js-lee-AI/DART) ([arXiv 2606.23181](https://arxiv.org/abs/2606.23181)) | **effort**, continuous budget | **none — training-free.** K=2 no-think drafts; agree → no-think; else an **isotonic map of draft entropy → budget** | **2.3× wall-clock** | ✅ real code, 17 ★ | **MIT** | **Best fit for the standing rule.** 13/14 pairs match-or-beat always-think at 15–69 % fewer thinking tokens. Qwen3-32B HumanEval **72.6 → 95.1 at −63 %**; +9.0 OlympiadBench. Its own baselines include supervised **MLP and GBT routers** — it beats them. Cross-family transfer 0.6B → 32B. |
| [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) ([arXiv 2510.08731](https://arxiv.org/abs/2510.08731), [2603.04444](https://arxiv.org/abs/2603.04444), [2603.12646](https://arxiv.org/abs/2603.12646)) | model + effort, but effort is a **static per-category config lookup** keyed by a learned **14-domain intent** classifier | mmBERT-32K **307 M**, ONNX (incl. fp16), `use_cpu: true` per classifier; ~14 ms @512 tok **on GPU** | **4,918 ms CPU baseline**; 50–108 ms only when **GPU-colocated** | ✅ 5.2k ★, 6,369 files, pushed 2026-08-19; weights [`llm-semantic-router/mmbert32k-intent-classifier-merged`](https://huggingface.co/llm-semantic-router/mmbert32k-intent-classifier-merged) **Apache-2.0, 12.9k dl**, trained on MMLU-Pro + 653 supplement, 14-class test acc 80.0 % | Apache-2.0 | The only shipping effort router. Steal the config schema; don't adopt the service. **Never touches `thinking_token_budget`**, and reports **no random or prompt-length baseline**. |
| [HRBench](https://github.com/usail-hkust/HRBench) ([arXiv 2605.28398](https://arxiv.org/abs/2605.28398)) | — (benchmark) | — | — | ✅ 9 ★ | — | Benchmarks 3 strategy families × 4 training regimes. **External routing ≈ 18–21 % token saving at preserved accuracy.** Checkpoints *unverified*. |
| [LLM-serving-with-proxy-models](https://github.com/James-QiuHaoran/LLM-serving-with-proxy-models) ([arXiv 2404.08509](https://arxiv.org/pdf/2404.08509)) | output-length buckets → vLLM priority scheduler (≥ v0.6.2) | BERT-base + 2-layer FC on `[CLS]` | not published | ⚠️ code only, **no weights** | Apache-2.0 | Proves prompt-only length prediction is deployable in vLLM. JCT +30.5–39.6 %, throughput 2.2–3.6×. |
| [TIE](https://github.com/Hyzheng-code/TIE) ([arXiv 2604.00499](https://arxiv.org/html/2604.00499v2)) | (μ,σ) of a log-t length distribution | DeBERTa-v3-base + multi-pool, 2 MLP heads | not published | ⚠️ code, 1 ★ | Apache-2.0 | **R² 0.82/0.76.** The right output *shape* for our asymmetric cost. |
| [Agent-as-a-Router](https://github.com/LanceZPF/agent-as-a-router) ([arXiv 2606.22902](https://arxiv.org/html/2606.22902v1)) | **models**, per coding task | Qwen3.5-0.8B policy + heuristics + kNN memory | not published | ✅ 947 ★ | MIT | Routes models. Its value to us is the **OOD collapse result** for static routers on coding. |
| [ARES](https://github.com/UCSB-NLP-Chang/Ares) ([arXiv 2603.07915](https://arxiv.org/abs/2603.07915)) | **per-step effort** in an agent loop | Qwen3-1.7B router | — | ❌ **repo = one 104-byte README**, 1 ★ | none | Exactly our problem; −52.7 % reasoning tokens. **Nothing to download.** Reuse the label definition only. |
| [RouteLLM](https://github.com/lm-sys/RouteLLM) ([arXiv 2406.18665](https://arxiv.org/abs/2406.18665)) | strong vs weak **model** | `mf` (0.8 M params) / `bert` / `causal_llm` (Llama-3-8B) | — | ✅ 5.4k ★, **dead since 2024-08-10** | Apache-2.0 | ⚠️ **`mf` and `sw_ranking` call the OpenAI embeddings API per request**; `sw_ranking` also refits a logistic-regression Elo over ~100k Arena battles *per request*. Only `bert` is self-contained. Its `causal_llm` prompt asks for a **native 1–5 rating** that RouteLLM then collapses to binary — worth reading. |
| [arXiv 2511.03808](https://arxiv.org/html/2511.03808v1) | 5-level difficulty → model choice | 3-layer MLP over **s1.1-32B layer-45 hidden states** | — | ❌ | — | **Not pre-generation-cheap**: needs a 32 B forward pass. Baseline is *random only*. Math only. Unusable. |
| AdaptThink ([2505.13417](https://arxiv.org/abs/2505.13417)), Thinkless ([2505.13379](https://arxiv.org/abs/2505.13379)), AutoThink ([2505.10832](https://arxiv.org/abs/2505.10832)), L1/LCPO ([2503.04697](https://arxiv.org/abs/2503.04697)), s1 ([2501.19393](https://arxiv.org/abs/2501.19393)) | think/no-think or length | **the LLM itself** (`Qwen2ForCausalLM` checkpoints) | — | ✅ full 1.5–7 B checkpoints | MIT / Apache-2.0 | ❌ **No separable classifier head.** You would have to swap the served model. Prior art, not a pre-filter. L1's length is *user-supplied*, and s1 has no decision model at all. |
| Self-Route ([2505.20664](https://arxiv.org/abs/2505.20664)), Router-R1 ([2506.09033](https://arxiv.org/abs/2506.09033)) | think/no-think; models | coupled head on the served model; a **3 B LLM** | — | ⚠️ / ✅ | — | Neither is a free-standing CPU gate. |
| [TwinRouterBench](https://github.com/CommonstackAI/TwinRouterBench) ([arXiv 2605.18859](https://arxiv.org/html/2605.18859v1)) | **4-tier per-step effort** on agentic coding | benchmark + trained logistic router | — | ✅ real code, pushed 2026-08-18 | **Apache-2.0** | **The only 4-tier execution-verified agentic-coding-step label set.** Trained logistic router **−53.1 % cost at matched resolution**; Claude Opus 4.6 as a prompted router caught **7/147** high steps and failed all 40 trajectories. See §1.2. |
| [Plan-and-Budget](https://github.com/junhongmit/P-and-B) ([arXiv 2505.16122](https://arxiv.org/abs/2505.16122), ICLR'26) | per-sub-question share of a **user-supplied** global budget | LLaMA-3.1-8B planner emits sub-question complexity; decay schedules; **training-free** | one 8B planner call | ✅ real pipeline | **MIT** | Does **not** predict the total budget — it splits one you give it. But the **8B planner successfully budgets DS-Qwen-32B, QwQ-32B, DS-LLaMA-70B and o4-mini**, which is genuine cross-target transfer. Baselines include a fixed-global-budget arm. |
| [RADAR](https://arxiv.org/abs/2509.25426) | correctness per **(model, budget)** | **IRT-2PL over frozen query embeddings, ~7 ms** | ~7 ms | ❌ paper-only | — | Cheapest architecture in the survey and the right output shape. **Onboards an unseen Qwen3-14B from 12 % of queries** — the best answer to "what happens when we change the served model". Nothing to download. |
| [Sketch-of-Thought](https://github.com/SimonAytes/SoT) | 3 reasoning paradigms | DistilBERT 67 M | — | ✅ weights + code | MIT | See §3a. |
| **[TUNES-TARGET] — produce a new checkpoint, not a classifier**: AdaptThink ([2505.13417](https://arxiv.org/abs/2505.13417), MIT code + `THU-KEG/AdaptThink-*` weights), Thinkless ([2505.13379](https://arxiv.org/abs/2505.13379), `Vinnnf/Thinkless-1.5B-RL-DeepScaleR`), AutoL2S (`amandaa/AutoL2S-7b`), AutoThink, ARM (`arm-team/ARM-*`), Ada-R1, OThink-R1, LHRM, AdaCoT, TOPS (`Keven16/Qwen2.5-32B-TOPS`), SelfBudgeter, SABER, BudgetThinker, BET, L1/LCPO (`l3lab/L1-Qwen3-8B-{Max,Exact}` etc., MIT, 265 ★), CogRouter | think/no-think or length | **the served model itself** | — | ✅ mostly | MIT/Apache | ❌ **Not drop-in.** You would swap Qwen3.8-27B for their 1.5–7 B checkpoint. Prior art only. **L1 does not predict a budget** — the user supplies it. **AdaCoT states outright that its decision boundaries do not transfer across models.** LHRM's 1.5 B "hybrid accuracy" is **54.4 %, ≈ a coin flip, with no random baseline reported.** |
| **[NEEDS-ROLLOUT] — cannot decide pre-generation**: ThinkLess (PKU), DEER ([iie-ycx/DEER](https://github.com/iie-ycx/DEER), MIT, 203 ★), Dynasor/Certaindex ([hao-ai-lab/Dynasor](https://github.com/hao-ai-lab/Dynasor), MIT, 232 ★), DeepConf ([facebookresearch/deepconf](https://github.com/facebookresearch/deepconf), MIT, 410 ★), SeerSC, Budget Guidance, SAT, TRAIL, EGTP/ForeLen, ProD, Self-Route, DiffAdapt, Predictive Scheduling (MLP variant), [2511.03808](https://arxiv.org/abs/2511.03808) | early exit / mid-gen budget | target-model hidden states or drafts | — | ✅ several | MIT | Good engineering, **wrong layer for us** — these belong to the in-flight controller (plan §4, signals S1–S8), not to S10. Worth reading for the escalation logic, not as a prompt-time gate. |
| **ThinkSwitcher** ([2505.14183](https://arxiv.org/abs/2505.14183)) | 2 pass-rate regressions → binary | **5-layer MLP, 2.9–6.6 M params, on the target LRM's query embedding** | needs LRM prefill | ❌ **paper-only, no repo found** | — | Its own **ModernBERT text-only baseline (60.7 % @ 6,021 tok) loses to its hidden-state version (62.8 % @ 5,405)** — and SAT ([2604.07922](https://arxiv.org/abs/2604.07922)) reports ThinkSwitcher **underperforming always-CoT by 2.5 pts** on MATH500/Qwen3-14B. |
| SelfBudgeter, TON, Switch-Reasoner, RouterDC, GraphRouter, EmbedLLM, IRT-Router, Zooter, S³, TRAIL, ProD | — | — | — | ❌ **no HF weights found** | — | **Nothing to download.** |
| [`ulab-uiuc/LLMRouter`](https://github.com/ulab-uiuc/LLMRouter) | — (framework) | 16+ methods (KNN, SVM, MLP, MF, Elo, graph, BERT) | — | ✅ 2.4k ★ | **MIT** | A **training framework**, not a source of drop-in weights — `ulab-ai` ships no meaningful lightweight checkpoints. |

### 3c. Commercial routers — weights public or not

| product | weights public? | exact ID | verdict |
|---|---|---|---|
| **Not Diamond** | **the only one that ever shipped weights** | `notdiamond/notdiamond-0001` (BERT seq-cls, Apache-2.0, **32 DL/30 d, last modified 2024-07-30**) + 12× `notdiamond/rorf-*` | Obsolete: hardcoded to a 2024 GPT-3.5-vs-GPT-4 pair. The `rorf-*` weights are **pickled sklearn random forests with no card and no licence tag**; SDKs archived Nov/Dec 2025. |
| **Martian** | **no** | `withmartian` org has 106 models, **none a router** — interpretability research + SFT checkpoints | API-only. `withmartian/routerbench` is a dataset (no licence tag, 2024-03-27). |
| **Unify.ai** | **no** | — | API-only; pivoted away from routing. |
| **OpenRouter `auto`** | **no** | — | Now openly described as *"powered by the market: the aggregate spend of millions of people… trailing 7-day window"* + a closed ~30-way task classifier. Not Diamond no longer involved. |
| **Requesty** | **no** | — | API-only. |
| **LiteLLM Router** | **no ML model exists** | — | MIT. Strategies are deterministic. Its `auto_router/complexity_router` (SIMPLE/MEDIUM/COMPLEX/REASONING) defaults to a **hand-written heuristic scorer** over 7 surface features — instructive as exactly the approach the project rules out. |
| **Portkey** | **no ML model exists** | — | JSON rule engine (`$and $or $eq $regex`). `?author=portkey` on HF is empty. |

---

## 4. Latency — the constraint that decides this

**Measured, not estimated:** the vLLM Semantic Router's own follow-up paper
([arXiv 2603.12646](https://arxiv.org/abs/2603.12646)) reports a **4,918 ms CPU
baseline** for one routing decision, optimised to 50 ms *by moving it onto the
serving GPU*. Corroborating per-length CPU figures from the v0.2 Athena writeup
(≈853 ms ONNX-CPU / 1,053 ms Candle-CPU at ~500 tokens, 4,796 ms at 8k) are
**second-hand but consistent** with that baseline.

The arithmetic explains why. Encoder cost ≈ `2·P_nonemb·T` (matmuls) +
`4·L·T²·d` (attention, when attention is global rather than windowed).

- `jina-embeddings-v2-small-en` (verified `config.json`: 4 layers, d=512,
  intermediate 2048, vocab 30528 → ~13 M non-embedding params, **global ALiBi
  attention, no windowing**) at T=8192: matmuls ≈ 2·13e6·8192 ≈ **2.1e11 FLOPs**;
  attention ≈ 4·4·8192²·512 ≈ **5.5e11 FLOPs**. Total ~7.6e11 — **attention
  dominates, and it is quadratic.**
- At 100–500 GFLOP/s effective CPU GEMM throughput that is **≈ 1.5–8 s per
  prompt.** At T=512 the same model is ~50–100× cheaper: **tens of ms.**
- ModernBERT-family models (mmBERT-small, ModernBERT-base, LFM2.5 encoders) use
  *alternating local/global* attention (`local_attention: 128`,
  `global_attn_every_n_layers: 3` in mmBERT-small's `config.json`), which kills
  the T² term for most layers — that is why Liquid can claim "3.3× faster than
  ModernBERT-base at 8k". It does **not** remove the linear term.

**Conclusion: a transformer encoder over the *full* 8k agentic prompt is too slow
on CPU regardless of parameter count.** Three viable escapes, in order:

1. **Static embeddings (potion / model2vec, MIT).** No context limit, no T² term,
   ~1 ms. **Best usefulness-per-parameter in the survey.**
2. **Truncate/reduce the prompt to ~512 tokens** and use any of the 22–70 M
   encoders (`deberta-v3-xsmall`, `bge-small`, `all-MiniLM-L6-v2`) at tens of ms.
   But choosing *what* to keep is itself a hand-written heuristic, which the
   project rules out unless the reduction is learned.
3. **Co-locate a small classifier on the serving GPU**, as vLLM SR concluded. On
   a box already at the HBM ceiling this competes with the model — probably a
   non-starter, but it is what the one system that measured this actually did.

**The single measurement to take before any further modelling work:** time
`LFM2.5-Encoder-230M`, `jina-v2-small` and `potion-base-32M` on a real 8k-token
agentic step on this box. Latency will likely eliminate more candidates than
accuracy does.

---

## 5. Vendor "thinking" routers — nothing reusable, but one useful convergence

**Every hybrid-reasoning vendor bakes the decision into the weights and exposes
only a caller-set flag. Zero vendor GitHub or HF org contains a router or
classifier repo.**

| vendor | control surface | auto/adaptive? | separate artifact? |
|---|---|---|---|
| **Qwen3 / 3.5 / 3.8** | Qwen3: `/think` `/no_think` are **literal user text the model was trained to obey** (not in the template; the template's only thinking logic is `enable_thinking=false` → emit an empty `<think></think>`). **Qwen3.5 dropped the soft switch.** Qwen3.8: `reasoning_effort ∈ {low, medium, xhigh}`, hard-errors otherwise | ❌ no `auto` | ❌ none |
| **DeepSeek V3.1/V3.2/V4** | V3.1: prompt prefix (`<think>` vs a pre-filled `</think>`). V4: `thinking_mode ∈ {chat, thinking}`, `reasoning_effort ∈ {low, high, max}` as **`REASONING_EFFORT_PROMPTS` prefix strings** (`low` → `""`) — mirrored in our tree at `vllm/tokenizers/deepseek_v4_encoding.py:250` | ❌ | ❌ (18 repos, none) |
| **Kimi K2 / K2.5 / K3** | `thinking.type ∈ {enabled, disabled}`; K3 always thinks with `reasoning_effort ∈ {low, high, max}`, default `max`. **"Heavy mode" is 8 parallel trajectories aggregated** — more compute you opt into, not a decision | ❌ no `auto` | ❌ |
| **GLM-4.5 → 5.2** | template literally appends the string `/nothink` to the user turn when `enable_thinking` is false; `thinking.type ∈ {enabled, disabled}`; GLM-5 adds `reasoning_effort ∈ {max, high}` | ❌ (a doc phrase "dynamic thinking is enabled" is **unverified** and unsupported by the schema) | ❌ |
| **Anthropic** | `thinking:{type:"adaptive"}` + `output_config:{effort: low…max}`; `type:"enabled"`+`budget_tokens` **400s on 4.7+** | ✅ model decides | ❌ — the **Opus 4.6 System Card (213 pp) contains "router" zero times** |
| **OpenAI** | `reasoning_effort ∈ none…max` | ✅ | ⚠️ one sentence, GPT-5 System Card ([arXiv 2601.03267](https://arxiv.org/abs/2601.03267)): *"a real-time router that quickly decides which model to use based on conversation type, complexity, tool needs, and explicit intent"*, *"continuously trained on real signals, including when users switch models, preference rates… and measured correctness."* Routes **between models in ChatGPT**, not effort in the API. No architecture, size or weights. |
| **Google** | `thinking_level` / `thinkingBudget: -1` = *"the model will adjust the budget based on the complexity of the request"* | ✅ | ❌ nothing published |

**The one useful convergence:** everyone has settled on a **discrete effort
ladder**, and in the open models it is literally swappable system-prompt text
(Qwen3.8: three hardcoded strings; DeepSeek V4: `REASONING_EFFORT_PROMPTS`;
GLM: `/nothink`). Our ladder is in good company. And every one of them pays the
prefix-cache cost described in §0.

---

## 6. Engine-native pre-generation effort routing

**The actuator is native and battle-tested. The decider does not exist. We would
be building the join.**

> **Provenance:** `/shared/vllm` is not stock upstream — HEAD is upstream
> `acb0f1dcd` (2026-08-15, `0.27.2rc1.dev110`) plus our 9 local patches.
> `reasoning_effort: "dynamic"` is **patch 0009, local and unreported upstream**,
> not a vLLM feature.

### vLLM

**✅ `thinking_token_budget` is genuinely engine-native, per-request,
pre-generation effort control within one model** — not template theatre. It is a
`SamplingParams` field enforced by forcing `reasoning_end_str` at sample time:
`vllm/sampling_params.py:365` (field) and `:51-77` (validator, `-1` = unlimited),
`vllm/v1/sample/thinking_budget_state.py` (`ThinkingBudgetStateHolder`, applied
after penalties on the V1 `BatchUpdate` logits-processor path),
`vllm/v1/worker/gpu/sample/thinking_budget.py` (Triton/MRv2),
`vllm/config/reasoning.py` (token IDs from the reasoning parser). Docs:
`docs/features/reasoning_outputs.md` § "Thinking Budget Control".

| PR | what | merged |
|---|---|---|
| [#20859](https://github.com/vllm-project/vllm/pull/20859) | limit thinking tokens (hard limit) | 2026-03-24 |
| [#34668](https://github.com/vllm-project/vllm/pull/34668) | spec-decoding support | 2026-04-29 |
| [#42116](https://github.com/vllm-project/vllm/pull/42116) | completions endpoint | 2026-05-13 |
| [#41674](https://github.com/vllm-project/vllm/pull/41674) | fix inverted condition (silently ignored) | 2026-05-15 |
| [#43402](https://github.com/vllm-project/vllm/pull/43402) | reject invalid values | 2026-05-26 |
| [#46137](https://github.com/vllm-project/vllm/pull/46137) | Rust frontend | 2026-06-22 |
| [#43757](https://github.com/vllm-project/vllm/pull/43757), [#45984](https://github.com/vllm-project/vllm/pull/45984) | re-entry enforcement fixes | 2026-06-30 / 07-10 |
| [#46727](https://github.com/vllm-project/vllm/pull/46727) | **Model Runner V2 support** | 2026-08-07 |

Open bugs worth tracking: [#44676](https://github.com/vllm-project/vllm/issues/44676)
(budget forces `</think>` into the middle of tool-call args on Qwen3.5+ — directly
relevant to an agentic loop) and [#48201](https://github.com/vllm-project/vllm/issues/48201).

**⚠️ `reasoning_effort` is accepted, validated, then mostly discarded.**
`chat_completion/protocol.py:245` declares the 7-value ladder;
`build_chat_params()` does exactly two things — inject it as a Jinja variable and
derive `enable_thinking = (effort != "none")`. `vllm/renderers/hf.py:644-673`
then **silently drops** any kwarg the template doesn't declare, so for most models
it is a no-op. Genuinely prompt-changing only for GPT-OSS/Harmony
(`parser/harmony_utils.py:67-132`) and DeepSeek-V4 (`tokenizers/deepseek_v4.py:43-59`).
Merged via [#31956](https://github.com/vllm-project/vllm/pull/31956),
[#36238](https://github.com/vllm-project/vllm/pull/36238),
[#40982](https://github.com/vllm-project/vllm/pull/40982),
[#43401](https://github.com/vllm-project/vllm/pull/43401). See also open issue
[#52738](https://github.com/vllm-project/vllm/issues/52738) (2026-08-18): vLLM
accepts all 7 values but Qwen3.8 400s at render time on anything but three.

**❌ The missing link:** `reasoning_effort` and `thinking_token_budget` are
**completely disconnected upstream** — there is no `effort → budget` table
anywhere in `vllm/`, and no upstream RFC or open PR proposes automatic effort
selection. **That gap is exactly what patch 0009 fills.**

**Where a classifier should hook in — two seams, ranked:**

| hook | scope | can set effort? |
|---|---|---|
| **`ReasoningParser.adjust_request()`** (`abs_reasoning_parsers.py:184`, called from `renderers/online_renderer.py:206-211`) | API-server process, **pre-tokenization, per request, CPU-side** | **Yes — the best in-tree seam.** It can mutate any request field including `thinking_token_budget`. No in-tree parser does. |
| Our own `apply_dynamic_effort()` (`chat_completion/serving.py`, called from `_create_chat_completion`; `dynamic_effort.py` already defines `_BIAS_KEY = "effort_bias"` riding through to the scheduler in `SamplingParams.extra_args["dynamic_effort"]`) | frontend, per request | **Yes, and already wired.** |
| Endpoint plugins (`docs/design/endpoint_plugins.md`) | HTTP surface only, not loaded by default, `VLLM_PLUGINS`-allowlisted | Only by adding a *new* route; cannot intercept `/v1/chat/completions` |
| Custom logits processors (V1) | engine core, per decode step | shapes generation, not pre-scheduling |
| IO Processor plugins | **pooling models only** | no |
| `Platform.validate_request` | engine input processor | signature `-> None`; raises, cannot mutate |

**A classifier should write `effort_bias`, not a start rung.** That plugs into
the existing controller's `bias` term (plan §4), lowers the escalation threshold θ
rather than jumping the ladder, degrades to exactly today's behaviour when the
classifier is wrong or absent, and is shadow-mode testable from day one. No new
control path, no new failure mode.

### Every other vLLM-ecosystem router: models/endpoints, not effort

| system | routes on | effort-aware? |
|---|---|---|
| `vllm-project/production-stack` | endpoints per model, round-robin, session-ID, prefix-aware (WIP) | **No** — 0 code hits for `reasoning_effort` |
| `vllm-project/router` (Rust, 363 ★) | cache-aware, power-of-two, consistent-hash, P/D disagg | deserializes `reasoning_effort` (`src/protocols/spec.rs:522`) as **pass-through only** |
| llm-d EPP/Router | KV-cache locality, load, priority | **No** — 0 code hits |
| AIBrix | gateway + LoRA density + autoscaling | model-level; effort only via bolted-on semantic-router |
| NVIDIA Dynamo | KV-aware worker selection | `frontend/thinking.py` defines `THINKING_CONTROL_KEYS = ("thinking","enable_thinking","thinking_mode","reasoning_effort")` and merges a **deployment-level default**, skipped if the request already controls it. Static, not classification. `frontend/prepost.py` is a real Python pre-processing layer over vLLM — a viable hook point, ships no classifier. |
| NVIDIA `llm-router` blueprint | — | **deprecated** → `NVIDIA-NeMo/Switchyard` (Apache-2.0, 1.9k ★, but *"pre-alpha… not for production"* and **ships no classifier weights**; its strategies are an LLM call, rules, post-hoc escalation, or random). **`nvidia/prompt-router*` does not exist** — verified empty on the HF API. |

### SGLang — rich caller-specified effort, zero auto
`thinking_budget` first landed [#6089](https://github.com/sgl-project/sglang/pull/6089)
(2025-05-09) and was **reverted the next day** by [#6181](https://github.com/sgl-project/sglang/pull/6181);
re-landed as custom logit processors in [#11416](https://github.com/sgl-project/sglang/pull/11416)
(2025-10-21); per-model processors since (e.g. [#33146](https://github.com/sgl-project/sglang/pull/33146),
2026-08-10, which also fixed a base-class bug where the end token appearing
anywhere in prompt+output silently voided the budget).
`reasoning_effort` schema aligned in [#31784](https://github.com/sgl-project/sglang/pull/31784)
(2026-07-20) — accepts the OpenAI string tiers **plus an SGLang float extension in
`[0.0, 0.99]`**, which is a strictly better interface than a 3-string enum and
worth copying. Also `--enable-strict-thinking` ([#23953](https://github.com/sgl-project/sglang/pull/23953),
a THINKING→GENERATION grammar state machine) and `--default-chat-template-kwargs`
([#29579](https://github.com/sgl-project/sglang/pull/29579)).
**`sgl-router` has been renamed `sgl-model-gateway`**; its strategies are
`random/round_robin/cache_aware/power_of_two/bucket` + P/D disagg — **no effort
routing**, and a code search for router+effort is empty.

### llama.cpp / Ollama / TGI
**llama.cpp has the richest caller-side reasoning-budget machinery of any
engine** — all merged, all caller-specified:
[#20297](https://github.com/ggml-org/llama.cpp/pull/20297) (`--reasoning on/off`,
`--reasoning-budget-message`, delayed-launch grammar, 2026-03-11),
[#23949](https://github.com/ggml-org/llama.cpp/pull/23949)
(`common_sampler_reasoning_budget_force()`, 2026-06-01),
**[#23971](https://github.com/ggml-org/llama.cpp/pull/23971) — real-time reasoning
interruption via `POST /v1/chat/completions/control` (2026-06-02), i.e.
mid-generation effort termination, which no other engine has**,
[#23116](https://github.com/ggml-org/llama.cpp/pull/23116) (per-request
`reasoning_budget_tokens`), [#26045](https://github.com/ggml-org/llama.cpp/pull/26045),
[#25544](https://github.com/ggml-org/llama.cpp/pull/25544) (multiple end
sequences), [#26941](https://github.com/ggml-org/llama.cpp/pull/26941).
**Ollama**: `think: false|true|"low"|"medium"|"high"|"max"`
([#15787](https://github.com/ollama/ollama/pull/15787),
[#15789](https://github.com/ollama/ollama/pull/15789)) — caller-specified.
**TGI is archived** (`"archived": true`, last push 2026-03-21) — not a path forward.

### Proxies
**LiteLLM** is pure translation: documented mapping for older Claude —
`minimal`/`low`→1024, `medium`→2048, `high`→4096, `xhigh`→8192, `max`→16384
`budget_tokens`, each overridable via `DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET`.
**A ready-made effort→budget table worth stealing as a starting ladder.**
**OpenRouter** uses static ratios of `max_tokens`: max/xhigh ≈95 %, high ≈80 %,
medium ≈50 %, low ≈20 %, minimal ≈10 %; Anthropic formula
`budget_tokens = max(min(max_tokens * ratio, 128000), 1024)`. **Neither
auto-selects.** No shipping proxy anywhere automatically picks an effort level
for a single model.

---

## 7. Datasets and label design

### 7a. Two published papers are literally this project

**ARES** ([arXiv 2603.07915](https://arxiv.org/html/2603.07915v1), UCSB, Mar 2026)
— router = **Qwen3-1.7B**, sees **interaction history + current observation** (no
hidden states), emits a rationale plus one of **3 effort levels**.
**Label derivation — copy this exactly:** collect successful max-effort
trajectories → for each step, sample each effort level **K=3** times → accept a
level as *sufficient* if ≥ M/K trials produce a functionally-equivalent action to
ground truth → label = **the lowest sufficient level**. Scale: **43,358 labelled
steps** (τ-Bench), 12,366 (BrowseComp-Plus), 1,718 (WebArena). Results: τ-Bench
Retail 54.8 % acc at **−35.2 % tokens**; BrowseComp-Plus 41.3 % / −41.8 %;
WebArena 46.5 % (beating the 45.0 % fixed-high baseline) / −45.3 %; an RL variant
reaches **−80 %** on τ-Bench Airline. Ablations: removing SFT drops 54.8 → 41.7 %;
removing the rationale costs 3.5 points. **No data or code released** (the repo is
a 104-byte stub, §3b), and its benchmarks are tool-use/web, not repo-level coding.

**CogRouter** ([arXiv 2602.12662](https://arxiv.org/html/2602.12662v1), Feb 2026,
[github.com/rhyang2021/CogRouter](https://github.com/rhyang2021/CogRouter)) —
**four** cognitive-depth levels (instinctive / situational / experience-integration
/ strategic-planning), the same rung count we want. Router sees interaction
history only; labels bootstrapped by prompting GPT-4o to write per-level thinking
on expert trajectories. ALFWorld w/ Qwen2.5-7B: **92.5 % success at 1,739 tokens**
vs GRPO's 83.5 % at 4,995. ALFWorld/ScienceWorld only — not coding.

**TwinRouterBench** ([arXiv 2605.18859](https://arxiv.org/html/2605.18859v1),
[CommonstackAI/TwinRouterBench](https://github.com/CommonstackAI/TwinRouterBench),
**Apache-2.0, real code, pushed 2026-08-18**) — the third, and the only one with
usable data. **Four tiers `low`/`mid`/`mid_high`/`high`, 970 execution-verified
step rows**, 336 of them SWE-bench Verified (high 168 / low 94 / mid_high 41 /
mid 33), plus BFCL 248, mtRAG 193, QMSum 145, PinchBench 48, and 100 held-out
SWE-bench Verified instances for dynamic evaluation. See §1.2 for its findings —
the important ones for data design are that **LLM-as-router labels are not a
shortcut** (Claude Opus 4.6 caught 7 of 147 high steps) and that **one
under-routed step fails a whole trajectory**.

**So: a 4-rung effort schema for agentic coding steps now exists, with 970
verified rows and an Apache-2.0 licence — but nobody has trained and shipped a
sub-100 M prompt-only router against it.** Use TwinRouterBench as the evaluation
set and label schema; generate training volume with the ARES recipe on our own
traffic.

### 7b. Our own telemetry is still the best dataset

Patch 0009's `reasoning_effort: "dynamic"` sink already records, per request, the
prompt, the realised think-token count, the rung and the outcome. That is
in-distribution agentic coding data with free labels. Everything below is a
distant second — and per §2c, **the think-token count should be used to measure
the noise floor, not as the training target.**

### 7c. Per-turn reasoning-token counts, already materialised

[`nvidia/Nemotron-SFT-Agentic-v2`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2)
— **991,900 rows, CC-BY-4.0**, and its schema carries verbatim:

```
turn_token_count:      List(struct{reasoning: int64, content: int64})
all_turns_token_count: struct{reasoning: int64, content: int64, all: int64}
chat_template_kwargs:  struct{thinking: bool}
```

**Per-turn reasoning-token counts, pre-computed, permissively licensed.** Subsets:
tool_calling 707,052 / customer_service 278,880 / search_graph_walk 5,968;
generated with DeepSeek-V3.2 and GLM-4.6. Blunt caveat: **tool-calling and
customer service, not repo-level coding.**

### 7d. Agentic step-level trajectories — the part that matters most

| dataset | scale | thinking tokens? | licence |
|---|---|---|---|
| **[TwinRouterBench](https://github.com/CommonstackAI/TwinRouterBench)** | **970 rows** + 100 held-out SWE-bench Verified instances | **n/a — carries a direct 4-tier effort label**, execution-verified | **Apache-2.0** |
| [`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | **207,489 trajectories** | **yes — Minimax-M2.5 runs are in thinking mode** (Qwen3.5-122B runs are not) | **CC-BY-4.0** |
| [`nebius/SWE-rebench-openhands-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | **67,074 trajectories, avg 64.3 turns (max 100) ⇒ ~4.3 M steps** | no per-step counts | **CC-BY-4.0** |
| [`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | **80,036** (13,389 resolved / 66,647 not) | card reports **avg context 8,352 tok resolved vs 15,241 unresolved** — the length↔failure signal of §2c, already measured | CC-BY-4.0 |
| [`nvidia/Nemotron-SFT-SWE-v3`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3) | 237,970 (`messages` len 9–353 ⇒ turn count is a free difficulty proxy) | not documented | CC-BY-4.0 |
| [`SWE-bench/SWE-smith-trajectories`](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | **76,002** | no | **MIT** |
| [`nvidia/SWE-Hero-openhands-trajectories`](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories) | 34,269 across 11,766 issues | no | CC-BY-4.0 |
| [`SWE-bench/experiments`](https://github.com/SWE-bench/experiments) (GitHub) | every leaderboard submission | **reasoning traces are now a mandatory submission requirement**, but the format is free-form | not stated |

**The highest-leverage data move available:** `nvidia/Open-SWE-Traces` and
`nebius/SWE-rebench-openhands-trajectories` are the only public corpora *shaped
like our prompts*. Neither ships token counts, but the thinking blocks can be
tokenised. Critically, both contain **multiple rollouts per instance**, so they
let us measure within-prompt variance on the real distribution — **that is the
experiment that decides whether this project is worth running.**

### 7e. Difficulty-labelled prompt sets

**`princeton-nlp/SWE-bench_Verified`** has a first-class **`difficulty`** column
with exactly four classes. Verified distribution over all 500 instances
(HF datasets-server, 2026-08-19). **No licence declared on the card.**

| label | count | share |
|---|---|---|
| `15 min - 1 hour` | 261 | 52.2 % |
| `<15 min fix` | 194 | 38.8 % |
| `1-4 hours` | 42 | 8.4 % |
| `>4 hours` | **3** | **0.6 %** |

A majority-class predictor already scores **52.2 %** — read any 4-class accuracy
figure against that, not against 25 %. The **full 1,699-instance annotation
release** is live at
`https://cdn.openai.com/introducing-swe-bench-verified/swe-bench-annotation-results.zip`
(HTTP 200, 2,397,074 bytes, last modified 2024-08-12), containing
`ensembled_annotations_public.csv` (1,699 rows; difficulty `<15 min` 417 /
`15 min-1 hr` 906 / `1-4 hr` 329 / `>4 hr` 47) and
`samples_with_3_annotations_public.csv` (**three annotators per instance →
soft labels and measurable disagreement**; use ordinal regression on these, not
hard 4-way classification).

**Prior art worth copying:** SWE-smith already built this rater —
`swesmith/train/difficulty_rater/create_datasets.py` uses the same four buckets
and then does `if label == ">4 hours": label = "1-4 hours"`. **They collapse to
three classes because `>4 hours` is unlearnable.** Their model is published as
`SWE-bench/SWE-Rater-32B` (no card, 9 downloads). Caveat: their rater conditions
on `problem_statement` **plus the patch**; we only have the prompt, so our ceiling
is lower.

| dataset | size | label | licence | verdict |
|---|---|---|---|---|
| [`nvidia/Nemotron-SFT-OpenCode-v1`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1) | 459 k, 6 agentic subsets | **`complexity_level` beginner/intermediate/advanced** + `question_category`, `agent_prompt`, `enabled_tools` | **CC-BY-4.0** | **The only permissively-licensed off-the-shelf effort label on agentic coding prompts.** Start here. |
| [`nvidia/OpenCodeReasoning-2`](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) | 2.57 M (34,799 uniq q ⇒ **~74 traces/q**) | **`pass_rate`** ∈ [0,1] (`-1` sentinel) + `difficulty` | **CC-BY-4.0** | Best clean-licensed continuous label, and the repeated-sampling structure §2b demands |
| [`mlfoundations-dev/base_get_difficulty_seed_code`](https://huggingface.co/datasets/mlfoundations-dev/base_get_difficulty_seed_code) | **979,935**, all code | **`difficulty` int 1–10** + free-text reasoning | ⚠️ **none declared** | **Best-shaped set found** — labelled *from the prompt alone*, matching our inference condition, no solution leakage. Histogram `[109603, 242265, 145440, 117954, 104236, 115073, 49592, 58973, 27452, 9347]`, mean 3.95. **Resolve the licence before it touches anything shipped.** |
| [`Skywork/Skywork-OR1-RL-Data`](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) | 14,057 code | **`extra_info.model_difficulty`** = failed rollouts out of 16, for **three model scales (1.5B/7B/32B)** | none declared | **Closest published proxy to our 4 rungs**: solved-by-1.5B → low, only-by-32B → high, none → xhigh. Small. Pin the commit. |
| [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces) | 9,556 / 468 | **`rating`** Elo 800–3500 | CC-BY-4.0 | Cleanest ordinal. `codeforces-cots` has **no** rating — join on `id`. |
| [`codeparrot/apps`](https://huggingface.co/datasets/codeparrot/apps) | 10,000 | introductory 3,639 / interview 5,000 / competition 1,361 | **MIT** | Clean anchor; train/test difficulty distributions differ deliberately — re-split. |
| [`BAAI/TACO`](https://huggingface.co/datasets/BAAI/TACO) | 26,443 | 5 tiers EASY→VERY_HARD | Apache-2.0 | Viewer disabled; mirror `likaixin/TACO-verified` (MIT, 12,898). |
| [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | **5,000,000** | `average_test_score` 0–1 | CC-BY-4.0 | Largest permissive pass-rate signal. |
| `deepmind/code_contests` | 13,328 | ⚠️ `difficulty` ClassLabel's **29 values A–V are Codeforces problem *indices*, not difficulty**; use `cf_rating` and **drop the 0 sentinels** | CC-BY-4.0 | Or just use `open-r1/codeforces`. |
| `EleutherAI/hendrycks_math` (12,500, **MIT**, `level`), `HuggingFaceH4/MATH-500` (keeps `level` int 1–5 + `subject`) | | MATH levels 1–5 | | Auxiliary at best — wrong domain. Original `hendrycks/competition_math` is **disabled**. |
| [`danfperam/castillo`](https://huggingface.co/datasets/danfperam/castillo) | **280,000** | 10 samples per ⟨prompt, model⟩ incl. **MBPP** | CC-BY-4.0 | Not a difficulty label — the **noise-floor measurement set**. |
| `bigcode/bigcodebench-hard` | 148 unique tasks | binary; its `score` is SO-query similarity, **not difficulty** | — | **Skip.** |
| `PrimeIntellect/verifiable-coding-problems`, `agentica-org/DeepCoder-Preview-Dataset` | | **difficulty stripped during curation** | | Skip as-is. |

**Licence-blocked — research probes only, never in a shipped model:**
`a-m-team/AM-Thinking-v1-Distilled` (1.89 M, ideal `think_content`/`answer_content`
schema, 17.1 % code, **no licence field, README says research-only**),
`a-m-team/AM-DeepSeek-R1-Distilled-1.4M` (**CC-BY-NC-4.0**),
`ServiceNow-AI/R1-Distill-SFT` (**CC-BY-NC-SA-4.0**),
`KodCode/KodCode-V1` (484 k with `gpt_pass_percentage` over 10 trials —
excellent labels, **CC-BY-NC-4.0**),
`Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B` (249,922; the
**richest difficulty metadata found** — `difficulty` 5 classes,
`difficulty_generator`, `input_quality`, `task_category`, `intent`, `knowledge` —
**CC-BY-NC-4.0 + Llama community licences**).

**Reasoning-trace corpora with usable length targets and permissive licences:**
`open-r1/Mixture-of-Thoughts` (349,317 rows, **`num_tokens` int64**, `code` config
83,070), `nvidia/OpenCodeReasoning` (735,255; ~26 traces/question; CC-BY-4.0),
`open-thoughts/OpenThoughts3-1.2M` (75 k uniq q × **16 samples**, Apache-2.0,
250 k code), `nvidia/Nemotron-SFT-Competitive-Programming-v2` (844,935,
**`reasoning_content` as its own field**, CC-BY-4.0),
`PrimeIntellect/SYNTHETIC-1` (1.99 M, Apache-2.0, 144 k algo + **70 k real SWE**),
`open-r1/codeforces-cots` (up to 5 traces/problem, CC-BY-4.0),
`open-r1/OpenR1-Math-220k` (1–6 generations/problem, Apache-2.0).

**`nvidia/Llama-Nemotron-Post-Training-Dataset` reasoning on/off — yes, but not
for code.** Per-split counts from datasets-server: `safety` **15,713 on /
15,713 off** (perfectly paired); `chat` 8,574 on / 31,218 off with 8,574 genuinely
paired; **`code`, `math`, `science` are reasoning-ON only in every sampled row.**
~24 k paired prompts total, **none of them code.** The successor
`nvidia/Nemotron-Post-Training-Dataset-v1` (25.7 M rows, code 1.90 M, ungated,
CC-BY-4.0) is the better one.

**Cross-cutting warning:** trace length is *generator-specific*. QwQ-32B, R1,
Gemini Flash Thinking and AM-Thinking have very different verbosity priors, and
none of them is Qwen3.8-27B. Pool them without a generator covariate and you train
a generator classifier wearing an effort-predictor costume.

### 7f. Preference / routing data
`routellm/gpt4_dataset` — **119,101 rows, Apache-2.0**, columns
`prompt, source, gpt4_response, mixtral_response, mixtral_score (int 1–5)`; the
score is a weak-vs-strong quality gap, usable as a coarse difficulty proxy.
`lmsys/lmsys-chat-1m` is **gated**, 2023-era, avg 214.5 output / 69.5 prompt
tokens — badly stale for agentic coding.
[`AmirMohseni/reasoning-router-data-v2`](https://huggingface.co/datasets/AmirMohseni/reasoning-router-data-v2)
— ~70 k samples labelled by scoring think vs no-think outputs with
Skywork-Reward-V2-Llama-3.1-8B; binary, non-agentic, **40 downloads**, no licence
tag; the **label-generation pipeline generalises to 4 rungs**.
`PromptComplexityEstimator`'s training mix (`BatsResearch/Cross-Difficulty`,
`furonghuang-lab/Easy2Hard-Bench`, `hendrycks_math`, `ai2_arc`, `race`, `anli`) is
the most reusable public *cross-model difficulty* set.

### 7g. Recommended data plan

1. **Measure the floor first, on our own traffic.** Replay ~500 real steps at
   K=8 per rung; compute within-prompt σ vs between-prompt σ. If within-prompt σ
   dominates, stop and retarget to a distributional/quantile head. Free
   comparators with repeated sampling: `danfperam/castillo` (10/prompt, has MBPP),
   `nvidia/OpenCodeReasoning` (~26/question), `OpenThoughts3-1.2M` (16/question).
2. **Label = lowest rung that still yields a correct action**, K=3–8 per rung,
   functional-equivalence check against the max-effort trajectory (ARES).
3. **Pretrain on cheap proxies**, in this order:
   `nvidia/Nemotron-SFT-OpenCode-v1` `complexity_level` → `OpenCodeReasoning-2`
   `pass_rate` → `open-r1/codeforces` `rating` → `codeparrot/apps` / `BAAI/TACO`.
4. **Domain-adapt on real agent steps** from `nvidia/Open-SWE-Traces` and
   `nebius/SWE-rebench-openhands-trajectories`.
5. **Evaluate on TwinRouterBench** (§1.2) — it is the only held-out, four-tier,
   execution-verified agentic-coding-step set in existence, and its 336
   SWE-bench-Verified step rows are the closest public proxy for our traffic.
   Report against its published rule-router range (4.2–62.5 % on high steps) and
   its trained-logistic result (−53.1 % cost at matched resolution).
6. **Calibrate on SWE-bench Verified** using the per-annotator soft labels, and
   **merge `>4 hours` into `1-4 hours`** — 3 examples in the 500, and SWE-smith
   already reached that conclusion. That leaves **three** usable rungs from this
   source, not four.
7. **Add a cheap escape hatch.** Prompt-only tops out near r ≈ 0.44 and the served
   model's own hidden states buy ≈ +0.3 — consider reading an early-layer state
   after ~50 generated tokens as a second-stage correction. Even then AUC on the
   AIME non-convergence case was only 0.615, so pair it with a hard token cap
   rather than trusting the predictor.

---

## 8. Multimodal requests (text + screenshots / computer-use turns)

Agentic turns in this loop are **predominantly long text with occasional
images**. The right architecture is therefore *text-primary with a cheap vision
side-channel*, not a VLM. Two ways to get there: one model with a shared space,
or two towers whose embeddings you concatenate.

### 8a. Ranked multimodal options

**1. AIST-87M (TriEmbed) — already on this box, and it wins on every axis.**
Read directly from the GGUF header at
`/shared/cortext.cpp/models/AIST-87M-GGUF/` (reassembled from its three chunks):

```
general.architecture          = triembed
general.description           = TriEmbed compact multimodal embedding model
triembed.text_model           = MongoDB/mdbr-leaf-ir
triembed.image_encoder_name   = mobilenetv4_conv_medium.e180_r384_in12k
triembed.audio_encoder_name   = efficientat_mn20_as_native_merged
triembed.matryoshka_dims      = [1280, 768, 512, 256, 128, 64, 32]
triembed.embed_dim            = 1280      (shared)
triembed.text_encoder_dim     = 768
triembed.image_encoder_dim    = 1280
triembed.original_params      = 87186755
```

**87,186,755 params for text + image + audio combined**, one shared 1280-d
retrieval space, q8_0 GGUF = 141,491,936 bytes, engine
[augmem/cortext.cpp](https://github.com/augmem/cortext.cpp) is **Apache-2.0** with
a C ABI (`EmbedText` / `EmbedImage` are embedding-only calls that skip the memory
store). Component licences check out: text tower
[`MongoDB/mdbr-leaf-ir`](https://huggingface.co/MongoDB/mdbr-leaf-ir) is
**22,565,376 params, Apache-2.0, 584 k downloads/30 d**; vision tower
[`timm/mobilenetv4_conv_medium.e180_r384_in12k`](https://huggingface.co/timm/mobilenetv4_conv_medium.e180_r384_in12k)
is 23.6 M, Apache-2.0.

**Why it is the right answer here, and not just the convenient one:**
- **One model, one space.** No tower alignment, no separate projection to train,
  no second runtime. Text and screenshot land in the same 1280-d vector, so a
  single linear/MLP head over `[e_text ; e_image]` — or even over their sum — is
  a coherent effort classifier.
- **Matryoshka dims down to 32.** Truncate to 128 or 256 before the head and the
  classifier is a few thousand parameters. That is the cheapest possible
  multi-level output, and it makes the head trivially retrainable as the ladder
  is retuned.
- **It is C++ with a stable C ABI and GGUF quantisation**, i.e. it fits the
  serving path without dragging a PyTorch stack into the API server process.
- **87 M total is under budget** even counting the audio tower we do not need.

**Caveats to test, not to assume.** No published CPU latency for this model —
**measure it**, on a real screenshot and a real 8k-token step, before committing
(the same measurement §4 demands of the text-only candidates). The text tower is
a retrieval encoder with a retrieval-length context, so the §4 truncation problem
does not go away. And see §8c on whether *any* of these vision towers can read a
screenshot at all.

**2. `nomic-embed-vision-v1.5` + `nomic-embed-text-v1.5` — best Apache-2.0
two-tower pairing.** 92,946,688 + 136,731,648 params (both verified), both
**Apache-2.0**, and crucially **aligned in the same 768-d space** — the text
tower's embeddings are directly comparable to the vision tower's, so you can
concatenate or average without training a projection. 62,955 and **16.4 M**
downloads/30 d respectively: by far the best-supported option in this section.
Total ~230 M is over budget, but the vision tower alone (93 M) can be bolted onto
a cheaper text tower *only if* you retrain a projection — which loses the whole
advantage.

**3. MobileCLIP-S0 — smallest and fastest, but the licence is a blocker.**
Verified from the [model card](https://huggingface.co/apple/MobileCLIP-S0):

| variant | img params | text params | latency img | latency text | IN-1k 0-shot |
|---|---|---|---|---|---|
| **S0** | **11.4 M** | **42.4 M** | **1.5 ms** | **1.6 ms** | 67.8 % |
| S1 | 21.5 M | 63.4 M | 2.5 ms | 3.3 ms | 72.6 % |
| S2 | 35.7 M | 63.4 M | 3.6 ms | 3.3 ms | 74.4 % |
| B | 86.3 M | 63.4 M | 10.4 ms | 3.3 ms | 76.8 % |

**S0 is 53.8 M total** — the smallest genuinely-joint text+image space available.
The card does **not** state the latency hardware (the paper measures on an iPhone
Neural Engine, so these are **not** x86 CPU numbers — do not quote them as such).
**Licence is `apple-amlr`** (Apple ML Research), which is **not** a standard
permissive licence; it needs legal review before anything ships. Downloads are
low (199/30 d for `apple/MobileCLIP-S0`; the OpenCLIP repackagings
`apple/MobileCLIP-S1-OpenCLIP` 11 k and `-S2-OpenCLIP` 20 k are the ones people
actually use). ONNX/transformers.js ports exist at `Xenova/mobileclip_s0`.

**4. `jinaai/jina-clip-v1`** — 222,672,128 params, **Apache-2.0**, joint 768-d
projection (verified `config.json`: `projection_dim: 768`, vision ViT-B/16 @224,
text = jina-bert-flash). 86 k downloads/30 d. Notable because its **text tower is
a real long-context text encoder**, not a 77-token CLIP stub — the one CLIP-family
model that does not throw away the text side. **`jina-clip-v2` is CC-BY-NC-4.0 —
blocked.**

**5. SigLIP / SigLIP2 — Apache-2.0, best quality, but there is no small one.**
`google/siglip-base-patch16-224` = **203,155,970 params**;
`google/siglip2-base-patch16-224` = **375,187,970**;
`google/siglip2-so400m-patch14-384` = **1,136,008,498**. Verified: **no `small`
or `tiny` SigLIP variant exists** in the `google/` org. Enormous adoption
(1.2–1.4 M downloads/30 d each). Use only if quality turns out to matter more
than latency, which §4 suggests it will not.

**6. `openai/clip-vit-base-patch32`** (~151 M, 20.7 M downloads/30 d) — the
default everyone reaches for. Its **77-token text limit** makes it useless as the
text half here; it is a vision-tower-only option, and MobileCLIP-S0 beats it on
both size and speed. **TinyCLIP (`wkcn/TinyCLIP-*`) could not be verified — the
HF API returned an auth error for those IDs. Treat as unavailable until checked.**

### 8b. Which text-only options pair cheaply with which vision tower

A joint space is only *required* if you want to compare text and image vectors.
For a **classifier** you do not — you just need both as features. So:

| text side | vision side | how they join | cost |
|---|---|---|---|
| **AIST text tower** | **AIST image tower** | already the same space | **free — one model, one call** |
| `nomic-embed-text-v1.5` | `nomic-embed-vision-v1.5` | already aligned, same 768-d | free, but 230 M combined |
| **potion-base-32M** (§1.3) | **AIST image tower** or MobileCLIP-S0 image tower | **concatenate `[e_text(256) ; e_img(256)]` and fit one head** — no alignment needed | ~33 M + ~12–24 M, and the text side has no context limit |
| `jina-embeddings-v2-small-en` (33 M, 8192 ctx) | MobileCLIP-S0 image tower (11.4 M) | concatenate | **~45 M total, the cheapest 8k-text + image combination** — but MobileCLIP's licence applies |
| any 512-token encoder | any | concatenate | works, but §4's truncation problem dominates |

**The concatenate-and-fit-one-head trick is the important point:** effort
classification does not need a shared embedding space, so the "which CLIP" debate
mostly evaporates. Pick the cheapest text tower that handles our context and the
cheapest vision tower that handles a screenshot, and let a single head learn the
joint decision. AIST is recommended first only because it removes the integration
work entirely, not because a shared space is technically necessary.

### 8c. The skeptical caveat that matters most

**No published evidence exists that any of these vision towers predicts task
difficulty from a screenshot**, and there is good reason to doubt it:

- CLIP/SigLIP/MobileCLIP/MobileNetV4 are trained on **natural images with
  captions**. An IDE window, a terminal, or a browser DOM render is far outside
  that distribution.
- **Resolution kills the signal.** AIST's vision tower runs at **384 px**;
  MobileCLIP-S0 and SigLIP-base at **224 px**. A 1080p screenshot downsampled to
  224–384 px has **no legible text left**. Whatever makes a computer-use step
  hard — an error dialog's wording, a stack trace, a diff — is exactly what is
  destroyed.
- Therefore, for computer-use turns the difficulty signal almost certainly lives
  in the **text** channel (accessibility tree, OCR output, tool result, prior
  turns), not in the pixels. The image embedding is more plausibly useful as a
  coarse *context* feature ("this is a terminal" vs "this is a browser" vs "this
  is a rendering diff") than as a difficulty feature.

**Recommended test, and it is cheap:** in the §1 probe, fit the head twice — once
on text features alone, once on `[text ; image]` — over the multimodal subset of
logged steps. **If the image features add nothing, drop the vision tower
entirely** and keep a text-only classifier that simply ignores the attachments.
Given the resolution argument, that is the outcome to expect, and finding it out
costs one extra column in an experiment we are running anyway.

---

## 9. What is marketing and what is real

| claim | status |
|---|---|
| **DART** training-free effort routing, Qwen3-32B HumanEval 72.6→95.1 at −63 % thinking | **paper + real MIT code**; the strongest concrete result in the survey |
| ARES per-step effort router, −52.7 % tokens | paper real, **repo is a 104-byte README stub** |
| vLLM Semantic Router "reasoning mode" | **real, Apache-2.0, actively developed, genuinely multi-level** — but the fast path is GPU-colocated (**4,918 ms on CPU**), it never touches `thinking_token_budget`, and its weights are licence-untagged |
| RouteLLM's tiny 0.8 MB `mf` router | **calls the OpenAI embeddings API per request**; `sw_ranking` also refits an Elo logistic regression per request; repo dead since 2024-08-10 |
| Arch-Router-1.5B | real weights, but the licence changed 2026-04-02 to require a **separate commercial grant from DigitalOcean**, and a 1.5 B decoder per request is a second model, not a router |
| NVIDIA `llm-router` blueprint | **deprecated**; successor Switchyard is *"pre-alpha, not for production"* and **ships no classifier weights**. `nvidia/prompt-router*` **does not exist** |
| NVIDIA classifier "reasoning" dimension | **real but binary**; the graded signal is `prompt_complexity_score`, whose formula weights *creativity* at 0.35 — the wrong quantity here. Trained on 4,024 prompts, **185 of them code** |
| Its reported 0.997 accuracy on the `reasoning` head | n-fold CV on the same 4k annotated pool — measures **annotator consistency, not generalisation** |
| LFM2.5 Prompt-Router "zero-shot lanes" | real weights, best adoption on-target (4,391 DL/30 d); **no published latency, training data, or benchmark**; lanes are topical, not effort-graded; $10 M revenue-capped licence |
| `ThinkingBudgetRouter` / `bge-small-prompt-difficulty` | real Apache-2.0/MIT weights at the right size, **zero published evaluation**; 4 and 12 downloads |
| Not Diamond / Martian / Unify / OpenRouter auto / Requesty / LiteLLM / Portkey | **no usable public router weights.** Not Diamond's 2024 artifacts are obsolete; LiteLLM's "complexity router" defaults to a **hand-written heuristic scorer** |
| Vendor "adaptive thinking" (Qwen Auto, Kimi adaptive, GLM dynamic) | **undisclosed or unverified**; the Claude Opus 4.6 System Card contains "router" **zero times** in 213 pages |
| [zylos.ai adaptive reasoning depth](https://zylos.ai/research/2026-04-13-adaptive-reasoning-depth-ai-agent-systems/) | **marketing blog**; cites arXiv IDs without links, no reproducible detail |
| "SWE-bench Verified has 4 difficulty classes so we have 4 training rungs" | **false in practice** — the top class has **3 of 500 examples**, and SWE-smith's own rater collapses `>4 hours` into `1-4 hours`. Three usable rungs, not four |
| "trace length is a free label" | **actively harmful** — length correlates **−0.72** with correctness; regressing on it rewards flailing |
| "a difficulty classifier will cut agentic tokens a lot" | a **gold human difficulty label** reaches only **τ = 0.32** vs real agentic spend; HRBench measured external routing at **−13 % tokens for −3.5 pp accuracy, last of three strategies** |
| MobileCLIP-S0's "1.5 ms" image latency | **not an x86 CPU number** — the card omits the hardware; the paper measures an iPhone Neural Engine |
| `apple-amlr` licence on MobileCLIP | **not a standard permissive licence**; needs legal review before shipping |
| `mlfoundations-dev/base_get_difficulty_seed_code` (best-shaped 980 k code difficulty set) | **no declared licence** — research probe only until resolved |
| TALE's "token-budget estimator" ([arXiv 2412.18547](https://arxiv.org/abs/2412.18547)) | **not a separate model.** TALE-EP is a zero-shot *prompt to the same big LLM*; the binary search is offline-only (~354 A100-min for 7.5k GSM8K); the LLaMA-3-8B regression estimator appears in arXiv v2 with no results and was dropped. **Official repo `GeniusHTX/TALE` is 404 — the account is deleted.** |
| arXiv 2604.14853's XGBoost router | paper real and its baseline table is the most honest in the survey — but **the claimed repo `zhiyuanZhai20/AdaCompute-LLM` is 404.** Paper-only |
| "a frontier LLM can just judge how hard the step is" | **measured false.** Claude Opus 4.6 prompted as a router flagged **7 of 147** verified-high SWE steps and failed all 40 trajectories (TwinRouterBench) |
| "learned routers beat simple baselines" | **not established.** Across 21 routers × 6 encoders, **kNN over embeddings is top-2 everywhere and the top five are within 0.22 pp** (Routing Plateau). On 415k agent outcomes a **benchmark-identity baseline (ρ .519) beat the learned ridge (ρ .225 OOD)** |
| ThinkSwitcher | **paper-only, no repo found**; its own ModernBERT text baseline *loses* to its hidden-state version, and a follow-up reports it **underperforming always-CoT by 2.5 pts** |
| LHRM's "hybrid accuracy" | 1.5 B variant scores **54.4 %** — approximately a coin flip — **with no random baseline reported** |
| CLIP-family embeddings of a screenshot as a difficulty feature | **untested and probably empty** — 224–384 px destroys every legible character (§8c) |

---

## 10. Known gaps in this survey

Stated so the next pass knows where to look, rather than mistaking absence for
a negative result:

- **Now resolved** (the budget-predictor lane returned after the first draft):
  TALE, Self-Budgeter, SABER, Plan-and-Budget, ThinkLess, AutoL2S, DEER, Dynasor,
  DeepConf, L1/LCPO and the QPP literature are all covered in §3b. Summary:
  **none is a drop-in prompt-only classifier.** ~70 % fine-tune the served model
  [TUNES-TARGET]; the rest need target-model prefill or drafts [NEEDS-ROLLOUT].
  On QPP specifically: pre-retrieval predictors reach ρ ≈ 0.40–0.51, post-retrieval
  ρ ≈ 0.59–0.76 but need a ranked list; **no usable pretrained QPP checkpoint
  exists** (HF yields only `sadjadeb/post-nn-qpp`, empty card, 9 downloads).
  Borrow QPP's *methodology* — Pearson + Kendall + Spearman reported together,
  and sMARE per-query error distributions — not its models.
- **Not verified:** TinyCLIP (`wkcn/TinyCLIP-*` returned an HF API auth error),
  AIST-87M's CPU latency, MobileCLIP's x86 latency, and `BAAI/TACO`'s viewer
  counts.
- **Snippet-sourced, not read in the source:** the three rows marked as such in
  §2, and the reasoning-vs-non-reasoning variance multipliers.
- **Licence unresolved:** `mlfoundations-dev/base_get_difficulty_seed_code`,
  `princeton-nlp/SWE-bench_Verified`, `Skywork/Skywork-OR1-RL-Data`,
  `AmirMohseni/*`, and the `llm-semantic-router/*` classifier weights all lack a
  declared licence.
- **Never measured by anyone, including us:** whether prompt-only effort
  prediction beats a constant baseline on *agentic coding steps*. Every number in
  §2a is on chat, math, or standalone competitive-programming prompts. That is the
  gap this project would be filling, and step 1 of §1's plan is the cheapest way
  to find out whether it is fillable.
