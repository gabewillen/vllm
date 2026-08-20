# Qwen's own prefill hidden states as a free effort signal

Status: **measured on the trial 4xL4 server, 2026-08-19.** Companion to
[`effort-router-prototype.md`](effort-router-prototype.md) (small encoders, CPU)
and [`effort-classifier-survey.md`](effort-classifier-survey.md) (the field).
Same dataset, same targets, same split, same metrics — see
`/shared/vllm/work/router-proto/dataset/README.md`.

The question here is narrower and, if it works, much cheaper: **the model is
already going to prefill this prompt. Can the hidden state it produces on the
way tell us how hard the step is, for free, with no second encoder and no
trained head?**

"For free" is the whole point:

- no extra model, no extra memory, no extra CPU on the request path;
- **multimodal by construction** — a screenshot in the prompt is already in the
  hidden state, so nothing about images has to be bolted on (the §8 problem in
  the survey disappears, though this dataset is text-only so that is an argument,
  not a measurement);
- the constraint from the user stands: **label-free and algorithmic**. The
  shippable signals below are geometric statistics over the model's own vectors.
  A logistic/ridge probe appears only as a **diagnostic upper bound, explicitly
  not for shipping**.

## 0. What was built, and where

| path | what |
|---|---|
| `work/router-proto/hidden/hidden_capture.py` | the capture module (installed into the venv source as `vllm/v1/worker/gpu/hidden_capture.py`) |
| `work/router-proto/hidden/hidden-capture.patch` | the same, as a diff against `/shared/vllm-dflash2` — ready to become a numbered `serve-configs/patches/00NN-*.patch` |
| `work/router-proto/hidden/replay.py` | replays the shared dataset through the server, prefill only |
| `work/router-proto/hidden/signals.py` | the label-free signals (surprisal, streaming/session quantile rank, novelty, kNN) |
| `work/router-proto/hidden/load_hidden.py` | joins shards → dataset rows |
| `work/router-proto/hidden/analyze_hidden.py` | scoring; writes `results_dynamicv2/metrics.{json,csv}`, `deep_rate.json`, `deep_rate_knn16.json`, `sigmoid_sensitivity.json`, `controls.json`, `policy.json`, `signal_values.npz` |
| `work/router-proto/hidden/summarize.py` | renders the tables below |
| `work/router-proto/hidden/controls.py` | the "is it just prompt length" controls, k sweep, memory-size sweep |
| `work/router-proto/hidden/policy.py` | the asymmetric quantile map and its under-routing table |
| `work/router-proto/hidden/twin_replay.py`, `twin_analyze.py` | the TwinRouterBench external yardstick |
| `work/router-proto/hidden/results_dynamicv2/`, `results_twin/` | every number below, as JSON |

The patch stays applied to the venv source and is **env-gated**: with
`VLLM_EFFORT_HIDDEN_CAPTURE` unset nothing is constructed and nothing runs. One
line is not gated — `execute_model` now unpacks the model output by
`isinstance(..., tuple)` instead of asserting on
`use_aux_hidden_state_outputs`, and sets that flag from what the model actually
returned. For every configuration here that is behaviour-preserving (the flag and
the output type already agree); it exists so a model that ignores an
auxiliary-output request cannot crash the engine at startup.

Server: the trial arm on `:8013` from `/shared/vllm/.venv-dflash2` (P6b build of
`qwen3.8-27B`, editable source at `/shared/vllm-dflash2`), config
`/tmp/dflash2-arms/B_mtp.yaml` — the MTP latency profile, V2 model runner, TP4,
FP8, `NCCL_P2P_LEVEL=SYS`, `TRITON_ATTN`. Production (`vllm-qwen38`, `:8012`)
stayed stopped throughout. The server was restored to that exact config with the
capture env unset when the measurement finished.

## 1. The capture mechanism

### Where it sits

`GPUModelRunner.sample_tokens` (V2 runner, `vllm/v1/worker/gpu/model_runner.py`),
immediately after `pcp.maybe_restore_pcp_for_sampling` and before `self.sample`.
That is the one place where the *full-width* hidden state and the batch's request
identity are both in hand:

```python
if self.hidden_capture is not None:
    self.hidden_capture.observe(hidden_states, input_batch, aux_hidden_states)
    self.hidden_capture.drop(finished_req_ids)
```

`hidden_states` there is `[num_tokens, 5120]` — the tensor `compute_logits` feeds
to `lm_head`. Under TP4 it is **already full width on every rank** (the final
row-parallel all-reduce has happened), so no gather is needed and only
`get_tp_group().rank_in_group == 0` writes.

Per request the capture keeps a running fp32 sum over the prompt rows scheduled
in each chunk (`query_start_loc_np[i] : + min(num_scheduled, prefill_len -
num_computed_prefill)`), and emits one record on the step where
`num_computed_prefill + n == prefill_len` — i.e. exactly when prefill finishes
and the last prompt token's row is present:

- `last_final` — the last-prompt-token final hidden state (the vector `lm_head`
  consumes);
- `mean_final` — mean over the prompt tokens this engine actually ran.

### Joining it to the HTTP request

The existing effort telemetry sink (`VLLM_EFFORT_TELEMETRY`,
`vllm/v1/sample/effort_signals.EffortTelemetrySink`) writes the engine `req_id`,
which the other track found is **not joinable to the traces** at all. The fix is
one line on the client side: vLLM's `_base_request_id`
(`vllm/entrypoints/serve/engine/serving.py:117`) takes the `X-Request-Id` header,
so the engine id becomes `chatcmpl-<your id>-<8 hex>`. Every record here is keyed
that way and joins 1:1. **Any future effort telemetry should do the same**; it
costs nothing and removes the whole "not reliably linkable" gap.

### Cost

Per prefill chunk: one fp32 reduction over `[n_tokens, 5120]`. Per finished
prefill: a `[2, 5120]` fp16 device→host copy (20 KB). Against a 27 B forward over
the same tokens this is ~7 orders of magnitude of arithmetic apart, and the
observed prompt throughput during the replay (2 000–2 700 tok/s, `Avg prompt
throughput` in the engine log) sits in the same band this config already reports.
Storage is 20 KB/request; the whole 3 258-request replay is 65 MB.

### What did *not* work: the mid-layer (layer 32) pooling

Two mechanisms were tried on this build and **both are inert**, which is worth
recording because it constrains any future "read layer L" design:

1. `layers[32].register_forward_hook(...)` — the hook never fires. vLLM's
   `support_torch_compile` dispatches the compiled artifact directly, so
   Python-level `nn.Module` hooks inside the compiled region are never executed.
   Captured vectors came back exactly zero.
2. Setting `language_model.model.aux_hidden_state_layers = (32,)` — the model's
   own EAGLE3 mechanism, which returns the layer-32 residual stream through the
   forward's *return value* rather than a hook. The attribute is set correctly
   (verified in the log: `aux layer 32 of 64 on language_model.model`), but the
   compiled forward still returns a bare tensor, so `execute_model`'s
   `assert isinstance(model_output, tuple)` fires during `profile_run` — with and
   without `VLLM_DISABLE_COMPILE_CACHE=1`.

So on this build the *only* free hidden state is the final one. Getting layer 32
needs either `--enforce-eager` (a different serving configuration, not the one
being characterised) or a small change inside `Qwen3_5Model.forward` itself. The
mid-layer row is therefore **absent** from the tables below, not measured-and-bad.
Note that the survey's one hidden-state datapoint (arXiv 2510.07571, r = 0.742
from a 2-layer MLP on layer-16 states) is a *trained* probe, which the shipping
constraint rules out anyway.

## 2. Replay method

Every dataset request was replayed at **its own recorded effort level** (so the
rendered prompt matches the trace: `low`/`medium`/`xhigh` verbatim,
`dynamic`/`dynamic-v2` → `reasoning_effort: "dynamic"`, which patch 0009 rewrites
into template-`medium` plus the low sentence on the last user turn), with the
harness's tool schemas attached and `max_tokens=1`. Nothing thinks; only the
prefill matters.

Requests of one run are issued **in step order on one worker**, eight runs in
parallel, so vLLM's prefix cache carries the shared conversation prefix — the
observed hit rate is 45–65 %, i.e. a step typically recomputes only its new tool
output. Two consequences the reader must keep:

- `mean_final` is a mean over the tokens **this step actually recomputed**, not
  over the whole prompt. `n_pooled` is recorded next to `prompt_tokens` so this is
  never ambiguous. For a per-step router this is arguably the more relevant
  quantity ("what is new since the last step"), but it is *not* the same thing as
  full-prompt mean pooling, and it is labelled `mean_final` throughout.
- `cum_mean_final` reconstructs an approximate whole-prompt mean by summing
  `mean_final * n_pooled` along the session. It over-weights regions that were
  evicted and recomputed, so it is an approximation, reported as such.
  A `--cache-salt` mode exists in `replay.py` to defeat the prefix cache entirely
  and get the exact full-prompt mean; at ~2 500 tok/s and 52 M dataset prompt
  tokens that is a multi-hour run and was not done here.

**Scope of the replay.** All **791 `dynamic-v2`** requests were replayed and all
791 joined to a captured vector, 0 errors — that is the whole labelled
evaluation, because `dynamic-v2` is the only column carrying reasoning tokens.
The four fixed-effort columns were replayed only partially (1 447 of 2 467) and
are **not used** in any table here: they add no think-length target, and the one
target they would enlarge (`next_step_trouble`) is at chance on the dynamic-v2
rows already. `next_step_trouble` below is therefore over the 791 dynamic-v2
requests (70 positives), not the 3 258 the other track uses.

**Self-check on the prompt reconstruction:** the server's reported
`prompt_tokens` equals the dataset's recorded `prompt_tokens` for every replayed
request. The prompts sent are the prompts the traces saw.

**Metric protocol reproduced.** Running this track's metric code over the free
controls returns the other track's published numbers exactly — prompt length
AUC 0.7286 [0.690, 0.767], Spearman 0.4945, within-run 0.581, trouble AUC 0.4785
— so the two tracks' tables are directly comparable.

## 3. The label-free signals

Two families, both computed from the pooled vectors alone. Neither is trained;
neither has a fitted parameter.

**(1) Within-session surprisal** — cortext's prediction error against an EMA of
the same conversation's earlier requests:

```
err_t = 1 - cos(h_t, EMA_t)          EMA_t <- (1-a) EMA_{t-1} + a h_t,  a in {0.2, 0.5, 1.0}
```

reported three ways: raw `err`, `sigmoid((err - err_ref)/k)` (cortext's mapping),
and — the shippable form — the **percentile rank of `err` in a streaming digest**,
either the session's own history or the server-wide stream. Because sigmoid is
monotone, `err_ref` and `k` cannot move AUC or Spearman at all; what they move is
the *fraction of requests a threshold escalates*, which is reported separately in
`results_dynamicv2/sigmoid_sensitivity.json` (e.g. at the median `err_ref = 0.101`: `k=0.01`
escalates 41.7 % of requests above 0.8, `k=0.05` 15.0 %, `k=0.1` 3.8 %; the
`>0.5` rate is fixed by `err_ref` alone at 50 / 30 / 10 % for the 50th/70th/90th
percentile). The quantile-rank form has no such constants at all, which is why it
is the one to ship.

**(2) Novelty and kNN against an online memory** — a memory of pooled vectors,
each with the reasoning length that request actually needed:

```
novelty_t = (1 - max_j cos(h_t, m_j)) / 2
knn_t     = softmax(cos/T)-weighted mean of log1p(reasoning_tokens) over the k nearest
```

evaluated **leave-one-task-out**: every memory entry from the task under test is
masked, so the estimate always answers "I have never seen this task before".

### Results — long think (top quartile of reasoning tokens at natural close)

689 natural closes, 172 positives. Chance = 0.500. The bar set by the other track
is the free control **prompt length: 0.729 raw / 0.581 within-run**.

| signal (label-free unless noted) | pooling | AUC | 95 % CI | AUC within-run |
|---|---|---:|---|---:|
| **kNN-16, leave-one-task-out** | `last_final` | **0.850** | [0.818, 0.877] | **0.762** |
| kNN-8, leave-one-task-out | `last_final` | 0.839 | [0.805, 0.867] | 0.741 |
| kNN-8, **online** memory, 512-entry ring | `last_final` | 0.842 | [0.811, 0.872] | — |
| kNN-8, online memory, 128-entry ring | `last_final` | 0.808 | [0.772, 0.843] | — |
| *(diagnostic, not for shipping)* logistic probe | `last_final` | *0.866* | *[0.833, 0.896]* | *0.778* |
| — free control: prompt length | — | 0.729 | [0.690, 0.767] | 0.581 |
| — free control: message count | — | 0.708 | [0.668, 0.750] | 0.581 |
| — free control: step index | — | 0.703 | [0.663, 0.745] | 0.581 |
| novelty (LOTO) | `mean_final` | 0.661 | [0.615, 0.708] | 0.570 |
| novelty (LOTO) | `last_final` | 0.655 | [0.612, 0.702] | 0.573 |
| kNN-8 (LOTO) | `cum_mean_final` | 0.644 | [0.602, 0.685] | 0.608 |
| kNN-8 (LOTO) | `mean_final` | 0.622 | [0.575, 0.672] | 0.557 |
| session surprisal, **inverted** (a=0.2, raw) | `last_final` | 0.716 | [0.674, 0.759] | 0.656 |
| session surprisal (a=0.2, raw, as defined) | `last_final` | 0.284 | [0.241, 0.326] | 0.344 |
| session surprisal, session-quantile rank (a=1.0) | `mean_final` | 0.552 | [0.504, 0.599] | 0.518 |
| — control: random | — | 0.542 | [0.489, 0.593] | 0.532 |

### Results — reasoning tokens at natural close (Spearman)

| signal | pooling | Spearman | within-run | partial, given prompt length |
|---|---|---:|---:|---:|
| **kNN-32, LOTO** | `last_final` | **0.689** | 0.595 | 0.565 |
| kNN-16, LOTO | `last_final` | 0.685 | 0.587 | 0.577 |
| kNN-8, LOTO | `last_final` | 0.675 | 0.573 | 0.572 |
| *(diagnostic)* ridge probe | `last_final` | *0.657* | *0.585* | — |
| — free control: prompt length | — | 0.494 | 0.307 | — |
| novelty (LOTO) | `last_final` | 0.345 | 0.197 | — |
| session surprisal, inverted (a=0.2) | `last_final` | 0.404 | 0.302 | — |
| — control: random | — | 0.077 | 0.075 | — |

### Held-out tasks only (fold 0, 193 natural closes)

| signal | AUC | 95 % CI | within-run |
|---|---:|---|---:|
| kNN-8, LOTO, `last_final` | **0.818** | [0.756, 0.873] | 0.739 |
| prompt length | 0.804 | [0.740, 0.869] | 0.674 |
| novelty, LOTO, `last_final` | 0.714 | [0.641, 0.784] | 0.637 |
| random | 0.551 | [0.465, 0.642] | 0.548 |

### Is the kNN just a prompt-length lookup? No.

The obvious worry is that the last-token hidden state encodes position/length and
the retrieval is a dressed-up length table. Three checks say otherwise
(`results_dynamicv2/controls.json`):

| check | number |
|---|---|
| Spearman(kNN-8, reasoning tokens) **after removing prompt length** (rank-partial) | **0.572** (vs 0.675 unconditioned) |
| the *same* retrieval run over the free scalars instead of the hidden state — Euclidean kNN-8 on standardized `prompt_len` | AUC 0.660 [0.619, 0.703] |
| … on `prompt_len + step` | AUC 0.648 [0.605, 0.685] |
| … on `prompt_len + step + n_messages` | AUC 0.634 [0.592, 0.675] |
| within-run AUC (removes every per-conversation level, including length) | 0.741 vs 0.581 for length |

The within-run number is the decisive one: inside a single conversation, prompt
length, step and message count are the *same ranking*, so they are pinned at
0.581 by construction. The hidden-state kNN reaches **0.741** on that same
ranking, i.e. it distinguishes the hard steps *of this conversation*, which is
exactly what a per-request policy needs and what the length control cannot do.

### Per-task "deep" rate vs which efforts solved the task

Taking the top tertile of the kNN-16 estimate as "deep" and aggregating per task
(`results_dynamicv2/deep_rate_knn16.json`, 23 tasks):

| relation | Spearman | p |
|---|---:|---:|
| deep rate vs **number of effort columns that solved the task** | **−0.590** | 0.003 |
| deep rate vs `extra-high` solved it | −0.578 | 0.004 |
| deep rate vs `low` solved it | −0.390 | 0.066 |
| deep rate vs the task's median reasoning tokens | +0.695 | 0.0002 |

| task tier (from the observed score matrix) | tasks | mean deep rate |
|---|---:|---:|
| solved by every effort | 11 | 0.154 |
| effort-gated (low failed, some higher effort solved) | 11 | 0.358 |
| unsolved everywhere | 1 | 0.263 |

The signal calls "deep" the tasks that fewer effort levels solve — the sign a
router needs. It is a per-task aggregate over 23 tasks, so treat the exact
coefficients as indicative.

### Next-step trouble: nothing

Every signal, every pooling, sits at chance for "did this step's tool calls
error" (best 0.575 [0.503, 0.640]; random 0.529 [0.459, 0.600]; prompt length
0.483). Same conclusion the other track reached with encoders. The prompt does
not know that the command is about to fail, and neither does the model's state
before it thinks.

### What did not work, and the honest reading of surprisal

**Within-session surprisal, as cortext defines it, is not a positive effort
signal on this loop — it is a weak negative one.** Raw `err` at `a = 0.2` scores
AUC 0.284, i.e. *low* prediction error predicts long think. Inverted (1 − err)
that is AUC 0.716, close to the prompt-length control, and the reason is
mundane: late steps in a conversation have a long EMA history and land closer to
it, and late steps also think longer. It is a position proxy wearing a geometry
costume, and the within-run number (0.656 inverted) is only modestly better than
length's 0.581. The self-calibrating variants (streaming and per-session quantile
rank) sit at chance in both directions.

The two pooling variants also separate cleanly: **`last_final` beats `mean_final`
everywhere that matters** (kNN 0.839 vs 0.622). Mean pooling over the recomputed
tokens washes out the "what is the model about to do" content that lives in the
last position — the same failure mode the other track diagnosed for mean-pooled
encoders over JSON tool output.

## 4. The diagnostic probe (upper bound, NOT for shipping)

A logistic head on the L2-normalised `last_final` vector, out-of-fold over
GroupKFold(5) on `task_id`, reaches **AUC 0.866 [0.833, 0.896]**, within-run
0.778; ridge on `log1p(reasoning_tokens)` reaches Spearman **0.657**, within-run
0.585. On `mean_final` the same probes collapse (0.686 / −0.121).

Two things to take from this:

1. The other track's stated ceiling for prompt-side prediction (≈0.75, measured
   with deliberate leakage on hashed n-grams) is **not** the ceiling. Reading the
   served model's own state clears it by a wide margin with a proper task-grouped
   split.
2. The label-free retrieval gets **97 % of the trained probe's AUC** (0.839–0.850
   vs 0.866) and **beats it on Spearman** (0.689 vs 0.657) with no fitted
   parameters at all. There is very little left for a trained head to buy, which
   is a comfortable place to be given the constraint that rules one out.

## 4b. The decision map, and why it is asymmetric

The survey's rule (`effort-classifier-survey.md` §1.2, §6, from TwinRouterBench's
own measurement): **one under-routed step can kill a trajectory; over-routing
only costs tokens.** So the map must raise effort freely and lower it only on
confidence. Two confidence signals fall out of the retrieval for free and need no
labels:

- `novelty = (1 − max cos)/2` — is there anything in the memory like this at all?
- `spread` — the weighted stdev of the k neighbours' values: do they agree?

```
q = percentile rank of the kNN estimate in the running digest
q >= q_high (0.60)                                          -> rung 2 (high)
q >= q_mid  (0.35)                                          -> rung 1 (medium)
q <  q_mid AND novelty <= rank 0.6 AND spread <= rank 0.6   -> rung 0 (low)
otherwise                                                   -> rung 1  (unsure -> safe)
```

Every cut is a quantile of a running digest, so nothing here is a per-model
constant — the same discipline `effort_quantiles.py` already applies to entropy
and margin. Measured over the 689 natural closes
(`results_dynamicv2/policy.json`), against the ladder `[1024, 4096, 16384]`:

| starting policy | rung mix (low/med/high) | starts under-provisioned | of those, at rung 0 | p90 think tokens in the low band | mean granted budget |
|---|---|---:|---:|---:|---:|
| today: always rung 0 | 689 / 0 / 0 | **12.3 %** (85) | 85 | **1 471** | 1 024 |
| always rung 1 | 0 / 689 / 0 | 1.3 % (9) | — | — | 4 096 |
| symmetric tertiles of the estimate | 230 / 229 / 230 | 0.44 % (3) | 0 | 61 | 7 173 |
| **asymmetric, gated (0.35 / 0.60)** | 194 / 219 / 276 | **0.44 %** (3) | **0** | **59** | 8 153 |
| asymmetric, ungated (0.40 / 0.65) | 276 / 172 / 241 | 0.44 % (3) | **1** | 66 | 7 164 |

"Starts under-provisioned" = the request's actual reasoning tokens exceed the cap
of its starting rung, i.e. it can only finish by escalating. Today that is 12.3 %
of natural closes, and the 90th percentile of what today's rung-0 requests
actually want is **1 471** tokens against a 1 024 cap. The map cuts that to
0.44 %, and the requests it does send to rung 0 want a p90 of **59** tokens.

**The gate's own contribution is small, real, and honestly not significant on
this sample.** At the most aggressive downward cut it is the difference between
0 and 1 under-routed rung-0 start out of 689. It costs ~3–6 % of mean granted
budget. Ship it as insurance consistent with the asymmetry rule, not because
this table proves it.

**What this table cannot say:** the token *cost* of the higher starting rungs. A
cap is a ceiling, not an instruction — the thing that actually shortens thinking
is the rung-0 sentence, and its effect on a real trajectory needs a rerun, not a
lookup. That is the rollout item, not a number that can be extracted from these
traces.

## 4c. External yardstick — TwinRouterBench

[TwinRouterBench](https://github.com/CommonstackAI/TwinRouterBench) (Apache-2.0,
970 execution-verified rows, four tiers `low`/`mid`/`mid_high`/`high`) was cloned
and **replayed through the same capture path** — same prefill-only protocol,
`reasoning_effort: medium` so their prompts go through verbatim. 958 of 970 rows
returned a vector (§6 explains the 11 misses and the 1 rejected row).

The subset that matters is **`swebench`: 336 agentic-coding steps over 40
instances** (high 168 / low 94 / mid_high 41 / mid 33) — the same task family as
this loop. `all 970` mixes five benchmarks whose *identity* predicts the tier
(BFCL is short and almost all `low`; SWE-bench is long and mostly `high`), so any
signal that can tell the corpora apart scores enormously there — step index alone
reaches AUC 0.818 on `all`. **Read the swebench column; the `all` column is a
confound, reported only so nobody mistakes it for a result.**

| signal (label-free) | swebench AUC high-vs-rest | 95 % CI | Spearman vs tier | within-instance |
|---|---:|---|---:|---:|
| **kNN-16 over `last_final`, leave-one-instance-out on their own tiers** | **0.789** | [0.742, 0.836] | **0.569** | 0.440 |
| kNN-16, `mean_final` | 0.665 | [0.602, 0.722] | 0.307 | 0.321 |
| **transfer**: kNN-16 with the memory built from *our* loop (VulcanBench think lengths, no TwinRouterBench label anywhere) | 0.620 | [0.559, 0.678] | 0.243 | 0.320 |
| session surprisal (a=0.2, raw), `last_final` | 0.606 | [0.544, 0.669] | 0.240 | −0.007 |
| control: prompt length | 0.584 | [0.524, 0.644] | 0.116 | 0.196 |
| control: step index | 0.492 | [0.423, 0.554] | −0.058 | 0.196 |
| control: random | 0.510 | [0.448, 0.575] | −0.001 | 0.031 |

Three things this establishes and one it does not:

1. **The mechanism generalises.** On an external, execution-verified, four-tier
   label set from a *different* agent harness and *different* models, the same
   label-free retrieval over Qwen's pooled prefill state separates the `high`
   tier at AUC 0.789 against 0.584 for prompt length. Nothing was trained and
   nothing about this loop was used.
2. **`last_final` beats `mean_final` again** (0.789 vs 0.665), independently of
   the VulcanBench measurement. That is the pooling decision settled twice.
3. **Cross-corpus transfer is real but weak** (0.620, CI clear of 0.5). A memory
   filled on one loop does carry to another, but it is worth much less than a
   memory filled on the traffic being served — which is exactly what the online
   design does, and is the strongest argument for filling the memory *online*
   rather than shipping a fixed one.
4. **It does not** validate the tier→rung mapping. TwinRouterBench's tiers are
   about which *model* to route to; our rungs are thinking-token caps for one
   model. The agreement measured here is ordinal, not a claim that tier `high`
   means rung 2.

## 4d. What to ship

| decision | choice | why |
|---|---|---|
| **pooling** | `last_final` — the last prompt token's final hidden state | at matched k = 8, 0.839 (`last_final`) vs 0.622 (`mean_final`) on this loop and 0.789 vs 0.665 on TwinRouterBench; it is also the one vector that is already materialised, so pooling costs nothing |
| **signal** | **kNN over an online memory** (k = 16), with `novelty` and neighbour `spread` as confidence gates | 0.850 / 0.762 within-run / Spearman 0.685 (0.689 at k = 32), within 2 % of a trained probe, no fitted parameter |
| **not shipped** | within-session surprisal | as defined it is a mild *negative* signal (AUC 0.284); inverted it only recovers a position proxy, and its self-calibrating forms sit at chance |
| **not shipped** | `mean_final`, `cum_mean_final` | worse everywhere, and `cum_mean_final` needs session bookkeeping the last-token pooling does not |
| **not shipped** | any trained head | the constraint, and the measurement says it buys ≤ 0.02 AUC |
| **calibration** | percentile rank in a running t-digest, cuts 0.35 / 0.60, gates at rank 0.6 | no per-model constant; matches what `effort_quantiles.py` already does |
| **memory** | 4 096-entry ring, ≈42 MB, filled online at request finish | 512 entries already reach 0.842; 4 096 is headroom |


## 5. Integration sketch — where this would sit in the engine

### 5a. The decision point: two-phase prefill

Today the effort sentence is chosen in the **frontend**, before a single token is
tokenized: `apply_dynamic_effort`
(`vllm/entrypoints/openai/chat_completion/dynamic_effort.py`) appends
`cfg.low_effort_sentence` to the last user turn, sets `reasoning_effort` to
`cfg.render_effort` (medium — so block 0 is identical for every effort) and sets
`thinking_token_budget = ladder[0]`. A hidden-state signal cannot be computed
there, because the hidden state does not exist until the prompt is prefilled.

The fix is that the prompt already has the right shape. Split it at the point
where the effort text goes:

```
[ body ]                                   [ tail ]
system + all turns + last user content     effort sentence + <|im_start|>assistant\n<think>\n
~10^4 tokens, identical for every effort    ~20-40 tokens, one variant per rung
```

Phase 1 prefills `body` only. `body`'s last row is exactly the vector this
document measures. Phase 2 prefills the chosen `tail` and generation starts.

Concretely, on the V2 path:

1. **Frontend** (`dynamic_effort.py`): instead of committing to one sentence,
   render the tail token ids for each rung once (they are per-server constants,
   not per-request), submit the request with `prompt_token_ids = body`, and put
   `{"effort_tail_variants": [...ids...]}` into
   `SamplingParams.extra_args["dynamic_effort"]` — the same channel the ladder and
   theta already ride in.
2. **Runner** (`gpu/model_runner.py`, the exact line this patch already touches):
   when a request finishes its body prefill, pool the hidden state and put it in
   `ModelRunnerOutput` next to the existing `effort_signals` dict — the plumbing
   for "per-request scalar from the worker to the scheduler" is already there
   (`signals_to_dict`, `ModelRunnerOutput.effort_signals`). It would carry a
   short vector instead of four floats; at 5120 fp16 that is 10 KB per *deciding*
   request, once, on the existing IPC path.
3. **Scheduler** (`v1/core/sched/scheduler.py`): the request is in a new
   `AWAITING_EFFORT` state rather than `RUNNING`. On the output that carries its
   vector, the scheduler runs the label-free signal (§4), maps it to a rung
   through `SignalSketches` (percentile rank, exactly as entropy/margin are
   handled today), appends the chosen tail ids to the request's token ids, bumps
   `num_tokens`, and returns it to the waiting queue. The next scheduler step
   prefills the tail.
4. **Actuator**: unchanged. The rung sets `state["thinking_token_budget"]` through
   the same versioned update patch 0009 already added, and/or picks the tail
   sentence. Both actuators are reachable from that point; the sentence one is
   only reachable *because* the decision was deferred.

**Cost of the split.** One extra engine step per request. The *arithmetic* of a
20–40 token chunk is negligible, but the step itself is not free: at 40 tokens
the forward is latency-bound, so it costs about what one target forward costs on
this box — order 10–15 ms at TP4 on these L4s, plus one scheduler round trip.
Against a median 13 043-token prompt (≈5 s of prefill at the 2 500 tok/s this
config sustains) that is a few tenths of a percent of TTFT; on a short prompt it
is a few percent. Against a request about to spend 1 000–64 000 thinking tokens
it is noise. It is also strictly *less* work than today in one respect: the body
prefill is identical for every rung, so the prefix cache sees one body per
conversation instead of one per (conversation, effort).

**Prefix-cache safety is preserved, and slightly improved.** Today the sentence
sits at the very end of the last user turn, so it perturbs the trailing blocks.
With the split, the body is byte-identical across rungs and across the
`dynamic`/fixed columns; only the tail blocks differ, and those are the ones that
would be recomputed anyway.

**Cost of the probe itself.** The shipping signals are:

- session surprisal: one dot product against the session's EMA vector —
  5 120 multiply-adds, plus a 5 120-element EMA update. Per request. This is
  free in any meaningful sense (a single 5120x5120 projection inside the model is
  5 000x more arithmetic).
- memory kNN: one `[M, 5120] @ [5120]` matvec. At `M = 4096` that is 21 MFLOP —
  ~0.02 ms on an L4, or a few ms on one CPU core. Bounded and predictable.
- the t-digest rank lookups are the ones `effort_quantiles.py` already does.

### 5b. The online memory

Nothing is trained, so the "memory" is the only state, and the server fills it
itself as it serves:

| field | size | why |
|---|---|---|
| pooled vector (fp16, 5120, L2-normalised) | 10 KB | the key |
| `reasoning_tokens` at close | 4 B | the value the kNN averages |
| `close_kind` (natural / soft / forced) | 1 B | only natural closes are evidence of *wanted* think length; forced ones are censored and must be excluded from the value, though they may still count as keys |
| rung reached, escalation count | 8 B | lets the memory answer "did the ladder end up here" as well as "how long did it think" |
| tool outcome of the *next* step, when the client reports it | 1 B | optional; the `trouble` target. Not available in-engine today |
| session id + monotonic insert index | 16 B | eviction and same-session exclusion |

Sizing: a 4 096-entry ring at 10 KB is **41 MB of host RAM** — the same order as
the quantile sketches' JSON, and nowhere near the KV cache. Eviction is FIFO with
one refinement: keep a reservoir sample rather than pure FIFO, so a burst of one
conversation cannot evict the whole memory (a pure ring is fine at 4 096 entries
if entries are also deduplicated per session, which the session-EMA path already
needs).

Persistence: write it next to `dynamic_effort.quantile_path`, same
`atomic-replace-on-flush` pattern, so a restart warms instantly instead of
running blind. While the memory is colder than `min_entries`, the kNN term
returns `None` and the controller falls back to the session-surprisal term alone
— the exact "cold digest reports `None`" discipline `effort_quantiles.py` already
implements.

**Provenance.** The memory is filled from *the engine's own observations of its
own generations*: pooled vector in, observed reasoning tokens out. No human
labels, no other model's output, no offline corpus. It is an online cache of
"requests that looked like this needed this much", which is a retrieval structure
and not a trained head — the constraint the user set.

### 5c. What the controller would actually do

The existing controller escalates *during* generation from entropy/margin/MTP
signals. This adds one decision *before* generation, on the same machinery:

```
rung_0 = quantile_bucket(knn_estimate, sketch)      # low / medium / high, before <think>
```

with the ladder and the mid-generation escalation logic untouched. The prefill
signal sets the *starting* rung instead of always starting at rung 0; the live
signals still decide whether to climb. That composition matters, because it means
a wrong prefill decision is recoverable upward at the usual cost, and the
`min_entries`-cold path degrades to exactly today's behaviour (always start at
rung 0).

The bucket cuts come from the same streaming digest, not from constants: "top
tertile of the kNN estimate this server has seen" is model-, quantization- and
workload-agnostic in the way §4's raw `err_ref`/`k` are not.

## 6. Honest limits

- **One model, one loop, one box.** Every number is Qwen3.8-27B-FP8 on the
  VulcanBench v3 agent loop. The mechanism (retrieve over the model's own pooled
  prefill state) is general; the coefficients are not.
- **The labelled evaluation is 689 requests from 23 tasks.** Per-task aggregates
  (the deep-rate table) rest on 23 points. The AUC CIs are over requests, and
  requests inside a task are correlated, so the effective sample is smaller than
  689.
- **`dynamic-v2` is the only column with observed reasoning tokens**, so both the
  memory's values and the targets come from one engine configuration. A memory
  filled under a different ladder would hold different values.
- **The kNN memory here is the evaluation set itself, minus the task under test.**
  That is the correct simulation of a warm server that has never seen this task,
  and the online-ring variants (128 / 512 / 2048 entries, strictly earlier
  arrivals only) confirm it survives a realistic memory: 0.808 / 0.842 / 0.843.
  It is *not* a simulation of a cold server, which has no memory and therefore no
  signal — hence the `min_entries` fallback in §5b.
- **`mean_final` is a mean over recomputed tokens, not over the prompt** (§2). The
  exact full-prompt mean was not measured; on the evidence that `last_final`
  dominates every mean variant, it is unlikely to change the conclusion.
- **Layer-32 pooling was not measured at all** (§1) — two mechanisms are inert
  under vLLM's compiled dispatch on this build.
- **Trouble prediction is dead** on this data, from every direction anyone has
  tried.
- **A fully prefix-cached prompt can produce no record.** 11 of 970
  TwinRouterBench rows (all short BFCL repeats of an earlier prompt in the same
  run) never entered the pooling path, because the step scheduled no prompt
  tokens at all. None of the 791 VulcanBench requests hit this — their prompts
  always grow — but the shipped version must take the vector from the request's
  logit row rather than from the prefill accounting, so a 100 %-cached prompt
  still yields one. One further TwinRouterBench row was rejected by the server
  ("System message must be at the beginning"), a property of their data.
- **The `all 970` TwinRouterBench column is a benchmark-identity confound** and is
  reported only to make that explicit; the swebench subset is the result.
- **No policy was simulated here.** The other track's §6 lookup simulation applies
  unchanged if this signal replaces its lane scores; that simulation's own caveat
  (changing effort changes the trajectory) applies with equal force.

## 7. Reconciliation with the other tracks

| track | best label-free think-length signal | AUC long-think | within-run | Spearman |
|---|---|---:|---:|---:|
| free controls (server already has them) | prompt length | 0.729 | 0.581 | 0.494 |
| small encoders, CPU ([`effort-router-prototype.md`](effort-router-prototype.md)) | see that document's §4–5 | — | — | — |
| **this track** | kNN-16 over Qwen's own `last_final`, LOTO | **0.850** | **0.762** | **0.685** |
| this track, upper bound (trained, not shippable) | logistic on `last_final` | 0.866 | 0.778 | 0.657 |

External check (TwinRouterBench, 336 execution-verified agentic-coding steps,
`high` tier vs rest): the same label-free retrieval scores **AUC 0.789**
[0.742, 0.836] against **0.584** for prompt length and 0.492 for step index.

Both tracks share `dataset/requests.jsonl`, the GroupKFold-on-`task_id` split, the
mid-rank AUC with 1000-sample bootstrap CIs, and the within-run rank
normalisation; this track's implementation reproduces the other's control numbers
to the fourth decimal (§2), so the rows are directly comparable.

The practical difference is not only accuracy. A CPU encoder costs a model, a
truncation policy (the median request is 13 043 tokens and every candidate
truncates), a second latency budget, and a separate answer for screenshots. The
hidden state costs a dot product and is already multimodal because the prompt
already went through the model. The price is the two-phase prefill of §5a — one
extra engine step — which is the only thing about this approach that is not free.
