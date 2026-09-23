# GLM-5.3-Flash on 8x Gaudi2: progress report

*As of 2026-09-23. Work started 2026-09-22 17:14.*

We serve the FP8 `GLM-5.3-Flash` checkpoint (306 GiB) with vLLM 0.26 and vLLM-Gaudi 1.24.1. The model is hybrid: KDA linear attention, MLA, mHC residual mixing and a 288-expert MoE. It runs with TP=8 across 8 Gaudi2 cards (HL-225, 96 GB each, about 2.06 TB/s measured HBM bandwidth). Every change is committed in `~/work`.

## Headline

| Metric | Shipped image | Now | Change |
|---|---|---|---|
| Output quality | garbage (`"!!!"`, NaN logits) | top-1 matches an fp32 HF reference 12/12 | fixed |
| Single-stream decode | 0.038 tok/s (about 27 s/token) | **36.5 tok/s** (27.4 ms/token) | about 960x |
| Decode, 16 streams | – | 327 tok/s (48.9 ms/step) | |
| Decode, 64 streams | – | about 1,000 tok/s (65 ms/step) | |
| Decode, 128 streams | – | **1,582 tok/s** (81 ms/step) | |
| Serving, 1 concurrent (512 in / 256 out) | – | 33 tok/s | |
| Serving, 64 concurrent | – | 508 tok/s | |
| Serving, 128 concurrent | – | 587 tok/s | |
| Prefill, 1k-token prompt | – | about 250 ms | |

"Decode" numbers are pure decode steps measured by the offline harness (`tools/run_llm.py --bench-bs`). "Serving" numbers come from the OpenAI server benchmark (`tools/bench_serve.sh`).

## Timeline

Each row is a commit. The numbers are the decode benchmarks recorded right after it. L5 is the 5-layer truncated checkpoint used for fast iteration; "full" is the 45-layer model.

| Time | Change | Measured |
|---|---|---|
| 09-22 17:14 | Baseline image | garbage output, 0.038 tok/s |
| 18:03 | Correctness: mHC native op dropped its RMSNorm; KDA layers shared one state slot; masked chunked KDA prefill | full model correct (11/12 top-1; the miss is a known near-tie); eager, slow |
| 18:10 | MLA decode on the flat-PA kernel | L5 bs1: 1,385 ms/step (eager) |
| 18:56 | torch.compile (static shapes, conv weights built on host) | L5 bs1: 27.3 ms (**50x**) |
| 20:02 | Whole-model compile support, strided 6-D Gaudi bug isolated | L5 bs1: 23.9 ms |
| 20:17 | MoE rewritten: stacked per-channel FP8 experts, dense and gather paths, HF routing, swiglu clamp | L5 bs1: 16.4 ms; L5 bs16: 826 tok/s |
| 21:25 | Blocked triangular inverse for KDA, guard-free decoder, compiled sampler | full bs1: 32.8 ms, full bs16: 312 tok/s |
| 22:04 | Removed per-step recompiles (sampling metadata indexing) | full bs1: 28.3 ms; bs64: 1,085 tok/s; bs128: 1,582 tok/s |
| 23:28 | MLA cache write via `index_put_` (stopped a whole-cache copy every step) | whole-graph compile possible (no gain on the full model) |
| 09-23 00:54 | Private per-layer KDA state, 4-layer compiled groups | measured slightly slower, so left off by default |
| 04:03 | Production server script and serving buckets | serving 508 tok/s at 64 concurrent |
| 05:47 | KDA prefill with block-factorized intra-chunk scores | 1k-token prefill KDA stage **2.5x** faster (about 250 ms total) |
| 07:19 | Server raised to 128 sequences | serving 587 tok/s at 128 concurrent |
| 07:31 | KDA decode: one pass over the state for both mat-vecs | full bs1: **27.4 ms (36.5 tok/s)** |
| 10:28 | MTP speculative decoding on HPU: multi-token KDA verify with per-position state rollback, GLM MTP drafter | works, accepts 1.96 tokens/step, but a step is slower (see below) |

## Where the time goes

At TP=8, each of the 45 layers needs two all-reduces, for **91 HCCL all-reduces per step**. Each costs about 130–150 µs. On top of those, each step has about 175 graph (recipe) boundaries.

Together these set a **fixed floor of about 25–30 ms per step**, independent of batch size. Single-stream decode (27.4 ms) sits on that floor. The weights read per token at bs1 would take only a few ms at HBM speed, so bs1 is limited by communication and launch overhead, not memory bandwidth.

At large batch the MoE weight reads run close to bandwidth: the dense FP8 expert GEMMs reach about 2.1 TB/s. Even so, about 40% of each step is still the fixed overhead.

These attempts did **not** help:

| Attempt | Result |
|---|---|
| Skip attention all-reduces (experiment) | no change at bs1, −6 ms at bs64 |
| Compile 4 layers as one graph | slightly slower |
| Compile the whole model as one graph | no gain; about 16 min compile per bucket |
| FP8 attention and linear weights | no gain |
| PRIM collectives | no gain |
| Fused HPU MoE op | about 2.2 ms/layer host cost at 288 experts; replaced |

## Correctness fixes along the way

| Bug | Symptom | Fix |
|---|---|---|
| mHC native op ignored `norm_weight` | output about 13x too large, garbage | own HPU mHC implementation |
| All KDA layers shared one recurrent state | garbage after the first layer | per-layer cache groups, then private state tensors |
| Padded rows allocated with `torch.empty` | NaN logits | new KDA kernels |
| Gaudi strided 6-D broadcast | wrong values (contiguous inputs are fine) | flatten to at most 5-D contiguous |
| Fused MoE op lacks the swiglu clamp | position-0 logprob off by 4 | stacked MoE with the ±10 clamp |
| Unstable chunk inverse | drift in long prefill | exact blocked inverse |

## In progress: speculative decoding

The goal is to break the fixed per-step floor by producing more than one token per step.

| | Non-spec | MTP (k=1), current |
|---|---|---|
| Tokens per step | 1 | 1.96 |
| Device forward per step | about 28 ms | about 70 ms |
| Host sample + propose | small | about 87 ms + 11 ms |
| Effective ms/token (bs1) | 28.0 | 58–68 |

Acceptance is high (1.96 of 2 possible tokens per step). However, the verify step and host sampling are currently far more expensive than a normal step, so MTP is slower overall.

Next steps:

1. Profile the verify step to find why a 2-token step costs about 70 ms on device.
2. Cut the host sampling and propose cost.
3. Port the **DFlash2** drafter (block diffusion, 7 draft tokens per step). vLLM 0.26 supports only DFlash v1, so this needs a new HPU proposer and batched drafter attention kernels. It would use the Apache-2.0 `GLM-5.3-Flash-DFlash2-E` weights.

If a step stays at about 30 ms with 3–4 accepted tokens, single-stream speed would reach roughly 100+ tok/s.

## Remaining limits

- **Context capped at 2048 tokens.** The sparse kpool indexer has no Gaudi kernel, and dense MLA is exact only while all tokens fit in the indexer's top-k (2048).
- **Pipeline parallelism** (which would reduce all-reduce group size) is not supported by vLLM-Gaudi for this hybrid model.
- **Only eager and torch.compile** are available on this stack (no lazy mode or HPU graphs), so per-launch host cost stays significant.
