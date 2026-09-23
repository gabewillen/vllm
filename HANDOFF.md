# GLM-5.3-Flash on 8x Gaudi2: handoff / state

*Last updated 2026-09-23.* Goal: optimize vLLM inference of GLM-5.3-Flash (FP8, 306 GiB; a hybrid of KDA linear attention, MLA, mHC and a 288-expert MoE) on 8x Gaudi2 (TP=8) until hardware limits are reached. For the results history, see [PROGRESS.md](PROGRESS.md).

## Where things stand

| Area | State |
|---|---|
| Correctness (non-spec) | Matches the fp32 HF reference 12/12 top-1 on the short check. See the open question below about longer sequences. |
| Decode, bs1 | 27.4 ms/step (36.5 tok/s). Bound by a fixed ~25-30 ms/step floor: 91 HCCL all-reduces plus ~175 graph launches. |
| Decode, bs64 / bs128 | ~1,000 / 1,582 tok/s |
| Serving (512 in/256 out) | c1 33 tok/s, c64 508, c128 587 |
| MTP spec decode (k=1) | Works. 27.3 ms/tok steady state vs 28.0 non-spec, with 1.88 tokens accepted per step. The verify step is serialized with the host, so the gain is small. |
| DFlash2 spec decode (K=7) | Runs end to end, but **not lossless: the 8-token verify gives wrong tokens** (see the bug below). 32.4 ms/tok. |
| Production server | **Down** (dev container `glmdev` holds the devices). Restart: `/mnt/glm-models/serve-glm53-opt.sh` |

## Environment

- **Work tree:** `~/work` (git). The editable `vllm/` and `vllm_gaudi/` trees are bind-mounted into the dev container `glmdev` over site-packages. Image: `vllm-gaudi:glm53-patch`.
- **Branches:**
  - `gaudi2` on `git@github.com:gabewillen/vllm.git` is a clean source snapshot.
  - Local `main` has the full history, including profiler traces. Don't push `main`.
- **Checkpoints** (in `/mnt/glm-models/`, never commit these):
  - `GLM-5.3-Flash` is the full model.
  - `-L5` and `-L13` are truncated to 5 and 13 layers for fast iteration (~1 min load).
  - `-L5MTP` is the truncated model with an MTP layer.
  - DFlash2 drafters:
    - `GLM-5.3-Flash-DFlash2-E` (Apache-2.0, 9 taps, the one in use).
    - `-DFlash2-incoai` (CC-BY-NC-ND licence).
- **Reference data** in `~/work/ref/` (gitignored):
  - `full_capital.pt`: fp32 reference output for the 5-token prompt.
  - `dflash_taps.pt`: fp32 per-layer streams for a 69-token sequence (ids in `dflash_ids.json`).
  - `dflash_sim_E.pt`: reference DFlash2 drafts for that sequence.
- **Z-lab DFlash2 reference code:** `ref_src/dflash_model.py`.

## Dev loop

- **Harness:** `tools/t.sh LOGNAME <run_llm.py args>` runs the harness in `glmdev` and writes `logs/LOGNAME.txt`.
  - It calls `tools/freedev.sh` first, which **kills any running vLLM job**. Never start a second harness run while one is going.
- **run_llm.py options:**
  - `--ref` checks prompt logprobs against the reference.
  - `--bench-bs`, `--spin-warm N --spin-tokens N` for timing (always warm before timing).
  - `--no-lp` is required for spec decode.
  - `--profile-bs` captures a device profile.
- **Scripts:**
  - `tools/mtp_spin.sh`: MTP steady-state timing.
  - `tools/dflash_spin.sh`: DFlash2 run. Takes the environment variables `NAME`, `EXTRA_ENV`, `ARGS` and `PC` (`PC=""` turns prefix caching on).
  - `tools/verify_ab.sh`: non-spec vs ngram(k) losslessness check (`MODEL=` picks the checkpoint).
  - `tools/mtp_prof.sh`: profile plus timing.
- **Unit tests:**
  - `tools/test_kda_spec8.py [cpu|hpu] [compile]`: KDA rollback at T=8. Passes on CPU, HPU eager and HPU compiled.
  - `tools/test_dflash2.py ... [cpu|cpubf16|hpu]`: drafter port vs the reference. fp32 CPU 392/392, HPU bf16 378/392.
  - `tools/dflash_sim.py`: CPU acceptance simulation.
  - `tools/hf_ref.py`: layer-streaming CPU reference (`REF_LAYERS_ONLY=1`). Run it in a separate container with `--cpuset-cpus 96-151`.
- **Required environment:** `VLLM_SKIP_WARMUP=true VLLM_T_COMPILE_DYNAMIC_SHAPES=0 VLLM_HPU_COMPILE_CACHE_MULT=16 PT_HPU_RECIPE_CACHE_CONFIG=/work/recipe_cache,false,65536`
  - For long compiles, also set `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200`.
- **Debug switches:**
  - `GLM53_TIMING=1` prints a per-step breakdown but adds ~14 ms/step itself.
  - `GLM53_DFLASH_DEBUG=1` prints per-step accepted tokens and drafts.
  - `GLM53_DFLASH_RANDOM_DRAFTS=1` replaces drafts with random tokens.
  - `GLM53_MOE_GATHER_MAX_SLOTS` sets the MoE gather-vs-dense threshold (default 16).
- **Watch-outs:**
  - `pkill -f <pattern>` kills your own shell if the pattern appears in its command line. Use a script file.
  - This machine has no GitHub credentials; pushing needs a forwarded ssh agent.

## Open bug: DFlash2 8-token verify is not lossless (top priority)

**Symptom:** prompt `[785, 6722, 315, 9621, 374]`. The DFlash2 output departs from greedy at generated token 8: it produces `279` where the reference and MTP give `330`, and the reference's margin there is 1.5 nats.

What is established:
- **Not draft-dependent.** With `GLM53_DFLASH_RANDOM_DRAFTS=1`, every step accepts one token and resumes from row 0, and the output shows the same wrong `279`. Row 0 of an 8-token verify differs from row 0 of a 2-token (MTP) verify on the same prefix.
- **Not KDA rollback or KDA math.** `test_kda_spec8.py` is exact at all 8 accept positions, including HPU compiled.
- **Not prefix caching.** Same output with it on or off.
- **Not MLA write/read ordering.** The cache is written before attention, and contiguous PA is off, so each verify token gets its own causal block list.
- **Not the drafter.** It only chooses the tokens being verified.
- **Main suspect: the MoE path.** A 2-token verify has T*K = 16 expert slots and takes the bf16 **gather** path. An 8-token verify has 64 slots and takes the **dense** path, which quantizes activations and intermediates to FP8 per row (`moe_hpu.py:moe_dense`, `_dyn_quant_rows`).
  - Test in progress: `logs/full_dflash_rand_g.txt`, random drafts with `GLM53_MOE_GATHER_MAX_SLOTS=64`.
  - If that output gives `330` at index 8, the dense FP8-activation path is the cause. That would also affect normal decoding at bs≥3 (T*K > 16) and needs a precision fix, for example bf16 activations or finer quant scales.
  - If it's still `279`, the next suspects are the T=8 attention metadata (per-virtual-token `seq_lens`/block usage in `_create_decode_input_data`) and the target logits row mapping in the rejection sampler.

**Related:** the vLLM MTP greedy output disagrees with the fp32 reference at 6 of 64 positions (margins 0.1-0.7) on `ref/dflash_ids.json`. Run a non-spec prompt-logprob check over all 69 positions (`run_llm.py --ids "$(cat ref/dflash_ids.json)" --ref ref/dflash_taps.pt`) to see whether non-spec decode is also off on longer sequences.

**Known latent bug:** when an 8-token verify block crosses a KV block boundary (384 tokens), `get_habana_paged_attn_buffers` gives every verify token the same block table with only the last block's usage set per token. Tokens in the earlier block then see wrong or future context. It needs per-virtual-token block tables truncated at the token's own block.

## DFlash2 port (implemented)

- **`vllm_gaudi/v1/spec_decode/dflash2_model.py`:** the drafter.
  - Attention and MLP weights are TP-sharded; the fc projection, convs and selector are replicated.
  - Dense per-request context K/V cache with rows `slot*L + pos`.
  - Vocab-parallel top-16, then the candidate-selector walk.
  - Candidate ids are int32: a gather from an int64 tensor fails to compile on Gaudi.
- **`vllm_gaudi/v1/spec_decode/hpu_dflash2.py`:** the proposer.
  - Decode path writes context rows for the verified block and masks with `ctx_len = n_tokens - 1`.
  - Prefill path writes the prompt rows.
  - Slot = KDA base slot + 1 (slot 0 is a dummy).
- **Target taps:** `Glm5NextModel.set_aux_hidden_state_layers`. Each tap layer returns a 5th output, the stream mean `mean(post)·x + Σ_i mean_j(comb[i,j])·res_i`. The model returns `(hidden, cat(taps))`.
  - Tap convention verified on CPU: mean over the 4 mHC streams of each layer's output.
- **Runner:** `method == "dflash"` branches in `hpu_model_runner.py` (drafter creation, `propose_dflash`, `_maybe_compile_drafter`). The `DFlash2DraftModel` stub is registered in `vllm_gaudi/models/__init__.py`.
- **Acceptance:** 2.30 tokens/step on raw-completion text in the CPU simulation; the model card reports ~3.57 on chat prompts with thinking on.

## Performance levers (measured)

- **No gain:**
  - Skipping attention all-reduces.
  - 4-layer compiled groups.
  - Whole-model compile: slower for MTP (29.8 ms/tok vs 27.3).
  - FP8 attention/linears.
  - PRIM collectives.
- **Verify steps are host-serialized:** rejection sampling syncs the host, so host issue time (~36 ms for 46 compiled regions) doesn't overlap the device.
- **Small-batch MoE:** for 16 < T*K < ~200 (bs 3-16, and the 8-token verify at bs1), both the dense and gather paths cost ~14-16 ms/step. The gather path converts each selected expert's weights to bf16 copies. A grouped FP8 GEMM that reads the selected experts in place would help both. The HPU fused MoE op takes per-expert weight lists (high host cost) and lacks the swiglu clamp.

## Next steps (in order)

1. Read `logs/full_dflash_rand_g.txt` and fix the verify divergence (see the bug above). Re-verify with `GLM53_DFLASH_RANDOM_DRAFTS=1`: the output must equal non-spec greedy.
2. Fix the block-boundary bug in the verify attention metadata.
3. Measure DFlash2 acceptance and ms/token on a chat-style prompt. Then optimize the small-batch MoE for the verify step, and the drafter step.
4. Restart the production server when done: `/mnt/glm-models/serve-glm53-opt.sh`.
