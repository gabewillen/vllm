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
| DFlash2 spec decode (K=7) | Runs end to end and follows the fp32 reference at bs1 after the MoE fix (see below). 30.4 ms/tok on raw-completion text (low acceptance); not yet faster than non-spec. |
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
  - `GLM53_MOE_GATHER_MAX_SLOTS` sets the MoE gather-vs-dense threshold (default 64; was 16).
- **Watch-outs:**
  - `pkill -f <pattern>` kills your own shell if the pattern appears in its command line. Use a script file.
  - This machine has no GitHub credentials; pushing needs a forwarded ssh agent.

## Root cause found: MoE path inconsistency between verify and decode

**Symptom:** DFlash2 output departed from greedy at generated token 8 (`279` instead of the reference's `330`, a 1.5-nat margin). It did so even with random drafts, so it wasn't draft-dependent.

**Cause:** the 8-token verify took the MoE **dense** path, which quantizes activations and intermediates to FP8 with one scale per row (`moe_hpu.py:moe_dense`, `_dyn_quant_rows`). The bs1 decode it must reproduce took the bf16 **gather** path. The two paths round differently, so spec output couldn't equal non-spec output.
- The FP8 rounding itself is ordinary W8A8 behaviour: ~2.5% relative error on the MoE input per token (e4m3 mantissa; outliers are modest, row max ≈ 15-24x RMS, measured on the L5 reference). That is what GPU FP8 serving of this checkpoint does too.
- Finer activation scales (1x128 blocks) would not reduce the rounding error.
- The path is chosen when T*K > `GLM53_MOE_GATHER_MAX_SLOTS`, which was 16.
- The bf16 **gather** path is exact. With `GLM53_MOE_GATHER_MAX_SLOTS=64` and random drafts, the output equals MTP and the reference (`logs/full_dflash_rand_g.txt`).

**Mitigation applied:** the default threshold is now 64 (`model.py`, `Glm5NextMoE._GATHER_MAX_SLOTS`). That covers decode up to bs8, the 8-token verify at bs1, and short prompt prefills.

**Result** (`logs/full_dflash_g64.txt`): DFlash2 with real drafts now follows the fp32 reference. It first departs from the old MTP output at generated token 22, and there the fp32 reference agrees with DFlash2 (`1`, logprob -0.61) rather than MTP (`3263`, -1.32).
- The old MTP and non-spec runs were themselves slightly off, because under the old threshold of 16 the 5-token prompt prefill (40 slots) used the lossy dense path.
- That likely explains the 6 MTP-vs-reference mismatches on `ref/dflash_ids.json`.
- Every earlier number (MTP 27.3 ms/tok, non-spec 28.0, serving) should be re-measured with threshold 64.
- DFlash2 bs1 speed: 30.4 ms/tok on the raw-completion prompt, where acceptance is low (~2.3 tokens/step in the CPU simulation). It still needs a chat-style prompt measurement and verify-step and drafter optimization.

**Remaining caveat:** numerics depend on batch size. bs≤8 decode uses bf16 activations and bs>8 uses FP8 W8A8, like batch-variant kernels on GPU.
- Spec decode stays self-consistent while the verify's T*K ≤ 64. At bs>1 with K=7 the verify exceeds that and falls back to dense, while plain decode at the same batch is also dense unless bs≤8.
- **To do:** make the path choice depend on the batch size *without* the spec tokens, so verify and non-spec decode always agree.
- To remove FP8 activation rounding everywhere, the options are a grouped/gather FP8 GEMM with bf16 activations, or dequantized-bf16 dense weights (2x bandwidth).

`fp8_gemm_v2` does **not** accept a bf16 A with an fp8 B on this stack (tested: "not yet supported"). Afterwards, check the gather path's speed at 17-64 slots against dense.

Ruled out along the way: KDA rollback/math (`test_kda_spec8.py` exact), prefix caching, MLA write/read order, and the drafter itself.

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

Work stopped 2026-09-23. Nothing is running and the devices are free.

1. **Re-measure baselines** with the new MoE gather threshold of 64:
   - `SLOTS=64 tools/bench_gather.sh; SLOTS=16 tools/bench_gather.sh` benchmarks non-spec decode at bs1/4/8 under both thresholds, to check whether gather at 17-64 slots is slower than dense. It was started and cancelled before any result.
   - If gather is slower at bs4-8, pick the threshold per batch size.
2. **Chat-prompt comparison:** `tools/chat_ab.sh` runs non-spec vs MTP vs DFlash2 on `ref/chat_ids.json` (35-token chat template with thinking on). It prints ms/tok and the first divergence from non-spec for each. Not run yet.
   - DFlash2's acceptance on chat text should be far above the ~2.3 seen on raw completion.
3. **Speed up DFlash2** (30.4 ms/tok at bs1 on raw completion): profile the 8-token verify step (MoE gather at 64 slots, host serialization) and the drafter step (`GLM53_TIMING=1`, `tools/mtp_prof.sh` pattern).
4. **Batch-dependent MoE numerics** (optional): see "Remaining caveat" above.
5. **Verify block-boundary bug:** fix the paged-attention metadata when an 8-token block crosses a 384-token boundary (see "Known latent bug").
6. **Restart the production server:** `/mnt/glm-models/serve-glm53-opt.sh`. It is currently down. Its serving numbers were measured with the old threshold of 16.
