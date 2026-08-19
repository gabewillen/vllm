# DFlash2 trial: bench plan for the next GPU window

Decision this run has to answer: **should the Qwen3.8-27B-FP8 latency profile on
the 4x L4 box switch its drafter from Qwen3.8's built-in MTP to the DFlash2
block-diffusion drafter?**

Nothing in this plan has run on a GPU. Everything below is prepared CPU-side:
branch `qwen3.8-27B-dflash2` (upstream PR #52816 cherry-picked onto
`qwen3.8-27B-effort-v2`), venv `/shared/vllm/.venv-dflash2`, worktree
`/shared/vllm-dflash2`, drafter in `/data/huggingface`, serve config
`serve-configs/qwen3_8_27b_fp8_dflash2_latency.yaml`.

Upstream's own numbers (H200, SGLang, bf16 target, K=7) claim acceptance length
4.10-5.46 vs MTP's 3.74-5.02 and 1.28-1.37x the MTP throughput at concurrency 1
and 8, falling to roughly parity at concurrency 32. Our box is different in
every way that matters — L4 not H200, FP8 target not bf16, PCIe TP4 not NVLink,
vLLM not SGLang — so those numbers set the hypothesis, not the expectation.

## Ground rules

* **One engine at a time.** 4x L4 with a 262k-context TP4 model has no room for
  two. Stop the prod unit before starting anything, restore it afterwards.
* **Experiment port is 8013**, prod is 8012. Never bench against 8012.
* **Experiment venv/worktree only**: `/shared/vllm/.venv-dflash2` and
  `/shared/vllm-dflash2`. The prod unit keeps pointing at `.venv-qwen38` and
  `/shared/vllm/serve-configs/...`; nothing in this plan edits either.
* **Every arm runs from the same venv** (`.venv-dflash2`), including the MTP and
  no-spec arms, so the engine build is not a second variable. That venv is
  installed editable from `/shared/vllm-dflash2` and reports
  `0.26.1rc1.dev832+g4e6924f4c`, the cherry-pick commit; the prod venv is a
  different build, which is a second reason arm A0 is only a rough tie-back.
* **Tiered KV offload is off in every arm.** With a DFlash drafter the
  offloading connector marks *every* KV group EAGLE-volatile (patch 0003 only
  recognises an `mtp.` layer prefix, and vLLM only sets `is_eagle_group` for
  DeepSeek-V4), which changes decode-time scheduling and would confound the
  comparison. Turning it off in all arms keeps the drafter the only variable.
  Consequence: aggregate numbers here are **not** directly comparable to the
  2026-08-18 goal-run figures, which had offload on. Arm A0 below is the
  optional tie-back.
* Greedy (`temperature 0`) everywhere, so a spec-decode arm and the no-spec
  control must produce the *same* tokens. That is also the losslessness check.

## Arms

All arms: `Qwen/Qwen3.8-27B-FP8`, TP4, `max-model-len 262144`,
`kv-cache-dtype fp8`, `max-num-seqs 96`, `max-num-batched-tokens 8192`,
`attention-backend TRITON_ATTN`, `VLLM_USE_V2_MODEL_RUNNER=1`,
`NCCL_P2P_LEVEL=SYS`, no `kv-transfer-config`.

| Arm | Drafter | Extra | gpu-mem-util |
|---|---|---|---|
| **A** NOSPEC | none | control | 0.92 |
| **B** MTP | `qwen3_5_mtp` K=7, `adaptive_draft_length` + margin 2.0, `draft_lm_head_dtype int4`, batch-size K schedule | `additional-config lm_head_dtype fp8` | 0.92 |
| **C** DFLASH2 | `dflash` K=7, `z-lab/Qwen3.8-27B-DFlash2` | none (see below) | 0.88 |
| **B'** MTP-PLAIN *(optional)* | `qwen3_5_mtp` K=7 flat, no adaptive, no `draft_lm_head_dtype` | no `lm_head_dtype` | 0.92 |
| **C'** DFLASH2-ADAPT *(optional)* | arm C + `adaptive_draft_length` true, margin 2.0 | none | 0.88 |
| **A0** PROD *(optional)* | prod latency yaml unmodified, offload on, port 8012 | tie-back to 2026-08-18 | 0.92 |

**B vs C is the decision comparison**: each arm keeps every optimisation that is
legal for it. **B' vs C'-less-C** is the attribution comparison: it says how
much of any B-C gap is the drafter and how much is the quantised-head work in
patches 0005/0008.

Arm C deliberately drops `lm_head_dtype`, `draft_lm_head_dtype`,
`adaptive_draft_length` and the batch-size K schedule. Reasons are in the header
of `serve-configs/qwen3_8_27b_fp8_dflash2_latency.yaml`; the load-bearing one is
that the DFlash2 checkpoint ships no `lm_head`, so it shares the **target's**
head, and `compute_candidates` raises
`DFlash2 requires an unquantized target LM head for candidate TopK` if patch
0008 has replaced it with a `QuantizedDraftLMHead`. Arm C will not start with
`lm_head_dtype: fp8`. That is a real cost to arm C (~1 ms/verify step) and is
part of what B vs C measures.

## What to measure

Per arm, at concurrency **1 / 8 / 32**, on the *same* eight-prompt set:

* `serve-configs/bench_concurrency.py <port> <conc> 512` — per-request decode
  tok/s (median and mean), aggregate output tok/s, TTFT median/max, and the
  spec-decode deltas over exactly those requests.
* At concurrency 1, also `serve-configs/bench_single_stream.py 8013` for
  continuity with the four-prompt reason/prose/write/edit series that every
  earlier goal run reports (66/65/105/160 tok/s is the current latency-profile
  record).
* `scripts/burst_client.py 8013 128 1024 1024` (from
  `goal/sessions/23cb78c0-.../runs/20260818T131256Z-afa8720e/artifacts/scripts/`)
  for the 128x1k/1k aggregate figure, so the throughput ceiling is on the same
  axis as the 1170/912/615 tok/s record — with the offload caveat above.

Spec-decode metrics from `http://localhost:8013/metrics` (deltas across each
phase; `bench_concurrency.py` already does this):

| Quantity | From |
|---|---|
| acceptance length (tokens committed per target verify step) | `1 + vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_drafts_total` |
| mean draft length | `vllm:spec_decode_num_draft_tokens_total / vllm:spec_decode_num_drafts_total` |
| per-token acceptance | `accepted / draft_tokens` |
| per-position acceptance | `vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0..6"}` |

Acceptance length is the number to compare against upstream's table; mean draft
length is the one that distinguishes arm B (adaptive, so < 7) from arm C (always
7 by construction).

Two things to read once per arm, from the startup log, before trusting any
throughput number:

* `GPU KV cache size: N tokens`. The MTP profile gets ~1.39M. DSpark's real cost
  was here — it halved the pool to 652k — and DFlash2 spends an estimated
  ~2.1 GiB/rank on drafter weights (~1.45 GiB, only partly sharded) plus the
  unconditional fp32 `draft_logits` buffer (96 x 7 x 248320 x 4 B = ~0.67 GiB)
  plus its own 5-layer sliding-window KV group. **If the pool drops below ~900k
  tokens the profile cannot hold three concurrent 262k sessions and arm C is
  not deployable at `max-num-seqs 96` regardless of its tok/s.** First lever if
  that happens: `max-num-seqs 32`, which returns ~0.45 GiB/rank of
  `draft_logits`.
* Whether the drafter actually loaded as DFlash2 and not DFlash1. The selector
  only exists on the V2 speculator; `_is_dflash2_draft()` forces V2, so a V1 log
  line means something is wrong. Check for `dflash_head` compile tags and no
  `DFlashProposer`.

## Losslessness check (cheap, do it first)

Arm A and arm C, same eight prompts, `temperature 0`, `max_tokens 512`: the
completions must be **identical**. DFlash2 claims lossless greedy decoding and
vLLM verifies against the target, so any divergence is a bug, not a tradeoff.
`bench_concurrency.py` at concurrency 1 prints token counts; capture the actual
text with a diff run, or reuse
`goal/sessions/23cb78c0-.../artifacts/scripts/compare_logprobs.py`.

## Decision rule

Switch the production latency profile to DFlash2 only if **all** hold:

1. Greedy output identical to the no-spec control (losslessness).
2. Concurrency 1 and 8 per-request decode tok/s beat arm B by >10% on the
   *mean* of the eight prompts — not just on code-edit. MTP's weakness is prose;
   DSpark was rejected because it bought code and lost prose (39-42 -> 27-28).
   Report per-prompt, decide on the mean and the worst prompt.
3. Concurrency 32 aggregate is no worse than arm B. Upstream shows DFlash2's
   advantage shrinking with batch; on L4 the drafter's replicated fc/selector
   work does not shard, so this is where it is most likely to lose.
4. KV pool >= ~900k tokens (see above), and no OOM in the 128x1k/1k burst.

If 2 and 3 pass but 4 fails, the fallback is arm C at `max-num-seqs 32` as an
opt-in profile alongside prod, the way `qwen3_8_27b_fp8_dspark_code.yaml` is.

## Exact commands for the GPU window

```bash
# --- 0. prepare arm yamls (CPU, safe to do now) -------------------------------
W=/shared/vllm-dflash2
mkdir -p /tmp/dflash2-arms
# arm A: no spec, no offload
grep -v -e '^speculative-config:' -e '^kv-transfer-config:' -e '^additional-config:' \
  /shared/vllm/serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml \
  | sed 's/^port: 8012/port: 8013/' > /tmp/dflash2-arms/A_nospec.yaml
# arm B: prod MTP spec-config, no offload
grep -v -e '^kv-transfer-config:' \
  /shared/vllm/serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml \
  | sed 's/^port: 8012/port: 8013/' > /tmp/dflash2-arms/B_mtp.yaml
# arm C: the new profile (already on 8013, already has no kv-transfer-config)
cp $W/serve-configs/qwen3_8_27b_fp8_dflash2_latency.yaml /tmp/dflash2-arms/C_dflash2.yaml

# --- 1. stop prod -------------------------------------------------------------
sudo -n systemctl stop vllm-qwen38 vllm-qwen38-throughput; sleep 5
rm -f /dev/shm/vllm_offload_*.mmap
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # expect ~0 MiB on all four

# --- 2. start one arm ---------------------------------------------------------
# $1 = arm yaml, $2 = log path
set -a; . /etc/vllm/qwen38.env; set +a
HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 PYTHONHASHSEED=8012 \
  PYTHONPATH=/shared/vllm/serve-configs/middleware \
  VLLM_USE_V2_MODEL_RUNNER=1 NCCL_P2P_LEVEL=SYS \
  setsid nohup /shared/vllm/.venv-dflash2/bin/vllm serve --config "$1" > "$2" 2>&1 &
# wait for health (model load + inductor + graph capture: several minutes)
for i in $(seq 1 120); do sleep 10; \
  curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8013/health | grep -q 200 && { echo "up ~$((i*10))s"; break; }; \
  grep -q 'Engine core initialization failed\|CUDA out of memory\|Application startup failed' "$2" && { tail -30 "$2"; break; }; done
grep -E 'GPU KV cache size|Available KV cache memory|dflash|speculat' "$2" | head -20

# --- 3. warm ------------------------------------------------------------------
/shared/vllm/.venv-dflash2/bin/python $W/serve-configs/bench_concurrency.py 8013 1 128 >/dev/null
/shared/vllm/.venv-dflash2/bin/python $W/serve-configs/bench_concurrency.py 8013 8 128 >/dev/null

# --- 4. run -------------------------------------------------------------------
for C in 1 8 32; do
  /shared/vllm/.venv-dflash2/bin/python $W/serve-configs/bench_concurrency.py 8013 $C 512
done
/shared/vllm/.venv-dflash2/bin/python $W/serve-configs/bench_single_stream.py 8013
/shared/vllm/.venv-dflash2/bin/python \
  $W/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T131256Z-afa8720e/artifacts/scripts/burst_client.py \
  8013 128 1024 1024
curl -s http://localhost:8013/metrics | grep '^vllm:spec_decode'

# --- 5. stop the arm, next arm -----------------------------------------------
pkill -f '[.]venv-dflash2/bin/vllm serve'; sleep 3
pkill -9 -f '[.]venv-dflash2/bin/vllm serve'; pkill -9 -f 'VLLM::EngineCore'; sleep 2
rm -f /dev/shm/vllm_offload_*.mmap
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# --- 6. restore prod ----------------------------------------------------------
sudo -n systemctl start vllm-qwen38
until curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8012/health | grep -q 200; do \
  sleep 10; systemctl is-active --quiet vllm-qwen38 || { echo 'prod unit died'; break; }; done
echo "prod: $(systemctl is-active vllm-qwen38)"
```

Restoring prod is not optional and not last-if-there-is-time: `vllm-qwen38` is
the served endpoint behind the Cloudflare tunnel. If an arm wedges, step 5 then
step 6 — the prod unit's `ExecStartPre` clears `/dev/shm/vllm_offload_*.mmap`
itself, but a `VLLM::EngineCore` left alive will hold VRAM and the prod start
will OOM.

## Order of work

1. Arm C first, briefly, just to confirm it starts and to read the KV pool line.
   If the pool is unusable, everything after is academic and the run becomes
   "arm C at `max-num-seqs 32`" instead.
2. Losslessness (A vs C, concurrency 1).
3. Full A, B, C at 1/8/32 plus the burst.
4. B' and C' only if B and C are close enough that attribution matters.
5. Restore prod, write the numbers into the goal-run artifacts, update
   `serve-configs/patches/README.md` if the profile changes.
