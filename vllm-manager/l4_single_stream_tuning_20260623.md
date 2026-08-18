# L4 Single-Stream vLLM Tuning Log

Date: 2026-06-23 UTC

Goal: test whether Gemma4 single-stream text decode on NVIDIA L4 can be improved beyond the current vLLM GPTQ/Marlin path.

## Live Baseline

- Services active before tuning pass: `vllm-managed@hot-a`, `vllm-managed@hot-b`, `vllm-swapper`, `litellm-omni`.
- `hot-a`: `gemma4-e2b-w4a16` on L4 `GPU-019676ed-23c9-ad9a-20cb-7bdc13ac61ca`, port `9000`.
- `hot-b`: `gemma4-e4b-w4a16` on L4 `GPU-653b223f-f36b-2fcc-bdc9-78adcbab14eb`, port `9001`.
- Both advertise `max_model_len: 131072`.
- Current E2B profile settings: full context, `--max-num-seqs 8`, `--generation-config vllm`, KV cache dtype `auto`, structured outputs enabled, Gemma4 tool parser enabled.
- Current E4B profile settings: same as E2B.

## Baseline Measurements

Previous direct vLLM benchmarks:

| Model | Config | Single stream | Stream decode after TTFT | Aggregate result |
| --- | --- | ---: | ---: | --- |
| `gemma4-e2b-w4a16` | `max_num_seqs=1`, FP8 KV | 105.6 tok/s | 107.2 tok/s | serialized at about 105 tok/s |
| `gemma4-e4b-w4a16` | `max_num_seqs=1`, FP8 KV | 61.4 tok/s | 62.1 tok/s | serialized at about 61 tok/s |
| `gemma4-e2b-w4a16` | `max_num_seqs=8`, FP8 KV | 106.3 tok/s | 107.9 tok/s | concurrency 8: 519.1 tok/s; concurrency 16: 793.9 tok/s |
| `gemma4-e4b-w4a16` | `max_num_seqs=8`, FP8 KV | 61.5 tok/s | 62.3 tok/s | concurrency 8: 353.7 tok/s; concurrency 16: 461.6 tok/s |
| `gemma4-e2b-w4a16` | `max_num_seqs=8`, auto KV | 109.9 tok/s on 1024-token completion | not retested | not retested |
| `gemma4-e4b-w4a16` | `max_num_seqs=8`, auto KV | about 63 tok/s | not retested | not retested |

Ollama/llama.cpp comparison on the same L4 using local `gemma4:e2b-it-qat`:

| Runtime | Model | Context | Decode result |
| --- | --- | ---: | ---: |
| Ollama/llama.cpp | `gemma4:e2b-it-qat` GGUF/QAT | 65536 | 124.7 eval tok/s |
| vLLM | `gemma4-e2b-w4a16` AutoGPTQ | 131072 | 109.9 tok/s |

Interpretation so far: the L4 is not idle during single-stream decode; power reaches the 72 W L4 cap. The remaining gap is likely runtime/kernel/model-format behavior, not HTTP or LiteLLM overhead.

## Local vLLM Kernel Evidence

- AutoGPTQ currently selects `MarlinLinearKernel`.
- `MacheteLinearKernel` requires compute capability 90 in local vLLM code. L4 is SM89, so forcing `--linear-backend machete` is expected to fail.
- CUDA mixed-precision kernel priority in local vLLM is: `CutlassW4A8LinearKernel`, `MacheteLinearKernel`, `AllSparkLinearKernel`, `MarlinLinearKernel`, `HummingLinearKernel`, `ConchLinearKernel`, `ExllamaLinearKernel`, `TritonW4A16LinearKernel`.
- `ExllamaLinearKernel` has CUDA support and may support this W4A16 GPTQ shape.
- `TritonW4A16LinearKernel` exists and allows CUDA in `can_implement`, but its file header says it is tuned for ROCm MI300 and is after Exllama in CUDA priority.

## Experiment Matrix

| Area | Candidate | Status | Result |
| --- | --- | --- | --- |
| vLLM linear backend | `marlin` explicit | complete | E2B 110.29 tok/s, E4B 63.03 tok/s. |
| vLLM linear backend | `exllama` forced | complete | E2B 113.23 tok/s, E4B 63.48 tok/s. Small gain only. |
| vLLM linear backend | `triton` forced | complete | Startup/profile failed on AutoGPTQ qzeros layout. |
| vLLM linear backend | `machete` forced | skipped | Local code requires SM90/Hopper; L4 is SM89. |
| alternate quant format | AWQ, QAT compressed-tensors, NVFP4 | complete | AWQ slow, QAT slower than GPTQ, NVFP4 not loadable in local vLLM. |
| speculative decoding | Gemma4 MTP/draft | complete | Official E2B/E4B assistants work for text-only speculative decode with FlashInfer sampler disabled; measured 218.27 tok/s and 132.16 tok/s. |
| SM89 W4A16 kernel | port/enable path | complete | No simple SM89 W4A16 kernel switch gives a large win; Exllama is small, Triton/Machete need real kernel work. |

## Linear Backend Experiments

Method:

- Stopped one hot vLLM user service at a time.
- Started a disposable vLLM server on a test port with the same model/profile settings plus `--linear-backend`.
- Waited for `/v1/models`, ran a warmup completion, then measured direct `/v1/completions`.
- Killed the disposable server and restored the user service.

### E2B Backend Results

Result file: `/shared/eve/vllm-manager/linear_backend_e2b_results_20260623.jsonl`

| Backend | Startup | Selected kernel | Measured decode | Context | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `marlin` | OK | `MarlinLinearKernel` | 110.29 tok/s avg, 2 x 1024 tokens | 131072 | Matches current `auto` behavior. |
| `exllama` | OK | `ExllamaLinearKernel` | 113.23 tok/s avg, 2 x 1024 tokens | 131072 | About +2.7% over explicit Marlin. |
| `triton` | failed before serving | `TritonW4A16LinearKernel` path reached | n/a | n/a | Fails during profile/dummy run: `qzeros shape mismatch: torch.Size([320, 12])`. |

E2B conclusion: Exllama is supported and slightly faster for single-stream decode, but it is not a large gain. Forced Triton is not usable for this AutoGPTQ checkpoint without code changes.

### E4B Backend Results

Result file: `/shared/eve/vllm-manager/linear_backend_e4b_results_20260623.jsonl`

| Backend | Startup | Selected kernel | Measured decode | Context | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `marlin` | OK | `MarlinLinearKernel` | 63.03 tok/s avg, 2 x 512 tokens | 131072 | Matches current `auto` behavior. |
| `exllama` | OK | `ExllamaLinearKernel` | 63.48 tok/s avg, 2 x 512 tokens | 131072 | About +0.7% over explicit Marlin. |

E4B conclusion: Exllama is supported but the speedup is noise-level/small. It also uses a larger CUDA graph pool and slightly reduces available KV cache, though full 131072 context still fits.

Production implication: Exllama can be enabled if we want a small single-stream improvement, especially for E2B. It is not a path to 150+ tok/s single stream on L4 by itself.

## Alternate Quant Format Experiments

### Chunity E2B AWQ

Candidate: `Chunity/gemma-4-E2B-it-AWQ-4bit`

- Downloaded snapshot: `/shared/eve/huggingface/hub/models--Chunity--gemma-4-E2B-it-AWQ-4bit/snapshots/79fb33d4c2d52338e9e6de36021d1bba8342db91`
- Quantization: AWQ, 4-bit, group size 128, `zero_point=false`, `version=gemm`.
- Result file: `/shared/eve/vllm-manager/quant_format_e2b_awq_results_20260623.jsonl`
- Startup: OK, full `max_model_len: 131072`.
- Measured decode: 28.76 tok/s avg, 2 x 1024 tokens.
- Kernel/log evidence: vLLM reported `quantization=awq` but did not log `AutoAWQMarlinLinearMethod`; runtime generation logger stabilized around 28-29 tok/s.

Why this was slow:

- Local L4 Marlin compatibility check returns false for this checkpoint shape: AWQ `uint4` with `zero_point=false`.
- The same check returns true for asymmetric AWQ `uint4` with `zero_point=true` and for GPTQ-style `uint4b8`.
- This checkpoint therefore falls back to vLLM's unoptimized AWQ path on L4.

Conclusion: not viable for speed. It is about 4x slower than the current E2B GPTQ path.

### Google E2B QAT Compressed-Tensors

Candidate: `google/gemma-4-E2B-it-qat-w4a16-ct`

- Downloaded snapshot: `/shared/eve/huggingface/hub/models--google--gemma-4-E2B-it-qat-w4a16-ct/snapshots/a50d96741aa53e1e0f1e8b6ea73230dce02b3341`
- Quantization: compressed-tensors `pack-quantized`, symmetric int4, group size 32, W4A16 QAT lineage.
- Result file: `/shared/eve/vllm-manager/quant_format_e2b_qat_ct_results_20260623.jsonl`
- Startup: OK, full `max_model_len: 131072`.
- Selected kernel: `MarlinLinearKernel` via `CompressedTensorsWNA16`.
- Measured decode: 106.12 tok/s avg, 2 x 1024 tokens.

Conclusion: loads cleanly and uses Marlin, but it is slower than the current GPTQ model (`~110 tok/s` Marlin, `~113 tok/s` Exllama). It does not explain llama.cpp's faster QAT path.

### BG Digitalservices E2B NVFP4

Candidate: `bg-digitalservices/Gemma-4-E2B-it-NVFP4A16`

- Downloaded snapshot: `/shared/eve/huggingface/hub/models--bg-digitalservices--Gemma-4-E2B-it-NVFP4A16/snapshots/c66416818c9899ab7b97acac42e0373d5da03b57`
- Quantization metadata: `quant_algo: NVFP4`, 4-bit float weights and 4-bit float input activations, group size 16. Despite the repo name, the config is not W4A16; it is NVFP4 activation plus NVFP4 weight quantization.
- Result files:
  - `/shared/eve/vllm-manager/quant_format_e2b_nvfp4_results_20260623.jsonl`
  - `/shared/eve/vllm-manager/quant_format_e2b_nvfp4_fixed_results_20260623.jsonl`
- First startup attempt: failed before loading weights because the repo includes `model.safetensors.index.json` with an empty `weight_map`, causing vLLM to filter out the real monolithic `model.safetensors`.
- Fixed-view startup attempt: created `/shared/eve/vllm-manager/model-fixes/Gemma-4-E2B-it-NVFP4A16-no-empty-index` with the empty index omitted. This got past weight discovery but failed while loading weights:
  - `KeyError: 'layers.15.self_attn.k_proj.weight'`

Conclusion: this checkpoint is not currently loadable in local vLLM as a drop-in Gemma4 E2B serving candidate. It has a packaging issue and then a model/weight-layout mismatch with the local Gemma4 shared-KV loader. It therefore cannot be benchmarked as a faster L4 path without checkpoint-specific loader work.

## Speculative Decoding Experiment

Candidates:

- `google/gemma-4-E2B-it-assistant`
- `google/gemma-4-E4B-it-assistant`

- Method: `--spec-method mtp --spec-model google/gemma-4-E2B-it-assistant --spec-tokens 4`
- E2B result file: `/shared/eve/vllm-manager/speculative_e2b_results_20260623.jsonl`
- E2B successful log: `/shared/eve/vllm-manager/backend_test_logs/e2b_spec_mtp_native_sampler_20260623T051143Z.log`
- E4B result file: `/shared/eve/vllm-manager/speculative_e4b_results_20260623.jsonl`
- E4B successful log: `/shared/eve/vllm-manager/backend_test_logs/e4b_spec_mtp_native_sampler_20260623T051736Z.log`

Startup findings:

- vLLM resolved both assistants as `Gemma4MTPModel`.
- Full `max_model_len: 131072` still fits with the drafter resident for E2B and E4B.
- E2B GPU KV cache with the drafter: `2,197,287 tokens`; maximum concurrency for 131072-token requests: `16.76x`.
- E4B GPU KV cache with the drafter: `630,325 tokens`; maximum concurrency for 131072-token requests: `4.81x`.
- The drafter is text-only. vLLM logged: `Draft model does not support multimodal inputs, falling back to text-only mode`.

FlashInfer sampler issue:

- With default sampling, startup failed while compiling FlashInfer top-k/top-p sampling for SM89:
  - `BlockAdjacentDifference<__nv_bool, ...> has no member "FlagHeads"`
- Setting `VLLM_USE_FLASHINFER_SAMPLER=0` avoids that broken JIT path and lets the server start with native/Triton sampling.

Measured text-to-text decode:

| Config | Startup | Context | Decode |
| --- | --- | ---: | ---: |
| E2B GPTQ baseline, no speculative decode | OK | 131072 | ~109.9 tok/s |
| E2B GPTQ + official E2B MTP assistant, `spec_tokens=4`, native sampler | OK | 131072 | 218.27 tok/s avg, 2 x 494 completion tokens |
| E4B GPTQ baseline, no speculative decode | OK | 131072 | ~63 tok/s |
| E4B GPTQ + official E4B MTP assistant, `spec_tokens=4`, native sampler | OK | 131072 | 132.16 tok/s avg, 2 x 384 completion tokens |

Conclusion: speculative decoding is the first tested path that clearly beats the requested 150 tok/s single-stream target on L4 for E2B, and it more than doubles E4B. It is a text-to-text fast path. I did not make it the production default because the current assistant path is text-only and should be separated from multimodal serving until multimodal requests are tested under speculative decode.

## SM89 Kernel Feasibility

The tested routes do not expose a simple flag-level path to 150+ tok/s single-stream decode on L4:

- `MarlinLinearKernel` is the current AutoGPTQ default and gives about `110 tok/s` for E2B.
- `ExllamaLinearKernel` works and improves E2B to `113.23 tok/s`, about `+2.7%`; E4B only improves from `63.03` to `63.48 tok/s`.
- `TritonW4A16LinearKernel` can be forced, but this GPTQ checkpoint fails during the profile/dummy run with `qzeros shape mismatch: torch.Size([320, 12])`. Fixing that would require adapting the Triton path to this AutoGPTQ qzeros layout, then validating performance on CUDA SM89. The local file notes that path is primarily tuned for ROCm MI300, so the fix is not guaranteed to be faster even if it loads.
- `MacheteLinearKernel` is gated to compute capability 90. L4 is SM89. Enabling it is not a one-line config change; it would mean porting or replacing the kernel path for Ada.
- The symmetric AWQ checkpoint falls off the Marlin-compatible path and is much slower.
- The compressed-tensors QAT checkpoint uses Marlin but is slower than the current GPTQ model.
- The NVFP4 checkpoint is not loadable with the current Gemma4 loader.

Practical conclusion: kernel work could get higher, but it is real CUDA/vLLM kernel engineering, not tuning. The closest low-risk linear-kernel config change is `--linear-backend exllama`, which is only a small E2B win. The practical large win on this machine is speculative decoding with `VLLM_USE_FLASHINFER_SAMPLER=0`: E2B + the official MTP assistant reached `218.27 tok/s`, and E4B reached `132.16 tok/s`, both at full context.
