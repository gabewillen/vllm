# Venv-local vLLM patches for the Qwen3.8-27B deployment

Production runs the nightly wheel `vllm 0.27.2rc1.dev110+gacb0f1dcd` in
`/shared/vllm/.venv-qwen38` with three source patches that are NOT in any
release. A venv rebuild silently drops them; run `apply-to-venv.sh` afterwards.

| Patch | Fixes | Symptom without it |
|---|---|---|
| 0001 gpu_worker.py | flock-serialized, row-stride-aligned chunked `cudaHostRegister` of the /dev/shm offload region | startup deadlock with `cpu_bytes_to_use` >= ~16 GiB at TP4 (all ranks pin the whole region concurrently); naive 1 GiB chunking then breaks `cuMemcpyBatchAsync` (error 1) |
| 0002 shared_offload_region.py | unregister the same chunks on cleanup | leaked pinned mappings / errors at shutdown after 0001 |
| 0003 offloading/scheduler.py | detect the MTP draft tower by `mtp.` layer prefix instead of marking every KV group EAGLE-volatile | with `speculative-config` MTP enabled, tiered-offload restore never converges (fs tier hits, promotes, then full-recompute); resume of a 32k chat 20 s instead of 2-3 s |
| 0004 model_loader/weight_utils.py | `get_quant_config` tolerates a callable `hf_overrides` (draft models compose one) instead of raising | `speculative-config` with `"quantization"` on a dspark/eagle draft crashes at load with "hf_overrides must be a dict" (note: online-fp8 for the DSpark drafter still fails later in the DFlash precompute path on SM89 — Marlin-repacked weight; keep the drafter bf16) |

Evidence, repros and benchmarks: `goal/sessions/ea737f9a-4156-40c9-b1b8-1e900c68c3ee/runs/20260815T021827Z-475a50cd/artifacts/offload-debug/`.
All three are unreported upstream as of 2026-08-17 and are candidates for PRs
(human-owned per AGENTS.md).
