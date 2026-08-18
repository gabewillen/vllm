# SGLang 0.5.17 vs vLLM 0.27.2 nightly — Qwen3.8-27B-FP8 on 4x L4 (2026-08-18)

Same box, same benches (`bench_single_stream.py`, 768-tok greedy; `vllm bench serve` 128x1k/1k random),
SGLang venv `.venv-sglang` (torch 2.11 cu130), launcher in the goal artifacts (`sglang/sgl_launch.sh`):
TP4, context 262144, mem-fraction-static 0.88, kv fp8_e4m3, flashinfer attention.

| | vLLM no-spec | SGLang no-spec | vLLM MTP K=7 (prod) | SGLang NEXTN steps=7 |
|---|---|---|---|---|
| reasoning tok/s | 26.7 | 27.7 | 42-50 | 41-47 |
| prose | 26.7 | 27.6 | 37-40 | 39.7 |
| code-write | 26.7 | 27.6 | 66-69 | 68.7 |
| code-edit | 26.7 | 27.4 | 103 | 108.5 |
| 128x1k/1k aggregate | 476.7 (96 running) | 328.6 (27 running, default) / 420.5 (48 running, `--max-mamba-cache-size 240`) | 424.7 | not run (cap 8 running) |
| KV pool tokens | 1.39M | 733k (default) / 254k (240 slots) | 1.39M | 662k |
| max running | 96 | 27 / 48 | 96 | 8 |

Conclusions: single-stream parity (both engines are HBM/all-reduce bound with equivalent
GDN + fp8 kernels on SM89); MTP depth transfers 1:1. SGLang's structural problem here is
the *static* split between the fp32 GDN state pool (35 MB per slot, 5 slots/request) and
the KV pool: raising concurrency destroys long-context capacity (48 running -> 254k tokens),
and NEXTN spec-decode caps it at 8 running. vLLM's unified 1600-token block pool serves 96
requests AND 1.39M tokens. SGLang also lacks our tiered RAM/disk KV offload for this
hybrid model. Verdict: stay on vLLM; kernel work is the next lever on both engines equally.
