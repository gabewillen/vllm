# Interconnect findings — 4x L4, 2026-08-18

Topology: all four GPUs at "NODE" distance (through the CPU root complex, no shared PCIe switch).
GPU0/GPU1 = PCIe Gen4 **x8**, GPU2/GPU3 = Gen4 x16. P2P access matrix all-True.

NCCL all-reduce (torch.distributed, bf16), default settings:
| group | 10 KB (decode-size) | 1 MB | 84 MB (8192 tok x 5120, one prefill-chunk activation) |
|---|---|---|---|
| 0,1,2,3 | 35.7 us | 361 us, 4.4 GB/s | **25.9 ms, 5.1 GB/s bus** |
| 2,3 (x16/x16) | 33 us | 85 us, 12.3 GB/s | 4.66 ms, 18.9 GB/s |
| 0,1 (x8/x8) | 33 us | 114 us, 9.2 GB/s | 8.0 ms, 11.0 GB/s |
| 0,2 / 1,3 (x8-x16) | 33 us | 115 us | 8.0 ms, 11.0 GB/s |

128 all-reduces per forward: prefill chunk = 128 x 25.9 ms = 3.3 s of the ~5 s per 8192-token
chunk (matches the 65.8% NCCL share profiled earlier); decode step = 128 x 36 us = 4.6 ms of ~37 ms.

NCCL_P2P_LEVEL=SYS: 84 MB all-reduce 25.9 -> 11.7 ms (2.2x); ~90k-token cold prefill TTFT
61 s -> 41.6 s (-32%); decode unchanged. BUT: CUDA illegal-memory-access on a TP rank under the
128x1k/1k batch load, reproducible 3/3 (also with NCCL_PROTO=Simple), while all-reduce
correctness (900 iters, bit-exact) and light load are fine and no Xid is logged. Not shippable.
NCCL_ALGO=Tree: much worse. vLLM custom all-reduce: disabled by policy (>2 PCIe GPUs);
SymmMem: SM90+ only; pass_config fuse_allreduce_rms / fuse_gemm_comms / enable_sp: off.

Implications: (1) moving GPU0/GPU1 to x16 slots (or fixing bifurcation) is the only clean way to
speed the 4-way ring; (2) TP2 x PP2 with the x16 pair {2,3} as one stage would cut per-chunk
comm ~6x at the cost of single-stream decode; (3) kernel fusion addresses only the ~35%
compute share of prefill and the ~88% non-comm share of the decode step.

## Update 2026-08-18 (later): P2P crash root-caused and P2P deployed

Bisection (all with NCCL_P2P_LEVEL=SYS, 128x1k/1k batch): plain TP4 -> stable, **620 tok/s** (+30%);
any MTP config -> CUDA illegal-memory-access, regardless of IOMMU (off), ACS (cleared on host),
memlock (unlimited), NCCL_P2P_DIRECT_DISABLE / CUMEM / LOCAL_REGISTER / GRAPH_REGISTER / PROTO,
eager mode, async scheduling, offload on/off, dynamic vs static K, draft TP1 vs TP4. Only
CUDA_LAUNCH_BLOCKING=1 hid it (race). GPU core dump (CUDA_ENABLE_COREDUMP_ON_EXCEPTION +
lightweight flags, read with cuda-gdb `info cuda kernels`) names the faulting kernel:
`flashinfer::PersistentVariableLengthMergeStatesKernel` on device 1 — FlashInfer's split-KV merge on
the spec-decode verify path (multi-query-token). Fix: `attention-backend: TRITON_ATTN` for the target
AND `"attention_backend": "TRITON_ATTN"` in speculative-config (the draft layer picks its own backend
and stayed FlashInfer). With that MTP K=7 + P2P is stable (128/128, several runs).

Deployed (unit Environment=NCCL_P2P_LEVEL=SYS, git next): single-stream reason/prose/write/edit
46/40/68/103 -> 51/42/80/115 tok/s; cold ~90k prefill 61 -> 42 s. Costs: Triton attention pool
1.39M -> 1.21M tokens; batch aggregate at K=7 313 tok/s because MTP on this hybrid needs (K+1)
GDN state checkpoints per request = (K+1) x 1600-token blocks x 3 GDN groups (K=7 -> max 34
concurrent, K=2 -> 88 on the same pool; measured, matches the arithmetic). Irrelevant at <=8 agents.
Host prerequisites the user set: amd_iommu=off iommu=off; ACS cleared via setpci on
00:03.1 20:03.1 40:03.1 60:03.1 03:00.0 (runtime — re-apply after a host reboot); memlock unlimited.
