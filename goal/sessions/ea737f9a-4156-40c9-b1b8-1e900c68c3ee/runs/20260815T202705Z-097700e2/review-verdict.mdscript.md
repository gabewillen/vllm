---
goal: >-
  Improve vLLM and the Qwen3.8-27B-FP8 deployment to achieve a 40-60% prefill
  throughput speedup on the 4x L4 box.
conversation_id: ea737f9a-4156-40c9-b1b8-1e900c68c3ee
run_id: 20260815T202705Z-097700e2
grade: >-
  Proven for: 40-60% prefill speedup via the committed TP1xPP4 + 4096-token-chunk
  profile (serve-configs/qwen3_8_27b_fp8_prefill.yaml @ git f5ed912) — 32k prompt
  20.1s -> 10.4s (-48.3%, worst-case noise floor -40.4%) and 200k prompt
  155s -> 90.4s (-41.7%, 0.8% repeat spread), needle retrieval correct on the
  committed config; root cause (NCCL all-reduce 65.5-66.0% of GPU time on all 4
  TP4 ranks) profiled and causally confirmed by A/B topology measurements and
  bubble-model consistency; decode tradeoff (-51%: 233.2 vs 474.6 tok/s)
  measured and disclosed; TP4 remains production default.
proof_decision: Proven for
blocking_findings: []
lanes:
  - lane: rules-completeness
    verdict: PROVEN (after two blocking fixes)
    notes: >-
      fixed: (1) enable_sp claim corrected to UNTESTABLE (vLLM SP heuristic
      silently disabled it; iter3 was an inadvertent iter2 repeat exposing ~15%
      32k restart noise — ledger_correction appended, yaml amended f5ed912);
      (2) needle-correctness evidence archived (probe-events-*.txt + provenance).
  - lane: eng-perf-adversarial
    verdict: PROVEN, no blocking findings
    notes: >-
      offload-baseline confound bounded at ~0.5% of profile window vs 65.87%
      NCCL; probes verified cold via server-side 0.0% prefix-hit rate; JIT
      contamination margin-safe; bubble math matches measurements in 4
      independent ratio tests; decode bench like-for-like.
proof_supplied:
  - artifacts/logs/baseline-32k.txt, progress.jsonl (full lever ledger with reverts and ledger_correction)
  - artifacts/logs/iter{2,3,4,5,6}-*serve.log (engine args per iteration)
  - artifacts/traces/profiler_out_{0..3}.txt (NCCL attribution, all ranks)
  - artifacts/logs/probe-events-{iter2-tp2pp2,iter5-pp4,iter6-pp4b4k}.txt + PROVENANCE (latencies + needle retrieval)
  - artifacts/logs/bench-decode-pp4b4k.{json,log} vs prior-run bench-iter8-seqs96.json (decode tradeoff)
proof_not_claimed:
  - enable_sp effect (untestable on this model via default heuristic)
  - per-lever 32k rankings inside the ~15-20% restart-noise band
  - decode throughput of current production-with-offload (reference predates offload)
artifact_paths:
  - /shared/vllm/goal/sessions/ea737f9a-4156-40c9-b1b8-1e900c68c3ee/runs/20260815T202705Z-097700e2/artifacts/
  - /shared/vllm/serve-configs/qwen3_8_27b_fp8_prefill.yaml
reviewed_at: 2026-08-15T22:55:00Z
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Verdict

* two blind lanes (rules+completeness, adversarial eng-perf) independently reviewed; the two blocking findings from lane 1 were fixed and re-confirmed by the originating lane; lane 2 found no blockers after attempting to refute the causal claim
* aggregate: **Proven for** the scope in front matter, blocking findings empty
