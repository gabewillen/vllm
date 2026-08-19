---
goal: >-
  Serve Qwen/Qwen3.8-27B (FP8) on the 4x L4 box via vLLM with the full 262144-token
  context enabled, maximize serving throughput, and optimize until diminishing
  returns (<5% primary-metric gain over two consecutive optimization rounds).
conversation_id: ea737f9a-4156-40c9-b1b8-1e900c68c3ee
run_id: 20260815T021827Z-475a50cd
grade: >-
  Proven for: Qwen/Qwen3.8-27B-FP8 served via vLLM 0.27.2rc1.dev110 at TP4 on
  4x L4 with max-model-len 262144 accepted and a ~200k-token request served
  end-to-end (needle retrieved) on the committed config; throughput optimized
  to 474.61 output tok/s (vllm bench serve, 128x 1024/1024, ignore-eos) with
  the loop stopped by the contract's diminishing-returns rule (+2.2% then
  -0.1%); winning config committed as serve-configs/qwen3_8_27b_fp8_max.yaml
  (git 71c15d0) and byte-equivalent (engine keys) to the measured winner.
proof_decision: Proven for
blocking_findings: []
lanes:
  - lane: security
    verdict: PROVEN
    notes: nightly-pin operational risk and house-wide LAN-open posture flagged non-blocking
  - lane: rules-completeness
    verdict: PROVEN
    notes: all iteration numbers recomputed from bench JSONs and matched; non-blocking observations recorded (single-run noise on iter8 keep, 200k-not-262k literal proof scope, iter3 probe gap in superseded branch)
  - lane: eng-config
    verdict: PROVEN (after one blocking fix)
    notes: header falsely cited #49757 as nightly-only; fixed and re-confirmed by the same lane
proof_supplied:
  - artifacts/logs/bench-iter1-baseline.json through bench-iter9-*.json (throughput evidence, all 9 iterations)
  - artifacts/logs/long-ctx-probe-iter{0,2,4,5,6,7,8,9}.log + long-ctx-probe-final.log (262k-gate evidence incl. committed config)
  - artifacts/logs/iter8-serve.log + final-serve.log (engine args + KV pool for winner and committed config)
  - progress.jsonl (full iteration ledger with kept/reverted decisions)
proof_not_claimed:
  - a literal 262,144-token single request (gate is ~199.6k tokens per contract)
  - MTP speculative decoding throughput (documented untested in config; latency-mode candidate)
  - repeat-run variance bounds on individual iterations
artifact_paths:
  - /shared/vllm/goal/sessions/ea737f9a-4156-40c9-b1b8-1e900c68c3ee/runs/20260815T021827Z-475a50cd/artifacts/
  - /shared/vllm/serve-configs/qwen3_8_27b_fp8_max.yaml
reviewed_at: 2026-08-15T06:14:00Z
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Verdict

* three blind lanes (security, rules+completeness, eng-config) each independently reviewed the run artifacts; the single blocking finding (false #49757 pin rationale in the config header) was fixed, commit amended to 71c15d0, and re-confirmed PROVEN by the originating lane
* aggregate: **Proven for** the scope in front matter, with empty blocking findings
