---
reviewer_id: "completeness"
reviewer_lane: "completeness"
review_round: 1
goal: "Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles: (1) LATENCY: max single-stream decode + min TTFT preserving K7 MTP, 262k ctx, stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high concurrency preserving 262k ctx, stability, quality; profile continuously, optimize the bottleneck, benchmark + stress each change vs best, stop only when remaining gains are negligible or hardware-bound"
conversation_id: "23cb78c0-b3a1-4ea3-89e2-332443ff1f8f"
run_id: "20260818T055754Z-8dd3de85"
review_head: "b3c9e81"
merge_base: "68dfda8"
proof_scope: "live-proof"
signed_off: false
verifier_summary: "Blind completeness attack on commit b3c9e81 (two Qwen3.8 profiles). Verified: all 25 manifest paths exist; live 8012 proof for both units matches the journal timeline (latency unit up 10:53:50, ss_bench 10:54; throughput unit up 10:59:08, c32 bench 10:59); installed unit files identical to repo copies; running unit carries VLLM_USE_V2_MODEL_RUNNER=1; health 200; patches 0005+0006 reproduce the venv byte-for-byte from the .orig0005 originals; final3 c128/c64/c32 JSONs are 128/128 successful at 827.9/680.2/479.6; both 200k needle logs show retrieval. Incomplete: the goal-listed cold long-prefill TTFT metric for the latency profile has no artifact (37k=17.1s and V1-vs-V2 90k comparison are prose-only / self-declared in flight); the latency 128-burst stress artifact shows 32/128 client-successful and is not in the manifest; 'greedy tokens identical' has no baseline-vs-final output artifact; qwen3_8_27b_fp8_max.yaml comment numbers (814/681, TTFT -25%) disagree with the final3 artifacts/commit (828/680, TTFT -30..-65%); DBO prefill timings (16k 7.8->5.7 s) have no artifact."
evidence:
  - "artifacts/manifest.json: 25/25 listed paths exist on disk (python os.path.exists loop)"
  - "artifacts/live/20260818T105400Z-prod-8012-ss_bench.log = 61.6/54.6/96.9/143.1 tok/s; journalctl -u vllm-qwen38 shows 'Starting vLLM server on http://0.0.0.0:8012' at 10:53:50 and 4 POST /v1 requests 10:50-10:58"
  - "artifacts/live/20260818T105911Z-prod-8012-throughput-c32.json: completed 128/128, output_throughput 475.96, mean_tpot 62.45; journalctl -u vllm-qwen38-throughput shows server start 10:59:08 and Running: 32 reqs"
  - "artifacts/logs/bench-tp_final3_c{128,64,32}.json: 128/128 ok, 827.9/680.2/479.6 tok/s, TPOT 136.8/87.3/62.6 (matches manifest, commit); baseline bench-tp_base_c{128,64,32}.json 621.7/632.6/452.6"
  - "artifacts/logs/needle200k-lat_final_v2.log and needle200k-tp_final.log: prompt_tokens 199628, completion '736251', needle retrieved: True (183.9 s / 158.0 s)"
  - "patch reproduction: copied 16 *.orig0005 files to a temp tree, patch -p1 0005 then 0006, cmp against .venv-qwen38 -> only difference is patch's own vllm/config/vllm.py.orig backup; new file draft_lm_head.py identical"
  - "systemctl show FragmentPath for both units -> /etc/systemd/system/*.service diff -q identical to serve-configs/systemd/*.service; /proc/<MainPID>/environ has VLLM_USE_V2_MODEL_RUNNER=1, NCCL_P2P_LEVEL=SYS, PYTHONPATH=serve-configs/middleware; vllm-qwen38 active/enabled, vllm-qwen38-throughput inactive/disabled; curl /health 200"
  - "artifacts/logs/bench-lat_final_v2_burst128.json: completed 32, failed 96 ('Never received a valid chunk to calculate TTFT'); server-lat_final_v2b.log shows the burst drained (Running 32 / Waiting 19 -> 0) and the EngineDeadError at 09:53:59 is the down.sh SIGTERM shutdown, not a crash"
  - "no artifact under artifacts/logs matches the '37k cold TTFT 17.1s' or DBO prefill '16k 7.8 -> 5.7 s' claims (grep of logs, only ttft_probe.py script present); goal.mdscript.md Next Steps: 'V1-vs-V2 90k TTFT comparison for the latency profile (in flight)'"
commands_run:
  - "python3 - <<EOF (json manifest exists loop) in run_dir"
  - "cat artifacts/live/*ss_bench.log artifacts/live/*throughput-c32.log; python3 json summary of artifacts/logs/bench-*.json and artifacts/live/*.json"
  - "cat artifacts/logs/ss-lat_base.log ss-lat_final_v2_offload.log ss2-lat_final_v2.log ss-lat_int4_v2_adaptive.log needle200k-*.log; cat progress.jsonl goal.mdscript.md; git show -s --format=%B b3c9e81"
  - "cat review-round1-serve-configs.diff (full); diff artifacts/configs/tp_final.yaml serve-configs/qwen3_8_27b_fp8_max.yaml; diff artifacts/configs/lat_final.yaml serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml"
  - "cp *.orig0005 -> scratchpad tree; patch -p1 -s < 0005; patch -p1 -s < 0006; cmp each file vs .venv-qwen38 site-packages"
  - "systemctl is-active/is-enabled vllm-qwen38 vllm-qwen38-throughput; systemctl show -p FragmentPath/MainPID/ActiveEnterTimestamp; diff -q installed vs repo units; tr '\\0' '\\n' < /proc/<pid>/environ | grep VLLM_USE_V2|NCCL|PYTHONPATH; curl -s http://127.0.0.1:8012/health; journalctl -u vllm-qwen38 -u vllm-qwen38-throughput --since 10:30 --until 11:10 (read-only)"
  - "grep -n -B3 -A25 EngineDeadError artifacts/logs/server-lat_final_v2b.log; grep -v Namespace artifacts/logs/bench-lat_final_v2_burst128.log; grep -rn ttft/17.1/90k goal.mdscript.md progress.jsonl; ls artifacts/logs"
attack_attempts:
  - "Attack: manifest paths missing or pointing at other runs -> FAILED (25/25 exist inside run_dir; each proves-string cross-checked against the file contents for the ss, needle, and bench-final3 entries)"
  - "Attack: patch files do not reproduce the venv (README claim 'verified to reproduce venv state') -> FAILED (independent re-application from .orig0005 originals cmp-identical; only patch's own .orig backup differs)"
  - "Attack: live proof was actually taken on the 8013 experiment server / not the systemd unit -> FAILED (journal shows the latency unit up at 10:53:50 and 4 POSTs before the 10:59 throughput unit start; throughput unit journal shows Running: 32 reqs; result_dir/port 8012 in the bench Namespace line; installed unit files identical to repo)"
  - "Attack: latency profile 'stress-tested / 128-burst all completed' -> PARTIAL SUCCESS (only artifact shows 32/128 client-successful, 96 marked failed by the bench client; server log shows the burst drained without crash; artifact not in the manifest, no rerun without middleware)"
  - "Attack: goal-listed metric 'cold long-prefill TTFT' for the latency profile is unevidenced -> SUCCESS (no ttft log for the final V2 latency profile; '37k cold TTFT 17.1s (= V1)' is prose only in goal.mdscript.md front matter; the run's own Next Steps marks the V1-vs-V2 90k comparison 'in flight'; needle 200k cold prefill on the final latency profile is 183.9 s vs 158 s on the throughput profile and the yaml's stated ~155 s cold, so a V2 prefill regression is possible and undocumented)"
  - "Attack: numbers in yaml comments / README / commit inconsistent with artifacts -> SUCCESS (qwen3_8_27b_fp8_max.yaml comment: '622 -> 814 tok/s at 128 concurrent ... 633 -> 681 at 64' while final3 artifacts and commit say 828/680; 'TTFT under load -25%' not derivable from artifacts: c128 50.6->17.7 s (-65%), c64 9.9->6.9 s (-30%), c32 6.1->4.3 s (-30%); DBO prefill timings 16k 7.8->5.7 s / 9k / 37k and '37k cold TTFT 17.1s' have no artifact)"
  - "Attack: 'greedy tokens identical / outputs unchanged' for the latency profile has no artifact -> SUCCESS (ss_bench.py does not save outputs; no baseline-vs-final cmp_out artifact; outputs-dbo.json is a single-config sanity dump with no baseline; progress iter2 itself notes single-prompt greedy runs can diverge between configs)"
  - "Attack: live latency numbers below the stated range -> minor (live reason 61.6 tok/s vs yaml/commit '62-66'; prose 54.6 vs '56'; within run-to-run noise but the published range excludes the only live sample)"
p_findings:
  - grade: "P1"
    location: "goal.mdscript.md (front matter best_latency '37k cold TTFT 17.1s (= V1)'; Next Steps 'V1-vs-V2 90k TTFT comparison ... (in flight)'); artifacts/logs (no ttft-*.log for the final latency profile); serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml comment block"
    summary: "The goal's latency profile is 'max single-stream decode + min TTFT' and the Goal Contract lists 'cold long-prefill TTFT' as a latency metric, but no artifact measures cold long-prefill TTFT for the final V2 latency profile vs the V1 baseline; the run's own record marks that comparison unfinished. The only long-prefill datapoint (200k needle, 183.9 s cold) is slower than the throughput profile (158 s) and the yaml's stated ~155 s cold, so a V2 prefill/TTFT regression is neither ruled out nor documented."
    contract: "LATENCY profile: min TTFT; Goal Contract metric 'cold long-prefill TTFT'; 'benchmark + stress each change vs best'"
    remediation: "Run artifacts/scripts/ttft_probe.py at 37k and 90k tokens against the final latency profile (V2) and against the same yaml with V1 (VLLM_USE_V2_MODEL_RUNNER unset), save logs under artifacts/logs, add them to manifest.json, and record the delta in the yaml comment/commit (document if >10% regression per the goal's own rule)."
  - grade: "P2"
    location: "artifacts/logs/bench-lat_final_v2_burst128.json/.log (not in artifacts/manifest.json); progress.jsonl iter7 '128-burst all completed'; commit message 'both re-benchmarked and stress-tested'"
    summary: "The latency profile's only 128-burst stress artifact records completed 32 / failed 96 (client mis-parses keepalive pings); the 'all completed, server healthy' claim rests on server-lat_final_v2b.log engine-log lines, and the burst artifact is not indexed in the manifest. No clean burst run of the final latency profile exists (unlike throughput final3 which was rerun without the middleware)."
    contract: "invariant 'no crashes under 128-burst stress'; 'benchmark + stress each change vs best'"
    remediation: "Rerun the 128-burst on the final latency config without the middleware (or with --temperature 0 and a ping-tolerant client) so the artifact shows 128/128 successful, add it (and the server log) to manifest.json."
  - grade: "P2"
    location: "serve-configs/qwen3_8_27b_fp8_max.yaml header comment (lines added 2026-08-18: '622 -> 814 tok/s at 128 concurrent (TPOT 137 ms), 633 -> 681 at 64', 'TTFT under load -25%', 'Prefill 16k: 7.8 -> 5.7 s; 9k: 3.5 -> 2.6 s; 37k ~-3%'); serve-configs/patches/README.md 0006 row; commit b3c9e81 body"
    summary: "Numbers in the shipped comments disagree with or are not backed by the artifacts: final3 JSONs and the commit say 828/680 (yaml says 814/681, the earlier dbo128_ch1 run); TTFT change per artifacts is -65%/-30%/-30% at c128/c64/c32, not '-25%'; the DBO prefill timings (16k/9k/37k) and the 16k baseline exist only as prose in progress.jsonl - no ttft/prefill log artifact on disk."
    contract: "'benchmark ... each change vs best' with evidence; yaml/README are the operator-facing record of what was measured"
    remediation: "Update the yaml comment to the final3 numbers (828/680/480) and the actual TTFT deltas, and either add the DBO prefill timing logs (ttft_probe outputs for 9k/16k/37k, DBO on/off) to artifacts/logs + manifest or mark those figures as unlogged."
  - grade: "P2"
    location: "serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml comment ('quality unchanged (greedy tokens identical: drafts only)'); commit body 'outputs unchanged (drafter-only changes)'; artifacts/logs (no baseline-vs-final output comparison)"
    summary: "'Greedy tokens identical' is asserted from construction, not shown: ss_bench.py does not persist outputs, there is no cmp_out.py capture on the baseline vs the final latency profile, and progress iter2 notes single-prompt greedy runs can diverge between configs. outputs-dbo.json (throughput) is a single-config sanity dump with nothing to compare against."
    contract: "invariant 'quality unchanged (greedy outputs / needle probe)'"
    remediation: "Capture cmp_out.py (or ss_bench with saved completions) on the 68dfda8 baseline config and on the final latency profile, diff the greedy outputs, save both JSONs to artifacts/logs and index them; do the same for DBO on/off on the throughput profile."
  - grade: "P3"
    location: "artifacts/live/20260818T105400Z-prod-8012-ss_bench.log vs yaml/commit '62-66/56/95/144'; artifacts/live/*throughput-c32.* (c32 only)"
    summary: "The single live latency sample (61.6/54.6/96.9/143.1) sits just under the published reason/prose range; the throughput unit's live proof is at 32 concurrent only (128-concurrent number is from the 8013 experiment server with an equivalent yaml). Both are disclosed in the packet; noted for completeness of the published ranges."
    contract: "live proof for the primary user action; published numbers should include the live sample"
    remediation: "State the range as 61-66/55-56 or take a second live ss_bench sample; optionally run one c128 vllm bench serve against the live throughput unit in a maintenance window."
rules_reviewed:
  - "/shared/vllm/AGENTS.md (uv/venv-only Python; accountability wording in commits; venv patches human-owned for upstream) - commit body carries the AI-assistance statement and Co-authored-by trailer"
  - "review packet Claim Under Review / Proof Supplied / Proof Not Claimed / Open Questions"
  - "goal.mdscript.md Goal Contract (metrics, invariants, stop condition) and front matter best_* claims"
  - "self-review completeness lane MDScript (attack surface + sign-off decision)"
artifact_paths:
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/manifest.json"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/20260818T105400Z-prod-8012-ss_bench.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/20260818T105911Z-prod-8012-throughput-c32.json"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/bench-tp_final3_c128.json"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/bench-lat_final_v2_burst128.json"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/server-lat_final_v2b.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/needle200k-lat_final_v2.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/needle200k-tp_final.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-serve-configs.diff"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/goal.mdscript.md"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/progress.jsonl"
  - "/shared/vllm/serve-configs/qwen3_8_27b_fp8_max.yaml"
  - "/shared/vllm/serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml"
  - "/shared/vllm/serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch"
  - "/shared/vllm/serve-configs/patches/0006-dense-tp-dual-batch-overlap-prefill-allreduce.patch"
  - "/etc/systemd/system/vllm-qwen38.service"
  - "/etc/systemd/system/vllm-qwen38-throughput.service"
objectives_checked:
  - "latency profile single-stream decode gain with K7 MTP kept: EVIDENCED (ss logs + live 8012)"
  - "latency profile min TTFT / cold long-prefill TTFT vs baseline: NOT EVIDENCED (P1)"
  - "latency profile 262k ctx: EVIDENCED (200k needle retrieved at gpu-mem 0.92)"
  - "latency profile stability under 128-burst: PARTIAL (server drained, client artifact 32/128) (P2)"
  - "latency profile quality unchanged: ASSERTED, no output diff artifact (P2)"
  - "throughput profile aggregate tok/s at 128/64/32 vs baseline: EVIDENCED (final3 JSONs 128/128 ok) ; live unit proof at c32: EVIDENCED"
  - "throughput profile 262k ctx: EVIDENCED (200k needle retrieved)"
  - "patches reproduce venv state: EVIDENCED (independent cmp)"
  - "systemd units installed and match repo; latency unit active with V2 env; health 200: EVIDENCED"
  - "yaml/README/commit numbers consistent with artifacts: NOT FULLY (P2)"
remaining_gaps:
  - "cold long-prefill TTFT (37k/90k) for the final latency profile vs V1 baseline, with artifacts"
  - "clean 128-burst artifact (128/128 successful) for the final latency profile, indexed in the manifest"
  - "baseline-vs-final greedy output comparison artifacts for both profiles"
  - "qwen3_8_27b_fp8_max.yaml comment numbers reconciled with final3 artifacts; DBO prefill timing artifacts or an explicit 'unlogged' note"
signed_off_at: "2026-08-18T11:15:30Z"
repair_resume_command: "/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/goal.mdscript.md#pursue-goal"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Signoff

* lane verdict: completeness — `signed_off: false` (round 1). Live proof for the primary user action exists for both units on port 8012, all manifest paths exist, patches reproduce the venv, and the throughput numbers are artifact-backed; the goal's latency TTFT criterion, the latency stress artifact, the quality-unchanged claim, and several shipped comment numbers are not backed by artifacts.
* P1 — goal.mdscript.md best_latency / Next Steps; artifacts/logs; qwen3_8_27b_fp8_mtp_latency.yaml comment: no cold long-prefill TTFT measurement for the final V2 latency profile vs V1 (comparison self-declared 'in flight'; 200k cold 183.9 s vs 158 s / ~155 s). Remediation: run ttft_probe.py at 37k/90k on V2 vs V1 for the latency yaml, save + index the logs, document the delta.
* P2 — artifacts/logs/bench-lat_final_v2_burst128.json (32/128 ok, not in manifest) vs 'stress-tested / all completed'. Remediation: rerun the 128-burst on the final latency config without the middleware, add 128/128 artifact + server log to manifest.json.
* P2 — serve-configs/qwen3_8_27b_fp8_max.yaml comment (814/681, TTFT -25%, DBO prefill 16k 7.8->5.7 s) vs final3 artifacts/commit (828/680; TTFT -65/-30/-30%) and no prefill-timing artifacts. Remediation: reconcile the comment to the final3 numbers and add or mark-unlogged the DBO prefill timing logs.
* P2 — 'greedy tokens identical / outputs unchanged' (latency yaml comment, commit) has no baseline-vs-final output artifact; outputs-dbo.json has no baseline. Remediation: capture cmp_out.py on baseline and final for both profiles, diff, save + index.
* P3 — live ss_bench 61.6/54.6 sits under the published '62-66/56'; throughput live proof is c32 only. Remediation: widen the stated range or take a second live sample; optional c128 live run in a maintenance window.

## Resume From Signoff

* `signed_off` is `false`: the next jump is `/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/goal.mdscript.md#pursue-goal` (repair_resume_command) to close the P1/P2 gaps above, then the composer must spawn a fresh blind completeness reviewer with a new packet and a new sign-off path — never re-enter this lane's own review from this sign-off
* when a later round reaches `signed_off: true`, continue at `/mdscript-exec /home/gwillen/.agents/skills/self-review/workflows/triple-adversarial-blind-review.mdscript.md#aggregate-triple-signoffs`
