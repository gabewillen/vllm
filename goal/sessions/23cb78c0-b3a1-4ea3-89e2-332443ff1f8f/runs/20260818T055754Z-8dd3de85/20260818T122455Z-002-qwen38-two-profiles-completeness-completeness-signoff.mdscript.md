---
reviewer_id: "completeness"
reviewer_lane: "completeness"
review_round: 2
goal: "Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles: (1) LATENCY: max single-stream decode + min TTFT preserving K7 MTP, 262k ctx, stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high concurrency preserving 262k ctx, stability, quality; profile continuously, optimize the bottleneck, benchmark + stress each change vs best, stop only when remaining gains are negligible or hardware-bound"
conversation_id: "23cb78c0-b3a1-4ea3-89e2-332443ff1f8f"
run_id: "20260818T055754Z-8dd3de85"
review_head: "1f3c16e"
merge_base: "68dfda8"
proof_scope: "live-proof"
signed_off: false
verifier_summary: "Round-2 completeness attack on head 1f3c16e: all 39 manifest paths exist and their contents match the numbers in the packet, yaml comments and commit messages (ss 44.4/39.2/77.4/111.6 -> 61.6-68.2/54.6-56.4/95-106/143-144; c128/c64/c32 621.7/632.6/452.6 -> 827.9/680.2/479.6; TTFT lat 17.53/56.95 vs 17.82/58.57; DBO 3.65/6.85/14.44 -> 2.88/5.97/13.61; needle 200k retrieved on both; burst 128/128; logprob 0.079/3.19% and 0.087/3.25% vs reference 0.092/3.52%; 0.94 OOM on 200k prefill confirmed in server-lat_final_v2_offload.log). Patches 0005+0006 applied to the 17 *.orig0005 originals reproduce all 18 venv files byte-for-byte; 26 CPU tests pass; latency unit is active, hardened, identical to the repo file, /health 200, keepalive counters on /metrics; live r2 latency proof (12:22) postdates the fix wave. Attacks that landed: (1) the throughput profile has no live/aggregate proof at head - bench-tp_final3 (10:26-10:34) and the only live 8012 throughput run (10:55-11:04) predate the 11:23-11:29 rework of ubatching.py/parallel_state.py/gpu_ubatch_wrapper.py and the unit hardening; the hardened throughput unit has never been active (InactiveEnterTimestamp 11:04); (2) both 200k needle probes (09:50/10:01) predate the fix wave, so the 262k-ctx functional invariant is not re-proven on head (only KV pool capacity 1.148M/1.365M is shown post-rework); (3) README 0006 row cites 'prefill 16k 7.8 -> 5.7 s' which no manifest artifact backs (artifacts show 6.85 -> 5.97; prof_dbo/prof_nodbo from iter6 are not in the commit). Default-false kept; findings are P2/P3."
evidence:
  - "artifacts/manifest.json: 39/39 listed paths exist (checked with os.path.exists from the run dir); contents match the packet/yaml/README/commit numbers listed above"
  - "artifacts/live/20260818T122159Z-prod-8012-ss_bench-r2.log (12:22, after the 11:23-11:29 venv rework, hardened unit active since 12:18:51): 65.3/54.7/98.7/144.4 tok/s; systemctl show vllm-qwen38: NoNewPrivileges/PrivateTmp/ProtectSystem=full, VLLM_USE_V2_MODEL_RUNNER=1; /etc unit files identical to serve-configs/systemd/*.service; curl /health 200"
  - "patch reproduction: 17 *.orig0005 originals copied to a scratch tree, patch -p1 0005 then 0006 applied cleanly, cmp against the venv for all 18 files touched: 0 differences"
  - "serve-configs/tests: 26 passed (venv python, run from the tests dir)"
  - "artifacts/logs/server-lat_final_v2_offload.log: gpu_memory_utilization 0.94 + torch.OutOfMemoryError at 09:44 -> supports the 0.92 rationale; server-lat_final_r3.log KV 1,148,106 tokens (4.38x 262k), server-tp_final_r3.log 1,364,956 tokens (5.21x)"
  - "venv mtimes: split-state files (ubatch_utils.py, backend.py, gdn/mamba/linear_attn.py, gpu_model_runner.py) unchanged since 08:47-08:54, so outputs-dbo.json (08:57) still covers head; overlap gating files (ubatching.py 11:26, parallel_state.py 11:26, gpu_ubatch_wrapper.py 11:26) postdate every aggregate throughput bench and the live throughput run"
  - "systemctl show vllm-qwen38-throughput: ActiveEnterTimestamp 10:55:01, InactiveEnterTimestamp 11:04:11 - the hardened unit (commit 4957ee2, 12:23) has never run; commit message honestly says only 'the latency unit restarts and serves' but the packet claims live proof on 8012 for both units"
commands_run:
  - "systemctl is-active vllm-qwen38 vllm-qwen38-throughput; systemctl show ... -p ActiveEnterTimestamp -p NoNewPrivileges -p PrivateTmp -p ProtectSystem -p FragmentPath -p Environment; diff /etc/systemd/system/<unit> serve-configs/systemd/<unit>"
  - "curl -s http://127.0.0.1:8012/health ; curl -s http://127.0.0.1:8012/metrics | grep -E 'keepalive|spec_decode_num_draft'"
  - "cd serve-configs/tests && /shared/vllm/.venv-qwen38/bin/python -m pytest -q . -> 26 passed"
  - "python (venv) loop over artifacts/manifest.json paths -> os.path.exists for each (39 OK); cat of all small logs/json in the manifest"
  - "copy *.orig0005 -> scratch tree; patch -p1 < 0005; patch -p1 < 0006; cmp each of 18 files vs venv -> 0 differs"
  - "stat -c '%y' on the 18 patched venv files; ls -la --time-style=full-iso artifacts/logs artifacts/live; git log 68dfda8..1f3c16e -- serve-configs"
  - "grep OutOfMemory / gpu_memory_utilization in server-lat_final_v2*.log; grep 'GPU KV cache size' in server-*_r3.log"
attack_attempts:
  - "Manifest paths missing or not proving what they claim: FAILED - 39/39 exist and every quoted number (ss, c128/c64/c32, TTFT, DBO prefill, needle, burst, logprob, reference noise floor) matches the artifact content"
  - "Patch files drift from the venv actually benchmarked: FAILED - originals + 0005 + 0006 reproduce all 18 venv files byte-for-byte"
  - "Latency profile at head unproven live: FAILED - live r2 ss_bench on 8012 at 12:22 postdates the 11:23-11:29 rework and the hardened unit start (12:18:51); TTFT/burst/greedy/logprob artifacts (11:40-12:18) also postdate it"
  - "Throughput profile at head unproven: LANDED - all aggregate benches (bench-tp_final3 10:26-10:34) and the only live throughput run on 8012 (10:55-11:04) predate the 11:26 rework of the DBO overlap gating and the unit hardening; post-rework evidence is only ttft-tp_final_dbo (11:51) + greedy/logprob on 8013; hardened throughput unit never started"
  - "262k-ctx invariant not re-proven after the fix wave: LANDED - needle200k-lat_final_v2 (09:50) and needle200k-tp_final (10:01) predate the V2 speculator/scheduler/draft-head/DBO-gating rework; only KV pool capacity at head is shown"
  - "Numbers in README/yaml/commits inconsistent with artifacts: PARTIALLY LANDED - README 0006 row '16k 7.8 -> 5.7 s' is unbacked (artifacts 6.85 -> 5.97; commit af18ad7 uses the artifact numbers); packet claim '.../56/...' prose vs live 54.6/54.7 (commit a4703eb states 54.6 correctly); max.yaml comment still says '~1.39M-token GPU pool' while head servers report 1,364,956"
  - "Keepalive telemetry claim ('/metrics exposes vllm_keepalive_* counters'): FAILED - HELP/TYPE lines for vllm_keepalive_pings_total and vllm_keepalive_early_commits_total are on the live /metrics"
  - "0.94 OOM rationale for gpu-memory-utilization 0.92 unbacked: FAILED - server-lat_final_v2_offload.log shows gpu_memory_utilization 0.94 and torch.OutOfMemoryError at 09:44"
p_findings:
  - severity: "P2"
    location: "review packet 'Claim Under Review' (live proof on 8012 for both units) / artifacts/manifest.json entries artifacts/live/20260818T105911Z-prod-8012-throughput-c32.* and artifacts/logs/bench-tp_final3_c{128,64,32}.json / serve-configs/systemd/vllm-qwen38-throughput.service"
    summary: "The throughput profile's aggregate numbers (827.9/680.2/479.6) and its only live 8012 run were measured 10:26-11:04, before the 11:26 rework of patch 0006's overlap gating (ubatching.set_overlap_tp_all_reduce, parallel_state.py, gpu_ubatch_wrapper.py) and before the unit hardening; the hardened vllm-qwen38-throughput unit has never been active. Post-rework evidence for this profile is only ttft-tp_final_dbo.log (prefill TTFT on 8013) plus greedy/logprob spot checks."
    contract: "goal: 'benchmark + stress each change vs best'; packet: live proof required for the primary user action (serve each profile via the systemd unit, run vllm bench serve, compare with prior best)"
    remediation: "On head: start vllm-qwen38-throughput (hardened unit) and run vllm bench serve at c32 on 8012 (and c128 on 8013 with the identical yaml), record both under artifacts/live and artifacts/logs, add to the manifest, and align the packet claim with what was measured post-rework."
  - severity: "P2"
    location: "artifacts/logs/needle200k-lat_final_v2.log (09:50), artifacts/logs/needle200k-tp_final.log (10:01) vs venv rework mtimes 11:23-11:29"
    summary: "Both 200k needle proofs predate the fix wave that changed the V2 speculator/model_runner (drafted-columns return), scheduler adaptive-K functions, draft_lm_head.py (Marlin path rewrite), cudagraph_utils, and DBO overlap gating; the 262k-ctx functional invariant is not re-proven on head (only KV pool capacity 1.148M / 1.365M tokens is visible in server-*_r3.log)."
    contract: "goal invariant '262k ctx' + 'quality unchanged (greedy outputs / needle probe)' per goal.mdscript.md Goal Contract; primary_user_action names the 200k needle"
    remediation: "Rerun artifacts/scripts/long_ctx_probe.py --target-tokens 200000 against a head latency server (8012 prod or 8013 lat_final.yaml with V2) and a head throughput server (8013 tp_final.yaml), store as needle200k-*-r2.log, add to the manifest."
  - severity: "P3"
    location: "serve-configs/patches/README.md 0006 row ('prefill 16k 7.8 -> 5.7 s'); progress.jsonl iter6 artifacts prof_dbo/ prof_nodbo/"
    summary: "The README's DBO prefill figure is not backed by any manifest artifact (ttft-tp_base.log / ttft-tp_final_dbo.log show 6.85 -> 5.97 s at 16k; commit af18ad7 uses those). The 7.79/5.71 s figures come from iter6 whose prof_dbo/ prof_nodbo/ directories exist on disk but are not in commit 1f3c16e (review-round2-stat.txt lists only prof_base/prof_int4_v2/prof_tp_base). Also max.yaml still says '~1.39M-token GPU pool' while head servers report 1,364,956, and the packet's prose figure '56' is the 8013 number while live 8012 shows 54.6-54.7 (commit a4703eb states 54.6)."
    contract: "packet open question: 'is any claimed measurement not backed by an artifact in the manifest'"
    remediation: "Change the README row to the manifested 6.85 -> 5.97 s (or manifest/commit the iter6 measurement and its prof_dbo/prof_nodbo profiles); refresh the max.yaml pool comment; quote the live prose figure as 54.6-56."
rules_reviewed:
  - "/home/gwillen/.agents/skills/self-review/workflows/blind-reviewers/completeness.mdscript.md (this lane)"
  - "review packet 20260818T122455Z-002-qwen38-two-profiles-main-review-packet.mdscript.md (claim, in-scope paths, proof supplied / not claimed, open questions)"
  - "goal.mdscript.md Goal Contract (metrics, invariants: 262k ctx, K7 MTP, quality unchanged, no crashes under 128-burst; primary_user_action)"
  - "AGENTS.md: venv-only python (all reproduce commands use /shared/vllm/.venv-qwen38/bin/python), AI-assistance statement in commits (present)"
artifact_paths:
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/manifest.json"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/20260818T122159Z-prod-8012-ss_bench-r2.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/20260818T105911Z-prod-8012-throughput-c32.json"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/bench-tp_final3_c128.json"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/needle200k-lat_final_v2.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/needle200k-tp_final.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/ttft-tp_final_dbo.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/logprob-agreement-latency.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/logprob-agreement-reference-latbase-vs-tpbase.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/server-lat_final_v2_offload.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round2-serve-configs.diff"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/progress.jsonl"
  - "/shared/vllm/serve-configs/patches/README.md"
  - "/shared/vllm/serve-configs/systemd/vllm-qwen38-throughput.service"
  - "/shared/vllm/serve-configs/tests/"
objectives_checked:
  - "latency profile: single-stream decode gain vs baseline with K7 MTP kept - PROVEN (ss-lat_base 44.4/39.2/77.4/111.6 -> live 8012 r2 65.3/54.7/98.7/144.4, K7 schedule and MTP in yaml)"
  - "latency profile: TTFT preserved - PROVEN (17.53/56.95 vs 17.82/58.57 s cold 37k/90k)"
  - "latency profile: stability - PROVEN (burst 128/128 ok, unit active since 12:18 with hardening)"
  - "throughput profile: aggregate gain vs baseline - MEASURED pre-rework only (621.7 -> 827.9 c128); NOT re-proven at head"
  - "both profiles: 262k ctx - capacity at head PROVEN (1.148M / 1.365M tokens); functional 200k needle only pre-rework"
  - "quality: logprob agreement vs reference noise floor - PROVEN post-rework (0.079/3.19% and 0.087/3.25% vs 0.092/3.52%, same 1817 positions); greedy 2/8 and 3/8 identical honestly stated"
  - "live proof for the primary user action - latency unit PROVEN at head; throughput unit only pre-rework/pre-hardening"
  - "patches reproduce venv, tests pass, manifest paths exist - PROVEN"
remaining_gaps:
  - "Head-state proof for the throughput profile (aggregate bench and hardened-unit live run) after the 11:26 DBO overlap-gating rework"
  - "200k needle on head for both profiles"
  - "README 0006 prefill figure vs manifested artifacts; max.yaml pool comment"
signed_off_at: "2026-08-18T12:41:00Z"
repair_resume_command: "/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/goal.mdscript.md#pursue-goal"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Signoff

* lane verdict: completeness - `signed_off: false` (Blocked-for head 1f3c16e; the latency profile is fully proven at head, the throughput profile and the 262k needle are proven only for the pre-fix-wave venv)
* P2 - throughput profile head proof missing: location artifacts/manifest.json live throughput entry + bench-tp_final3_c*.json (10:26-11:04) vs venv rework 11:26 and hardened unit never active; remediation: start the hardened vllm-qwen38-throughput unit, run vllm bench serve c32 on 8012 (+ c128 on 8013), manifest the results, align the packet claim
* P2 - 200k needle predates the fix wave for both profiles: location artifacts/logs/needle200k-lat_final_v2.log (09:50) and needle200k-tp_final.log (10:01); remediation: rerun long_ctx_probe.py --target-tokens 200000 on head servers for both profiles and manifest them
* P3 - README 0006 row '16k 7.8 -> 5.7 s' unbacked by manifested artifacts (6.85 -> 5.97 s), iter6 prof_dbo/prof_nodbo not committed, max.yaml '~1.39M-token' comment stale, prose '56' vs live 54.6-54.7; remediation: align the README/yaml/packet numbers with the manifested artifacts or manifest the iter6 measurement

## Resume From Signoff

* `signed_off` is false: continue at `/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/goal.mdscript.md#pursue-goal` (repair_resume_command; the packet supplied no separate repair command) to run the repair wave for the findings above
* after repair, a fresh blind completeness reviewer must be spawned on a new packet/round; never re-enter this lane's own review from this sign-off
* only when a later round's completeness sign-off is `signed_off: true` does the composer continue at `/mdscript-exec /home/gwillen/.agents/skills/self-review/workflows/triple-adversarial-blind-review.mdscript.md#aggregate-triple-signoffs`
