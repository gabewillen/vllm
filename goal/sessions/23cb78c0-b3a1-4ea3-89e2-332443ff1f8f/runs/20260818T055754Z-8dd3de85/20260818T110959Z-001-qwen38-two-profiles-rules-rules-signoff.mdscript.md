---
reviewer_id: "rules"
reviewer_lane: "rules"
review_round: 1
goal: "Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles: (1) LATENCY: max single-stream decode + min TTFT preserving K7 MTP, 262k ctx, stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high concurrency preserving 262k ctx, stability, quality; profile continuously, optimize the bottleneck, benchmark + stress each change vs best, stop only when remaining gains are negligible or hardware-bound"
conversation_id: 23cb78c0-b3a1-4ea3-89e2-332443ff1f8f
signed_off: false
verifier_summary: "Blind rules lane over commit b3c9e81 (base 68dfda8): read AGENTS.md/CLAUDE.md end-to-end, searched for Cursor/VS Code/Windsurf/Cline/aider/GEMINI/CODEX rule surfaces (none exist), attacked the diff and run history for bare pip / system python, secrets in configs/logs/scripts, commit attribution, unit safety, manifest path integrity and patch reproducibility. Most attacks failed (no pip, no secrets, units mutually Conflicts= and match /etc/systemd, all 25 manifest paths exist, 0005+0006 reproduce the venv byte-for-byte). One real AGENTS.md violation: the manifest/packet reproduce commands invoke bare `python`, which AGENTS.md forbids (uv/.venv/bin/python only) and which does not even exist on this box (only /usr/bin/python3), so the live-proof reproduce lines are both non-compliant and non-runnable as written. Plus two P3 documentation/reproducibility gaps."
evidence:
  - "AGENTS.md section 2: 'Never use system python3 or bare pip/pip install. All Python commands must go through uv and .venv/bin/python.' — artifacts/manifest.json lines 10 and 31 say 'VLLM_API_KEY=... python artifacts/scripts/ss_bench.py 8012|8013'; `which python` -> not found, `which python3` -> /usr/bin/python3 (system)."
  - "Commit b3c9e81 body ends with 'Generated with AI assistance (Claude) ...' and 'Co-authored-by: Claude Fable 5 <noreply@anthropic.com>' — satisfies AGENTS.md commit-trailer attribution; no PR opened, so PR-description rules and duplicate-work checks are N/A (README marks patches human-owned per AGENTS.md)."
  - "grep over run dir + serve-configs for 'pip install', '/usr/bin/python', 'sk-', 'Bearer <literal>', 'VLLM_API_KEY=<literal>' returned only env-var reads in scripts and 'VLLM_API_KEY=...' placeholders; /etc/vllm/qwen38.env is root:root 0600 and not in the diff."
  - "systemctl: vllm-qwen38 active/enabled, vllm-qwen38-throughput inactive/disabled, both units carry mutual Conflicts= and byte-match the committed serve-configs/systemd/*.service; curl /health on 8012 -> 200."
  - "Patch reproduction: copied the 16 *.orig0005 originals to a scratch tree, applied 0005 then 0006 with patch -p1 (no rejects), cmp against .venv-qwen38 files -> 0 differences."
  - "manifest.json: 25 artifacts, every 'path' exists on disk; bench-tp_final3_c128.json output_throughput 827.93 (c64 680.17, c32 479.59), tp_base_c128 621.73, tp_dbo128_ch1_c128 814.26."
commands_run:
  - "ls -d .cursor .cursorrules .vscode .windsurf .windsurfrules .clinerules .aider.conf.yml GEMINI.md CODEX.md .agents .github/copilot-instructions.md .github/instructions (all absent)"
  - "git log 68dfda8..b3c9e81 --format='%H %s%n%b'"
  - "grep -rnE 'pip install|python3 |/usr/bin/python|sk-...|VLLM_API_KEY=...|Bearer ...' <run_dir> serve-configs"
  - "which python python3 uv"
  - ".venv-qwen38/bin/python (json walk of artifacts/manifest.json checking os.path.exists for every artifact path)"
  - "systemctl is-active/is-enabled/show vllm-qwen38 vllm-qwen38-throughput; diff committed units vs /etc/systemd/system/*.service; curl -s http://127.0.0.1:8012/health"
  - "patch -p1 of 0005/0006 onto *.orig0005 copies in scratchpad + cmp against .venv-qwen38"
attack_attempts:
  - "Bare pip / system python in the diff, scripts, progress.jsonl, goal.mdscript.md, goal-log: no pip usage anywhere; all server/bench launches use /shared/vllm/.venv-qwen38/bin/vllm; BUT manifest reproduce strings use bare `python` (P2 below)."
  - "Secret leakage: searched configs, units, patches, scripts, live logs, bench logs and committed session/goal logs for API keys / Bearer tokens; none — key lives only in root-only /etc/vllm/qwen38.env and scripts read os.environ."
  - "Commit/PR rules: attribution trailer present, AI-assistance stated; no upstream PR so accountability/dup-check clauses N/A; attempted to argue committing directly to master violates 'branch first' — rejected as a finding because every prior goal-run commit on this repo (68dfda8, 817a425, fc4f184 ...) follows the same user-established master workflow and the goal MDScript Next Steps explicitly schedules the commit."
  - "Domain-specific guides in AGENTS.md: only editing-agent-instructions is listed; the diff does not touch AGENTS.md or its referenced guides — no violation."
  - "Unit safety: tried to find a way both profiles run at once — mutual Conflicts= present in both committed and installed units; throughput unit disabled; V2 env only in latency unit as the yaml comments require."
  - "Rule-family sweep: .cursor/, .cursorrules, .vscode/, .windsurf/, .windsurfrules, .clinerules, .aider.conf.yml, GEMINI.md, CODEX.md, .agents/, .github/copilot-instructions.md, .github/instructions/ — none present (noted, not silently skipped)."
  - "Manifest integrity / patch honesty: all 25 artifact paths exist; 0005+0006 reproduce the venv byte-for-byte; bench JSON numbers match the claim (828/680/480 vs 622)."
p_findings:
  - severity: P2
    location: "goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/manifest.json:10,31 (reproduce fields) and the packet Proof Supplied line for the live ss_bench"
    summary: "Reproduce commands invoke bare `python artifacts/scripts/ss_bench.py`; AGENTS.md forbids system python / requires .venv/bin/python, and `python` is not on PATH here (only /usr/bin/python3), so the documented live-proof reproduction is non-compliant and does not run as written; the actual interpreter used for the live proof is not recorded."
    contract: "/shared/vllm/AGENTS.md §2 Development Workflow: 'Never use system python3 or bare pip. All Python commands must go through uv and .venv/bin/python.'"
    remediation: "Change the reproduce strings to '/shared/vllm/.venv-qwen38/bin/python artifacts/scripts/ss_bench.py <port>' (and the same for any other python invocation), and record the interpreter used in the live log or manifest."
  - severity: P3
    location: "serve-configs/qwen3_8_27b_fp8_max.yaml header (622 -> 814) and serve-configs/patches/README.md 0006 row (757 -> 814)"
    summary: "Committed docs quote the intermediate tp_dbo128_ch1 number (814) while the commit message, claim and final evidence (bench-tp_final3_c128.json) are 828 with offload on; readers of the yaml/README get a stale figure for the shipped config."
    contract: "AGENTS.md coding style: 'Code should be self-documenting' / keep comments accurate; packet claim vs committed docs consistency"
    remediation: "Update the yaml header and README row to the final3 numbers (828/680/480) or label 814 explicitly as the no-offload run."
  - severity: P3
    location: "goal/sessions/.../artifacts/scripts/up.sh, tp_bench.sh (S=/tmp/claude-1000/.../scratchpad; reads $S/run_dir and $S/launch.sh)"
    summary: "Evidence scripts referenced by manifest reproduce commands depend on the session scratchpad under /tmp, which is wiped on reboot (memory note box-is-lxc), so the recorded reproduction path is not durable."
    contract: "Goal Contract in goal.mdscript.md (proof artifacts must let the primary user action be re-run); AGENTS.md test-command reproducibility expectation"
    remediation: "Make up.sh/tp_bench.sh resolve launch.sh and run_dir relative to $(dirname $0) / the run dir instead of the scratchpad."
rules_reviewed:
  - "/shared/vllm/AGENTS.md (end-to-end)"
  - "/shared/vllm/CLAUDE.md (@AGENTS.md include)"
  - "/home/gwillen/.claude/CLAUDE.md (global router rule; process-only, not verifiable from artifacts)"
  - ".cursor/rules/ (none), .cursorrules (none), .cursor/AGENTS.md (none)"
  - ".vscode/ (none), .github/copilot-instructions.md (none), .github/instructions/ (none)"
  - ".windsurf/ (none), .windsurfrules (none)"
  - ".clinerules (none), .aider.conf.yml (none), GEMINI.md (none), CODEX.md (none), .agents/ (none)"
artifact_paths:
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-serve-configs.diff"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-stat.txt"
  - "/shared/vllm/serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml"
  - "/shared/vllm/serve-configs/qwen3_8_27b_fp8_max.yaml"
  - "/shared/vllm/serve-configs/systemd/vllm-qwen38.service"
  - "/shared/vllm/serve-configs/systemd/vllm-qwen38-throughput.service"
  - "/shared/vllm/serve-configs/patches/README.md"
  - "/shared/vllm/serve-configs/patches/apply-to-venv.sh"
  - "/shared/vllm/serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch"
  - "/shared/vllm/serve-configs/patches/0006-dense-tp-dual-batch-overlap-prefill-allreduce.patch"
  - "/shared/vllm/serve-configs/middleware/vllm_keepalive.py"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/manifest.json"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/progress.jsonl"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/goal.mdscript.md"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/20260818T105400Z-prod-8012-ss_bench.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/20260818T105911Z-prod-8012-throughput-c32.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/scripts/{ss_bench.py,up.sh,tp_bench.sh,launch.sh,mkpatch.sh,apply_dbo_patch.py}"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/bench-tp_{final3,base,dbo128_ch1}_c*.json"
objectives_checked:
  - "AGENTS.md: no bare pip / system python in diff, scripts, or documented commands"
  - "AGENTS.md: commit attribution trailer and AI-assistance statement; PR/dup-check clauses N/A (no PR)"
  - "AGENTS.md: domain-specific guide gate not triggered by the diff"
  - "Secrets / private endpoints not committed in configs, units, patches, scripts, logs"
  - "systemd units cannot run both profiles at once; installed units match committed units; env split (V2 vs NCCL_MAX_NCHANNELS) as documented"
  - "Every manifest artifact path exists; claimed numbers match bench JSON"
  - "Patches 0005/0006 reproduce the venv state (honesty of the patch artifacts)"
  - "Other agent rule families (Cursor/VS Code/Windsurf/Cline/aider/Gemini/Codex) searched"
remaining_gaps:
  - "The interpreter actually used for the live ss_bench proof is not recorded anywhere; if it was /usr/bin/python3 the run itself breached AGENTS.md (cannot be determined from artifacts)."
signed_off_at: "2026-08-18T11:20:00Z"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Signoff

* lane verdict: rules lane NOT signed off (signed_off: false) for commit b3c9e81 — one P2 AGENTS.md violation plus two P3 documentation/reproducibility gaps; every other rules attack (pip, secrets, attribution, unit safety, manifest integrity, patch reproduction, other rule families) failed
* P2 — artifacts/manifest.json reproduce fields (lines 10, 31) and the packet's live-proof line use bare `python`, which AGENTS.md forbids and which is not on PATH; remediation: use /shared/vllm/.venv-qwen38/bin/python and record the interpreter used for the live proof
* P3 — serve-configs/qwen3_8_27b_fp8_max.yaml header and patches/README.md 0006 row quote 814 tok/s while the shipped-config evidence is 828 (bench-tp_final3_c128.json); remediation: update or label the figure
* P3 — artifacts/scripts/up.sh and tp_bench.sh resolve launch.sh / run_dir via the /tmp scratchpad, so the recorded reproduction path does not survive a reboot; remediation: resolve relative to the script/run dir

## Resume From Signoff

* signed_off is false: the composer should route repair through the packet re-entry `/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T110959Z-001-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change` (no repair_resume_command was supplied by the packet) and, after repair, spawn a fresh blind rules reviewer for round 2 — never re-enter this lane's own review from this sign-off
* when a later round reaches signed_off true, continue at `/mdscript-exec /home/gwillen/.agents/skills/self-review/workflows/triple-adversarial-blind-review.mdscript.md#aggregate-triple-signoffs`
