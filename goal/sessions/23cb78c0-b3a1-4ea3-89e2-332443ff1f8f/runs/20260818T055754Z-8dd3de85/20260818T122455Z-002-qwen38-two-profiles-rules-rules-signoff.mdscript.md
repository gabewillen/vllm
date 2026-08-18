---
reviewer_id: "rules"
reviewer_lane: "rules"
review_round: 2
goal: "Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles: (1) LATENCY: max single-stream decode + min TTFT preserving K7 MTP, 262k ctx, stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high concurrency preserving 262k ctx, stability, quality; profile continuously, optimize the bottleneck, benchmark + stress each change vs best, stop only when remaining gains are negligible or hardware-bound"
conversation_id: "23cb78c0-b3a1-4ea3-89e2-332443ff1f8f"
review_head: "1f3c16e"
merge_base: "68dfda8"
proof_scope: "live-proof"
signed_off: false
verifier_summary: "Blind rules lane, round 2, head 1f3c16e (7 commits). Read AGENTS.md/CLAUDE.md end-to-end; searched for Cursor/VS Code/Windsurf/Cline/Aider/Gemini/Codex rule surfaces (none exist). Attacked: bare pip/system python in diff, scripts, docs and run logs; secret/endpoint leakage in configs, units, patches, logs; commit atomicity + AI-assistance attribution per commit; systemd unit safety (Conflicts, env separation, installed units == repo); manifest path existence (38/38); patch-reproduces-venv claim (all 18 files byte-identical after applying 0005+0006 to the *.orig0005 originals); docs-vs-artifact numeric consistency; python line length (88) in middleware/tests/patch hunks; CPU tests (26 passed with the deployment venv). Two P3 findings remain (README 0006 prefill numbers disagree with the round-2 TTFT artifacts and the patch commit message; a helper script docstring documents bare `python`), so per lane policy signed_off stays false."
evidence:
  - "git log 68dfda8..1f3c16e: 7 commits, each atomic by concern (patch 0005 + its tests/README; latency yaml + unit; patch 0006 + test/README; throughput yaml + new unit; middleware + test; unit hardening; evidence). All 7 carry `Co-authored-by: Claude Fable 5 <noreply@anthropic.com>`; the 6 product commits also state `Generated with AI assistance (Claude)`; the two patch commits state venv-local / unreported upstream (no PR opened -> AGENTS.md duplicate-check/PR rules not triggered)."
  - "git grep over serve-configs and the run dir at 1f3c16e: no `pip install`, no `/usr/bin/python`, no literal API keys (`VLLM_API_KEY=<value>`, `Bearer <token>`, `sk-`); ss_bench.py reads VLLM_API_KEY from the environment; only non-local URL hits are 0.0.0.0:8013 experiment-server log lines and pytorch doc links; /etc/vllm/qwen38.env is referenced by EnvironmentFile only, not committed."
  - "systemctl: vllm-qwen38 active+enabled, vllm-qwen38-throughput inactive+disabled; `systemctl cat` of both units equals the repo files (comment-stripped diff empty); mutual Conflicts= present; NCCL_MAX_NCHANNELS=1 and VLLM_USE_V2_MODEL_RUNNER=1 are set only in their own unit as the yaml comments require; curl http://127.0.0.1:8012/health -> 200."
  - "artifacts/manifest.json: 38 distinct referenced paths, 0 missing on disk (checked with /shared/vllm/.venv-qwen38/bin/python)."
  - "Patch reproduction: copied every *.orig0005 for the 18 files named in 0005/0006 into a scratch tree, `patch -p1` both patches, `cmp` against .venv-qwen38 -> all 18 identical (draft_lm_head.py is a new file in 0005, no .orig needed). README/apply-to-venv.sh wording (`six source patches`, `All were written against ...`) is consistent with the 6 patch files present."
  - "serve-configs/tests: 26 passed in 13.8 s with /shared/vllm/.venv-qwen38/bin/python -m pytest -q . (README documents that interpreter and `uv pip install pytest`, no bare pip). No line >88 cols in vllm_keepalive.py, tests/*.py, or added lines of the 0005/0006 patch hunks."
  - "Docs-vs-artifact: README 0006 row says `prefill 16k 7.8 -> 5.7 s`, sourced from an earlier progress.jsonl measurement (control 7.79 s / dbo+NCHANNELS 5.71 s), while the round-2 proof artifacts artifacts/logs/ttft-tp_base.log / ttft-tp_final_dbo.log (cited by the packet) and commit af18ad7 report 16k 6.85 -> 5.97 s. Both numbers are backed by artifacts, but the same claim is stated two ways in the same commit range."
commands_run:
  - "cat rules.mdscript.md; cat <run_dir>/20260818T122455Z-002-qwen38-two-profiles-main-review-packet.mdscript.md; cat review-round2-commits.txt review-round2-stat.txt"
  - "git log --format='%h %an <%ae>%n%B----' 68dfda8..1f3c16e; git show --stat --format= <each of 7 commits>"
  - "ls -d .cursor .vscode .windsurf .clinerules .cursorrules .windsurfrules .github/copilot-instructions.md .github/instructions GEMINI.md CODEX.md .agents (all absent); ls .github"
  - "git diff 68dfda8 1f3c16e -- serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml serve-configs/qwen3_8_27b_fp8_max.yaml serve-configs/systemd serve-configs/patches/README.md serve-configs/patches/apply-to-venv.sh serve-configs/tests/README.md"
  - "git grep -n -E '(python3?|pip)( |$)' 1f3c16e -- <run_dir> serve-configs (filtered .venv/uv pip); git grep -n -i -E 'VLLM_API_KEY=[A-Za-z0-9]|api[_-]?key...|Bearer [A-Za-z0-9]{8,}|sk-[A-Za-z0-9]{10,}' 1f3c16e -- <run_dir> serve-configs; git grep -o -E 'https?://[^ \"]+' 1f3c16e -- <run_dir> serve-configs"
  - "grep -rn -i 'pip install|pip3' progress.jsonl goal.mdscript.md session-log.jsonl artifacts/scripts serve-configs; grep -n -E '(^|[ \"(])(python3?)( |$)' artifacts/manifest.json artifacts/scripts/* goal.mdscript.md"
  - "/shared/vllm/.venv-qwen38/bin/python <manifest path-existence walk> -> 38 paths, 0 missing"
  - "systemctl is-active/is-enabled vllm-qwen38 vllm-qwen38-throughput; curl -s -m 5 http://127.0.0.1:8012/health; diff <(systemctl cat <unit> | grep -v '^#') <(grep -v '^#' serve-configs/systemd/<unit>)"
  - "awk 'length>88' serve-configs/middleware/vllm_keepalive.py serve-configs/tests/*.py; awk over '+' lines of 0005/0006 patches; cd serve-configs/tests && /shared/vllm/.venv-qwen38/bin/python -m pytest -q ."
  - "scratchpad patch reproduction: cp <venv>/<file>.orig0005 -> tree; patch -p1 < 0005; patch -p1 < 0006; cmp each of 18 files against the venv"
  - "cat artifacts/logs/ttft-tp_base.log artifacts/logs/ttft-tp_final_dbo.log; grep '7\\.8\\|5\\.7' serve-configs/patches/README.md; grep 16k progress.jsonl"
attack_attempts:
  - "Bare pip / system python (AGENTS.md section 2): searched diff, committed scripts, manifest reproduce commands, tests README, progress.jsonl, session-log.jsonl, goal.mdscript.md. Failed except one docstring: artifacts/scripts/long_ctx_probe.py line 5 documents `python long_ctx_probe.py` (P3). All server/bench launches and the tests README use /shared/vllm/.venv-qwen38/bin/{vllm,python}; only `uv pip install pytest` is documented."
  - "Secret / private-endpoint leakage in configs, units, patches, logs: failed. No literal keys or bearer tokens; EnvironmentFile=/etc/vllm/qwen38.env (root 0600 per unit comment, not committed); ss_bench.py reads VLLM_API_KEY from env; only 0.0.0.0:8013 experiment-server bind lines appear in server logs."
  - "Commit atomicity + attribution: failed. Each of the 6 product commits changes one concern (patch+tests+README / yaml+unit / middleware+test / hardening); every commit has a Co-authored-by Claude trailer and the product commits state AI assistance; the evidence commit 1f3c16e is goal-run files only. Duplicate-work checks and PR-description rules do not apply because no PR is opened (patches explicitly venv-local, human-owned per README)."
  - "Systemd safety: tried to show both units can hold the GPUs at once or that env leaks across profiles: failed. Mutual Conflicts= present, throughput unit disabled, V2-runner env only in the latency unit and NCCL_MAX_NCHANNELS only in the throughput unit; installed units byte-equal (comments stripped) to the repo files."
  - "Docs vs artifacts: tried to find claimed numbers without artifacts or manifest paths missing: 38/38 manifest paths exist; single-stream 65.3/54.7/98.7/144.4 present in the live r2 log; BUT README 0006 `16k 7.8 -> 5.7 s` disagrees with the packet's ttft-tp artifacts (6.85 -> 5.97 s) and commit af18ad7 (P3 inconsistency, both numbers artifact-backed)."
  - "Patch/venv drift (README claim that 0005/0006 reproduce the venv): failed to break; 18/18 files byte-identical after applying both patches to the *.orig0005 originals; apply-to-venv.sh iterates 000*.patch so 0005/0006 are covered."
  - "Style rules (88-col limit, Google docstrings, no reST) on committed python: no over-length lines in middleware, tests, or patch hunks; middleware module docstring is prose, no :param: fields. Not enforced by pre-commit here (no ruff/pre-commit in .venv-qwen38; upstream tooling rules only bind PRs to vllm-project/vllm)."
  - "Other rule families: .cursor/, .vscode/, .windsurf/, .clinerules, .cursorrules, .windsurfrules, .github/copilot-instructions.md, .github/instructions/, GEMINI.md, CODEX.md, .agents/ all absent (verified with ls); .github/ holds only CODEOWNERS/templates/workflows (upstream CI, no agent rules); ~/.claude/CLAUDE.md only mandates the self router (process rule, not attackable from the diff)."
p_findings:
  - severity: "P3"
    location: "serve-configs/patches/README.md (0006 row, `throughput profile: prefill 16k 7.8 -> 5.7 s`)"
    summary: "The README quotes an earlier progress.jsonl measurement (control 7.79 s vs 5.71 s) while the round-2 proof artifacts the packet cites (artifacts/logs/ttft-tp_base.log 6.85 s, ttft-tp_final_dbo.log 5.97 s) and the patch commit message af18ad7 report a different pair for the same 16k prefill claim; the durable doc and the artifacts disagree."
    contract: "AGENTS.md section 2 coding style (`Code should be self-documenting`; docs must match evidence) + packet claim `documented in serve-configs/patches/README.md` with evidence-backed numbers"
    remediation: "State the round-2 ttft-tp numbers (16k 6.85 -> 5.97 s, or the 9k/16k/37k triple) in the README row, or name both measurements with their conditions and cite the exact log for each."
  - severity: "P3"
    location: "goal/sessions/.../artifacts/scripts/long_ctx_probe.py:5 (module docstring `Usage: python long_ctx_probe.py ...`)"
    summary: "A committed helper script documents invocation with bare `python`; the deployment convention and AGENTS.md require running Python via the venv interpreter (`.venv/bin/python` / uv), and the rest of the run (manifest, tests README) was already corrected to /shared/vllm/.venv-qwen38/bin/python."
    contract: "AGENTS.md section 2 Development Workflow: `Never use system python3 or bare pip ... All Python commands must go through uv and .venv/bin/python`"
    remediation: "Change the usage line to `/shared/vllm/.venv-qwen38/bin/python long_ctx_probe.py ...` (or `.venv-qwen38/bin/python`)."
rules_reviewed:
  - "/shared/vllm/AGENTS.md (end-to-end: contribution policy, duplicate checks, accountability, dev workflow uv/venv, tests, linters, 88 cols, Google docstrings, commit trailers, domain guides)"
  - "/shared/vllm/CLAUDE.md (@AGENTS.md include only)"
  - "/home/gwillen/.claude/CLAUDE.md (self router mandate)"
  - "/shared/vllm/serve-configs/patches/README.md and serve-configs/tests/README.md (deployment-local operating rules: apply-to-venv after rebuild, run tests with the deployment venv, never system python)"
  - ".cursor/rules/ (none), .cursor/AGENTS.md / .cursorrules / .cursor/rules.md (none)"
  - ".vscode/ (none), .github/copilot-instructions.md (none), .github/instructions/ (none)"
  - ".windsurf/ (none), .windsurfrules (none)"
  - ".clinerules (none), .aider.conf.yml (none), GEMINI.md (none), CODEX.md (none), .agents/AGENTS.md (none)"
  - ".github/ (CODEOWNERS, ISSUE_TEMPLATE, PULL_REQUEST_TEMPLATE.md, mergify.yml, workflows: upstream CI config, no agent rules)"
artifact_paths:
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T122455Z-002-qwen38-two-profiles-main-review-packet.mdscript.md"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round2-serve-configs.diff"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round2-stat.txt"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round2-commits.txt"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/manifest.json (38 paths verified)"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/ttft-tp_base.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/logs/ttft-tp_final_dbo.log"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/ (ss_bench-r2 log)"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/scripts/ss_bench.py, long_ctx_probe.py"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/progress.jsonl"
  - "/shared/vllm/serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml, qwen3_8_27b_fp8_max.yaml, systemd/vllm-qwen38.service, systemd/vllm-qwen38-throughput.service, patches/README.md, patches/apply-to-venv.sh, patches/0005-*.patch, patches/0006-*.patch, tests/, middleware/vllm_keepalive.py"
  - "/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/ (18 patched files + *.orig0005, read-only cmp)"
objectives_checked:
  - "AGENTS.md dev workflow: no bare pip / system python in diff, committed scripts, docs, or run history"
  - "AGENTS.md contribution policy: no PR opened; venv-local patches marked human-owned; AI-assistance stated in product commits; Co-authored-by trailer on all commits"
  - "Commit atomicity: one concern per commit, evidence separated"
  - "Secrets/private endpoints not committed in configs, units, patches, logs"
  - "Systemd units: mutual exclusion, profile-specific env isolation, installed == repo"
  - "Every packet-claimed measurement has an on-disk artifact; manifest paths exist"
  - "Patch files reproduce the venv state (docs vs deployed code)"
  - "Durable docs (README/yaml comments) agree with artifacts"
  - "Python style rules (88 cols, docstring style) on committed python and patch hunks"
  - "All agent rule families discovered and searched (Cursor/VS Code/Windsurf/Cline/Aider/Gemini/Codex)"
remaining_gaps:
  - "Two P3 findings above must be repaired (README 0006 prefill numbers; long_ctx_probe.py usage line) before this lane can sign off; no P0-P2 rule violations were found."
  - "Blindness note: a broad `git grep` over the run dir at 1f3c16e (which contains the committed round-1 sign-offs) surfaced a few round-1 sign-off lines in its output; they were not read further or relied on, and progress.jsonl (in scope) already lists the round-1 findings. Nothing in this sign-off derives from them."
signed_off_at: "2026-08-18T12:40:00Z"
repair_resume_command: "/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/goal.mdscript.md#resume-goal"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Signoff

* lane verdict: rules lane, round 2, head 1f3c16e — `signed_off: false` (Blocked-for: docs-vs-artifact consistency and one bare-`python` usage line; no P0-P2 rule violations)
* P3 — serve-configs/patches/README.md 0006 row `prefill 16k 7.8 -> 5.7 s`: disagrees with artifacts/logs/ttft-tp_base.log / ttft-tp_final_dbo.log (6.85 -> 5.97 s) and commit af18ad7; remediation: quote the round-2 ttft-tp numbers or name both measurements with conditions and exact log paths
* P3 — artifacts/scripts/long_ctx_probe.py:5 documents `python long_ctx_probe.py`; remediation: use `/shared/vllm/.venv-qwen38/bin/python long_ctx_probe.py` per AGENTS.md dev workflow
* all other rules attacks (bare pip/system python, secrets, commit atomicity/attribution, unit safety, manifest completeness, patch reproduction, style limits, absent rule families) failed to find a violation

## Resume From Signoff

* `signed_off` is `false`: next jump is the repair path `/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/goal.mdscript.md#resume-goal` (repair the two P3 findings), then a FRESH blind rules reviewer must review the repaired head in a new round — never re-enter this lane's own review from this sign-off
* when a later round reaches `signed_off: true`, aggregation continues at `/mdscript-exec /home/gwillen/.agents/skills/self-review/workflows/triple-adversarial-blind-review.mdscript.md#aggregate-triple-signoffs`
