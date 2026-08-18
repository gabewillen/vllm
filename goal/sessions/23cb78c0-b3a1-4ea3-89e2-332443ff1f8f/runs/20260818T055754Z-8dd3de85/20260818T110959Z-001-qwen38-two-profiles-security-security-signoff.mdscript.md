---
reviewer_id: "security"
reviewer_lane: "security"
review_round: 1
goal: "Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles: (1) LATENCY: max single-stream decode + min TTFT preserving K7 MTP, 262k ctx, stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high concurrency preserving 262k ctx, stability, quality; profile continuously, optimize the bottleneck, benchmark + stress each change vs best, stop only when remaining gains are negligible or hardware-bound"
conversation_id: 23cb78c0-b3a1-4ea3-89e2-332443ff1f8f
run_id: 20260818T055754Z-8dd3de85
review_head: b3c9e81
merge_base: 68dfda8
proof_scope: live-proof
signed_off: false
verifier_summary: "No exploitable issue found in the serve-configs diff: no secrets/keys in the commit or run artifacts (env-file value grepped against the commit tree, absent; /etc/vllm/qwen38.env is root 0600), installed systemd units match the repo, Conflicts= is symmetric, API-key auth sits inside the keepalive middleware and rejects unauthenticated/malformed bodies in <1 ms on live 8012, patches 0005/0006 have no request-controlled unbounded allocation (draft graph sets and EMA dict are config/req-bounded). Residual P3 hardening items block a clean sign-off under the packet's all-findings rule: the middleware now buffers a full second copy of every authenticated POST /v1/* body with no size cap; the experiment servers on 8013 during the run were unauthenticated on 0.0.0.0 (LAN-visible, now down, not part of the product change); the new throughput unit copies the unhardened service template."
evidence:
  - "git show b3c9e81 | grep for sk-/hf_/Bearer/api_key/secret patterns: only env-var references (os.environ VLLM_API_KEY) and needle-test prose; the actual key value from /etc/vllm/qwen38.env (read via sudo) does not appear anywhere in the b3c9e81 tree (git grep)"
  - "ls -la /etc/vllm: qwen38.env is -rw------- root:root; both units run User=gwillen with EnvironmentFile read by systemd (no key on the command line or in unit files)"
  - "systemctl cat vllm-qwen38 / vllm-qwen38-throughput diffed against serve-configs/systemd/*.service: identical (comments stripped); latency unit active+enabled, throughput unit inactive+disabled; Conflicts= declared in both directions"
  - "live probes on 127.0.0.1:8012 through the keepalive middleware: /health 200; unauthenticated truncated-JSON body -> 401 in 0.5 ms; unauthenticated JSON-list body -> 401; authenticated JSON-list body -> 400 validation error in 18 ms; authenticated truncated JSON -> 400; authenticated stream=true max_tokens=4 -> 200 text/event-stream with normal data: chunks (no early-commit header, no ping injected); service still active afterwards"
  - "vllm/entrypoints/openai/api_server.py in .venv-qwen38: --middleware entries are add_middleware'd after AuthenticationMiddleware, so keepalive is outermost and auth (header-only, never reads the body) runs before any body reaches _KeepAliveResponder._receive"
  - "patch 0005 review: _accepted_ema is keyed by request_id, populated only for running requests and popped in _free_request; decode_cudagraph_managers count = num_speculative_steps - max(2, min_tokens) + 1 (6 for K7), config-bounded and measured (1.59 GiB, gpu-memory-utilization 0.92); adaptive K is clamped to [min_tokens, schedule K] and speculator clamps _num_steps to [1, num_speculative_steps]"
  - "patch 0006 review: DP=1 micro-batching decision is local and guarded by is_last_ubatch_empty; dp_metadata None is threaded through gpu_ubatch_wrapper (lines 475-535) as per-ubatch None; no new eval/shell/deserialisation surface"
  - "run artifacts grep for URLs/IPs: only 127.0.0.1, localhost, http://0.0.0.0:8013 (experiment servers) and the LAN IP 192.168.2.210 in untracked server-*.log files (mq_connect_ip log line); no private endpoints or credentials in tracked files"
commands_run:
  - "git show b3c9e81 | grep -nEi '(sk-[A-Za-z0-9]{8,}|api[_-]?key|VLLM_API_KEY|Bearer |hf_[A-Za-z0-9]{20,}|password|secret|BEGIN (RSA|OPENSSH)|\\.env\\b)'"
  - "grep -rnoE '(sk-[A-Za-z0-9_-]{8,}|Bearer [A-Za-z0-9._-]{8,}|hf_[A-Za-z0-9]{20,})' <run_dir> goal/goal-log.jsonl"
  - "sudo -n grep VLLM_API_KEY /etc/vllm/qwen38.env | cut -d= -f2- -> git grep -n <value> b3c9e81 (no hits); ls -la /etc/vllm"
  - "systemctl is-active/is-enabled vllm-qwen38 vllm-qwen38-throughput; diff <(systemctl cat <unit>) serve-configs/systemd/<unit>.service; systemctl show vllm-qwen38 -p User -p NoNewPrivileges -p ProtectSystem"
  - "curl -s http://127.0.0.1:8012/health; curl -X POST http://127.0.0.1:8012/v1/chat/completions with (a) no auth + truncated JSON, (b) no auth + JSON list, (c) auth + JSON list, (d) auth + truncated JSON, (e) auth + stream=true max_tokens=4"
  - "ss -ltnp | grep 801[23]; sudo -n iptables -S; sudo -n nft list ruleset; hostname -I"
  - "grep -n 'ubatch_dp_metadata|dp_metadata' .venv-qwen38/.../vllm/v1/worker/gpu_ubatch_wrapper.py; grep -n 'add_middleware|args.middleware' .venv-qwen38/.../vllm/entrypoints/openai/api_server.py"
  - "cat <run_dir>/artifacts/configs/{tp_final,lat_final}.yaml <run_dir>/artifacts/scripts/{up,launch,down,tp_bench}.sh"
attack_attempts:
  - attack: "Secret exfiltration via the commit: search b3c9e81 (serve-configs, run artifacts, goal-log, session-log) for API keys, HF tokens, Bearer values, .env contents, and the literal production key value"
    result: "failed - only environment-variable references and the needle-test phrase 'secret launch code'; the real key value is absent from the tree; env file is root 0600"
  - attack: "Bypass or crash the keepalive middleware body peek: send unauthenticated and authenticated non-object JSON, truncated JSON, and a real stream=true request to 8012"
    result: "failed - auth rejects before the body is read (401 <1 ms); authenticated malformed bodies return 400 from vLLM's validator; the middleware's json.loads/.get failure is swallowed (expect_sse=False); streaming path unchanged (SSE headers from the app, no early commit at 40 s threshold for a fast request)"
  - attack: "Start both profiles at once / cross-contaminate env: check Conflicts=, installed-vs-repo drift, and whether the throughput unit inherits VLLM_USE_V2_MODEL_RUNNER or the latency unit inherits NCCL_MAX_NCHANNELS"
    result: "failed - Conflicts= symmetric, units identical to repo, env sets are disjoint per unit; only the latency unit is enabled"
  - attack: "Request-controlled memory blow-up in patch 0005 (per-K CUDA graph sets, EMA dict) and 0006 (micro-batch split with empty second slice, dp_metadata None)"
    result: "failed - graph set count and capture sizes derive from config only; EMA entries are per running request and freed; ubatch split guarded by is_last_ubatch_empty and dp_metadata None is handled per slice; no path lets a client choose K or ubatch count"
  - attack: "Privilege escalation via unit files: ExecStartPre rm -f /dev/shm/vllm_offload_*.mmap, User/Group, EnvironmentFile"
    result: "failed - ExecStartPre runs as User=gwillen (no '+' prefix), glob confined to /dev/shm/vllm_offload_*, EnvironmentFile read by systemd only"
  - attack: "Unauthenticated network exposure during the run: experiment servers on 8013"
    result: "partially succeeded (historical, out of product diff) - launch.sh started vllm serve on port 8013 with no VLLM_API_KEY and no host: setting, so the server bound 0.0.0.0:8013 on 192.168.2.210 with no host firewall (iptables/nft policy accept) for the duration of each experiment; nothing listens on 8013 now"
p_findings:
  - severity: P3
    location: "serve-configs/middleware/vllm_keepalive.py:_KeepAliveResponder._receive"
    summary: "The stream=true peek accumulates a full second copy of every POST /v1/* request body (bytearray) until the last chunk, with no size cap; an authenticated client can double the per-request body memory of the API server (Starlette already buffers the body once)."
    contract: "threat: authenticated memory amplification / DoS on the API process; control: bounded resource use in request handling"
    remediation: "Cap the peek (e.g. stop appending and set expect_sse=False, or parse only if len(_body) <= a few MiB) or scan for the stream flag with a bounded regex on the first chunk; document the cap in the module docstring."
  - severity: P3
    location: "goal-run artifacts/scripts/launch.sh + artifacts/configs/*.yaml (port 8013 experiments; not in the serve-configs product diff)"
    summary: "Experiment servers were launched without VLLM_API_KEY and without host: 127.0.0.1, so during the run an unauthenticated model API listened on 0.0.0.0:8013 (LAN 192.168.2.210, no host firewall). Transient and now down, but the run history is in scope."
    contract: "threat: unauthenticated LAN access to the model/offload tier during experiments; control: least exposure for non-production listeners"
    remediation: "Add host: 127.0.0.1 to the experiment yamls (or --host 127.0.0.1 in launch.sh) and/or export VLLM_API_KEY for experiment servers; note it in the run scripts."
  - severity: P3
    location: "serve-configs/systemd/vllm-qwen38-throughput.service (new) and vllm-qwen38.service"
    summary: "New unit copies the existing template with no service hardening (NoNewPrivileges=no, ProtectSystem=no, no PrivateTmp/ProtectHome/CapabilityBoundingSet); pre-existing pattern, duplicated rather than fixed."
    contract: "threat: post-compromise blast radius of the vLLM process; control: systemd sandboxing"
    remediation: "Optional: add NoNewPrivileges=yes and a conservative ProtectSystem=full/ProtectKernelTunables=yes to both units (keep /dev/shm, /data, /home/gwillen writable), verify a restart on the latency unit at a maintenance window."
rules_reviewed:
  - "AGENTS.md section 1-2 (no bare pip/python3, uv/.venv only; accountability; no low-value PRs): no violations in the diff (patches are venv-local, README marks them human-owned PR candidates)"
  - "OWASP-style API hardening: authn before body handling, bounded request buffering, no status masking of auth errors, no secrets in repo/logs"
  - "systemd unit safety: User/Group, EnvironmentFile perms, ExecStartPre privilege, Conflicts= symmetry"
artifact_paths:
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-serve-configs.diff"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round1-stat.txt"
  - "/shared/vllm/serve-configs/middleware/vllm_keepalive.py"
  - "/shared/vllm/serve-configs/systemd/vllm-qwen38.service"
  - "/shared/vllm/serve-configs/systemd/vllm-qwen38-throughput.service"
  - "/shared/vllm/serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch"
  - "/shared/vllm/serve-configs/patches/0006-dense-tp-dual-batch-overlap-prefill-allreduce.patch"
  - "/shared/vllm/serve-configs/patches/apply-to-venv.sh"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/scripts/launch.sh"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/configs/tp_final.yaml"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/live/20260818T105400Z-prod-8012-ss_bench.log"
  - "/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py"
  - "/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/gpu_ubatch_wrapper.py"
objectives_checked:
  - "no secrets, keys, env-file contents or private endpoints in configs, patches, logs, or the commit"
  - "systemd units: mutual exclusion, env isolation, privilege of ExecStartPre, env-file permissions, installed == repo"
  - "middleware: request-body handling, auth ordering, malformed-input robustness, no body logging, non-streaming behavior unchanged"
  - "patches 0005/0006: request-controlled allocation, unbounded growth, unsafe eval/shell/deserialisation, privilege"
  - "AGENTS.md compliance of the diff and run history"
remaining_gaps:
  - "The keepalive body peek has no size cap (P3 above); a one-line cap makes the middleware strictly no worse than the app it wraps."
  - "Experiment launcher exposes unauthenticated servers on 0.0.0.0:8013 during runs (P3 above); bind to loopback or set the API key for future runs."
signed_off_at: "2026-08-18T11:14:00Z"
repair_resume_command: "/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T110959Z-001-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Signoff

* lane verdict: security lane `signed_off: false` (no exploitable issue in the product change; three P3 hardening findings and the packet's all-findings blocking rule keep this round unsigned)
* P3 `serve-configs/middleware/vllm_keepalive.py:_KeepAliveResponder._receive` — uncapped second copy of every authenticated POST /v1/* body; remediation: cap the peek size (or bounded scan of the first chunk) and document it
* P3 goal-run `artifacts/scripts/launch.sh` + `artifacts/configs/*.yaml` — experiment servers ran unauthenticated on 0.0.0.0:8013 (LAN-visible, transient, now down; not in the serve-configs diff); remediation: `host: 127.0.0.1` and/or `VLLM_API_KEY` for experiment servers
* P3 `serve-configs/systemd/vllm-qwen38-throughput.service` (and the latency unit) — no systemd sandboxing, template duplicated; remediation: optional `NoNewPrivileges=yes` + conservative `ProtectSystem`, verified at a maintenance restart

## Resume From Signoff

* `signed_off` is `false`: the next jump is the `repair_resume_command` in the front matter (`/mdscript-exec /shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/20260818T110959Z-001-qwen38-two-profiles-main-review-packet.mdscript.md#review-this-change`) after the composer's repair; a fresh blind security reviewer must re-review the repaired head — never re-enter this lane's own review from this sign-off
* when a later round reaches `signed_off: true`, continue at `/mdscript-exec /home/gwillen/.agents/skills/self-review/workflows/triple-adversarial-blind-review.mdscript.md#aggregate-triple-signoffs`
