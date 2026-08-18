---
reviewer_id: "security"
reviewer_lane: "security"
review_round: 2
goal: "Maximize Qwen3.8-27B FP8 on 4x L4 for two independent production profiles: (1) LATENCY: max single-stream decode + min TTFT preserving K7 MTP, 262k ctx, stability, quality; (2) THROUGHPUT: max aggregate tok/s under realistic high concurrency preserving 262k ctx, stability, quality; profile continuously, optimize the bottleneck, benchmark + stress each change vs best, stop only when remaining gains are negligible or hardware-bound"
conversation_id: 23cb78c0-b3a1-4ea3-89e2-332443ff1f8f
review_head: 1f3c16e
merge_base: 68dfda8
signed_off: true
verifier_summary: "Blind security review of 68dfda8..1f3c16e (serve-configs diff + run evidence). No secret leakage: the only key lives in root:root 0600 /etc/vllm/qwen38.env; committed configs, units, patches, scripts, server/bench logs, manifest and progress files contain only VLLM_API_KEY=... placeholders and os.environ reads (vLLM startup logs show api_key=None). Middleware: keepalive is the outermost ASGI layer but never touches the request body until the app itself calls receive(), and vLLM's AuthenticationMiddleware answers 401 from headers alone (live: unauth POST /v1/chat/completions -> 401), so the 8 MiB body peek and json.loads add no pre-auth CPU/memory; peek buffer is capped, bodies are never logged, prometheus/OTEL labels are bounded to {sse,json}. Systemd: both units are installed byte-identical to the repo, mutually Conflicts= (systemd confirms ConflictedBy both ways), User=gwillen, NoNewPrivileges/PrivateTmp/ProtectSystem=full live on the running unit, no secrets in unit text. Experiment launcher (archived launch.sh) binds --host 127.0.0.1 on 8013 and nothing else listens on 801x now; only prod 8012 (0.0.0.0, API-key guarded, pre-existing). Patches 0005/0006 add no I/O, env, subprocess, eval, or deserialisation surfaces (pure torch/scheduler code). Residual, non-security nits found and NOT blocking: (a) _body_requests_stream does not catch RecursionError from pathologically nested JSON - the exception propagates exactly as vLLM's own json.loads would (500 either way, authenticated clients only); (b) a truthy non-bool stream value such as \"stream\":\"false\" is treated as streaming and, only after a 40 s idle early commit, yields an SSE-wrapped JSON body where pydantic would have coerced it to False. Pre-existing and out of this diff: /data/kv-offload/qwen38 fs-tier directories are 0755, and PYTHONHASHSEED is fixed (predictable dict hashing for authenticated request parsing) - both predate 68dfda8."
evidence:
  - "git diff 68dfda8 1f3c16e -- serve-configs: middleware peek capped at 8 MiB, no body logging, bounded label sets; units add NoNewPrivileges/PrivateTmp/ProtectSystem=full and mutual Conflicts=; patches 0005/0006 contain no import of os/subprocess/pickle/tempfile and no env or file access"
  - "ls -l /etc/vllm -> qwen38.env root:root 0600; systemctl show vllm-qwen38 -> NoNewPrivileges=yes PrivateTmp=yes ProtectSystem=full User=gwillen; diff /etc/systemd/system/vllm-qwen38{,-throughput}.service against repo -> identical"
  - "git grep over 1f3c16e (run dir + serve-configs) for VLLM_API_KEY=<literal>, Bearer <literal>, sk-/hf_/ghp_ tokens -> only '...' placeholders in manifest/packet and os.environ reads in scripts; committed server-*.log contain no api_key/host lines"
  - "vLLM api_server.build_app: --middleware classes are add_middleware'd after AuthenticationMiddleware (outermost); AuthenticationMiddleware.verify_token reads only the Authorization header; live curl unauth POST /v1/chat/completions -> 401, /health -> 200"
  - "ss -ltnp: only 0.0.0.0:8012 (prod unit pid 3359384) listens on 801x; archived launch.sh uses --host 127.0.0.1 for 8013 experiment servers; up.sh polls 127.0.0.1:8013"
  - "serve-configs/tests: 26 passed with /shared/vllm/.venv-qwen38/bin/python -m pytest -q ."
commands_run:
  - "git diff 68dfda8 1f3c16e -- serve-configs/qwen3_8_27b_fp8_mtp_latency.yaml serve-configs/qwen3_8_27b_fp8_max.yaml serve-configs/systemd serve-configs/patches/README.md serve-configs/patches/apply-to-venv.sh serve-configs/middleware/vllm_keepalive.py"
  - "git grep -n -i -E 'VLLM_API_KEY=[^ $]|Bearer [A-Za-z0-9]|sk-[A-Za-z0-9]{10,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}' 1f3c16e -- <run_dir> serve-configs"
  - "ls -l /etc/vllm/; diff -u /etc/systemd/system/vllm-qwen38{,-throughput}.service serve-configs/systemd/...; systemctl show vllm-qwen38 -p NoNewPrivileges -p PrivateTmp -p ProtectSystem -p User -p EnvironmentFiles; systemctl show ... -p Conflicts -p ConflictedBy; systemctl is-active/is-enabled both units"
  - "ss -ltnp | grep ':801'; pgrep -af 'vllm serve'"
  - "grep -n -E '^\\+.*(subprocess|eval\\(|exec\\(|pickle|os\\.system|torch\\.load|open\\(|__import__|shell=|tempfile|/tmp|getenv|environ)' serve-configs/patches/0005-*.patch serve-configs/patches/0006-*.patch -> no hits"
  - "cd serve-configs/tests && /shared/vllm/.venv-qwen38/bin/python -m pytest -q . -> 26 passed"
  - "/shared/vllm/.venv-qwen38/bin/python probe of vllm_keepalive._body_requests_stream: {\"stream\":\"false\"} -> True; [1]/5/'' -> False; 200k-deep nested JSON -> RecursionError; 9 MiB body -> False in 0.0 s (skipped); 7 MiB body -> True in 0.009 s"
  - "curl http://127.0.0.1:8012/health -> 200; curl -X POST http://127.0.0.1:8012/v1/chat/completions (no Authorization) -> 401; curl /metrics"
attack_attempts:
  - attack: "Secret exfiltration through the commit: search every committed file of 1f3c16e (serve-configs, run artifacts, manifest, progress, goal-log, server and bench logs) for the production API key, Bearer values, HF/GitHub tokens or .env contents"
    result: "failed - only 'VLLM_API_KEY=...' placeholders and os.environ.get reads; the key exists only in root:root 0600 /etc/vllm/qwen38.env, which is outside the repo"
  - attack: "Pre-auth resource exhaustion via the middleware body peek: send large / pathological JSON bodies to POST /v1/* without credentials to make the keepalive layer buffer and json.loads them"
    result: "failed - the peek only runs when the app calls receive(); AuthenticationMiddleware (inside keepalive) rejects on headers alone (live 401) without reading the body, so no unauthenticated CPU/memory is spent; buffer capped at 8 MiB (+ one chunk) and bodies > 8 MiB skip parsing"
  - attack: "Authenticated pathological bodies: 200k-deep nested JSON, 9 MiB body, non-object JSON, stream given as string"
    result: "no security impact - deep nesting raises RecursionError which propagates the same way vLLM's own json.loads would (500, request scoped, pinger cancelled in finally); non-object/empty bodies return False; oversize bodies skip; 'stream':'false' misclassifies as SSE only after a 40 s idle early commit (correctness nit, not exploitable)"
  - attack: "Request-body / prompt leakage into logs or telemetry: inspect every logger call and metric label in the middleware"
    result: "failed - only byte counts, path, ping counts and durations are logged; labels bounded to content_type in {sse,json}; no request body or headers emitted"
  - attack: "Systemd privilege / co-scheduling: start both profiles at once, escalate from the service, or read the env file as the service user"
    result: "failed - Conflicts= is bidirectional (systemd ConflictedBy confirmed), throughput unit disabled/inactive, NoNewPrivileges/PrivateTmp/ProtectSystem=full active on the running unit, EnvironmentFile read by systemd (root) and env only visible to gwillen/root; ExecStartPre rm runs as gwillen on a gwillen-owned glob"
  - attack: "Experiment-server exposure: check whether 8013 experiment servers listen on all interfaces without an API key"
    result: "failed - launch.sh passes --host 127.0.0.1 and nothing listens on 8013 now; only prod 8012 (API-key guarded) is bound to 0.0.0.0, unchanged from base"
  - attack: "Injection / unsafe code in patches 0005/0006: search added lines for subprocess, eval/exec, pickle, torch.load, file or env access, string-built shell"
    result: "failed - none present; changes are scheduler/torch/quantization code operating on in-memory weights and config fields validated at load"
p_findings: []
rules_reviewed:
  - "/shared/vllm/AGENTS.md (no bare pip, uv/.venv only, accountability trailers, domain guides)"
  - "vLLM AuthenticationMiddleware contract (header-only guard on /v1 prefix) in .venv-qwen38 vllm/entrypoints/serve/utils/server_utils.py"
  - "systemd hardening directives (NoNewPrivileges, PrivateTmp, ProtectSystem, Conflicts, EnvironmentFile permissions)"
artifact_paths:
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/review-round2-serve-configs.diff"
  - "/shared/vllm/serve-configs/middleware/vllm_keepalive.py"
  - "/shared/vllm/serve-configs/systemd/vllm-qwen38.service"
  - "/shared/vllm/serve-configs/systemd/vllm-qwen38-throughput.service"
  - "/shared/vllm/serve-configs/patches/0005-spec-decode-adaptive-draft-length-quantized-draft-lm-head-v2-runner.patch"
  - "/shared/vllm/serve-configs/patches/0006-dense-tp-dual-batch-overlap-prefill-allreduce.patch"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/scripts/launch.sh"
  - "/shared/vllm/goal/sessions/23cb78c0-b3a1-4ea3-89e2-332443ff1f8f/runs/20260818T055754Z-8dd3de85/artifacts/manifest.json"
  - "/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py"
objectives_checked:
  - "no secrets in configs, units, patches, scripts, logs, commits"
  - "systemd unit hardening and mutual exclusion; privilege boundary of the service user"
  - "middleware request-body handling: pre-auth exposure, peek cap, error paths, log/telemetry leakage"
  - "injection / unsafe-code / DoS surfaces in patches 0005 and 0006"
  - "experiment-server network exposure"
remaining_gaps: []
signed_off_at: "2026-08-18T12:40:00Z"
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Signoff

* lane verdict: security lane signs off `signed_off: true` for head 1f3c16e (base 68dfda8) — every serious attack failed; `p_findings` is empty
* non-blocking nits recorded in `verifier_summary` only: middleware `_body_requests_stream` could also catch `RecursionError` and treat non-bool `stream` values as pydantic would; fs-tier directory modes and fixed `PYTHONHASHSEED` predate this change

## Resume From Signoff

* `signed_off` is `true`: continue at `/mdscript-exec /home/gwillen/.agents/skills/self-review/workflows/triple-adversarial-blind-review.mdscript.md#aggregate-triple-signoffs`
* do not re-enter this lane's review from this sign-off; a repair (if any other lane requires one) needs a fresh blind reviewer
