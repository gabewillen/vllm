# CLIProxyAPI front for vllm-qwen38

Public endpoint: https://cpa.willen.dev (cloudflared -> localhost:8317).
Web UI: https://cpa.willen.dev/management.html (management secret-key).
vLLM (:8012) is internal-only; its key lives only in /etc/vllm/qwen38.env and
/etc/cliproxyapi/config.yaml (root, 0600). Client + management keys are in
serve-configs/cliproxyapi.env (gitignored). Live config: /etc/cliproxyapi/config.yaml
(this dir holds a redacted copy). The UI writes config edits back to that file.

    docker run -d --name cliproxyapi -p 8317:8317 \
      --network cliproxy-net --add-host host.docker.internal:host-gateway \
      -v /etc/cliproxyapi/config.yaml:/CLIProxyAPI/config.yaml \
      -v /etc/cliproxyapi/auth:/root/.cli-proxy-api \
      -v /etc/cliproxyapi/static:/CLIProxyAPI/static \
      -v /etc/cliproxyapi/plugins:/CLIProxyAPI/plugins \
      --restart unless-stopped eceasy/cli-proxy-api:latest

Rotating the vLLM key: edit /etc/vllm/qwen38.env, restart vllm-qwen38, update
api-key in /etc/cliproxyapi/config.yaml (file watcher reloads it live).
Replaced the LiteLLM proxy (litellm.willen.dev) on 2026-08-24.

## Cloudflare Tunnel 125s timeout on slow /v1/responses calls

2026-08-24: DeepSWE-via-pier tasks (routed through https://cpa.willen.dev)
were dying with HTTP 524 at exactly 125s (Cloudflare Tunnel's non-Enterprise
Proxy Read Timeout, not configurable via originRequest knobs). Root-caused by
tracing the full stack: vLLM's own [[cloudflare-524-keepalive]] middleware
works fine and CPA proxies fine locally, but CPA's non-streaming
/v1/responses path (OpenAICompatExecutor) reads+translates the *entire*
upstream response before writing anything to its own client, so it produces
zero response bytes for the full duration of a slow call — CPA already has a
first-party fix for exactly this (`StartNonStreamingKeepAlive`, wired into
the responses handler), it's just **off by default**. Fixed by adding to
config.yaml:

    nonstream-keepalive-interval: 15

Verified end-to-end through the real tunnel: an 8000-word/16k-token
generation (~198s) that reliably 524'd before now completes cleanly. See
[[cloudflare-tunnel-125s-limit]] for the full investigation (including a
detour building a from-scratch Gin middleware fork before finding this
existing config flag — reverted, not needed).

## Cloudflare Tunnel: QUIC stalls on large responses (fixed 2026-08-24)

cpamp.willen.dev (a 5.6 MB management.html) intermittently hung for external
clients while small API calls through cpa.willen.dev worked. cloudflared's
journal showed every "stream canceled by remote with error code 0" on ONE of
its four edge connections (connIndex=2: 576 errors vs ~80 on the others), and
a restart just moved the problem to the new connIndex=2 — i.e. a QUIC/UDP
path problem from this LXC (path-MTU black-hole), not a bad edge server.
**Actual cause: an unpaid Cloudflare bill** (account degraded for external
clients while the origin looked healthy). `protocol: http2` was applied at the
same time and is kept (harmless, slightly more robust in an LXC), but do not
treat it as the fix — check the Cloudflare account/billing status first. Note a cloudflared restart
drops in-flight cpa.willen.dev requests (pi retries 530s, but 10 consecutive
failures end a pi session), so restart it only between benchmark runs.

## Key Policy plugin: persist its key store

`cpa-plugin-key-policy` keeps its keys/aliases/usage in a JSON state file
that defaults to `/CLIProxyAPI/cpa-key-policy-state.json` INSIDE the container
— recreating the container wipes every key made in its UI (happened
2026-08-24 13:49). config.yaml now sets
`plugins.configs.cpa-key-policy.state_file: /root/.cli-proxy-api/cpa-key-policy-state.json`
(the mounted /etc/cliproxyapi/auth dir). The path is read at startup only;
when changing it, first `docker cp cliproxyapi:/CLIProxyAPI/cpa-key-policy-state.json
/etc/cliproxyapi/auth/` then `docker restart cliproxyapi`. Verify with
`GET /v0/management/plugins/cpa-key-policy/status` (`state_file` field).

## Effort levels and sampling params through CPA (2026-08-25)

Two silent translation gaps, both found while ablating effort on DeepSWE:

- **`reasoning.effort: none` became `low`.** CPA's thinking pipeline validates
  the requested budget against the model registry; a user-defined
  openai-compatibility model has no `thinking` spec, so budget 0 is "not
  allowed" (`validate.go` warns "budget zero not allowed") and the effort is
  clamped to the lowest level. Fix in config.yaml (needs a container restart,
  hot reload does not re-register the model):

      models:
        - name: "Qwen3.8-27B"
          alias: "Qwen3.8-27B"
          thinking:
            zero-allowed: true
            levels: ["none", "low", "medium", "high", "xhigh"]

  Verified: `none` now renders the 29-token bare prompt with 0 reasoning
  tokens (vLLM sets `enable_thinking=false` for an explicit `none`).
- **`temperature`/`top_p`/`top_k`… were dropped on `/v1/responses`.** The
  responses→chat translator mapped only `max_output_tokens`, so every client
  ran at the model's `generation_config.json` defaults (t1.0/p0.95/k20).
  Fixed upstream in https://github.com/router-for-me/CLIProxyAPI/pull/5231;
  until it ships, the container runs the fork build `cli-proxy-api:sampling-fix`
  (`docker build -t cli-proxy-api:sampling-fix /shared/CLIProxyAPI` on branch
  `fix/responses-sampling-params`). Verified: `temperature: 0` through CPA is
  now deterministic. Related: https://github.com/router-for-me/CLIProxyAPI/pull/5224
  (non-stream `reasoning_tokens`).

## Provider cooldown: opencode single key (2026-08-25)

`opencode` (openai-compatibility, one key) got two one-off upstream
`server_error: Endpoint is unavailable` responses; each made CPA cool the key
down for 60 s (`transientErrorCooldown`, `sdk/cliproxy/auth/conductor_refresh.go`),
so every request on that model returned `503 auth_unavailable: no auth
available` for a minute. With a single credential the cooldown is pure
amplification, so the provider now carries `disable-cooling: true` — upstream
5xx is returned to the client as-is and the next request retries immediately.

**Editing config.yaml:** it is bind-mounted into the container as a single
file. `sed -i` and most editors write a new inode, which the container never
sees (no reload, and the running config silently diverges). Write in place:
`sudo cat new.yaml | docker exec -i cliproxyapi sh -c 'cat > /CLIProxyAPI/config.yaml'`
(and keep the host copy identical) — the watcher then logs
`config successfully reloaded`.

## Codex remote compaction v2 through compat providers (2026-08-25)

Codex Desktop compacts by sending a normal streaming `/v1/responses` whose
`input` ends with `{"type":"compaction_trigger"}` and expects exactly one
`{"type":"compaction","encrypted_content":…}` output item (codex-rs
`compact_remote_v2*.rs`). Upstream CPA translates compat providers to
`/chat/completions` and drops the trigger, so Codex got message items and
failed with `expected exactly one compaction output item, got 0 from N`.
Fork branch `feat/compat-responses-compaction` (image `cli-proxy-api:compaction`)
makes CPA do the compaction: strips tools, appends a summarization
instruction, returns the text as one `compaction` item (`encrypted_content` =
`cpa1:` + base64url JSON), and re-injects that item as a summary message on
later turns. Verified against vLLM Qwen: summary produced, replay answered
from it. Only the openai-compatibility path is affected.

## xAI tool schema normalization (fork, 2026-08-26)

Codex Desktop on `grok-4.6` got `400 invalid_client_tool_schema
mcp__codex_app__automation_update: tool parameter root must be an object
type` from xAI's cli-chat-proxy on every turn. That tool's `parameters` is
`{"type":"object","oneOf":[{"$ref":"#/$defs/..."},...],"$defs":{...}}`;
upstream CPA only special-cased the `codex_app` namespace by name (Codex now
sends `mcp__codex_app`) and stamped `"type":"object"` onto the `$ref`
branches, which xAI still rejects. Fork commit `960be823` (same branch
`feat/compat-responses-compaction`, image `cli-proxy-api:compaction`)
replaces that with a general normalizer in
`internal/runtime/executor/xai_tool_schema.go`, applied to every function
tool on the xAI path (`/v1/responses`, the responses websocket, and
`/v1/chat/completions`):

- root `type: object` with inline object union branches: add missing
  `type: object` to the branches, keep constraints (upstream behaviour);
- root union whose branches all resolve (following local `$ref`) to objects:
  merge into one object with the union of their properties, no `required`,
  `$defs` kept (this is the Codex automation_update case);
- anything else (missing/scalar/array root type, a non-object branch): wrap
  the original schema as `{"type":"object","properties":{"input":<schema>},
  "required":["input"]}`; wrapped tool names are tracked per request and the
  `{"input":...}` in returned `function_call.arguments` is unwrapped in
  `response.output_item.done` and terminal `response.*` payloads (the
  `function_call_arguments.delta/done` events stay wrapped).

Verified: the real Codex schema returns 200 on both endpoints (was 400), a
forced call on the merged tool yields `{"id":"abc123","mode":"view"}`, and a
forced call on a wrapped `anyOf[object,string]` tool yields `{"a":"hello"}`.

## Auto model context/output limits (fork, 2026-08-26)

Codex Desktop reads `GET /v1/models?client_version=1`; for models that are not
in CPA's Codex template list, upstream fell back to the template default and
reported `context_window` 272000 for everything from an `openai-compatibility`
provider. Upstream's only source for those models is the per-model config field
`max-context-length`.

Fork commit `09f9f3d9` (branch `feat/compat-responses-compaction`, image
`cli-proxy-api:compaction`) adds `internal/modellimits`, a best-effort resolver
wired into `sdk/cliproxy/service_model_limits.go`. It runs at provider model
registration and on every config reload (never on the request path), with a 3 s
timeout per fetch and results cached in-process. Precedence per model:

1. explicit `max-context-length` in config (unchanged upstream behaviour);
2. the provider's own `GET {base-url}/models`, if the entry for that model id
   carries `max_model_len` (vLLM), `context_length` (OpenRouter/LiteLLM),
   `context_window`, `max_context_length`, `max_input_tokens`, or a nested
   `top_provider`/`limit` object; output from `max_output_tokens`,
   `max_completion_tokens`, or `max_tokens`;
3. models.dev (`https://models.dev/api.json`), matched by model id, preferring
   the provider whose `api` URL matches the config `base-url` (so
   `opencode.ai/zen/go/v1` -> `opencode-go`), then one whose id/name matches the
   config provider name, then the first provider listing that id. `limit.context`
   -> context, `limit.output` -> max output.

The models.dev payload is fetched once at startup, refreshed every 24 h, and
persisted to `<auth-dir>/models-dev-cache.json` (in the container:
`/root/.cli-proxy-api/`, i.e. `/etc/cliproxyapi/auth/` on the host), so a
restart while offline still resolves. Resolution is logged at info, e.g.
`model limits: provider "vllm-qwen38" resolved Qwen3.8-27B ctx=262144 out=0
(upstream:vllm-qwen38)`, deduplicated per provider.

Values flow into the Codex catalog (`context_window`/`max_context_window`,
`max_tokens`) and the Anthropic-format list (`max_input_tokens`/`max_tokens`).

New global config keys (all optional, defaults shown):

- `auto-model-limits: true` — turn the whole resolver off with `false`;
- `models-dev-url: https://models.dev/api.json`;
- `models-dev-refresh: 24h`;
- `openai-models-extended-fields: true` — **diverges from upstream policy.**
  Upstream deliberately strips the plain `GET /v1/models` response to
  id/object/created/owned_by and declined to add limits (won't-fix #5119). With
  this flag on, `context_length`, `max_context_length` and
  `max_completion_tokens` are passed through when known (for OAuth/static
  models too). Set it to `false` for upstream's strict four-field behaviour.

Live after deploy (`cli-proxy-api:compaction`, 2026-08-26):

| model | Codex `context_window` / `max_tokens` | `/v1/models` | Anthropic `max_input_tokens` / `max_tokens` |
| --- | --- | --- | --- |
| `Qwen3.8-27B` | 262144 / — (was 272000) | context_length 262144 (was absent) | 262144 / 64000 (default output) |
| `ox-alpha-free` | 1000000 / 131072 (was 272000) | context_length 1000000, max_completion_tokens 131072 | 1000000 / 131072 |
| `claude-sonnet-4-5-20250929` (OAuth) | 200000 / 64000 (unchanged) | context_length 200000, max_completion_tokens 64000 (was absent) | 200000 / 64000 |

vLLM's `/models` carries no output-token field, so `Qwen3.8-27B` keeps the
per-format default output; set `max-completion-tokens` upstream or an explicit
config value if that matters.

## Fork sync with upstream (2026-08-26)

Fork branch `feat/compat-responses-compaction` (remote `origin-fork`,
gabewillen/CLIProxyAPI) is rebased on upstream `origin/main` = `1f53b2eb`
(v7.2.142, 2026-08-26); previous base `dc3c3b1e` (v7.2.141). Fork HEAD
`79e181c9`; our four commits (sampling forward, compaction v2, xAI schema
normalizer, auto model limits) applied cleanly, no file overlap with the
upstream delta (dc3c3b1e..1f53b2eb: quota-signal observation + fair credential
rotation in `sdk/cliproxy/auth`, antigravity finish-reason/tool-result fixes,
plugin quiesce on hot reload, GitHub-token release checks, video duration
fix).

Sync procedure: `git fetch --all --tags && git rebase origin/main`, then in
`golang:1.26-bookworm` (`-v /shared/CLIProxyAPI:/src -w /src`, add
`git config --global --add safe.directory /src` for VCS stamping):
`gofmt -l .`, `go build ./...`, `go test ./internal/... ./sdk/...` — only the
three known host-fingerprint failures in `internal/runtime/executor`
(`TestApplyClaudeHeaders_*`, `TestClaudeExecutor_NonClaudeRequest…`) are
expected. Then `git push --force-with-lease origin-fork
feat/compat-responses-compaction`, `docker build -t cli-proxy-api:compaction
/shared/CLIProxyAPI`, and recreate `cliproxyapi` with the flags/mounts from
`docker inspect`. Post-deploy checks: `/v1/models?client_version=1`
(Qwen3.8-27B 262144, ox-alpha-free 1000000), Codex `oneOf`-of-`$ref` schema
against grok-4.6 -> 200, compaction_trigger on Qwen3.8-27B -> one
`compaction` item. All passed 2026-08-26.

## Live Codex model catalog per OAuth account (fork, 2026-08-26)

CPA's Codex (ChatGPT OAuth) model list was static (`internal/registry/models/
models.json`), so account-gated models such as `gpt-daybreak-blue-latest`
("Daybreak Blue") and the hidden `gpt-reserve` that OpenAI's live catalog
returns for this account were unknown to CPA and Codex Desktop could not select
them. Fork commits `55cba9b2` + `e9417066` (branch
`feat/compat-responses-compaction`, image `cli-proxy-api:compaction`) add
`internal/codexmodels` wired in `sdk/cliproxy/service_codex_live_models.go`:

- For every Codex OAuth auth, at model registration (startup, auth reload,
  config reload) CPA fetches
  `GET https://chatgpt.com/backend-api/codex/models?client_version=0.146.0`
  with that auth's bearer token, `Chatgpt-Account-Id`, the executor's
  `codex-tui` User-Agent/Originator and any custom header attrs (8 s timeout,
  best-effort; an expired access token is refreshed the way the executor does
  and persisted). A 15 min ticker re-registers auths whose catalog is older
  than the refresh interval; a failed fetch is not retried for 15 min.
- Cache: `<auth-dir>/codex-models-cache.json` (`/etc/cliproxyapi/auth/` on the
  host), keyed by `account_id` with `fetched_at`, so an offline restart keeps
  the last catalog (log line says `(cache)` instead of `(live)`).
- Merge, per account: union of static plan models + live slugs. Live entries
  add unknown slugs (routable to the codex executor, `type: openai`, 128k
  max output) and refresh display name, `context_window` -> `context_length`,
  reasoning levels and input modalities on known ones; static-only models are
  kept. `max_context_window` is deliberately NOT mapped to CPA's
  `MaxContextLength` (that field is a config override that would force
  `context_window` to 872000 in the Codex catalog). Hidden live models
  (`visibility: hide`, e.g. `gpt-reserve`) are registered and routable but
  stay hidden in the `?client_version=` catalog.
- Codex catalog (`/v1/models?client_version=...`): live entries overlay the
  static templates (`registry.SetCodexLiveClientModels`, union across
  accounts): unknown slugs become full templates (including OpenAI's
  `model_messages` instructions template), known slugs get `display_name`,
  `description`, `visibility`, `context_window`, `max_context_window`,
  `priority`, `supported_reasoning_levels`, `default_reasoning_level` refreshed.
- One info line per account when the result changes, e.g.
  `codex live models: account codex-3660ee59-... (live) added
  gpt-daybreak-blue-latest,gpt-reserve; updated codex-auto-review,...`.
- Config (config.yaml top level, both optional): `codex-live-models: false`
  disables everything (static list only); `codex-live-models-refresh: 6h`
  (Go duration) sets the per-account refresh interval. Precedence for a Codex
  model's metadata: live catalog > static models.json / codex_client_models.json
  templates; `excluded_models` and OAuth model aliases still apply after the
  merge.
- Static fallback: `gpt-daybreak-blue-latest` was also added to models.json
  (codex-team/plus/pro, 272000 ctx, sol's shape). Note upstream's 3 h remote
  models.json refresh can replace the static list; the live merge re-adds it.

Verified 2026-08-26/27: `/v1/models?client_version=0.149.0` lists
`gpt-daybreak-blue-latest` (Daybreak Blue, list, 272000/872000, priority 3,
low..ultra) and `gpt-reserve` (hide); plain `/v1/models` and the Anthropic list
(cloaked id) include both; `POST /v1/responses` model
`gpt-daybreak-blue-latest` stream:false -> 200 completed "pong" (upstream echoes
`model: gpt-5.6-sol` for this slug -- verified the same directly against
chatgpt.com, so it is an upstream alias, not a CPA rewrite); `gpt-5.6-sol`
still 200. openai-compatibility providers are untouched by this feature.

## Egress isolation
Container runs on its own bridge `cliproxy-net` (172.30.0.0/24, `docker network
create --subnet 172.30.0.0/24 cliproxy-net`). `firewall.sh` (installed at
/etc/cliproxyapi/, run at boot by `cliproxyapi-firewall.service` after docker)
allows only: host :8012 (vLLM), DNS to 192.168.2.1, and WAN. All other host
ports and RFC1918 ranges are dropped. Verified 2026-08-24 from a curl container
on the network: 8012 200; host 8317/22, router, LAN peer blocked; github 200.

## CPA Manager Plus (observability dashboard)

Added 2026-08-24, full mode (separate service + SQLite, not the lightweight
panel). Persistent request history / cost analytics on top of CPA's
management API.

    docker run -d --name cpa-manager-plus --restart unless-stopped \
      --network cliproxy-net -p 127.0.0.1:18317:18317 \
      -v cpa-manager-plus-data:/data seakee/cpa-manager-plus:latest

Public endpoint: https://cpamp.willen.dev (cloudflared -> localhost:18317,
own ingress hostname added 2026-08-24; separate from cpa.willen.dev since
CPAMP's admin key has broader access than the client key). First-run wizard
needs:
- Admin key: from `docker logs cpa-manager-plus` (grep "admin key generated"),
  also stashed in `serve-configs/cliproxyapi.env` as `CPAMP_ADMIN_KEY`.
- CPA URL: `http://cliproxyapi:8317` (container DNS name on `cliproxy-net`).
- CPA management key: `CLIPROXY_MGMT_KEY` in `serve-configs/cliproxyapi.env`.
