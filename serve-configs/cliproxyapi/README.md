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
Fix: `protocol: http2` at the top of /etc/cloudflared/config.yml (TCP
transport), then `systemctl restart cloudflared`. Verified: 5.6 MB page
0.18-0.35 s through the tunnel, zero cancels. Note a cloudflared restart
drops in-flight cpa.willen.dev requests (pi retries 530s, but 10 consecutive
failures end a pi session), so restart it only between benchmark runs.

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
