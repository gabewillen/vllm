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

Bound to localhost only, not exposed via cloudflared. UI at
http://127.0.0.1:18317/management.html. First-run wizard needs:
- Admin key: from `docker logs cpa-manager-plus` (grep "admin key generated"),
  also stashed in `serve-configs/cliproxyapi.env` as `CPAMP_ADMIN_KEY`.
- CPA URL: `http://cliproxyapi:8317` (container DNS name on `cliproxy-net`).
- CPA management key: `CLIPROXY_MGMT_KEY` in `serve-configs/cliproxyapi.env`.
