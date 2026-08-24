# CLIProxyAPI front for vllm-qwen38

Public endpoint: https://cliproxyapi.willen.dev (cloudflared -> localhost:8317).
Web UI: https://cliproxyapi.willen.dev/management.html (management secret-key).
vLLM (:8012) is internal-only; its key lives only in /etc/vllm/qwen38.env and
/etc/cliproxyapi/config.yaml (root, 0600). Client + management keys are in
serve-configs/cliproxyapi.env (gitignored). Live config: /etc/cliproxyapi/config.yaml
(this dir holds a redacted copy). The UI writes config edits back to that file.

    docker run -d --name cliproxyapi -p 8317:8317 \
      --add-host host.docker.internal:host-gateway \
      -v /etc/cliproxyapi/config.yaml:/CLIProxyAPI/config.yaml \
      -v /etc/cliproxyapi/auth:/root/.cli-proxy-api \
      -v /etc/cliproxyapi/static:/CLIProxyAPI/static \
      -v /etc/cliproxyapi/plugins:/CLIProxyAPI/plugins \
      --restart unless-stopped eceasy/cli-proxy-api:latest

Rotating the vLLM key: edit /etc/vllm/qwen38.env, restart vllm-qwen38, update
api-key in /etc/cliproxyapi/config.yaml (file watcher reloads it live).
Replaced the LiteLLM proxy (litellm.willen.dev) on 2026-08-24.
