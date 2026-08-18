# vLLM Model Manager

This directory defines vLLM model switching for the LiteLLM proxy at
`omni.willen.dev`.

There are two switching paths:

- `vllm_model_swapper.py`: request-path switching. LiteLLM sends requests to a
  stable local swapper. The swapper inspects the request `model`, restarts the
  managed vLLM worker when the requested model maps to a different profile,
  rewrites the backend request to the profile's served model name, and proxies
  the inference response.
- `vllm_model_manager.py switch`: blue/green operator switching. It starts a
  candidate vLLM slot on another port/GPU, checks it, rewrites LiteLLM routing,
  restarts LiteLLM, and optionally stops the previous backend.

The swapper does not reuse Python objects or CUDA tensors across restarts. Its
`cpu_ram_cache_max_bytes` setting warms up to that many bytes into the Linux page
cache, so recently loaded model files can skip physical disk reads when memory
pressure has not evicted them.

## Current State

The initial active backend is the existing unmanaged service:

- unit: `qwen-vllm.service`
- port: `8999`
- GPU UUID: `GPU-019676ed-23c9-ad9a-20cb-7bdc13ac61ca`
- profile: `qwen2_5_omni_awq`

The request-path swapper listens on `127.0.0.1:8998` and manages the worker slot
`vllm-managed@hot.service` on port `9000`.

## Commands

Show state and relevant user services:

```bash
/shared/eve/scripts/vllm_model_manager.py status
```

Show request-path swapper state:

```bash
curl http://127.0.0.1:8998/_status
```

Warm a model into the page cache:

```bash
/shared/eve/scripts/vllm_model_swapper.py warm-cache qwen2_5_omni_awq --max-bytes 32GB
```

Smoke-test the current direct vLLM backend:

```bash
/shared/eve/scripts/vllm_model_manager.py smoke qwen2_5_omni_awq --port 8999
```

Smoke-test through LiteLLM:

```bash
/shared/eve/scripts/vllm_model_manager.py smoke qwen2_5_omni_awq --port 8999 --via-litellm
```

Start the Qwen profile in managed slot `blue` on the spare L4 and switch
LiteLLM to it:

```bash
/shared/eve/scripts/vllm_model_manager.py switch qwen2_5_omni_awq \
  --slot blue \
  --port 9000 \
  --gpu-uuid GPU-653b223f-f36b-2fcc-bdc9-78adcbab14eb
```

After a successful switch, retire the previous backend too:

```bash
/shared/eve/scripts/vllm_model_manager.py switch qwen2_5_omni_awq \
  --slot blue \
  --port 9000 \
  --gpu-uuid GPU-653b223f-f36b-2fcc-bdc9-78adcbab14eb \
  --retire-old \
  --disable-old
```

## Adding Models

Add a profile under `profiles/<name>.yaml`. Required fields:

- `executable`
- `args`
- `served_model_name`
- `litellm_model`
- `aliases`

The manager appends `--host`, `--port`, and `--served-model-name` at runtime.
