# canada-quant/vllm-glm53-flash-sm121 — v2 (the PROVEN path)
#
# GLM-5.3-Flash W4A16 (INT4 GPTQ) + DFlash2 speculative-decoding drafter serving
# on 2x NVIDIA DGX Spark (GB10, SM121a, aarch64, 128 GB UMA per node, TP=2 over RoCE).
#
# v2 DESIGN (2026-09-17): fork base + the two serving-critical fork patches BAKED IN,
# byte-identical (sha256-gated) to the patches the production stack ran as runtime
# bind-mounts since 2026-09-12. This is byte-equivalent to the serving stack that
# passed the full adoption gate (boot fuses + smoke + graphs-ON g4 config), with the
# bind-mounts eliminated: no host-side patch files needed to serve.
#
#   base: ghcr.io/canada-quant/vllm-glm53-flash-base:sm121-v11-dflash2
#         (re-hosted community DGX-Spark GLM-5.3-Flash bring-up image)
#         (Apache-2.0 vLLM fork build with the SM121 fixes; vLLM 0.1.dev20051+g487ecf187,
#          FlashInfer 0.6.18.dev20260819, torch 2.13.0+cu130, CUDA 13.0)
#   baked patches (exact production bytes, sha256-gated by the RUN below):
#     - vllm/model_executor/layers/sparse_attn_indexer_kpool.py  (sha256 8a3ecfb0bab2…)
#       the target-side top-k init fix for the NoPE sparse indexer
#     - vllm/v1/core/kv_cache_utils.py                           (sha256 b894ad440cd4…)
#       DFLASH2-DRAFTER-GROUP: GLM-5 KV fast path with the drafter's full-attention
#       layers (the G1 port, rev 3 — see patches/fork/PROVENANCE.md)
#
# Why not the official upstream nightly base? That path (v1, kept in
# Dockerfile.experimental-upstream) boots through model + drafter load and then DIES
# at FlashInfer sparse-MLA warmup ("Failed to run MLA, error: invalid argument",
# 2026-09-17 gate on real SM121 hardware, with and without autotune). The two fork
# patches above are load-bearing for the sparse-MLA path; the upstream nightly base
# has no equivalent. Rebase onto upstream remains blocked on that.
#
# The drafter is NOT baked in: it is bind-mounted at runtime (README "Drafter
# pluggability") so an enhanced drafter is a drop-in swap with no image rebuild.
#
# Build (on an aarch64 SM121 host — e.g. a DGX Spark; cross-builds are not supported):
#   docker build -t ghcr.io/canada-quant/vllm-glm53-flash-sm121:v2-w4a16-dflash2e .

ARG BASE_IMAGE=ghcr.io/canada-quant/vllm-glm53-flash-base:sm121-v11-dflash2@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
FROM ${BASE_IMAGE}

# The two serving-critical fork patches, baked at their import paths:
COPY patches/fork/sparse_attn_indexer_kpool.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py
COPY patches/fork/kv_cache_utils.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py

# SHA GATE — the build FAILS unless the installed bytes are exactly the
# production-verified patch bytes (and both files compile, and the GLM-5.3 target
# arch is registered in this base):
RUN python3 - <<'PY'
import hashlib, py_compile, sys

D = "/usr/local/lib/python3.12/dist-packages"
EXPECT = {
    f"{D}/vllm/model_executor/layers/sparse_attn_indexer_kpool.py":
        "8a3ecfb0bab2441dd7417ed00a10d142191496149f88e5fe79fcfaea4b160980",
    f"{D}/vllm/v1/core/kv_cache_utils.py":
        "b894ad440cd4722342484cc751ba65a480c6903368af168386b86ca6768288e0",
}
for path, sha in EXPECT.items():
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert got == sha, f"SHA GATE FAILED: {path}\n  got      {got}\n  expected {sha}"
    py_compile.compile(path, doraise=True)
    print("SHA_OK", path.rsplit("/", 1)[-1], sha[:16])

kvcu = open(f"{D}/vllm/v1/core/kv_cache_utils.py").read()
assert "DFLASH2-DRAFTER-GROUP" in kvcu, "drafter-group marker missing from kv_cache_utils"
kpool = open(f"{D}/vllm/model_executor/layers/sparse_attn_indexer_kpool.py").read()
assert "topk_indices_buffer" in kpool, "top-k fix missing from kpool"
print("MARKERS_OK: DFLASH2-DRAFTER-GROUP + topk fix present")

sys.path.insert(0, D)
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert "Glm5NextForConditionalGeneration" in archs, "GLM-5.3 target arch not registered"
print("registry OK: Glm5NextForConditionalGeneration present")
print("V2_PATCHES_VERIFIED")
PY

LABEL org.opencontainers.image.title="vllm-glm53-flash-sm121" \
      org.opencontainers.image.description="GLM-5.3-Flash W4A16 + DFlash2 drafter serving on DGX Spark (SM121a), TP=2 — fork base + sha-gated production patches" \
      org.opencontainers.image.vendor="canada-quant" \
      org.opencontainers.image.source="https://github.com/canada-quant/vllm-glm53-flash-sm121" \
      org.opencontainers.image.base.name="ghcr.io/canada-quant/vllm-glm53-flash-base" \
      org.opencontainers.image.base.digest="sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6"
