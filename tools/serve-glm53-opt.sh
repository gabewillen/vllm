#!/usr/bin/env bash
# Optimized GLM-5.3-Flash server on 8x Gaudi2 (see ~/work git log for changes).
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-/mnt/glm-models/GLM-5.3-Flash}
IMAGE=${IMAGE:-vllm-gaudi:glm53-patch}
PORT=${PORT:-8000}
WORK=${WORK:-/home/shadeform/work}
SP=/usr/local/lib/python3.10/dist-packages
exec sudo docker run -d --name glm53-server --runtime=habana --ipc=host --network=host \
  -e HABANA_VISIBLE_DEVICES=all \
  -e VLLM_T_COMPILE_DYNAMIC_SHAPES=0 \
  -e VLLM_HPU_COMPILE_CACHE_MULT=16 \
  -e VLLM_BUCKETING_FROM_FILE=/work/tools/buckets_serve.txt \
  -e PT_HPU_RECIPE_CACHE_CONFIG=/work/recipe_cache,false,65536 \
  -e VLLM_RPC_TIMEOUT=3600000 \
  -e VLLM_PROMPT_BS_BUCKET_MAX=2 \
  -v "${MODEL_DIR}:/models/GLM-5.3-Flash:ro" \
  -v "${WORK}:/work" \
  -v "${WORK}/vllm:${SP}/vllm:ro" \
  -v "${WORK}/vllm_gaudi:${SP}/vllm_gaudi:ro" \
  --entrypoint vllm "${IMAGE}" serve /models/GLM-5.3-Flash \
  --tensor-parallel-size 8 \
  --max-model-len 2048 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.35 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --host 0.0.0.0 --port "${PORT}"
