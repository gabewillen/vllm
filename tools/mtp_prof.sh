#!/bin/bash
# MTP k=1 device profile (bs1) + timing, full model
cd ~/work
sudo rm -rf prof/mtp
DEXTRA="-e VLLM_SKIP_WARMUP=true -e VLLM_T_COMPILE_DYNAMIC_SHAPES=0 -e VLLM_HPU_COMPILE_CACHE_MULT=16 -e PT_HPU_RECIPE_CACHE_CONFIG=/work/recipe_cache,false,65536 -e VLLM_TORCH_PROFILER_DIR=/work/prof/mtp -e GLM53_TIMING=1 ${EXTRA_ENV}" \
 TMO=7200 tools/t.sh ${NAME:-full_mtp_prof} --model /mnt/glm-models/GLM-5.3-Flash --gen 64 --no-lp --spin-tokens 300 --profile-bs 1 \
 --max-model-len 1024 --max-num-seqs 4 --extra "'{\"speculative_config\": {\"method\": \"mtp\", \"num_speculative_tokens\": 1}}'" > /dev/null
grep -h "harness\]" logs/${NAME:-full_mtp_prof}.txt | cut -c1-200
grep -m3 "Error" logs/${NAME:-full_mtp_prof}.txt | cut -c1-250
sudo find prof/mtp -type f | head
