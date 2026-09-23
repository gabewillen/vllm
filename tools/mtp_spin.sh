#!/bin/bash
# MTP k=1 bs1 spin (no timing/profile instrumentation). EXTRA_ENV adds docker -e flags.
cd ~/work
DEXTRA="-e VLLM_SKIP_WARMUP=true -e VLLM_T_COMPILE_DYNAMIC_SHAPES=0 -e VLLM_HPU_COMPILE_CACHE_MULT=16 -e PT_HPU_RECIPE_CACHE_CONFIG=/work/recipe_cache,false,65536 ${EXTRA_ENV}" \
 TMO=7200 tools/t.sh ${NAME:-full_mtp_spin} --model /mnt/glm-models/GLM-5.3-Flash --gen 64 --no-lp --spin-warm 600 --spin-tokens 600 \
 --max-model-len 1024 --max-num-seqs 4 --extra "'{\"speculative_config\": {\"method\": \"mtp\", \"num_speculative_tokens\": 1}}'" > /dev/null
grep -h "harness\] \(gen\|spin \)" logs/${NAME:-full_mtp_spin}.txt | cut -c1-400
grep -m3 "Error" logs/${NAME:-full_mtp_spin}.txt | grep -v "_C\|commit hash" | cut -c1-250
