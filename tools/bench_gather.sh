#!/bin/bash
# non-spec decode bench bs1/4/8 at a given MoE gather threshold (SLOTS)
cd ~/work
DEXTRA="-e VLLM_SKIP_WARMUP=true -e VLLM_T_COMPILE_DYNAMIC_SHAPES=0 -e VLLM_HPU_COMPILE_CACHE_MULT=16 -e PT_HPU_RECIPE_CACHE_CONFIG=/work/recipe_cache,false,65536 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 -e GLM53_MOE_GATHER_MAX_SLOTS=$SLOTS" \
 TMO=7200 tools/t.sh bench_g$SLOTS --model /mnt/glm-models/GLM-5.3-Flash --gen 64 --no-lp --bench-bs 1,4,8 --bench-tokens 48 \
 --max-model-len 1024 --max-num-seqs 8 > /dev/null
grep -a "harness\] \(bs=\|generated\)" logs/bench_g$SLOTS.txt | cut -c1-160
