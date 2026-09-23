#!/bin/bash
# losslessness + acceptance: non-spec vs MTP spec on the full model
cd ~/work
export DEXTRA="-e VLLM_SKIP_WARMUP=true -e VLLM_T_COMPILE_DYNAMIC_SHAPES=0 -e VLLM_HPU_COMPILE_CACHE_MULT=16 -e PT_HPU_RECIPE_CACHE_CONFIG=/work/recipe_cache,false,65536"
export TMO=7200
IDS="'[785, 6722, 315, 9621, 374, 12089, 13, 576, 6722, 315, 9851, 374]'"
#tools/t.sh full_mtp0 --model /mnt/glm-models/GLM-5.3-Flash --gen 64 --no-lp --spin-tokens 256 --ids "$IDS" --max-model-len 1024 --max-num-seqs 4 > /dev/null
tools/t.sh full_mtp1 --model /mnt/glm-models/GLM-5.3-Flash --gen 64 --no-lp --spin-tokens 256 --ids "$IDS" --max-model-len 1024 --max-num-seqs 4 --extra "'{\"speculative_config\": {\"method\": \"mtp\", \"num_speculative_tokens\": 1}}'" > /dev/null
for SP in 0 1; do echo "== spec=$SP"; grep -a "harness" logs/full_mtp$SP.txt; grep -a "SpecDecoding\|acceptance" logs/full_mtp$SP.txt | tail -2 | cut -c1-250; done
