#!/bin/bash
# DFlash2 (K=7) bs1: greedy gen + warm spin. EXTRA_ENV adds docker -e flags; ARGS adds harness args.
cd ~/work
SPEC='{"speculative_config": {"method": "dflash", "model": "/mnt/glm-models/GLM-5.3-Flash-DFlash2-E", "num_speculative_tokens": 7}'"${PC-, \"enable_prefix_caching\": false}"'}'
DEXTRA="-e VLLM_SKIP_WARMUP=true -e VLLM_T_COMPILE_DYNAMIC_SHAPES=0 -e VLLM_HPU_COMPILE_CACHE_MULT=16 -e PT_HPU_RECIPE_CACHE_CONFIG=/work/recipe_cache,false,65536 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 ${EXTRA_ENV}" \
 TMO=7200 tools/t.sh ${NAME:-full_dflash} --model /mnt/glm-models/GLM-5.3-Flash --gen 64 --no-lp ${ARGS:---spin-warm 300 --spin-tokens 600} \
 --max-model-len 1024 --max-num-seqs 4 --extra "'$SPEC'" > /dev/null
grep -h "harness\] \(gen\|spin \)\|DFlash2 drafter\|Compiled DFlash2" logs/${NAME:-full_dflash}.txt | sort -u | cut -c1-400
grep -a "Traceback\|Error" logs/${NAME:-full_dflash}.txt | grep -v "_C'\|commit hash" | head -5 | cut -c1-250
