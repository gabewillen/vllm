#!/bin/bash
# non-spec vs MTP vs DFlash2 on the chat prompt (ref/chat_ids.json): greedy gen + warm 600-token spin
cd ~/work
IDS="'$(cat ref/chat_ids.json)'"
export DEXTRA="-e VLLM_SKIP_WARMUP=true -e VLLM_T_COMPILE_DYNAMIC_SHAPES=0 -e VLLM_HPU_COMPILE_CACHE_MULT=16 -e PT_HPU_RECIPE_CACHE_CONFIG=/work/recipe_cache,false,65536 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 ${EXTRA_ENV}"
export TMO=7200
COMMON="--model /mnt/glm-models/GLM-5.3-Flash --gen 128 --no-lp --ids $IDS --spin-warm 600 --spin-tokens 600 --max-model-len 2048 --max-num-seqs 4"
tools/t.sh chat_ns $COMMON > /dev/null
tools/t.sh chat_mtp $COMMON --extra "'{\"speculative_config\": {\"method\": \"mtp\", \"num_speculative_tokens\": 1}}'" > /dev/null
tools/t.sh chat_dfl $COMMON --extra "'{\"speculative_config\": {\"method\": \"dflash\", \"model\": \"/mnt/glm-models/GLM-5.3-Flash-DFlash2-E\", \"num_speculative_tokens\": 7}, \"enable_prefix_caching\": false}'" > /dev/null
python3 - <<'PY'
import re
g = {}
for n in ("chat_ns", "chat_mtp", "chat_dfl"):
    t = open(f"/home/shadeform/work/logs/{n}.txt", errors="ignore").read()
    m = re.search(r"generated (\[[0-9, ]*\])", t); s = re.search(r"spin 600 tok in [0-9.]+s -> ([0-9.]+) ms/tok", t)
    g[n] = eval(m.group(1)) if m else None
    print(n, "ms/tok", s.group(1) if s else "FAILED")
for n in ("chat_mtp", "chat_dfl"):
    a, b = g["chat_ns"], g[n]
    if a and b:
        print(n, "first divergence vs non-spec:", next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), "none"))
PY
