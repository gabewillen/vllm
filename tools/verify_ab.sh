#!/bin/bash
# Losslessness of multi-token verify: non-spec vs ngram(k) greedy on MODEL (default L5).
cd ~/work
MODEL=${MODEL:-/mnt/glm-models/GLM-5.3-Flash-L5}
K=${K:-7}
IDS="'[785, 6722, 315, 9621, 374, 12089, 13, 576, 6722, 315, 9621, 374, 12089, 13, 576, 6722, 315, 9621, 374]'"
export DEXTRA="-e VLLM_SKIP_WARMUP=true -e VLLM_T_COMPILE_DYNAMIC_SHAPES=0 -e VLLM_HPU_COMPILE_CACHE_MULT=16 -e PT_HPU_RECIPE_CACHE_CONFIG=/work/recipe_cache,false,65536 ${EXTRA_ENV}"
export TMO=3600
tools/t.sh vab_ns --model $MODEL --gen 64 --no-lp --ids "$IDS" --max-model-len 1024 --max-num-seqs 4 > /dev/null
tools/t.sh vab_ng --model $MODEL --gen 64 --no-lp --ids "$IDS" --max-model-len 1024 --max-num-seqs 4 --extra "'{\"speculative_config\": {\"method\": \"ngram\", \"num_speculative_tokens\": $K, \"prompt_lookup_max\": 4, \"prompt_lookup_min\": 2}}'" > /dev/null
python3 - <<'PY'
import re
g = {}
for n in ("vab_ns", "vab_ng"):
    t = open(f"/home/shadeform/work/logs/{n}.txt").read()
    m = re.search(r"generated (\[[0-9, ]*\])", t)
    g[n] = eval(m.group(1)) if m else None
    if not m:
        print(n, "FAILED:", [l for l in t.splitlines() if "Error" in l and "_C" not in l][:3])
a, b = g["vab_ns"], g["vab_ng"]
if a and b:
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print("non-spec:", a[:24]); print("ngram   :", b[:24]); print("first divergence:", d)
PY
