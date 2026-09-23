#!/bin/bash
# usage: t.sh LOGNAME [run_llm args...]  -- runs harness in dev container
LOG=/work/logs/$1.txt; shift
$HOME/work/tools/freedev.sh
sudo docker exec -e GLM53_DENSE_MLA_TORCH=${GLM53_DENSE_MLA_TORCH:-0} ${DEXTRA} glmdev bash -c "cd /work && timeout ${TMO:-1800} python -u tools/run_llm.py $* > $LOG 2>&1"
grep -aE "^\[harness\]|^pos[0-9]|^       ref|Error|error:" ${LOG/\/work/$HOME/work} | grep -av "vllm._C\|commit hash" | head -${HEADN:-60}
