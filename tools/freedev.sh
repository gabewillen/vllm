#!/bin/bash
sudo docker exec glmdev bash -c "pkill -9 -f VLLM; pkill -9 -f run_llm.py" 2>/dev/null
for i in $(seq 60); do
  u=$(hl-smi -Q memory.used -f csv,noheader | awk '{s+=$1} END {print s}')
  [ "$u" -lt 10000 ] && exit 0; sleep 2
done; echo "devices still busy"
