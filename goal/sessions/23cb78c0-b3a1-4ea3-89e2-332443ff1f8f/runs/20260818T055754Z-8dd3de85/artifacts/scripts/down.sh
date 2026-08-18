#!/bin/bash
# kill experiment vllm on 8013 by pid of the vllm binary process tree
for p in $(pgrep -f "[.]venv-qwen38/bin/vllm serve" ); do kill $p 2>/dev/null; done
sleep 3
for p in $(pgrep -f "[.]venv-qwen38/bin/vllm serve" ); do kill -9 $p 2>/dev/null; done
pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 2
rm -f /dev/shm/vllm_offload_*.mmap; nvidia-smi --query-gpu=memory.used --format=csv,noheader
