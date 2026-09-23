#!/bin/bash
# usage: bench_serve.sh CONC NUM_PROMPTS [IN] [OUT]
C=$1; N=$2; IN=${3:-512}; OUT=${4:-256}
sudo docker run --rm --network host --entrypoint vllm -v /mnt/glm-models/GLM-5.3-Flash:/models/GLM-5.3-Flash:ro \
  vllm-gaudi:glm53-patch bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
  --model /models/GLM-5.3-Flash --dataset-name random --random-input-len $IN --random-output-len $OUT \
  --num-prompts $N --max-concurrency $C --ignore-eos 2>&1 | grep -E "Successful|Benchmark duration|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|Mean ITL|Request throughput"
