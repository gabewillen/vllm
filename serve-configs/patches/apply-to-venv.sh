#!/bin/sh
# Re-apply the venv-local vLLM patches after a rebuild of .venv-qwen38.
# Usage: serve-configs/patches/apply-to-venv.sh [venv_dir]
# All were written against vllm 0.27.2rc1.dev110+gacb0f1dcd; on a newer
# wheel check `patch --dry-run` output and re-port if hunks fail.
set -e
VENV="${1:-/shared/vllm/.venv-qwen38}"
SP=$(ls -d "$VENV"/lib/python3.*/site-packages)
cd "$SP"
for p in "$(dirname "$0")"/000*.patch; do
  echo "== $(basename "$p")"
  patch -p1 -N -r - < "$p" || true
done
# L4-tuned block-fp8 GEMM configs (per-rank Qwen3.8-27B shapes; see README)
cp "$(dirname "$0")"/l4-configs/*.json "$SP/vllm/model_executor/layers/quantization/utils/configs/"
find "$SP/vllm" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null
echo "done — restart vllm-qwen38 to pick up."
