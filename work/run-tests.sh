#!/bin/sh
# Run pytest against a lane worktree copy of the installed vLLM package.
# Usage: work/run-tests.sh <worktree-dir> [pytest args...]
# The worktree shadows site-packages/vllm via PYTHONPATH; cwd is neutral so
# the upstream source tree at /shared/vllm/vllm cannot shadow it.
set -e
WT=$(cd "$1" && pwd); shift
cd /shared/vllm/work
PYTHONPATH="$WT" exec /shared/vllm/.venv-qwen38/bin/python -m pytest -q "$@"
