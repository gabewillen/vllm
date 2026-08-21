#!/bin/sh
# Emit a venv patch (paths a/vllm/... b/vllm/...) from a lane worktree.
# Usage: work/make-patch.sh <worktree> <out.patch> [base]
# `base` defaults to the branch the venv wheel was built from, so re-running
# reproduces the same patch. Patch artifacts are excluded from their own diff.
set -e
cd "$1"
BASE=${3:-master}
git add -A
git diff --cached "$BASE" -- . ':(exclude)*.so' ':(exclude)serve-configs/patches/*.patch' > "$2"
git reset -q
echo "$(grep -c '^diff --git' "$2") files -> $2"
