#!/bin/sh
# Emit a venv patch (paths a/vllm/... b/vllm/...) from a lane worktree vs master.
# Usage: work/make-patch.sh <worktree> <out.patch>
set -e
cd "$1"
git add -A
git diff --cached master -- . ':(exclude)*.so' > "$2"
git reset -q
echo "$(grep -c '^diff --git' "$2") files -> $2"
