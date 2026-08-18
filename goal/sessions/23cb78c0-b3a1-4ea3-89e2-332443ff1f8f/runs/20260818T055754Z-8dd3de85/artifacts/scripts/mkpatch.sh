#!/bin/bash
# usage: mkpatch.sh <out-name> <rel-file>...   (rel to site-packages, files must have .orig0005 backups; new files diffed against /dev/null)
SP=/shared/vllm/.venv-qwen38/lib/python3.12/site-packages; OUT=/shared/vllm/serve-configs/patches/$1; shift
cd $SP; : > $OUT
for f in "$@"; do
  if [ -f $f.orig0005 ]; then diff -u --label a/$f --label b/$f $f.orig0005 $f >> $OUT; else diff -u --label a/$f --label b/$f /dev/null $f >> $OUT; fi
done
echo "wrote $OUT ($(grep -c '^[-+]' $OUT) +/- lines)"
