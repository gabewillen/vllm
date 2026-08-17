#!/bin/sh
# Prune vLLM fs KV-offload tier: delete block files not accessed in TTL_HOURS
# (rolling, per file — the tier is content-addressed so a shared prefix stays
# alive as long as ANY session touches it), then drop empty shard dirs.
# Safe while vLLM runs: the tier treats a missing file as a cache miss.
ROOT="${1:-/data/kv-offload/qwen38}"
TTL_HOURS="${2:-48}"
before=$(du -sm "$ROOT" | cut -f1)
n=$(find "$ROOT" -type f -name '*.bin' -amin +$((TTL_HOURS*60)) -print -delete | wc -l)
find "$ROOT" -mindepth 2 -type d -empty -delete
after=$(du -sm "$ROOT" | cut -f1)
echo "kv-prune: removed $n block files older than ${TTL_HOURS}h (last access); ${before}MB -> ${after}MB"
