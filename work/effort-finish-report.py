"""Summarise `event: finish` records from the effort telemetry sink.

Usage: .venv-qwen38/bin/python work/effort-finish-report.py [path] [last_n]
Shows what clients sent (`requested_effort`), what was served
(`effective_effort`) and, for dynamic requests, the v3 report: level
distribution, how many levels came from the hidden-state memory (`decided`),
how the think block closed (`close_kind`) and the reasoning-token spend.
"""

import collections
import json
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/data/effort-telemetry/latency.jsonl"
last_n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rows = []
with open(path, "rb") as f:
    f.seek(0, 2)
    size = f.tell()
    f.seek(max(0, size - 400_000_000))
    for line in f:
        if b'"event": "finish"' not in line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
rows = rows[-last_n:]
if not rows:
    sys.exit("no finish records yet")
req = collections.Counter(str(r.get("requested_effort")) for r in rows)
eff = collections.Counter(str(r.get("effective_effort")) for r in rows)
print(f"{len(rows)} finished requests")
print("requested_effort (as sent):", dict(req.most_common()))
print("effective_effort (served): ", dict(eff.most_common()))
dyn = [r["dynamic"] for r in rows if r.get("dynamic")]
legacy = sum("level" not in d for d in dyn)
dyn = [d for d in dyn if "level" in d]
if legacy:
    print(f"skipped {legacy} pre-v3 (rung-ladder) records")
if dyn:
    levels = collections.Counter(d["level"] for d in dyn)
    decided = sum(int(d.get("decided", 0)) for d in dyn)
    close = collections.Counter(str(d.get("close_kind")) for d in dyn)
    rt = [d["reasoning_tokens"] for d in dyn]
    rt_sorted = sorted(rt)
    p90 = rt_sorted[min(len(rt_sorted) - 1, int(0.9 * len(rt_sorted)))]
    print(
        f"dynamic: {len(dyn)} reqs, level dist {dict(sorted(levels.items()))}, "
        f"mean level {sum(d['level'] for d in dyn) / len(dyn):.2f}, "
        f"decided by memory {decided}/{len(dyn)}, close_kind {dict(close)}"
    )
    print(
        f"reasoning tokens mean {statistics.mean(rt):.0f} "
        f"median {statistics.median(rt):.0f} p90 {p90} max {max(rt)}; "
        f"memory entries at last finish {dyn[-1].get('memory_entries')}"
    )
    for level in sorted(levels):
        sub = [d["reasoning_tokens"] for d in dyn if d["level"] == level]
        print(
            f"  level {level}: {len(sub)} reqs, reasoning tokens "
            f"median {statistics.median(sub):.0f} max {max(sub)}"
        )
fin = collections.Counter(str(r.get("finish_reason")) for r in rows)
print("finish_reason:", dict(fin.most_common()))
