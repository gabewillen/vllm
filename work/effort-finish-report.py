"""Summarise `event: finish` records from the effort telemetry sink.

Usage: .venv-qwen38/bin/python work/effort-finish-report.py [path] [last_n]
Shows what clients sent (`requested_effort`), what was served
(`effective_effort`) and, for dynamic requests, the rung/escalation report.
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
dyn = [r for r in rows if r.get("dynamic")]
if dyn:
    rungs = collections.Counter(r["dynamic"]["rung"] for r in dyn)
    esc = sum(r["dynamic"]["escalations"] for r in dyn)
    rt = [r["dynamic"]["reasoning_tokens"] for r in dyn]
    print(f"dynamic: {len(dyn)} reqs, final rung dist {dict(sorted(rungs.items()))}, "
          f"mean rung {sum(r['dynamic']['rung'] for r in dyn)/len(dyn):.2f}, "
          f"escalations {esc}, reasoning tokens mean {statistics.mean(rt):.0f} "
          f"median {statistics.median(rt):.0f}, late {sum(r['dynamic']['late'] for r in dyn)}")
fin = collections.Counter(str(r.get("finish_reason")) for r in rows)
print("finish_reason:", dict(fin.most_common()))
