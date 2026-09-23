import gzip, json, sys, collections
tr = json.load(gzip.open(sys.argv[1]))
ev = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
steps = [e for e in ev if e.get("cat") == "python_function" and "hpu_model_runner.py" in e["name"] and ": sample_tokens" in e["name"]]
st = steps[-2]; s0, s1 = st["ts"], st["ts"] + st["dur"]
d = sorted([e for e in ev if e.get("pid") == 0 and e["ts"] >= s0 and e["ts"] <= s1], key=lambda e: e["ts"])
print("device events in step:", len(d), "span", (d[-1]["ts"] - d[0]["ts"]) / 1e3, "ms; step starts", (d[0]["ts"] - s0) / 1e3, "ms after host step start")
gaps = []; end = d[0]["ts"]; prev = d[0]
for e in d:
    if e["ts"] - end > 100: gaps.append((e["ts"] - end, prev["name"][:60], e["name"][:60], (end - s0) / 1e3))
    if e["ts"] + e["dur"] > end: end = e["ts"] + e["dur"]; prev = e
gaps.sort(reverse=True)
print("total gap >100us:", sum(g[0] for g in gaps) / 1e3, "ms in", len(gaps), "gaps")
for g in gaps[:15]: print(f"  gap {g[0]/1e3:6.2f}ms at +{g[3]:6.2f}ms  after [{g[1]}] before [{g[2]}]")
# host-side collective / sync ops
hc = collections.Counter()
for e in ev:
    if e.get("cat") in ("cpu_op", "python_function") and e["ts"] >= s0 and e["ts"] <= s1 and any(k in e["name"] for k in ("allreduce", "all_reduce", "synchronize", "_copy_from", "graph_launch", "item", "all_gather")):
        hc[e["name"][:70]] += e["dur"]
for n, t in hc.most_common(12): print(f"host {t/1e3:7.2f} ms {n}")
