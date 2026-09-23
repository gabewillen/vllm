import gzip, json, sys, collections
tr = json.load(gzip.open(sys.argv[1]))
names = {}
for e in tr["traceEvents"]:
    if e.get("ph") == "M" and e.get("name") == "thread_name": names[(e["pid"], e.get("tid"))] = e["args"]["name"]
ev = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
steps = [e for e in ev if e.get("cat") == "python_function" and "hpu_model_runner.py" in e["name"] and ": sample_tokens" in e["name"]]
st = steps[-2]; s0, s1 = st["ts"], st["ts"] + st["dur"]
k = [e for e in ev if e.get("cat") == "kernel" and e.get("pid") == 0 and e["ts"] >= s0 - 5000 and e["ts"] <= s1]
def union(es):
    es = sorted(es, key=lambda e: e["ts"]); b = 0; cs = ce = None
    for e in es:
        s, en = e["ts"], e["ts"] + e["dur"]
        if ce is None or s > ce:
            if ce is not None: b += ce - cs
            cs, ce = s, en
        else: ce = max(ce, en)
    return b + (ce - cs if ce else 0)
eng = collections.defaultdict(list)
for e in k:
    nm = names.get((0, e["tid"]), str(e["tid"]))
    grp = nm.split(" ")[0] if not nm.startswith("[D") else nm.split("]")[1].strip().split(" ")[0]
    eng[grp].append(e)
print(f"step {st['dur']/1e3:.1f}ms; kernels {len(k)}; any-engine busy {union(k)/1e3:.2f} ms")
for g, es in sorted(eng.items(), key=lambda kv: -union(kv[1])):
    print(f"  {g:12s} busy {union(es)/1e3:7.2f} ms  n={len(es)}")
agg = collections.defaultdict(lambda: [0, 0.0])
for e in k: agg[e["name"].split(" ")[0]][0] += 1; agg[e["name"].split(" ")[0]][1] += e["dur"]
for n, (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:25]: print(f"{d/1e3:8.2f} ms n={c:5d} {n[:100]}")
