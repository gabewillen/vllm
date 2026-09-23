import gzip, json, sys, collections
tr = json.load(gzip.open(sys.argv[1]))
ev = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
cats = collections.Counter((e.get("cat"), e.get("pid")) for e in ev)
if len(sys.argv) > 2 and sys.argv[2] == "cats":
    for k, v in cats.most_common(30): print(v, k)
    names = {}
    for e in tr["traceEvents"]:
        if e.get("ph") == "M" and e.get("name") in ("process_name", "thread_name"): names[(e["pid"], e.get("tid"))] = e["args"]["name"]
    for k, v in list(names.items())[:40]: print(k, v)
    sys.exit()
dev = sorted([e for e in ev if e.get("cat") == "hpu_op" and e.get("pid") == 0], key=lambda e: e["ts"])
t0, t1 = dev[0]["ts"], max(e["ts"] + e["dur"] for e in dev)
busy = 0; cur_s = cur_e = None
for e in dev:
    s, en = e["ts"], e["ts"] + e["dur"]
    if cur_e is None or s > cur_e:
        if cur_e is not None: busy += cur_e - cur_s
        cur_s, cur_e = s, en
    else: cur_e = max(cur_e, en)
busy += cur_e - cur_s
print(f"window {(t1-t0)/1e3:.1f} ms  device busy {busy/1e3:.1f} ms ({busy/(t1-t0)*100:.0f}%)  n_ops={len(dev)}")
agg = collections.defaultdict(lambda: [0, 0.0])
for e in dev:
    n = e["name"]; agg[n][0] += 1; agg[n][1] += e["dur"]
tot = sum(v[1] for v in agg.values())
for n, (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:int(sys.argv[2]) if len(sys.argv) > 2 else 25]:
    print(f"{d/1e3:9.2f} ms {d/tot*100:5.1f}%  n={c:5d}  avg={d/c:8.1f}us  {n[:110]}")
# host: model forward / step markers
host = [e for e in ev if e.get("cat") in ("cpu_op", "python_function") and ("execute_model" in e["name"] or "sample_tokens" in e["name"])]
for e in host[:12]: print("host", e["name"][:80], f"{e['dur']/1e3:.1f}ms")
