import gzip, json, sys, collections
tr = json.load(gzip.open(sys.argv[1]))
names = {(e["pid"], e.get("tid")): e["args"]["name"] for e in tr["traceEvents"] if e.get("ph") == "M" and e.get("name") == "thread_name"}
ev = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
steps = [e for e in ev if e.get("cat") == "python_function" and "hpu_model_runner.py" in e["name"] and ": sample_tokens" in e["name"]]
st = steps[-2]; s0, s1 = st["ts"], st["ts"] + st["dur"]
by = collections.defaultdict(list)
for e in ev:
    if e["ts"] >= s0 and e["ts"] <= s1 and e.get("pid") != 0: by[(e["pid"], e["tid"])].append(e)
for k, es in by.items():
    top = [e for e in es]
    tot = collections.Counter()
    for e in es: tot[e["name"][:90]] += e["dur"]
    print(f"== {names.get(k, k)} ({k}) n={len(es)}")
    for n, d in tot.most_common(6): print(f"   {d/1e3:7.2f} ms {n}")
