import gzip, json, sys, collections
tr = json.load(gzip.open(sys.argv[1])); pid_filter = None
ev = [e for e in tr["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("python_function", "cpu_op")]
steps = [e for e in ev if "hpu_model_runner.py" in e["name"] and ": sample_tokens" in e["name"]]
st = steps[-2]; s0, s1 = st["ts"], st["ts"] + st["dur"]; tid = st["tid"]
inside = [e for e in ev if e["tid"] == tid and e["ts"] >= s0 and e["ts"] + e["dur"] <= s1]
inside.sort(key=lambda e: (e["ts"], -e["dur"]))
# self time via stack
selft = collections.Counter(); incl = collections.Counter(); stack = []
for e in inside:
    while stack and stack[-1]["ts"] + stack[-1]["dur"] <= e["ts"]: stack.pop()
    if stack: selft[stack[-1]["name"]] -= e["dur"]
    selft[e["name"]] += e["dur"]; stack.append(e)
for e in inside: incl[e["name"]] += e["dur"]
print(f"step {st['dur']/1e3:.1f} ms")
print("--- top self"); [print(f"{d/1e3:7.2f} ms  {n[:120]}") for n, d in selft.most_common(25)]
print("--- top inclusive (depth-limited)")
for n, d in incl.most_common(60):
    if any(k in n for k in ("hpu_model_runner", "glm5next", "sampler", "model_runner", "logits", "compile", "eval_frame")): print(f"{d/1e3:7.2f} ms  {n[:120]}")
