"""Cold vs restored equivalence: same 32k prompt, greedy, 48 tokens; compare text."""
import json, sys, time, urllib.request
BASE = "http://localhost:8013"
def post(path, payload, timeout=3600):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)
salt = sys.argv[1]
filler = ("Telemetry snapshot: subsystem nominal, latency within budget, "
          "queue depth stable, cache hit ratio steady. ")
per_block = post("/tokenize", {"prompt": filler * 50})["count"] / 50
blocks = int((32000 - 200) / per_block)
prompt = (f"[session {salt}] The secret launch code is 736251.\n" + filler * blocks
          + "\n\nWhat is the secret launch code? Answer with the number only, no thinking.")
def run():
    t0 = time.time()
    out = post("/v1/completions", {"model": "Qwen3.8-27B", "prompt": prompt,
                                   "max_tokens": 48, "temperature": 0,
                                   "chat_template_kwargs": {"enable_thinking": False}})
    return time.time() - t0, out["choices"][0]["text"]
mode = sys.argv[2]
dt, text = run()
print(f"{mode}: {dt:.1f}s {text!r}")
open(f"restore_equiv_{salt}_{mode}.txt", "w").write(text)
