"""Broader single-stream bench: 8 prompts, 512 tokens greedy; prints per-prompt decode tok/s, TTFT, tok/pass and a summary."""
import json, os, re, sys, time, urllib.request, statistics
PORT = sys.argv[1] if len(sys.argv) > 1 else "8013"
KEY = os.environ.get('VLLM_API_KEY', '')
URL = f"http://localhost:{PORT}/v1/chat/completions"
CODE = open('/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/kv_offload/tiering/async_lookup.py').read()
CODE2 = open('/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/worker/ubatch_utils.py').read()[:6000]
PROMPTS = {
 "reason-math": ("A train leaves city A at 9:00 going 80 km/h; another leaves city B (300 km away) at 9:30 going 100 km/h toward A. When and where do they meet? Then generalize to symbolic speeds v1, v2, gap D, delay t.", True),
 "reason-sys": ("A ZFS pool on 2 NVMe drives shows 12 GiB ARC but the box has 120 GB RAM; explain what limits ARC, whether raising it helps a KV-cache file store of 55 MB files, and give a recommendation.", True),
 "prose-essay": ("Write a detailed essay about the history of container shipping.", False),
 "prose-story": ("Write a short story about a lighthouse keeper who discovers the light is being used to send messages.", False),
 "code-write": ("Write a Python module implementing an LRU cache with TTL expiry, thread safety, and statistics; include docstrings and a small test suite.", False),
 "code-edit": ("Here is a Python file:\n\n```python\n" + CODE + "\n```\n\nRewrite the ENTIRE file, changing only: rename `_pending_results` to `_result_queue` everywhere and add a `size()` method returning len(self._lookup_state). Output the full file in one code block, no commentary.", False),
 "code-explain": ("Explain what this code does, function by function, then list three possible bugs:\n\n```python\n" + CODE2 + "\n```", False),
 "json-extract": ("Extract every product name, price and quantity from this order email as a JSON array of objects with keys name, price, qty. Email: 'Hi, please send 3x Widget Pro at $19.99, 12 units of the Gizmo Mini ($4.50 each), and one Deluxe Stand for $89. Also 2 packs of Cable Set B, $7.25/pack. Thanks!'", False),
}
def metrics():
    t = urllib.request.urlopen(f"http://localhost:{PORT}/metrics").read().decode()
    g = lambda n: sum(float(m.group(1)) for m in re.finditer(rf'^vllm:{n}\{{[^}}]*\}} ([0-9.e+]+)$', t, re.M))
    return g("spec_decode_num_accepted_tokens_total"), g("spec_decode_num_drafts_total"), g("spec_decode_num_draft_tokens_total")
rows=[]
for name, (p, think) in PROMPTS.items():
    a0, d0, dt0 = metrics()
    body = json.dumps({"model":"Qwen3.8-27B","messages":[{"role":"user","content":p}],"max_tokens":512,"temperature":0,"stream":True,
                       "stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking": think}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.time(); first=None; usage=None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data: ") or line=="data: [DONE]": continue
            c=json.loads(line[6:])
            if c.get("usage"): usage=c["usage"]
            if first is None and c.get("choices") and (c["choices"][0].get("delta",{}).get("content") or c["choices"][0].get("delta",{}).get("reasoning")): first=time.time()
    end=time.time(); a1,d1,dt1 = metrics()
    out=usage["completion_tokens"]; dec=end-first
    per_pass = 1+(a1-a0)/(d1-d0) if d1>d0 else 1
    rows.append((name, first-t0, out/dec, per_pass))
    print(f"{name:13s} ttft={first-t0:5.2f}s  decode={out/dec:6.1f} tok/s  ({out} tok)  tok/pass={per_pass:4.2f}", flush=True)
print(f"MEAN decode={statistics.mean(r[2] for r in rows):.1f} tok/s  median={statistics.median(r[2] for r in rows):.1f}  mean ttft={statistics.mean(r[1] for r in rows):.2f}s")
