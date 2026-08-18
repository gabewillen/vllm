"""Single-stream bench: prose / code-write / code-edit; TTFT, decode tok/s, acceptance."""
import json, os, re, sys, time, urllib.request
PORT = sys.argv[1] if len(sys.argv) > 1 else "8013"
KEY = os.environ.get('VLLM_API_KEY', '')
URL = f"http://localhost:{PORT}/v1/chat/completions"
CODE = open('/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/kv_offload/tiering/async_lookup.py').read()
PROMPTS = {
 "reason": "A ZFS pool on 2 NVMe drives shows 12 GiB ARC but the box has 120 GB RAM; explain what limits ARC, whether raising it helps a KV-cache file store of 55 MB files, and give a recommendation.",
 "prose": "Write a detailed essay about the history of container shipping.",
 "code-write": "Write a Python module implementing an LRU cache with TTL expiry, thread safety, and statistics; include docstrings and a small test suite.",
 "code-edit": "Here is a Python file:\n\n```python\n" + CODE + "\n```\n\nRewrite the ENTIRE file, changing only: rename `_pending_results` to `_result_queue` everywhere and add a `size()` method returning len(self._lookup_state). Output the full file in one code block, no commentary.",
}
def metrics():
    t = urllib.request.urlopen(f"http://localhost:{PORT}/metrics").read().decode()
    g = lambda n: sum(float(m.group(1)) for m in re.finditer(rf'^vllm:{n}\{{[^}}]*\}} ([0-9.e+]+)$', t, re.M))
    return g("spec_decode_num_accepted_tokens_total"), g("spec_decode_num_drafts_total"), g("spec_decode_num_draft_tokens_total")
for name, p in PROMPTS.items():
    a0, d0, dt0 = metrics()
    body = json.dumps({"model":"Qwen3.8-27B","messages":[{"role":"user","content":p}],"max_tokens":768,"temperature":0,"stream":True,
                       "stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking": name=="reason"}}).encode()
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
    acc = (a1-a0)/(dt1-dt0) if dt1>dt0 else 0; per_pass = 1+(a1-a0)/(d1-d0) if d1>d0 else 1
    print(f"{name:10s} ttft={first-t0:5.2f}s  decode={out/dec:6.1f} tok/s  ({out} tok)  accept={acc*100:4.1f}%  tok/pass={per_pass:4.2f}")
