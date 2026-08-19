import json, sys, time, urllib.request, random
PORT=sys.argv[1]; ntok=int(sys.argv[2]); seed=int(sys.argv[3]) if len(sys.argv)>3 else 1
U=f"http://localhost:{PORT}"
r=random.Random(seed)
words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
prompt=" ".join(r.choice(words) for _ in range(ntok))+"\n\nSummarize the above in one sentence."
body=json.dumps({"model":"Qwen3.8-27B","messages":[{"role":"user","content":prompt}],"max_tokens":32,"temperature":0,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}).encode()
req=urllib.request.Request(U+"/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
t0=time.time(); first=None; usage=None
with urllib.request.urlopen(req, timeout=1800) as resp:
    for line in resp:
        line=line.decode().strip()
        if not line.startswith("data: ") or line=="data: [DONE]": continue
        c=json.loads(line[6:])
        if c.get("usage"): usage=c["usage"]
        if first is None and c.get("choices") and c["choices"][0].get("delta",{}).get("content"): first=time.time()
print(f"prompt_tokens={usage['prompt_tokens']} ttft={first-t0:.2f}s prefill_tok_s={usage['prompt_tokens']/(first-t0):.0f}")
