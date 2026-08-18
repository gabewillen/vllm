import json, random, sys, time, http.client
port=sys.argv[1]; seed=int(sys.argv[2]); nw=int(sys.argv[3])
random.seed(seed); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
text=" ".join(f"{random.choice(words)}{random.randint(0,999)}" for _ in range(nw))
body=json.dumps({"model":"Qwen3.8-27B","stream":True,"max_tokens":4,"chat_template_kwargs":{"enable_thinking":False},"messages":[{"role":"user","content":text+"\n\nReply with the single word OK."}]}).encode()
c=http.client.HTTPConnection("127.0.0.1",int(port),timeout=900); t0=time.time()
c.request("POST","/v1/chat/completions",body,{"Content-Type":"application/json"}); r=c.getresponse(); first=None; usage=None
while True:
    line=r.readline()
    if not line: break
    if line.startswith(b"data:") and b'"content"' in line and first is None: first=time.time()
    if b'"usage"' in line:
        try: usage=json.loads(line[6:])["usage"]["prompt_tokens"]
        except Exception: pass
print(f"seed={seed} prompt_tokens={usage} TTFT={first-t0:.1f}s")
