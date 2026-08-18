import json, sys, time, urllib.request, random
PORT=sys.argv[1]; n=int(sys.argv[2]); seed=int(sys.argv[3])
U=f"http://localhost:{PORT}"
def post(p, body=None):
    req=urllib.request.Request(U+p, data=json.dumps(body).encode() if body is not None else b"", headers={"Content-Type":"application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=1800).read()
r=random.Random(seed); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
prompt=" ".join(r.choice(words) for _ in range(n))+"\n\nSummarize."
body={"model":"Qwen3.8-27B","messages":[{"role":"user","content":prompt}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
post("/start_profile"); t=time.time(); rr=json.loads(post("/v1/chat/completions", body)); dt=time.time()-t; post("/stop_profile")
print("prompt", rr["usage"]["prompt_tokens"], "in", round(dt,2), "s")
