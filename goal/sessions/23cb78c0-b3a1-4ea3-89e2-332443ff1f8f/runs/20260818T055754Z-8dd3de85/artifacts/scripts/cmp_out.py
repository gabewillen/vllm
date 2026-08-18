import json, sys, time, urllib.request, random
PORT=sys.argv[1]; out=sys.argv[2]
U=f"http://localhost:{PORT}"
words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
def soup(n, seed):
    r=random.Random(seed); return " ".join(r.choice(words) for _ in range(n))
CODE=open('/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm/v1/kv_offload/tiering/async_lookup.py').read()
cases={
 "soup9k": soup(8000,21)+"\n\nWhich Greek letter appears most often above? Answer with the letter name and a rough count.",
 "soup37k": soup(32000,22)+"\n\nWhich Greek letter appears most often above? Answer with the letter name and a rough count.",
 "code_review": "Review this Python file for bugs and list them tersely:\n\n```python\n"+CODE+"\n```",
 "essay": "Write a detailed essay about the history of container shipping.",
}
res={}
for name,p in cases.items():
    body=json.dumps({"model":"Qwen3.8-27B","messages":[{"role":"user","content":p}],"max_tokens":96,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(U+"/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
    t=time.time(); r=json.loads(urllib.request.urlopen(req, timeout=1800).read()); dt=time.time()-t
    res[name]={"content":r["choices"][0]["message"]["content"],"usage":r["usage"],"time":round(dt,2)}
    print(name, r["usage"]["prompt_tokens"], "tok", round(dt,2), "s ->", repr((r["choices"][0]["message"]["content"] or "")[:80]))
json.dump(res, open(out,"w"), indent=1)
