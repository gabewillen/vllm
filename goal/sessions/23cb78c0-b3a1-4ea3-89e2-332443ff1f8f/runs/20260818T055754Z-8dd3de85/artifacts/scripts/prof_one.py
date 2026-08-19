import json, sys, time, urllib.request
PORT=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 96
U=f"http://localhost:{PORT}"
def post(p, body=None):
    req=urllib.request.Request(U+p, data=json.dumps(body).encode() if body is not None else b"", headers={"Content-Type":"application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=900).read()
body={"model":"Qwen3.8-27B","messages":[{"role":"user","content":"Write a detailed essay about the history of container shipping."}],"max_tokens":n,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
post("/v1/chat/completions", body)  # warm / cache prompt
post("/start_profile"); t=time.time(); r=json.loads(post("/v1/chat/completions", body)); dt=time.time()-t; post("/stop_profile")
print("gen", r["usage"]["completion_tokens"], "tok in", round(dt,2), "s")
