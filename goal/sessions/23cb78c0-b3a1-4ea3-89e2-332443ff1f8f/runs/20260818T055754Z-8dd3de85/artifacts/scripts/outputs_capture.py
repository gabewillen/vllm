"""Capture greedy outputs (96 tokens) for the 8 ss_bench2 prompts -> JSON, for baseline-vs-final comparison."""
import json, sys, urllib.request, importlib.util, os
PORT=sys.argv[1]; out=sys.argv[2]
spec=importlib.util.spec_from_file_location("ssb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ss_bench2.py"))
src=open(spec.origin).read(); ns={}
exec(src.split("def metrics()")[0].replace('PORT = sys.argv[1] if len(sys.argv) > 1 else "8013"','PORT="'+PORT+'"'), ns)
res={}
for name,(p,think) in ns["PROMPTS"].items():
    body=json.dumps({"model":"Qwen3.8-27B","messages":[{"role":"user","content":p}],"max_tokens":96,"temperature":0,"chat_template_kwargs":{"enable_thinking":think}}).encode()
    req=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
    r=json.loads(urllib.request.urlopen(req, timeout=900).read())
    m=r["choices"][0]["message"]; res[name]={"content":m.get("content"),"reasoning":m.get("reasoning") or m.get("reasoning_content"),"usage":r["usage"]}
    print(name, r["usage"]["completion_tokens"], "tok")
json.dump(res, open(out,"w"), indent=1)
