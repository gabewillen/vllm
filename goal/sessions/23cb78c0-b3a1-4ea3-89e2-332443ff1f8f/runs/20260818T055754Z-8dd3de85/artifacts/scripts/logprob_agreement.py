"""Distribution-agreement probe: per-token prompt logprobs (and top-1 token) of fixed
texts under one server -> JSON. Compare two dumps with compare_logprobs.py."""
import json, sys, urllib.request, importlib.util, os
PORT=sys.argv[1]; out=sys.argv[2]; texts=json.load(open(sys.argv[3]))  # {name: text}
res={}
for name, text in texts.items():
    body=json.dumps({"model":"Qwen3.8-27B","prompt":text,"max_tokens":1,"temperature":0,"prompt_logprobs":1,"logprobs":1}).encode()
    req=urllib.request.Request(f"http://localhost:{PORT}/v1/completions", data=body, headers={"Content-Type":"application/json"})
    r=json.loads(urllib.request.urlopen(req, timeout=900).read())
    pl=r["choices"][0]["prompt_logprobs"]  # list of {token_id: {logprob, rank, decoded_token}} or None
    rows=[]
    for pos in pl:
        if not pos: rows.append(None); continue
        # actual token = the entry with rank possibly >1; top-1 = entry with rank 1
        items=[(int(tid), d) for tid,d in pos.items()]
        actual=None; top1=None
        for tid,d in items:
            if d.get("rank")==1: top1=tid
        # the actual prompt token is included; identify it as the one whose logprob key is present with rank>=1 - vLLM includes the actual token always; if two entries, the non-rank-1 one is the actual
        if len(items)==1: actual=items[0][0]; lp=items[0][1]["logprob"]
        else:
            others=[(tid,d) for tid,d in items if d.get("rank")!=1]
            actual, d = others[0] if others else items[0]; lp=d["logprob"]
        rows.append({"tok":actual,"lp":lp,"top1":top1})
    res[name]=rows
    print(name, len(rows), "positions")
json.dump(res, open(out,"w"))
