"""Burst stress client tolerant of SSE comment pings: N concurrent streaming
completions, reports successes, tokens, per-request TTFT/latency."""
import json, sys, time, threading, random, urllib.request, statistics
PORT=sys.argv[1]; N=int(sys.argv[2]); IN=int(sys.argv[3]); OUT=int(sys.argv[4])
U=f"http://localhost:{PORT}/v1/completions"
words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
res=[None]*N
def one(i):
    r=random.Random(1000+i); prompt=" ".join(r.choice(words) for _ in range(IN))
    body=json.dumps({"model":"Qwen3.8-27B","prompt":prompt,"max_tokens":OUT,"temperature":0,"stream":True,"ignore_eos":True,"stream_options":{"include_usage":True}}).encode()
    req=urllib.request.Request(U, data=body, headers={"Content-Type":"application/json"})
    t0=time.time(); first=None; toks=0; err=None
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for line in resp:
                s=line.decode().strip()
                if not s.startswith("data: ") or s=="data: [DONE]": continue
                c=json.loads(s[6:])
                if c.get("choices") and c["choices"][0].get("text"):
                    if first is None: first=time.time()
                if c.get("usage"): toks=c["usage"].get("completion_tokens",0)
    except Exception as e:
        err=repr(e)[:120]
    res[i]=(err, first-t0 if first else None, time.time()-t0, toks)
t=time.time(); ths=[threading.Thread(target=one,args=(i,)) for i in range(N)]
[x.start() for x in ths]; [x.join() for x in ths]; dt=time.time()-t
ok=[r for r in res if r and r[0] is None and r[3]>0]
print(f"burst N={N} in={IN} out={OUT}: ok={len(ok)}/{N} errors={[r[0] for r in res if r and r[0]][:3]} wall={dt:.1f}s agg={sum(r[3] for r in ok)/dt:.0f} tok/s ttft median={statistics.median(r[1] for r in ok):.1f}s max={max(r[1] for r in ok):.1f}s")
