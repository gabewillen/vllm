import json, sys, time, urllib.request, threading, random
PORT=sys.argv[1]; conc=int(sys.argv[2]); n_out=int(sys.argv[3]); n_in=int(sys.argv[4]) if len(sys.argv)>4 else 1024
U=f"http://localhost:{PORT}"
def post(p, body=None):
    req=urllib.request.Request(U+p, data=json.dumps(body).encode() if body is not None else b"", headers={"Content-Type":"application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=900).read()
random.seed(0)
words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
def prompt(i):
    r=random.Random(i)
    return " ".join(r.choice(words) for _ in range(n_in))
def run(i, prof):
    body={"model":"Qwen3.8-27B","prompt":prompt(i),"max_tokens":n_out,"temperature":0,"ignore_eos":True}
    post("/v1/completions", body)
# warm: run all prompts once so prefill is cached... no: random prompts differ; do a warm batch first w/o profiling
ths=[threading.Thread(target=run,args=(i,False)) for i in range(conc)]
[t.start() for t in ths]; [t.join() for t in ths]
post("/start_profile"); t=time.time()
ths=[threading.Thread(target=run,args=(i,True)) for i in range(conc)]
[t.start() for t in ths]; [t.join() for t in ths]
dt=time.time()-t; post("/stop_profile")
print(f"conc={conc} out={n_out}: {dt:.2f}s -> {conc*n_out/dt:.0f} tok/s (incl. prefill of cached prompts)")
