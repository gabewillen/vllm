import gzip, json, sys, collections
d=json.load(gzip.open(sys.argv[1]))
ev=[e for e in d["traceEvents"] if e.get("ph")=="X" and e.get("cat")=="kernel"]
by=collections.defaultdict(lambda:[0,0.0])
for e in ev:
    k=("nccl" if "nccl" in e["name"] else "gemm" if "block_scaled_mm" in e["name"] else "other")
    by[(e.get("tid"), k)][0]+=1; by[(e.get("tid"), k)][1]+=e["dur"]
for k,v in sorted(by.items()): print(f"stream {k[0]} {k[1]:6s} n={v[0]:6d} {v[1]/1e3:9.1f} ms")
# overlap: total time where nccl kernel running AND some non-nccl kernel running
nc=sorted([(e["ts"],e["ts"]+e["dur"]) for e in ev if "nccl" in e["name"]])
ot=sorted([(e["ts"],e["ts"]+e["dur"]) for e in ev if "nccl" not in e["name"]])
def merge(iv):
    out=[]; 
    for s,e in iv:
        if out and s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
nc=merge(nc); ot=merge(ot)
i=j=0; ov=0
while i<len(nc) and j<len(ot):
    s=max(nc[i][0],ot[j][0]); e=min(nc[i][1],ot[j][1])
    if e>s: ov+=e-s
    if nc[i][1]<ot[j][1]: i+=1
    else: j+=1
tn=sum(e-s for s,e in nc); to=sum(e-s for s,e in ot)
t0=min(e["ts"] for e in ev); t1=max(e["ts"]+e["dur"] for e in ev)
print(f"window {(t1-t0)/1e3:.0f} ms; nccl busy {tn/1e3:.0f} ms; compute busy {to/1e3:.0f} ms; overlap {ov/1e3:.0f} ms")
