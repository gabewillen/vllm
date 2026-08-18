import gzip, json, sys, collections
d=json.load(gzip.open(sys.argv[1]))
ev=[e for e in d["traceEvents"] if e.get("ph")=="X"]
k=sorted([e for e in ev if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")], key=lambda e:e["ts"])
# merge over streams: compute union busy
busy=0; gaps=[]; end=k[0]["ts"]+k[0]["dur"]; start=k[0]["ts"]
for e in k[1:]:
    if e["ts"]>end:
        gaps.append((e["ts"]-end, k[k.index(e)-1]["name"][:60], e["name"][:60])) if False else gaps.append((e["ts"]-end, e["name"][:70]))
        end=e["ts"]+e["dur"]
    else:
        end=max(end,e["ts"]+e["dur"])
tot=end-start; idle=sum(g[0] for g in gaps)
print(f"window {tot/1e3:.1f} ms, idle {idle/1e3:.1f} ms ({idle/tot*100:.1f}%), gaps {len(gaps)}")
h=collections.Counter()
for g,_ in gaps:
    b = "<20us" if g<20 else "<50us" if g<50 else "<100us" if g<100 else "<300us" if g<300 else "<1ms" if g<1000 else ">=1ms"
    h[b]+=g
for b in ["<20us","<50us","<100us","<300us","<1ms",">=1ms"]: print(f"  {b:7s} {h[b]/1e3:7.1f} ms")
# which kernels follow the big gaps
c=collections.defaultdict(lambda:[0,0.0])
for g,n in gaps:
    if g>=50: c[n][0]+=1; c[n][1]+=g
print("kernels after gaps>=50us (count, total gap ms):")
for n,v in sorted(c.items(), key=lambda x:-x[1][1])[:15]: print(f"  {v[1]/1e3:7.1f} ms n={v[0]:5d} {n}")
# CPU side: top cpu_op names by total time
cpu=[e for e in ev if e.get("cat")=="cpu_op"]
c2=collections.defaultdict(lambda:[0,0.0])
for e in cpu: c2[e["name"][:70]][0]+=1; c2[e["name"][:70]][1]+=e["dur"]
print("top cpu ops:")
for n,v in sorted(c2.items(), key=lambda x:-x[1][1])[:12]: print(f"  {v[1]/1e3:7.1f} ms n={v[0]:6d} {n}")
