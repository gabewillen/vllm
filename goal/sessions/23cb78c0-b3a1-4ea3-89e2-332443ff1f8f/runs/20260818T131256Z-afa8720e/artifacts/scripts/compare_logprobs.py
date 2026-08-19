import json, sys, math
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
tot=0; sad=0.0; mx=0.0; top1_diff=0; n=0
for k in a:
    for ra, rb in zip(a[k], b[k]):
        if ra is None or rb is None: continue
        assert ra["tok"]==rb["tok"], k
        d=abs(ra["lp"]-rb["lp"]); sad+=d; mx=max(mx,d); n+=1
        top1_diff += (ra["top1"]!=rb["top1"])
print(f"positions={n} mean|dlogprob|={sad/n:.4f} max|dlogprob|={mx:.3f} top1 disagreement={top1_diff}/{n} ({100*top1_diff/n:.2f}%)")
