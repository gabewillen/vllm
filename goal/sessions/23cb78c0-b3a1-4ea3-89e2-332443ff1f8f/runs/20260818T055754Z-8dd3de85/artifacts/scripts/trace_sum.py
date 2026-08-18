import gzip, json, sys, collections
f=sys.argv[1]; d=json.load(gzip.open(f))
ev=[e for e in d["traceEvents"] if e.get("ph")=="X"]
kern=[e for e in ev if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
t0=min(e["ts"] for e in kern); t1=max(e["ts"]+e["dur"] for e in kern)
print("gpu window ms", (t1-t0)/1000, "kernels", len(kern))
by=collections.defaultdict(lambda:[0,0.0])
for e in kern:
    n=e["name"]
    key=("nccl" if "nccl" in n.lower() or "AllReduce" in n else
         "gemm_fp8_triton" if "w8a8_block_fp8" in n or "fp8_matmul" in n.lower() else
         "marlin" if "marlin" in n.lower() else
         "cutlass/gemm" if "gemm" in n.lower() or "cutlass" in n.lower() or "sm80_xmma" in n or "nvjet" in n else
         "gdn/fla" if "chunk_" in n or "fused_recurrent" in n or "gated_delta" in n or "fla" in n.lower() or "causal_conv" in n else
         "attention" if "attn" in n.lower() or "attention" in n.lower() else
         "argmax/topk/softmax" if "argmax" in n.lower() or "topk" in n.lower() or "softmax" in n.lower() or "reduce_kernel" in n.lower() else
         "memcpy/memset" if e.get("cat")!="kernel" else
         "elementwise/other:"+n[:50])
    by[key][0]+=1; by[key][1]+=e["dur"]
tot=sum(v[1] for v in by.values())
for k,v in sorted(by.items(), key=lambda x:-x[1][1])[:30]:
    print(f"{v[1]/1000:8.1f} ms {v[1]/tot*100:5.1f}%  n={v[0]:6d}  {k}")
print("sum kernel ms", tot/1000)
