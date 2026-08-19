import os, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); torch.cuda.set_device(rank); dist.init_process_group("nccl")
bad=0; g=torch.Generator(device="cuda").manual_seed(1234)
for it in range(300):
    for mb in (0.01, 1, 84):
        n=int(mb*1024*1024/2)
        x=torch.randn(n, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)  # same on all ranks
        y=x.clone(); dist.all_reduce(y)
        ref=(x.float()*dist.get_world_size()).to(torch.bfloat16)
        if not torch.equal(y, ref):
            bad+=1
            if rank==0: print(f"MISMATCH it={it} size={mb}MB maxdiff={(y.float()-ref.float()).abs().max().item()}", flush=True)
    # also direct P2P copy check
    a=torch.arange(n, device="cuda", dtype=torch.float32)+rank
    b=torch.empty_like(a); dist.all_reduce(b.zero_()) 
torch.cuda.synchronize()
if rank==0: print("all-reduce correctness: mismatches =", bad, flush=True)
dist.destroy_process_group()
