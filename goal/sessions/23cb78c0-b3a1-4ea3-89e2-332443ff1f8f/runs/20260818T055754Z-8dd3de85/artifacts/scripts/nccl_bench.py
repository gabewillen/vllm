import os, time, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); ws=int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dist.init_process_group("nccl", rank=rank, world_size=ws)
res=[]
for n in [1*5120, 8*5120, 96*5120, 512*5120, 8192*5120]:
    x=torch.randn(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(30): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    it = 300 if n<1e6 else 40
    t=time.perf_counter()
    for _ in range(it): dist.all_reduce(x)
    torch.cuda.synchronize(); res.append(f"{n*2/1024:.0f}KB={(time.perf_counter()-t)/it*1e6:.0f}us")
if rank==0: print(os.environ.get("TAG","default"), " ".join(res), flush=True)
dist.barrier(); dist.destroy_process_group()
