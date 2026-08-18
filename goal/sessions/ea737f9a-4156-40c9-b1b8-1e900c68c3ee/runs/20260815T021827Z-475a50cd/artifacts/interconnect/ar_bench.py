import os, sys, time, torch, torch.distributed as dist
# usage: torchrun --nproc_per_node=N ar_bench.py <gpu_list e.g. 0,1,2,3>
gpus=[int(x) for x in sys.argv[1].split(",")]
rank=int(os.environ["RANK"]); torch.cuda.set_device(gpus[rank])
dist.init_process_group("nccl")
for mb in (0.01, 1, 84):   # 10 KB decode-like, 1 MB, 84 MB (8192 tok x 5120 x bf16)
    n=int(mb*1024*1024/2); x=torch.ones(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(5): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    iters=50 if mb>=1 else 500
    t=time.perf_counter()
    for _ in range(iters): dist.all_reduce(x)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/iters
    if rank==0:
        k=len(gpus); busbw = 2*(k-1)/k * mb*1024*1024/dt/1e9
        print(f"gpus={gpus} size={mb:>6} MB  latency={dt*1e6:8.1f} us  busBW={busbw:6.1f} GB/s", flush=True)
dist.destroy_process_group()
