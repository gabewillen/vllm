import os, time, torch, torch.distributed as dist
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm import platforms
def main():
    rank=int(os.environ["RANK"]); ws=int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    g=dist.new_group(ranks=list(range(ws)), backend="gloo")
    platforms.current_platform.is_fully_connected = lambda ids: True
    ca=CustomAllreduce(g, torch.device(f"cuda:{rank}"), max_size=8192*1024)
    if rank==0: print("custom AR disabled?", ca.disabled, flush=True)
    for n in [8*5120, 96*5120, 512*5120, 2048*5120]:
        x=torch.randn(n, dtype=torch.bfloat16, device="cuda")
        # NCCL
        for _ in range(20): dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        t=time.perf_counter()
        for _ in range(200): dist.all_reduce(x)
        torch.cuda.synchronize(); nccl=(time.perf_counter()-t)/200*1e6
        if not ca.disabled:
            ref=x.clone(); dist.all_reduce(ref)
            y=ca.custom_all_reduce(x.clone())
            ok = torch.allclose(y, ref, atol=1e-1, rtol=1e-2) if y is not None else None
            for _ in range(20): ca.custom_all_reduce(x)
            torch.cuda.synchronize(); dist.barrier()
            t=time.perf_counter()
            for _ in range(200): ca.custom_all_reduce(x)
            torch.cuda.synchronize(); cus=(time.perf_counter()-t)/200*1e6
        else: cus=None; ok=None
        if rank==0: print(f"bytes={n*2/1024:.0f}KB nccl={nccl:.1f}us custom={cus}us ok={ok}", flush=True)
    dist.barrier()
main()
