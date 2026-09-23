import os, time, torch, torch.distributed as dist
import habana_frameworks.torch as ht  # noqa
import habana_frameworks.torch.distributed.hccl  # noqa
dist.init_process_group("hccl")
r = dist.get_rank(); d = torch.device("hpu")
for n in (4096, 4096 * 4, 4096 * 64, 1 << 22, 1 << 24):
    x = torch.ones(n, dtype=torch.bfloat16, device=d)
    for _ in range(5): dist.all_reduce(x)
    torch.hpu.synchronize()
    it = 50; t = time.perf_counter()
    for _ in range(it): dist.all_reduce(x)
    torch.hpu.synchronize(); dt = (time.perf_counter() - t) / it
    if r == 0: print(f"allreduce bf16 n={n:9d} ({n*2/1e6:7.3f} MB): {dt*1e6:8.1f} us  busbw={2*7/8*n*2/dt/1e9:6.1f} GB/s", flush=True)
# HBM bandwidth: big copy / read
a = torch.empty(1 << 30, dtype=torch.bfloat16, device=d); b = torch.empty_like(a)
for _ in range(3): b.copy_(a)
torch.hpu.synchronize(); t = time.perf_counter()
for _ in range(10): b.copy_(a)
torch.hpu.synchronize(); dt = (time.perf_counter() - t) / 10
if r == 0: print(f"HBM copy 2GiB: {dt*1e3:.2f} ms -> {2*a.numel()*2/dt/1e12:.2f} TB/s (read+write)")
w = torch.randn(8192, 8192, dtype=torch.bfloat16, device=d); v = torch.randn(1, 8192, dtype=torch.bfloat16, device=d)
for _ in range(3): (v @ w)
torch.hpu.synchronize(); t = time.perf_counter()
for _ in range(50): y = v @ w
torch.hpu.synchronize(); dt = (time.perf_counter() - t) / 50
if r == 0: print(f"GEMV 8192x8192 bf16: {dt*1e6:.1f} us -> {w.numel()*2/dt/1e12:.2f} TB/s eager")
dist.destroy_process_group()
