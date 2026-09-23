import time, torch, torch.distributed as dist
import torch.distributed._functional_collectives as fc
import habana_frameworks.torch as ht  # noqa
import habana_frameworks.torch.distributed.hccl  # noqa
dist.init_process_group("hccl"); r = dist.get_rank(); d = "hpu"; G = dist.group.WORLD
x = torch.randn(64, 4096, device=d, dtype=torch.bfloat16)
bufs = [x.clone() for _ in range(64)]
for _ in range(3): [dist.all_reduce(b) for b in bufs]
torch.hpu.synchronize(); t = time.perf_counter()
hs = [dist.all_reduce(b, async_op=True) for b in bufs]; [h.wait() for h in hs]
torch.hpu.synchronize(); dt = (time.perf_counter() - t) / 64
if r == 0: print(f"64 independent async ARs: {dt*1e6:.0f} us each (pipelined)")
torch.hpu.synchronize(); t = time.perf_counter()
y = x
for _ in range(64): dist.all_reduce(y)
torch.hpu.synchronize(); dt = (time.perf_counter() - t) / 64
if r == 0: print(f"64 dependent eager ARs: {dt*1e6:.0f} us each")
def chain(x):
    for _ in range(16): x = fc.all_reduce(x * 1.0001, "sum", G)
    return x
cf = torch.compile(chain, backend="hpu_backend", dynamic=False)
for _ in range(3): cf(x)
torch.hpu.synchronize(); t = time.perf_counter()
for _ in range(10): cf(x)
torch.hpu.synchronize(); dt = (time.perf_counter() - t) / 160
if r == 0: print(f"compiled chain of 16 dependent (mul+AR): {dt*1e6:.0f} us per AR")
dist.destroy_process_group()
