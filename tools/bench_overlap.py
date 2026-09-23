import time, torch, torch.distributed as dist
import habana_frameworks.torch as ht  # noqa
import habana_frameworks.torch.distributed.hccl  # noqa
dist.init_process_group("hccl"); r = dist.get_rank(); d = "hpu"
w = torch.randn(4096, 16384, device=d, dtype=torch.bfloat16); x = torch.randn(64, 4096, device=d, dtype=torch.bfloat16)
a = torch.randn(64 * 4096, device=d, dtype=torch.bfloat16)
def gemms(n=4):
    y = x
    for _ in range(n): y = (y @ w)[:, :4096]
    return y
def t(fn, it=30):
    for _ in range(3): fn()
    torch.hpu.synchronize(); s = time.perf_counter()
    for _ in range(it): fn()
    torch.hpu.synchronize(); return (time.perf_counter() - s) / it * 1e3
g = t(lambda: gemms()); ar = t(lambda: dist.all_reduce(a))
seq = t(lambda: (gemms(), dist.all_reduce(a)))
def ov():
    h = dist.all_reduce(a, async_op=True); y = gemms(); h.wait(); return y
o = t(ov)
if r == 0: print(f"gemms {g:.3f} ms | allreduce {ar:.3f} ms | sequential {seq:.3f} ms | async-overlap {o:.3f} ms")
dist.destroy_process_group()
