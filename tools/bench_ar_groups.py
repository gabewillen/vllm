import time, torch, torch.distributed as dist
import habana_frameworks.torch as ht  # noqa
import habana_frameworks.torch.distributed.hccl  # noqa
dist.init_process_group("hccl"); r = dist.get_rank(); d = "hpu"
groups = {8: dist.group.WORLD}
for gs in (2, 4):
    for i in range(0, 8, gs):
        g = dist.new_group(list(range(i, i + gs)))
        if i <= r < i + gs: groups[gs] = g
x = torch.randn(64 * 4096, device=d, dtype=torch.bfloat16)
for gs, g in sorted(groups.items()):
    for n in (4096, 64 * 4096):
        y = x[:n].clone()
        for _ in range(5): dist.all_reduce(y, group=g)
        torch.hpu.synchronize(); t = time.perf_counter()
        hs = [dist.all_reduce(y, group=g, async_op=True) for _ in range(64)]; [h.wait() for h in hs]
        torch.hpu.synchronize(); dt = (time.perf_counter() - t) / 64
        if r == 0: print(f"group={gs} n={n:7d} ({n*2//1024} KB): {dt*1e6:.0f} us per AR (pipelined)", flush=True)
dist.destroy_process_group()
