import time, torch, torch.distributed as dist
import torch.distributed._functional_collectives as fc
import habana_frameworks.torch as ht  # noqa
import habana_frameworks.torch.distributed.hccl  # noqa
dist.init_process_group("hccl"); r = dist.get_rank(); d = "hpu"; G = dist.group.WORLD
L = 8
ws = [torch.randn(4096, 4096, device=d, dtype=torch.bfloat16) * 0.01 for _ in range(L)]
def layer_seq(x):  # per layer: gemm -> allreduce (serial dependency)
    for w in ws:
        x = fc.all_reduce(x @ w, "sum", G)
    return x
def layer_mb(x):   # two micro-batches, interleaved so each AR can hide behind the other's GEMM
    a, b = x.chunk(2)
    for w in ws:
        a = fc.all_reduce(a @ w, "sum", G)
        b = fc.all_reduce(b @ w, "sum", G)
    return torch.cat([a, b])
def gem_only(x):
    for w in ws: x = x @ w
    return x
def t(fn, x, it=30):
    cf = torch.compile(fn, backend="hpu_backend", dynamic=False)
    for _ in range(3): cf(x)
    torch.hpu.synchronize(); s = time.perf_counter()
    for _ in range(it): cf(x)
    torch.hpu.synchronize(); return (time.perf_counter() - s) / it * 1e3
for B in (64, 256):
    x = torch.randn(B, 4096, device=d, dtype=torch.bfloat16)
    g, s_, m = t(gem_only, x), t(layer_seq, x), t(layer_mb, x)
    if r == 0: print(f"B={B}: {L} layers  gemm-only {g:.2f} ms | gemm+AR serial {s_:.2f} ms | 2-microbatch {m:.2f} ms  (AR cost/layer serial {(s_-g)/L*1e3:.0f} us, mb {(m-g)/L*1e3:.0f} us)", flush=True)
dist.destroy_process_group()
