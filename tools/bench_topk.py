import time, torch
import habana_frameworks.torch  # noqa
d = "hpu"; E, K = 288, 8
def topk_std(s): return torch.topk(s, K, dim=-1).indices
def topk_unsorted(s): return torch.topk(s, K, dim=-1, sorted=False).indices
def topk_iter(s):
    ids = []
    for _ in range(K):
        i = s.argmax(dim=-1, keepdim=True); ids.append(i)
        s = s.scatter(1, i, float("-inf"))
    return torch.cat(ids, 1)
for T in (1, 64, 128, 512):
    s = torch.randn(T, E, device=d)
    ref = torch.sort(topk_std(s).cpu(), 1).values
    for name, fn in (("topk", topk_std), ("topk_unsorted", topk_unsorted), ("iter_argmax", topk_iter)):
        cf = torch.compile(lambda x: [x := x + 0 * fn(x)[:, :1].float() for _ in range(20)][-1] if False else fn(x), backend="hpu_backend", dynamic=False)
        def chain(x, cf=cf):
            acc = 0
            for _ in range(20): acc = acc + cf(x + 1e-3)[:, 0].sum()
            return acc
        out = torch.sort(cf(s).cpu(), 1).values
        for _ in range(3): chain(s)
        torch.hpu.synchronize(); t = time.perf_counter()
        for _ in range(10): chain(s)
        torch.hpu.synchronize(); dt = (time.perf_counter() - t) / 200
        print(f"T={T:4d} {name:14s} {dt*1e6:7.1f} us  exact={bool((out == ref).all())}", flush=True)
