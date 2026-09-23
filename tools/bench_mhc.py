import sys, time, torch
import habana_frameworks.torch  # noqa
sys.path.insert(0, "/work/tools")
import importlib.util
src = open("/work/vllm/models/glm5next/common/model.py").read()
start = src.index("def _mhc_mix("); end = src.index("class Glm5NextMLP(")
ns = {"torch": torch}; exec(src[start:end], ns)
d = "hpu"; T, n, D = 1, 4, 4096
res = torch.randn(T, n, D, device=d, dtype=torch.bfloat16)
fn = torch.randn(24, n * D, device=d) * 0.01; sc = torch.ones(3, device=d); base = torch.zeros(24, device=d)
nw = torch.ones(D, device=d, dtype=torch.bfloat16)
def f(r):
    for _ in range(10):
        post, comb, x = ns["_mhc_pre_hpu"](r, fn, sc, base, nw, 1e-5, n, 1e-6, 1e-6, 1e-6, 2.0, 20)
        r = r + x.unsqueeze(1) * 0.001
    return r
def f_no_sk(r):
    for _ in range(10):
        post, comb, x = ns["_mhc_pre_hpu"](r, fn, sc, base, nw, 1e-5, n, 1e-6, 1e-6, 1e-6, 2.0, 1)
        r = r + x.unsqueeze(1) * 0.001
    return r
for name, g in (("sinkhorn20", f), ("sinkhorn1", f_no_sk)):
    cf = torch.compile(g, backend="hpu_backend", dynamic=False)
    for _ in range(3): cf(res)
    torch.hpu.synchronize(); t = time.perf_counter()
    for _ in range(20): cf(res)
    torch.hpu.synchronize(); dt = (time.perf_counter() - t) / 20 / 10
    print(f"{name}: {dt*1e6:.1f} us per mHC pre call (T=1)")
