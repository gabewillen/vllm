import time, torch, importlib.util, sys
import habana_frameworks.torch  # noqa
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
new = load("/work/vllm/models/glm5next/common/kda_hpu.py", "kda_new")
old = load("/work/tools/kda_old.py", "kda_old")

d = "hpu"; H, K = 8, 128
for B in (1, 64, 128):
    q, k, v, g = (torch.randn(B, H, K, device=d) for _ in range(4)); g = -torch.sigmoid(g); beta = torch.rand(B, H, device=d)
    st = torch.randn(B + 10, H, K, K, device=d); idx = torch.arange(1, B + 1, device=d)
    for name, mod in (("old", old), ("new", new)):
        cf = torch.compile(lambda *a, m=mod: m.kda_decode_step(*a), backend="hpu_backend", dynamic=False)
        def run():
            o = None
            for _ in range(10): o = cf(q, k, v, g, beta, st, idx, K ** -0.5)
            return o
        for _ in range(3): run()
        torch.hpu.synchronize(); t = time.perf_counter()
        for _ in range(10): run()
        torch.hpu.synchronize(); dt = (time.perf_counter() - t) / 100
        print(f"B={B:4d} {name}: {dt*1e6:.0f} us per layer-step", flush=True)
