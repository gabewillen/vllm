import time, torch, importlib.util
import habana_frameworks.torch  # noqa
spec = importlib.util.spec_from_file_location("kda_hpu", "/work/vllm/models/glm5next/common/kda_hpu.py")
import sys; kh = importlib.util.module_from_spec(spec); sys.modules["kda_hpu"] = kh; spec.loader.exec_module(kh)
d = "hpu"; H, K = 8, 128
for S in (128, 1024):
    q, k, v = (torch.randn(1, S, H, K, device=d) for _ in range(3))
    g = -5 * torch.sigmoid(torch.randn(1, S, H, K, device=d)); beta = torch.sigmoid(torch.randn(1, S, H, device=d))
    init = torch.zeros(1, H, K, K, device=d)
    cf = torch.compile(lambda *a: kh.kda_chunk_prefill(*a, K ** -0.5), backend="hpu_backend", dynamic=False)
    for _ in range(3): cf(q, k, v, g, beta, init)
    torch.hpu.synchronize(); t = time.perf_counter()
    for _ in range(10): o = cf(q, k, v, g, beta, init)
    torch.hpu.synchronize(); dt = (time.perf_counter() - t) / 10
    print(f"S={S}: kda_chunk_prefill {dt*1e3:.2f} ms per layer -> x34 layers = {dt*34*1e3:.0f} ms", flush=True)
