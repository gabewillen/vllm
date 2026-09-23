import time, torch
import habana_frameworks.torch as ht  # noqa
d = "hpu"; H, I, K = 4096, 256, 8
torch.manual_seed(0)
def mk(E):
    w12 = [torch.randn(2 * I, H, device=d).to(torch.float8_e4m3fn) for _ in range(E)]
    w3 = [torch.randn(H, I, device=d).to(torch.float8_e4m3fn) for _ in range(E)]
    s12 = [torch.rand(2 * I, device=d) * 0.01 for _ in range(E)]
    s3 = [torch.rand(H, device=d) * 0.01 for _ in range(E)]
    return w12, w3, s12, s3
def moe(x, ids, wts, w12, w3, s12, s3, E):
    scale = x.abs().amax().float() / 240.0
    xq = (x / scale).to(torch.float8_e4m3fn)
    return torch.ops.hpu.mixture_of_experts.fp8_fused_weights_dynamic(
        xq, ids, wts, w12, w3, scale, s12, s3, True, "silu", 0, E - 1)
for E in (288, 36, 8):
    L = 4  # chain 4 layers in one graph
    ws = [mk(E) for _ in range(L)]
    def f(x, ids, wts):
        for w in ws: x = x + moe(x, ids, wts, *w, E)
        return x
    cf = torch.compile(f, backend="hpu_backend", dynamic=False)
    for T in (1, 32):
        x = torch.randn(T, H, device=d, dtype=torch.bfloat16)
        ids = torch.randint(0, E, (T, K), device=d, dtype=torch.int64)
        wts = torch.rand(T, K, device=d, dtype=torch.bfloat16)
        for _ in range(3): cf(x, ids, wts)
        torch.hpu.synchronize(); n = 50; t = time.perf_counter()
        for _ in range(n): y = cf(x, ids, wts)
        t_host = (time.perf_counter() - t) / n
        torch.hpu.synchronize(); t_all = (time.perf_counter() - t) / n
        print(f"E={E:3d} T={T:3d}: {L} MoE layers/graph  host-issue {t_host*1e3:6.2f} ms  wall {t_all*1e3:6.2f} ms  -> {t_all/L*1e3:.3f} ms/layer", flush=True)
