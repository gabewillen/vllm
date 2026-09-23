import time, torch
import habana_frameworks.torch as ht  # noqa
d = "hpu"; H, I, K, E = 4096, 256, 8, 288
FP8_MAX = 240.0
torch.manual_seed(0)
def dq(x):
    s = (x.abs().amax(-1, keepdim=True).float() + 1e-8) / FP8_MAX
    return torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / s, False, False, torch.float8_e4m3fn)[0], s
L = 4
W13 = [torch.randn(E * 2 * I, H, device=d).to(torch.float8_e4m3fn) for _ in range(L)]   # [E*2I, H]
S13 = [torch.rand(E * 2 * I, device=d) * 0.01 for _ in range(L)]
W2 = [torch.randn(H, E * I, device=d).to(torch.float8_e4m3fn) for _ in range(L)]        # [H, E*I]
S2 = [torch.rand(H, device=d) * 0.01 for _ in range(L)]
def dense(x, rw, l):
    xq, xs = dq(x)
    gu = torch.ops.hpu.fp8_gemm_v2(xq, False, W13[l], True, None, torch.bfloat16, xs, S13[l], None, False)
    gu = gu.view(-1, E, 2, I)
    h = torch.nn.functional.silu(gu[:, :, 0].clamp(max=10)) * gu[:, :, 1].clamp(-10, 10)
    h = (h * rw.unsqueeze(-1)).reshape(-1, E * I)
    hq, hs = dq(h)
    return torch.ops.hpu.fp8_gemm_v2(hq, False, W2[l], True, None, torch.bfloat16, hs, S2[l], None, False)
W13e = [w.view(E, 2 * I, H) for w in W13]; S13e = [s.view(E, 2 * I) for s in S13]
W2e = [w.view(H, E, I).permute(1, 0, 2).contiguous() for w in W2]  # [E, H, I]
def gather(x, ids, wts, l):
    T = x.shape[0]; f = ids.reshape(-1)
    w13 = W13e[l].index_select(0, f).to(torch.bfloat16) * S13e[l].index_select(0, f).unsqueeze(-1).to(torch.bfloat16)  # [TK,2I,H]
    w2 = W2e[l].index_select(0, f).to(torch.bfloat16)  # [TK,H,I]
    xx = x.unsqueeze(1).expand(T, K, H).reshape(T * K, 1, H)
    gu = torch.bmm(xx, w13.transpose(1, 2)).view(T * K, 2, I)
    h = torch.nn.functional.silu(gu[:, 0].clamp(max=10)) * gu[:, 1].clamp(-10, 10)
    o = torch.bmm(h.unsqueeze(1), w2.transpose(1, 2)).view(T, K, H) * S2[l]
    return (o * wts.unsqueeze(-1)).sum(1).to(x.dtype)
def run(fn, T, name):
    x = torch.randn(T, H, device=d, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (T, K), device=d); wts = torch.rand(T, K, device=d)
    rw = torch.zeros(T, E, device=d, dtype=torch.bfloat16).scatter_(1, ids, wts.to(torch.bfloat16))
    if name == "dense": f = lambda x: [x := x + dense(x, rw, l) for l in range(L)][-1]
    else: f = lambda x: [x := x + gather(x, ids, wts, l) for l in range(L)][-1]
    cf = torch.compile(f, backend="hpu_backend", dynamic=False)
    for _ in range(3): cf(x)
    torch.hpu.synchronize(); n = 30; t = time.perf_counter()
    for _ in range(n): y = cf(x)
    th = (time.perf_counter() - t) / n; torch.hpu.synchronize(); tw = (time.perf_counter() - t) / n
    by = (E * 3 * H * I) if name == "dense" else (T * K * 3 * H * I)
    print(f"{name:6s} T={T:3d}: {tw/L*1e3:.3f} ms/layer (host {th/L*1e3:.3f})  weights {by/1e6:.0f}MB/layer -> {by/(tw/L)/1e12:.2f} TB/s", flush=True)
for T in (128, 256, 512, 1024):
    run(dense, T, "dense")
for T in ():
    run(gather, T, "gather")
