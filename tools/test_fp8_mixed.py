"""Does fp8_gemm_v2 take a bf16 A with an fp8 B? Accuracy + speed vs per-row-FP8 A."""
import torch, time
import habana_frameworks.torch  # noqa
torch.manual_seed(0)
T, H, N = 64, 4096, 288 * 2 * 192 // 4   # a slice of the w13 GEMM
x = torch.randn(T, H, dtype=torch.bfloat16, device="hpu")
x[:, :4] *= 300.0                         # massive-activation outlier channels
w = (torch.randn(N, H, device="hpu") * 0.02)
ws = (w.abs().amax(1) / 448.0).float().view(1, -1)
wq = (w / ws.view(-1, 1)).to(torch.float8_e4m3fn)
ref = (x.float() @ (wq.float() * ws.view(-1, 1)).t())
def rowq(a):
    s = (a.abs().amax(1).float() / 448.0).clamp_min(1e-12)
    return (a.float() / s.unsqueeze(1)).to(torch.float8_e4m3fn), s.view(-1, 1)
xq, xs = rowq(x)
out_q = torch.ops.hpu.fp8_gemm_v2(xq, False, wq, True, None, torch.bfloat16, xs, ws, None, False)
print("per-row fp8 A: rel err", ((out_q.float() - ref).norm() / ref.norm()).item())
try:
    one = torch.ones((), device="hpu")
    out_m = torch.ops.hpu.fp8_gemm_v2(x, False, wq, True, None, torch.bfloat16, None, ws, None, False)
    print("bf16 A mixed : rel err", ((out_m.float() - ref).norm() / ref.norm()).item())
    for name, f in (("fp8A", lambda: torch.ops.hpu.fp8_gemm_v2(xq, False, wq, True, None, torch.bfloat16, xs, ws, None, False)),
                    ("bf16A", lambda: torch.ops.hpu.fp8_gemm_v2(x, False, wq, True, None, torch.bfloat16, None, ws, None, False))):
        for _ in range(3): f()
        torch.hpu.synchronize(); t = time.perf_counter()
        for _ in range(20): f()
        torch.hpu.synchronize(); print(name, f"{(time.perf_counter() - t) / 20 * 1e6:.0f} us")
except Exception as e:
    print("mixed failed:", str(e)[:200])
