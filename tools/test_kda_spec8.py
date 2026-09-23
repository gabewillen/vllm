"""kda_decode_multi rollback at T=8 for every accept position, CPU/HPU, eager/compiled."""
import sys, importlib.util, torch
import torch.nn.functional as F
spec = importlib.util.spec_from_file_location("kda_hpu", "/work/vllm/models/glm5next/common/kda_hpu.py")
kh = importlib.util.module_from_spec(spec); sys.modules["kda_hpu"] = kh; spec.loader.exec_module(kh)
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
comp = len(sys.argv) > 2 and sys.argv[2] == "compile"
if dev == "hpu":
    import habana_frameworks.torch  # noqa
torch.manual_seed(0)
H, K, W = 2, 16, 4; D = 3 * H * K; T = 8; P = 8; scale = K ** -0.5
wt = torch.randn(W, D) * 0.5
def seq_ref(xs, gs, bs):
    conv = torch.zeros(W - 1, D); h = torch.zeros(H, K, K); outs = []
    for x, g, b in zip(xs, gs, bs):
        win = torch.cat([conv, x[None]], 0); o = F.silu((win * wt).sum(0)); conv = win[1:]
        q, k, v = o.split(D // 3); q = kh._l2n(q.view(H, K)) * scale; k = kh._l2n(k.view(H, K)); v = v.view(H, K)
        h = h * torch.exp(g).unsqueeze(1)
        vn = (v - (h @ k.unsqueeze(-1)).squeeze(-1)) * b.unsqueeze(-1)
        h = h + vn.unsqueeze(-1) * k.unsqueeze(1)
        outs.append((h @ q.unsqueeze(-1)).squeeze(-1))
    return outs
fn = kh.kda_decode_multi
if comp:
    fn = torch.compile(fn, backend="hpu_backend" if dev == "hpu" else "inductor", dynamic=False)
N = 24
B = T  # one sequence per accept value a = 1..8
xs = torch.randn(B, N, D); gs = -torch.rand(B, N, H, K) * 3; bs = torch.rand(B, N, H)
conv_state = torch.zeros((B + 2) * P, W - 1, D, device=dev); rec = torch.zeros((B + 2) * P, H, K, K, device=dev)
slots = (torch.arange(B) + 1) * P
d = lambda t: t.to(dev)
o1 = fn(d(xs[:, 0:T]), d(gs[:, 0:T]), d(bs[:, 0:T]), conv_state, rec, d(wt), d(slots), d(slots), scale, H, K)
acc = torch.arange(B)             # seq i accepts a = i + 1 -> resume from row i
starts = [i + 1 for i in range(B)]
x2 = torch.stack([xs[i, s:s + T] for i, s in enumerate(starts)])
g2 = torch.stack([gs[i, s:s + T] for i, s in enumerate(starts)])
b2 = torch.stack([bs[i, s:s + T] for i, s in enumerate(starts)])
o2 = fn(d(x2), d(g2), d(b2), conv_state, rec, d(wt), d(slots + acc), d(slots), scale, H, K)
o1, o2 = o1.cpu(), o2.cpu()
for i in range(B):
    ref = seq_ref(xs[i], gs[i], bs[i])
    e1 = max((o1[i, t] - ref[t]).abs().max().item() for t in range(T))
    e2 = max((o2[i, t] - ref[starts[i] + t]).abs().max().item() for t in range(T))
    print(f"[{dev}{' compiled' if comp else ''}] accept a={i + 1}: step1 err {e1:.1e}  step2 err {e2:.1e}")
