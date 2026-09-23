import torch, importlib.util, sys
import torch.nn.functional as F
spec = importlib.util.spec_from_file_location("kda_hpu", "/work/vllm/models/glm5next/common/kda_hpu.py")
kh = importlib.util.module_from_spec(spec); sys.modules["kda_hpu"] = kh; spec.loader.exec_module(kh)
torch.manual_seed(0)
H, K, W = 2, 16, 4; D = 3 * H * K; P = 4; B = 2; scale = K ** -0.5
wt = torch.randn(W, D) * 0.5
def seq_ref(xs, gs, bs):  # sequential reference over a token list for 1 sequence
    conv = torch.zeros(W - 1, D); h = torch.zeros(H, K, K); outs = []
    for x, g, b in zip(xs, gs, bs):
        win = torch.cat([conv, x[None]], 0); o = F.silu((win * wt).sum(0)); conv = win[1:]
        q, k, v = o.split(D // 3); q = kh._l2n(q.view(H, K)) * scale; k = kh._l2n(k.view(H, K)); v = v.view(H, K)
        h = h * torch.exp(g).unsqueeze(1)
        vn = (v - (h @ k.unsqueeze(-1)).squeeze(-1)) * b.unsqueeze(-1)
        h = h + vn.unsqueeze(-1) * k.unsqueeze(1)
        outs.append((h @ q.unsqueeze(-1)).squeeze(-1))
    return outs
N = 10
xs = torch.randn(B, N, D); gs = -torch.rand(B, N, H, K) * 3; bs = torch.rand(B, N, H)
conv_state = torch.zeros(8 * P, W - 1, D); rec = torch.zeros(8 * P, H, K, K)
slots = torch.tensor([1, 3]) * P
# step 1: verify tokens 0..2 (T=3) from fresh state
o1 = kh.kda_decode_multi(xs[:, 0:3], gs[:, 0:3], bs[:, 0:3], conv_state, rec, wt, slots, slots, scale, H, K)
# seq 0 accepts a=2 (tokens 0,1), seq 1 accepts a=3 -> next inputs start at token 2 / 3
acc = torch.tensor([1, 2])
starts = [2, 3]
x2 = torch.stack([xs[0, 2:5], xs[1, 3:6]]); g2 = torch.stack([gs[0, 2:5], gs[1, 3:6]]); b2 = torch.stack([bs[0, 2:5], bs[1, 3:6]])
o2 = kh.kda_decode_multi(x2, g2, b2, conv_state, rec, wt, slots + acc, slots, scale, H, K)
for i in range(B):
    ref = seq_ref(xs[i], gs[i], bs[i])
    e1 = max((o1[i, t] - ref[t]).abs().max().item() for t in range(3))
    e2 = max((o2[i, t] - ref[starts[i] + t]).abs().max().item() for t in range(3))
    print(f"seq{i}: step1 err {e1:.2e}  step2 (after rollback) err {e2:.2e}")
