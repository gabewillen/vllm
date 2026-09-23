import torch
import habana_frameworks.torch  # noqa
torch.manual_seed(0)
B, S, H, K, C = 1, 128, 8, 128, 64; N = S // C
def prep(dev):
    k0 = torch.randn(B, S, H, K); g0 = -torch.rand(B, S, H, K)
    k0, g0 = k0.to(dev), g0.to(dev)
    k = k0.transpose(1, 2).reshape(B, H, N, C, K)          # strided-origin view
    g = g0.transpose(1, 2).reshape(B, H, N, C, K).cumsum(-2)
    return k, g
res = {}
for dev in ("cpu", "hpu"):
    torch.manual_seed(0); k, g = prep(dev)
    decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()          # 6-D
    res[dev, "6d_ksum"] = (k.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1).cpu()
    kc, dc = k.contiguous(), decay.contiguous()
    res[dev, "6d_contig"] = (kc.unsqueeze(-2) * kc.unsqueeze(-3) * dc).sum(-1).cpu()
    res[dev, "decay"] = decay.cpu()
for key in ("decay", "6d_ksum", "6d_contig"):
    r, h = res["cpu", key], res["hpu", key]
    print(f"{key:10s} max err {(r - h).abs().max().item():.3e}  scale {r.abs().max().item():.3e}")
