import torch, torch.nn.functional as F
import habana_frameworks.torch  # noqa
torch.manual_seed(0)
B, S, H, K = 1, 128, 8, 128
q0, k0, v0 = (torch.randn(B, S, H, K) for _ in range(3))
g0 = -5 * torch.sigmoid(torch.randn(B, S, H, K)); beta0 = torch.sigmoid(torch.randn(B, S, H))
def run(dev):
    st = {}
    l2n = lambda x: x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
    q = l2n(q0.to(dev)) * K ** -0.5; k = l2n(k0.to(dev)); v = v0.to(dev); g = g0.to(dev); beta = beta0.to(dev)
    q, k, v, g = (x.transpose(1, 2) for x in (q, k, v, g)); beta = beta.transpose(1, 2)
    C = 64; N = S // C
    v_beta = v * beta.unsqueeze(-1); k_beta = k * beta.unsqueeze(-1)
    q, k, v, g, k_beta, v_beta = (x.reshape(B, H, N, C, x.shape[-1]) for x in (q, k, v, g, k_beta, v_beta))
    g = g.cumsum(dim=-2); st["gcum"] = g
    tri_incl = torch.triu(torch.ones(C, C, dtype=torch.bool, device=dev), diagonal=0)
    tri_strict = torch.triu(torch.ones(C, C, dtype=torch.bool, device=dev), diagonal=1)
    diff = g.unsqueeze(-2) - g.unsqueeze(-3); st["diff"] = diff
    diff = diff.masked_fill(tri_strict.view(1, 1, 1, C, C, 1), 0.0); st["diffm"] = diff
    decay = diff.exp(); st["decay"] = decay
    attn = -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(dim=-1); st["attn0"] = attn
    attn = attn.masked_fill(tri_incl, 0.0); st["attn1"] = attn.clone()
    for i in range(1, C):
        row = attn[..., i, :i].clone(); sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    st["attn2"] = attn.clone()
    attn = attn + torch.eye(C, dtype=attn.dtype, device=dev)
    st["u"] = attn @ v_beta; st["w"] = attn @ (k_beta * g.exp())
    return {k_: v_.cpu() for k_, v_ in st.items()}
c = run("cpu"); h = run("hpu")
for k_ in c:
    print(f"{k_:6s} maxabs diff {(c[k_] - h[k_]).abs().max().item():.3e}  scale {c[k_].abs().max().item():.3e}")
