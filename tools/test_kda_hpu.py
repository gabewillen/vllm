import torch, importlib.util
import habana_frameworks.torch  # noqa
spec = importlib.util.spec_from_file_location("kda_hpu", "/work/vllm/models/glm5next/common/kda_hpu.py")
kh = importlib.util.module_from_spec(spec); spec.loader.exec_module(kh)
torch.manual_seed(0)
B, S, H, K = 1, 128, 8, 128
q, k, v = (torch.randn(B, S, H, K) for _ in range(3))
g = -5 * torch.sigmoid(torch.randn(B, S, H, K)); beta = torch.sigmoid(torch.randn(B, S, H))
init = torch.zeros(B, H, K, K)
o_c, s_c = kh.kda_chunk_prefill(q, k, v, g, beta, init, K ** -0.5)
d = "hpu"
o_h, s_h = kh.kda_chunk_prefill(q.to(d), k.to(d), v.to(d), g.to(d), beta.to(d), init.to(d), K ** -0.5)
print("chunk hpu vs cpu", (o_h.cpu() - o_c).abs().max().item(), (s_h.cpu() - s_c).abs().max().item(), o_c.abs().max().item())
# pieces
C = 64
a = torch.randn(2, 3, C, C).tril(-1) * 0.1
def solve(attn):
    attn = attn.clone()
    for i in range(1, C):
        row = attn[..., i, :i].clone(); sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return attn
print("tri-loop hpu vs cpu", (solve(a.to(d)).cpu() - solve(a)).abs().max().item())
x = torch.randn(2, 5, 64, 16)
print("cumsum", (x.to(d).cumsum(-2).cpu() - x.cumsum(-2)).abs().max().item())
m = torch.triu(torch.ones(C, C, dtype=torch.bool), 1)
y = torch.randn(2, 3, C, C, 16)
print("masked_fill6d", (y.to(d).masked_fill(m.to(d).view(1, 1, C, C, 1), 0).cpu() - y.masked_fill(m.view(1, 1, C, C, 1), 0)).abs().max().item())
