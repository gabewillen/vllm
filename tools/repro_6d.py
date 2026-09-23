import torch
import habana_frameworks.torch  # noqa
torch.manual_seed(0)
a = torch.randn(1, 8, 2, 64, 1, 128)   # [B,H,N,C,1,K]
b = torch.randn(1, 8, 2, 1, 64, 128)   # [B,H,N,1,C,K]
c = torch.rand(1, 8, 2, 64, 64, 128)   # [B,H,N,C,C,K]
ref = (a * b * c).sum(-1)
out6 = (a.to("hpu") * b.to("hpu") * c.to("hpu")).sum(-1).cpu()
f = lambda t: t.reshape(-1, *t.shape[3:])  # same data, 5-D
out5 = (f(a).to("hpu") * f(b).to("hpu") * f(c).to("hpu")).sum(-1).cpu().reshape(ref.shape)
two = (a.to("hpu") * b.to("hpu")).cpu(); 
print("6-D a*b*c sum  max err:", (out6 - ref).abs().max().item(), " (ref scale", ref.abs().max().item(), ")")
print("5-D a*b*c sum  max err:", (out5 - ref).abs().max().item())
print("6-D a*b only   max err:", (two - a * b).abs().max().item())
print("6-D (a*b)*c no-sum err:", ((a.to("hpu") * b.to("hpu") * c.to("hpu")).cpu() - a * b * c).abs().max().item())
