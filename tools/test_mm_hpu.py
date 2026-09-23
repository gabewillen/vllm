import torch
import habana_frameworks.torch  # noqa
torch.manual_seed(0)
a = torch.randn(2, 3, 64, 64); b = torch.randn(2, 3, 64, 128)
r = a @ b
print("fp32 bmm rel err", ((a.to("hpu") @ b.to("hpu")).cpu() - r).norm().item() / r.norm().item())
print("matmul precision", torch.get_float32_matmul_precision())
torch.set_float32_matmul_precision("highest")
print("fp32 bmm rel err (highest)", ((a.to("hpu") @ b.to("hpu")).cpu() - r).norm().item() / r.norm().item())
a5 = torch.randn(1, 8, 2, 64, 64); b5 = torch.randn(1, 8, 2, 64, 128)
print("5d matmul rel err", ((a5.to("hpu") @ b5.to("hpu")).cpu() - a5 @ b5).norm().item() / (a5 @ b5).norm().item())
x = torch.randn(1, 8, 2, 64, 64, 128)
print("6d sum(-1)", (x.to("hpu").sum(-1).cpu() - x.sum(-1)).abs().max().item())
y = torch.randn(1, 8, 2, 64, 1, 128) * torch.randn(1, 8, 2, 1, 64, 128)
print("bcast mul-sum", ((torch.randn(0) if False else None) is None))
p = torch.randn(1, 8, 2, 64, 128); q = torch.randn(1, 8, 2, 64, 128)
rr = (p.unsqueeze(-2) * q.unsqueeze(-3)).sum(-1)
print("bcast outer mul-sum", ((p.to("hpu").unsqueeze(-2) * q.to("hpu").unsqueeze(-3)).sum(-1).cpu() - rr).abs().max().item(), rr.abs().max().item())
