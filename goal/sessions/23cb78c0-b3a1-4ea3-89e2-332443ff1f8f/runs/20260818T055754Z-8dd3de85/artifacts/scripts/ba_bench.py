import torch, time
torch.manual_seed(0)
dev="cuda:0"
def bench(fn, iters=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/iters*1e6
for M in [1,8,96]:
    x=torch.randn(M,5120,dtype=torch.bfloat16,device=dev); W=torch.randn(24,5120,dtype=torch.bfloat16,device=dev)
    Wt=W.t().contiguous(); Wf=W.float(); xf=x.float()
    W32=W.to(torch.float32)
    print(f"M={M}: linear={bench(lambda: torch.nn.functional.linear(x,W)):.1f}us  mm_xWt={bench(lambda: torch.mm(x,Wt)):.1f}us  matmul_f32={bench(lambda: torch.mm(xf,Wf.t())):.1f}us  einsum={bench(lambda: torch.einsum('mk,nk->mn',x,W)):.1f}us  mul_sum={bench(lambda: (x.unsqueeze(1)*W.unsqueeze(0)).sum(-1)):.1f}us")
    # split-K via view: (M, 40, 128) x (24, 40, 128) -> bmm over 40 chunks
    xs=x.view(M,40,128).transpose(0,1).contiguous(); Ws=W.view(24,40,128).transpose(0,1).contiguous()
    print(f"      bmm_splitk={bench(lambda: torch.bmm(xs, Ws.transpose(1,2)).sum(0)):.1f}us")
