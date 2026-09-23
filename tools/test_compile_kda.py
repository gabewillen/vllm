import torch, importlib.util
import habana_frameworks.torch  # noqa
spec = importlib.util.spec_from_file_location("kda_hpu", "/work/vllm/models/glm5next/common/kda_hpu.py")
kh = importlib.util.module_from_spec(spec); spec.loader.exec_module(kh)
torch.manual_seed(0); d = "hpu"
B, S, D, W = 1, 128, 3072, 4
x = torch.randn(B, S, D, dtype=torch.bfloat16); x[:, 5:] = 0
lens = torch.tensor([5]); w = torch.randn(W, D) * 0.3
def mk_state():
    st = torch.full((18, W - 1, D), 1e4, dtype=torch.bfloat16); return st
li = torch.tensor([1]); has = torch.tensor([0], dtype=torch.int32)
st_c = mk_state(); ref = kh.conv_prefill(x, lens, st_c, w, li, li, has)
def run(fn):
    st = mk_state().to(d)
    out = fn(x.to(d), lens.to(d), st, w.to(d), li.to(d), li.to(d), has.to(d))
    return (out.cpu() - ref).abs().max().item(), (st.cpu() - st_c).abs().max().item()
print("eager hpu conv  (out, state) err:", run(kh.conv_prefill))
print("compiled conv   (out, state) err:", run(torch.compile(kh.conv_prefill, backend="hpu_backend", dynamic=False)))
# decode step
H, K = 8, 128
q, k, v, g = (torch.randn(4, H, K) for _ in range(4)); g = -torch.sigmoid(g); beta = torch.rand(4, H)
rs = torch.randn(10, H, K, K); idx = torch.tensor([1, 3, 5, 7])
rs_c = rs.clone(); o_ref = kh.kda_decode_step(q, k, v, g, beta, rs_c, idx, K ** -0.5)
for name, fn in (("eager", kh.kda_decode_step), ("compiled", torch.compile(kh.kda_decode_step, backend="hpu_backend", dynamic=False))):
    rs_h = rs.to(d); o = fn(q.to(d), k.to(d), v.to(d), g.to(d), beta.to(d), rs_h, idx.to(d), K ** -0.5)
    print(name, "decode out err", (o.cpu() - o_ref).abs().max().item(), "state err", (rs_h.cpu() - rs_c).abs().max().item())
