import sys, json, torch, importlib.util
import habana_frameworks.torch  # noqa
from safetensors import safe_open
spec = importlib.util.spec_from_file_location("dflash2_model", "/work/vllm_gaudi/v1/spec_decode/dflash2_model.py")
mod = importlib.util.module_from_spec(spec); sys.modules["dflash2_model"] = mod; spec.loader.exec_module(mod)
torch.set_grad_enabled(False)
dd = "/mnt/glm-models/GLM-5.3-Flash-DFlash2-E"; dev = "hpu"; dt = torch.bfloat16
cfg = mod.DFlash2Config.load(dd); L = 128
m = mod.DFlash2Drafter(cfg, mod.Comm(), max_len=L, n_slots=3, device=dev, dtype=dt).load(dd)
def sync(tag):
    torch.hpu.synchronize(); print("ok", tag, flush=True)
aux = torch.randn(20, 9 * 4096, dtype=dt, device=dev); pos = torch.arange(20, device=dev)
m.write_context(aux, pos, 2 * L + pos); sync("write_context")
emb = torch.randn(1, 4096, dtype=dt, device=dev)
# query step by step
B, T, D = 1, cfg.block, cfg.head_dim
h = torch.cat([emb.unsqueeze(1), m.mask_emb.view(1, 1, -1).expand(B, T - 1, -1)], dim=1); sync("cat")
x = mod._rms(h, m.l0_in_norm, cfg.eps); sync("rms")
x2, dyn = m._conv_prepare(0, "attention_conv", x); sync("conv_prepare")
hid = m.query(emb, torch.tensor([2], device=dev), torch.tensor([20], device=dev)); sync("query")
head = torch.randn(1000, 4096, dtype=dt, device=dev)
val, cand = m.topk_logits(hid.reshape(-1, 4096), head, 0); sync("topk")
import torch.nn.functional as F
cand = cand.view(1, -1, 16); unary = val.view(1, -1, 16)
hp = F.linear(hid, m.sel_hproj).float(); sync("hproj")
pred = torch.tensor([5], device=dev)
a = m.sel_pred.index_select(0, pred).float() * hp[:, 0]; sync("pred_lookup")
c0 = cand[:, 0].reshape(-1); sync("cand reshape")
succ = m.sel_succ.index_select(0, c0); sync("succ_lookup")
succ = succ.float().view(1, -1, 256); sync("succ view")
sc = unary[:, 0] + torch.einsum("br,bkr->bk", a, succ); sync("einsum")
j = torch.argmax(sc, dim=-1, keepdim=True); sync("argmax")
ct = cand[:, 0].contiguous(); sync("contig")
try:
    p1 = ct.gather(-1, j).squeeze(-1); sync("gather contiguous int64")
except Exception as e:
    print("FAIL contiguous int64", str(e)[:80])
p2 = ct.int().gather(-1, j).squeeze(-1); sync("gather int32")
p3 = (ct * torch.nn.functional.one_hot(j.squeeze(-1), 16)).sum(-1); sync("onehot")
