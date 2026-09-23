"""DFlash2Drafter (HPU port) vs the z-lab reference drafts from dflash_sim.py.

usage: python test_dflash2.py DRAFTER_DIR TAPS.pt SIM.pt TARGET_DIR [cpu|hpu]
"""
import json
import sys

import torch
from safetensors import safe_open

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("dflash2_model", "/work/vllm_gaudi/v1/spec_decode/dflash2_model.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["dflash2_model"] = _mod
_spec.loader.exec_module(_mod)
Comm, DFlash2Config, DFlash2Drafter = _mod.Comm, _mod.DFlash2Config, _mod.DFlash2Drafter

torch.set_grad_enabled(False)
ddir, taps_path, sim_path, tdir = sys.argv[1:5]
dev = sys.argv[5] if len(sys.argv) > 5 else "cpu"
if dev == "hpu":
    import habana_frameworks.torch  # noqa: F401
dtype = torch.float32 if dev == "cpu" else torch.bfloat16
if dev == "cpubf16":
    dev = "cpu"

cfg = DFlash2Config.load(ddir)
L = 128
m = DFlash2Drafter(cfg, Comm(), max_len=L, n_slots=3, device=dev, dtype=dtype).load(ddir)
wmap = json.load(open(f"{tdir}/model.safetensors.index.json"))["weight_map"]
def tget(name):
    return safe_open(f"{tdir}/{wmap[name]}", "pt").get_tensor(name)
emb = tget("model.language_model.embed_tokens.weight").to(dtype).to(dev)
head = tget("lm_head.weight").to(dtype).to(dev)

taps = torch.load(taps_path)
ids = taps["ids"]
S = ids.numel()
feat = torch.stack([taps[f"layer{i}"] for i in cfg.taps], 0).mean(2)  # [9, S, H]
aux = feat.permute(1, 0, 2).reshape(S, -1).to(dtype).to(dev)
sim = torch.load(sim_path)["mean"]["drafts"]

# slot 2 holds this sequence; slot 1 gets junk to check isolation
slot = 2
pos = torch.arange(S, device=dev)
m.write_context(aux, pos, slot * L + pos)
junk = torch.randn(S, aux.shape[1], dtype=dtype, device=dev)
m.write_context(junk, pos, 1 * L + pos)

agree = tot = full = 0
for s in sorted(sim):
    a = ids[s:s + 1].to(dev)
    hid = m.query(emb[a], torch.tensor([slot], device=dev), torch.tensor([s], device=dev))  # [1, 7, H]
    val, cand = m.topk_logits(hid.reshape(-1, hid.shape[-1]), head, 0)
    k = cand.shape[-1]
    drafts = m.select(hid, val.view(1, -1, k), cand.view(1, -1, k), a)[0].cpu().tolist()
    ref = sim[s]
    n = sum(int(x == y) for x, y in zip(drafts, ref))
    agree += n
    tot += len(ref)
    full += int(drafts == ref)
print(f"[{dev}] per-token draft agreement {agree}/{tot}  exact blocks {full}/{len(sim)}")
