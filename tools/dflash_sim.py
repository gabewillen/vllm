"""Teacher-forced DFlash2 acceptance on CPU from reference target taps.

For every anchor position s the drafter sees target taps for positions < s and
the block [tok[s], mask x (B-1)]; its greedy drafts are compared with the known
greedy continuation tok[s+1 : s+B].  Prints mean acceptance length (accepted
drafts + bonus) for several mHC stream reductions of the taps.

usage: python dflash_sim.py DRAFTER_DIR TAPS.pt TARGET_DIR [prompt_len] [out.pt]
"""
import json
import sys

import torch
from safetensors import safe_open
from transformers import Qwen3Config

sys.path.insert(0, "/work/ref_src")
from dflash_model import DFlash2DraftModel  # noqa: E402

torch.set_grad_enabled(False)
ddir, taps_path, tdir = sys.argv[1:4]
prompt_len = int(sys.argv[4]) if len(sys.argv) > 4 else 5
out_path = sys.argv[5] if len(sys.argv) > 5 else None

cfg = Qwen3Config.from_pretrained(ddir)
cfg._attn_implementation = "sdpa"
model = DFlash2DraftModel(cfg).float().eval()
f = safe_open(f"{ddir}/model.safetensors", "pt")
sd = {}
for k in f.keys():
    t = f.get_tensor(k).float()
    if k.endswith(("predecessor_codebook", "successor_codebook")):
        k = k + ".weight"
    sd[k] = t
emb_w = sd.pop("embed_tokens.weight")
head_w = sd.pop("lm_head.weight")
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing", [m for m in missing if "rotary" not in m], "unexpected", unexpected)
mask = torch.load(f"{ddir}/mask_embedding.pt", map_location="cpu")
mask_id = int(mask["mask_token_id"])
emb_w[mask_id] = mask["embedding"].float()

# compare the drafter's shipped embed/head with the target's
wmap = json.load(open(f"{tdir}/model.safetensors.index.json"))["weight_map"]
def tget(name):
    return safe_open(f"{tdir}/{wmap[name]}", "pt").get_tensor(name)
t_emb = tget("model.language_model.embed_tokens.weight")
t_head = tget("lm_head.weight")
rows = torch.randint(0, t_emb.shape[0], (256,))
print("embed max|diff| vs target", (t_emb[rows].float() - emb_w[rows]).abs().max().item(),
      " head max|diff|", (t_head[rows].float() - head_w[rows]).abs().max().item())

taps = torch.load(taps_path)
ids = taps["ids"]
S = ids.numel()
B = model.block_size
layer_ids = model.target_layer_ids
print("block", B, "taps", layer_ids, "S", S)
import os
_off = int(os.environ.get("TAP_OFFSET", "0"))
streams = torch.stack([taps[f"layer{i + _off}"] for i in layer_ids], 0)  # [9, S, hc, H]

reductions = {
    "mean": lambda x: x.mean(2),
    **({} if os.environ.get("ONLY_MEAN") else {"sum": lambda x: x.sum(2), "stream0": lambda x: x[:, :, 0]}),
}


class Head(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, h):
        return h @ self.w.T


head = Head(head_w)
results = {}
for name, red in reductions.items():
    feat = red(streams)  # [9, S, H]
    ctx_all = feat.permute(1, 0, 2).reshape(1, S, -1)  # [1, S, 9H], layer-major concat
    acc = []
    drafts = {}
    for s in range(prompt_len, S - B):
        block = torch.full((1, B), mask_id, dtype=torch.long)
        block[0, 0] = ids[s]
        noise = emb_w[block]
        pos = torch.arange(s + B)[None]
        hid = model(target_hidden=ctx_all[:, :s], noise_embedding=noise, position_ids=pos)[:, 1:]
        tok, _, _ = model.propose(hid, block[:, 0], head, 0.0)
        truth = ids[s + 1:s + B]
        a = int((tok[0] == truth).int().cumprod(0).sum())
        acc.append(a + 1)
        drafts[s] = tok[0].tolist()
    results[name] = {"acc": acc, "drafts": drafts}
    print(f"{name:8s} mean acceptance length {sum(acc) / len(acc):.3f} over {len(acc)} anchors  "
          f"hist={[acc.count(k) for k in range(1, B + 1)]}", flush=True)
if out_path:
    torch.save(results, out_path)
