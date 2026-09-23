"""Layer-streaming CPU reference for GLM-5.3-Flash built from the HF modules.

Runs a full (no-cache) forward over token ids and dumps final logits plus the
per-layer hidden streams.  Weights are loaded one decoder layer at a time and
FP8 block weights are dequantized, so the full 45-layer model fits in RAM.

usage: python hf_ref.py MODEL_DIR OUT.pt TOKEN_IDS_JSON [dtype]
"""
import json
import re
import sys
import time

import torch
from safetensors import safe_open
from transformers.models.glm5_next import modeling_glm5_next as M
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

torch.set_grad_enabled(False)
model_dir, out_path, ids_json = sys.argv[1:4]
dtype = getattr(torch, sys.argv[4]) if len(sys.argv) > 4 else torch.float32
ids = torch.tensor(json.loads(ids_json), dtype=torch.long)[None]

cfg = Glm5NextConfig.from_pretrained(model_dir)
tc = cfg.text_config
tc._attn_implementation = "eager"
wmap = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
PFX = "model.language_model."

_handles = {}


def get(name):
    f = wmap[name]
    if f not in _handles:
        _handles[f] = safe_open(f"{model_dir}/{f}", "pt")
    return _handles[f].get_tensor(name)


def weight(name):
    """Return a dequantized weight (handles block-FP8 + weight_scale_inv)."""
    w = get(name)
    sname = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn:
        s = get(sname).float()
        o, i = w.shape
        bo, bi = -(-o // s.shape[0]), -(-i // s.shape[1])
        s = s.repeat_interleave(bo, 0)[:o].repeat_interleave(bi, 1)[:, :i]
        return (w.float() * s).to(dtype)
    return w.to(dtype) if w.is_floating_point() else w


def load_layer(i):
    layer = M.Glm5NextTextDecoderLayer(tc, i).to(dtype)
    p = f"{PFX}layers.{i}."
    sd = {}
    for hc, src in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
        sd[f"{hc}.fn"] = get(p + f"{src}_fn").float()
        sd[f"{hc}.base"] = get(p + f"{src}_base").float()
        sd[f"{hc}.scale"] = get(p + f"{src}_scale").float()
    sd["input_layernorm.weight"] = weight(p + "input_layernorm.weight")
    sd["post_attention_layernorm.weight"] = weight(p + "post_attention_layernorm.weight")
    a = p + "self_attn."
    if layer.block_type == "linear_attention":
        for n in ("q_proj", "k_proj", "v_proj", "b_proj", "g_a_proj", "g_b_proj", "o_proj"):
            sd[f"self_attn.{n}.weight"] = weight(a + n + ".weight")
        sd["self_attn.o_norm.weight"] = weight(a + "o_norm.weight")
        sd["self_attn.forget_gate.f_a_proj.weight"] = weight(a + "f_a_proj.weight")
        sd["self_attn.forget_gate.f_b_proj.weight"] = weight(a + "f_b_proj.weight")
        sd["self_attn.forget_gate.dt_bias"] = get(a + "dt_bias").float()
        sd["self_attn.forget_gate.A_log"] = get(a + "A_log").float().reshape(-1)
        conv = [get(a + f"{c}_conv1d.weight").float() for c in "qkv"]
        conv = [c.reshape(c.shape[0], 1, -1) for c in conv]
        sd["self_attn.conv1d.weight"] = torch.cat(conv, 0)
    else:
        for n in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
            sd[f"self_attn.{n}.weight"] = weight(a + n + ".weight")
        sd["self_attn.q_a_layernorm.weight"] = weight(a + "q_a_layernorm.weight")
        sd["self_attn.kv_a_layernorm.weight"] = weight(a + "kv_a_layernorm.weight")
        ix = a + "indexer."
        for n in ("wq_b", "wk", "weights_proj"):
            sd[f"self_attn.indexer.{n}.weight"] = weight(ix + n + ".weight")
        sd["self_attn.indexer.k_norm.weight"] = get(ix + "k_norm.weight").to(dtype)
        sd["self_attn.indexer.k_norm.bias"] = get(ix + "k_norm.bias").to(dtype)
        sd["self_attn.indexer.index_kpool_compress_ape"] = get(ix + "index_kpool_compress_ape").to(dtype)
        sd["self_attn.indexer.index_kpool_compress_gate"] = get(ix + "index_kpool_compress_gate").to(dtype)
    m = p + "mlp."
    if tc.mlp_layer_types[i] == "sparse":
        sd["mlp.gate.weight"] = get(m + "gate.weight").float()
        sd["mlp.gate.e_score_correction_bias"] = get(m + "gate.e_score_correction_bias").float()
        for n in ("gate_proj", "up_proj", "down_proj"):
            sd[f"mlp.shared_experts.{n}.weight"] = weight(m + f"shared_experts.{n}.weight")
        E = tc.n_routed_experts
        gu = [torch.cat([weight(m + f"experts.{e}.gate_proj.weight"),
                         weight(m + f"experts.{e}.up_proj.weight")], 0) for e in range(E)]
        sd["mlp.experts.gate_up_proj"] = torch.stack(gu)
        del gu
        sd["mlp.experts.down_proj"] = torch.stack(
            [weight(m + f"experts.{e}.down_proj.weight") for e in range(E)])
    else:
        for n in ("gate_proj", "up_proj", "down_proj"):
            sd[f"mlp.{n}.weight"] = weight(m + f"{n}.weight")
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "rotary" not in k]
    assert not missing and not unexpected, (missing, unexpected)
    # router math is done in fp32 in HF regardless of module dtype
    if tc.mlp_layer_types[i] == "sparse":
        layer.mlp.gate.float()
    return layer.eval()


t0 = time.time()
emb = get(PFX + "embed_tokens.weight").to(dtype)
x = emb[ids]
del emb
S = ids.shape[1]
mask = torch.ones(1, S, dtype=torch.bool)
pos = torch.arange(S)[None]
h = x.unsqueeze(2).expand(-1, -1, tc.hc_mult, -1).contiguous()
dump = {"embed": x[0].float().clone()}
topk = None
import os
MAXL = int(os.environ.get("REF_MAX_LAYERS", tc.num_hidden_layers))


def _hook(name):
    def fn(mod, inp, out):
        x = inp[0] if inp else None
        if x is None:
            x = mod._kw_hidden
        dump[name + "_in"] = x[0].float().clone()
        o = out[0] if isinstance(out, tuple) else out
        dump[name + "_out"] = o[0].float().clone()
    return fn


def _pre(mod, args, kwargs):
    mod._kw_hidden = kwargs.get("hidden_states")


for i in range(min(MAXL, tc.num_hidden_layers)):
    layer = load_layer(i)
    if not os.environ.get("REF_LAYERS_ONLY"):
        layer.self_attn.register_forward_pre_hook(_pre, with_kwargs=True)
        layer.self_attn.register_forward_hook(_hook(f"l{i}_attn"))
        layer.mlp.register_forward_hook(_hook(f"l{i}_mlp"))
    h, topk = layer(h, attention_mask=mask, position_ids=pos, past_key_values=None,
                    prev_topk_indices=topk)
    dump[f"layer{i}"] = h[0].float().clone()
    print(f"layer {i} done {time.time() - t0:.1f}s  |h|={h.float().norm():.3f}", flush=True)
    del layer
norm_w = get(PFX + "norm.weight").to(dtype)
hs = h.mean(dim=2)
hs32 = hs.float()
hs = (hs32 * torch.rsqrt(hs32.pow(2).mean(-1, keepdim=True) + tc.rms_norm_eps)).to(dtype) * norm_w
lm = get("lm_head.weight").to(dtype)
logits = (hs @ lm.T).float()[0]
dump["final_hidden"] = hs[0].float()
dump["logits"] = logits
dump["ids"] = ids[0]
torch.save(dump, out_path)
print("top5 per position:", logits.topk(5, dim=-1).indices.tolist())
print(f"total {time.time() - t0:.1f}s")
