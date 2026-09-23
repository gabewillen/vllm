import json, os, re, sys, shutil
from safetensors import safe_open
from safetensors.torch import save_file
src = "/src"; N = int(sys.argv[1]); dst = f"/dst/GLM-5.3-Flash-L{N}"
os.makedirs(dst, exist_ok=True)
idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
keep = {}
for k, f in idx.items():
    m = re.match(r"model\.language_model\.layers\.(\d+)\.", k)
    if m and int(m.group(1)) >= N: continue
    keep.setdefault(f, []).append(k)
out_map = {}; shard = {}; nbytes = 0; si = 0
def flush():
    global shard, nbytes, si
    if not shard: return
    name = f"model-{si:05d}.safetensors"; save_file(shard, f"{dst}/{name}", metadata={"format":"pt"})
    for k in shard: out_map[k] = name
    si += 1; shard = {}; nbytes = 0
for f, ks in sorted(keep.items()):
    with safe_open(f"{src}/{f}", "pt") as fh:
        for k in ks:
            t = fh.get_tensor(k); shard[k] = t; nbytes += t.numel()*t.element_size()
            if nbytes > 4e9: flush()
flush()
json.dump({"metadata": {}, "weight_map": out_map}, open(f"{dst}/model.safetensors.index.json", "w"))
cfg = json.load(open(f"{src}/config.json")); tc = cfg["text_config"]
tc["num_hidden_layers"] = N
for key in ("layer_types", "mlp_layer_types", "indexer_types"): tc[key] = tc[key][:N]
lac = tc["linear_attn_config"]
lac["kda_layers"] = [i for i in lac["kda_layers"] if i < N]
lac["full_attn_layers"] = [i for i in lac["full_attn_layers"] if i < N]
tc["num_nextn_predict_layers"] = 0
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
for fn in os.listdir(src):
    if fn.endswith((".json", ".jinja")) and fn not in ("config.json", "model.safetensors.index.json"):
        shutil.copy(f"{src}/{fn}", dst)
print("done", len(out_map), "tensors", si, "shards")
