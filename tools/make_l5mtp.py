import json, os, shutil, re
from safetensors import safe_open
from safetensors.torch import save_file
src = "/src"; base = "/dst/GLM-5.3-Flash-L5"; dst = "/dst/GLM-5.3-Flash-L5MTP"
os.makedirs(dst, exist_ok=True)
for fn in os.listdir(base):
    if fn != "config.json" and fn != "model.safetensors.index.json":
        p = f"{dst}/{fn}"
        if not os.path.exists(p): os.symlink(f"/mnt/glm-models/GLM-5.3-Flash-L5/{fn}", p)
idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
wm = json.load(open(f"{base}/model.safetensors.index.json"))["weight_map"]
mtp = {}
for k, f in idx.items():
    if re.match(r"model\.language_model\.layers\.45\.", k):
        mtp.setdefault(f, []).append(k)
out = {}
for f, ks in sorted(mtp.items()):
    with safe_open(f"{src}/{f}", "pt") as fh:
        for k in ks:
            out[k.replace("layers.45.", "layers.5.")] = fh.get_tensor(k)
save_file(out, f"{dst}/mtp.safetensors", metadata={"format": "pt"})
for k in out: wm[k] = "mtp.safetensors"
json.dump({"metadata": {}, "weight_map": wm}, open(f"{dst}/model.safetensors.index.json", "w"))
cfg = json.load(open(f"{base}/config.json")); cfg["text_config"]["num_nextn_predict_layers"] = 1
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
print("mtp tensors", len(out), sorted({k.split("layers.5.")[1].split(".")[0] for k in out}))
