import torch, sys, glob, os
d = sys.argv[1]; ref = torch.load(sys.argv[2]); S = ref["ids"].numel()
def st(t): return f"nan={torch.isnan(t).sum().item():7d} absmax={t.nan_to_num().abs().max().item():.3g}"
for f in sorted(glob.glob(f"{d}/*.pt"), key=lambda x: (int(os.path.basename(x).split('_')[0][1:]), os.path.getmtime(x))):
    n = os.path.basename(f)[:-3]; t = torch.load(f)
    key = f"layer{n.split('_')[0][1:]}" if n.endswith("_stream") else n
    line = f"{n:14s} {tuple(t.shape)} {st(t)}"
    if key in ref:
        r = ref[key]; v = t.reshape(-1, *r.shape[1:])[:S]
        cos = torch.nn.functional.cosine_similarity(v.flatten(), r.flatten(), 0).item()
        line += f"  cos={cos:.5f} rel={((v-r).norm()/r.norm()).item():.4f} |ref|max={r.abs().max().item():.3g}"
        if cos < 0.99:
            line += "\n     per-token cos: " + " ".join(f"{torch.nn.functional.cosine_similarity(v[j].flatten(), r[j].flatten(), 0).item():.3f}" for j in range(S))
    print(line)
