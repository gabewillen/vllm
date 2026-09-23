"""Count swiglu-clamp activations per token/layer in the HF reference."""
import sys, json, torch, collections
sys.argv = ["x", "/mnt/glm-models/GLM-5.3-Flash", "/work/ref/clamp_dummy.pt", sys.argv[1]]
src = open("/work/tools/hf_ref.py").read()
pre, post = src.split("t0 = time.time()\nemb =", 1)
exec(pre)
stats = collections.defaultdict(lambda: torch.zeros(0))
cur = {"layer": -1}
orig = M.Glm5NextTextExperts._apply_gate
def patched(self, gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    lim = self.swiglu_limit
    n = ((gate > lim).sum(-1) + (up.abs() > lim).sum(-1))
    stats.setdefault(cur["layer"], []).append((self._tok_idx.clone(), n.clone(), gate.max().item(), up.abs().max().item()))
    return orig(self, gate_up)
M.Glm5NextTextExperts._apply_gate = patched
orig_fwd = M.Glm5NextTextExperts.forward
def fwd(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in hit:
        expert_idx = expert_idx[0]
        top_k_pos, token_idx = torch.where(mask[expert_idx])
        self._tok_idx = token_idx
        current = self._apply_gate(torch.nn.functional.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
        current = torch.nn.functional.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, current.to(final.dtype))
    return final
M.Glm5NextTextExperts.forward = fwd
# shared expert / dense MLP clamp too
orig_mlp = M.Glm5NextTextMLP.forward
def mlp_fwd(self, x):
    g = self.gate_proj(x); u = self.up_proj(x)
    n = ((g > self.swiglu_limit).sum(-1) + (u.abs() > self.swiglu_limit).sum(-1)).flatten()
    stats.setdefault(("mlp", cur["layer"]), []).append((torch.arange(n.numel()), n, g.max().item(), u.abs().max().item()))
    return orig_mlp(self, x)
M.Glm5NextTextMLP.forward = mlp_fwd
exec("t0 = time.time()\nemb =" + post.split("for i in range(min(MAXL")[0])
for i in range(tc.num_hidden_layers):
    cur["layer"] = i
    layer = load_layer(i)
    h, topk = layer(h, attention_mask=mask, position_ids=pos, past_key_values=None, prev_topk_indices=topk)
    del layer
    print(f"layer {i} done", flush=True)
S = ids.shape[1]
per_tok = torch.zeros(S, dtype=torch.long); per_tok_mlp = torch.zeros(S, dtype=torch.long); layers_hit = collections.Counter()
for k, lst in stats.items():
    for tok, n, gm, um in lst:
        tgt = per_tok_mlp if isinstance(k, tuple) else per_tok
        tgt.index_add_(0, tok, n.long())
        if n.sum() > 0: layers_hit[k] += int(n.sum())
print("routed-expert clamp counts per token:", per_tok.tolist())
print("shared/dense MLP clamp counts per token:", per_tok_mlp.tolist())
print("layers with clamps:", dict(layers_hit))
print("ids:", ids[0].tolist())
