import sys, torch, json
sys.argv = ["x", "/mnt/glm-models/GLM-5.3-Flash-L5", "/tmp/x.pt", "[1]"]
exec(open("/work/tools/hf_ref.py").read().split("t0 = time.time()")[0])
L = load_layer(0); A = L.self_attn
x = torch.load("/work/dbg/l5/l0_attn_in.pt")[:5].unsqueeze(0)  # vLLM input (matches ref)
S = 5
mixed = torch.cat([A.q_proj(x), A.k_proj(x), A.v_proj(x)], -1).transpose(1, 2)
conv = M.causal_conv1d_fn(mixed, weight=A.conv1d.weight.squeeze(1), bias=None, activation="silu")[:, :, -S:].transpose(1, 2)  # [1,S,3*8192]
g = A.forget_gate(x); beta = torch.sigmoid(A.b_proj(x))
qkvc = torch.load("/work/dbg/l5/l0_kda_qkvc.pt")[0, :S]  # rank0: [S, 3*1024]
ref_q, ref_k, ref_v = conv[0].split(8192, -1)
ref_r0 = torch.cat([ref_q[:, :1024], ref_k[:, :1024], ref_v[:, :1024]], -1)
cs = lambda a, b: torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0).item()
print("conv out cos", cs(qkvc, ref_r0), "absmax", qkvc.abs().max().item(), ref_r0.abs().max().item())
print("per-part cos", [round(cs(qkvc[:, i*1024:(i+1)*1024], ref_r0[:, i*1024:(i+1)*1024]), 4) for i in range(3)])
gv = torch.load("/work/dbg/l5/l0_kda_gate.pt")[:S]; print("gate cos", cs(gv, g[0, :, :8]))
bv = torch.load("/work/dbg/l5/l0_kda_beta.pt")[:S]; print("beta cos", cs(bv, beta[0, :, :8]))
print("mask", torch.load("/work/dbg/l5/l0_kda_mask.pt")[0, :8], "hasinit", torch.load("/work/dbg/l5/l0_kda_hasinit.pt"))
xv = torch.load("/work/dbg/l5/l0_kda_x.pt")[0, :S]; ref_x = torch.cat([m_[0][:, :1024] for m_ in (A.q_proj(x), A.k_proj(x), A.v_proj(x))], -1)
print("pre-conv x cos", cs(xv, ref_x))
