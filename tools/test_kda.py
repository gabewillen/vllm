import sys, torch
sys.path.insert(0, "/work/vllm/models/glm5next/common")
import importlib.util
spec = importlib.util.spec_from_file_location("kda_hpu", "/work/vllm/models/glm5next/common/kda_hpu.py")
kh = importlib.util.module_from_spec(spec); spec.loader.exec_module(kh)
from transformers.models.glm5_next import modeling_glm5_next as M
torch.manual_seed(0)
B, S, H, K = 2, 70, 3, 16
q, k, v = (torch.randn(B, S, H, K) for _ in range(3))
g = -5 * torch.sigmoid(torch.randn(B, S, H, K))
beta = torch.sigmoid(torch.randn(B, S, H))
ref_o, ref_s = M.chunk_kimi_delta_attention.__wrapped__(q, k, v, g, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True) if hasattr(M.chunk_kimi_delta_attention, "__wrapped__") else M.chunk_kimi_delta_attention(q, k, v, g, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
rec_o, rec_s = M.recurrent_kimi_delta_attention(q, k, v, g, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
print("hf chunk vs hf recurrent", (ref_o - rec_o).abs().max().item(), (ref_s - rec_s).abs().max().item())
o, s = kh.kda_chunk_prefill(q, k, v, g, beta, torch.zeros(B, H, K, K), K ** -0.5, 64)
print("mine chunk vs hf recurrent: out", (o - rec_o).abs().max().item(), "state(VK vs KV^T)", (s - rec_s.transpose(-1, -2)).abs().max().item())
# decode step chain
st = torch.zeros(B, H, K, K); outs = []
for t in range(S):
    outs.append(kh.kda_decode_step(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], st, torch.arange(B), K ** -0.5))
print("mine decode chain vs hf recurrent", (torch.stack(outs, 1) - rec_o).abs().max().item())
