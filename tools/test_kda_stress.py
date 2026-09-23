import torch, importlib.util, sys
spec = importlib.util.spec_from_file_location("kda_hpu", "/work/vllm/models/glm5next/common/kda_hpu.py")
kh = importlib.util.module_from_spec(spec); sys.modules["kda_hpu"] = kh; spec.loader.exec_module(kh)
from transformers.models.glm5_next import modeling_glm5_next as M
torch.manual_seed(0)
B, S, H, K = 1, 200, 2, 32
q, k, v = (torch.randn(B, S, H, K) for _ in range(3))
for gv in (-5.0, -0.001):
    g = torch.full((B, S, H, K), gv); beta = torch.sigmoid(torch.randn(B, S, H))
    ro, rs = M.recurrent_kimi_delta_attention(q, k, v, g, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o, s = kh.kda_chunk_prefill(q, k, v, g, beta, torch.zeros(B, H, K, K), K ** -0.5)
    print(f"g={gv}: out err {(o - ro).abs().max().item():.2e} finite={torch.isfinite(o).all().item()} state err {(s - rs.transpose(-1,-2)).abs().max().item():.2e}")
