# SPDX-License-Identifier: Apache-2.0
"""Gaudi MoE for GLM-5.3-Flash on stacked per-channel FP8 expert weights.

The HPU ``mixture_of_experts`` op takes one tensor per expert; with 288
experts its per-launch host cost (~2.2 ms/layer) dominates decode.  Here the
routed experts are evaluated either

* ``dense``:  two FP8 GEMMs over *all* local expert shards with a dense
  [T, E] routing-weight matrix (zeros for unselected experts).  Each expert
  weight is read exactly once, so for the small per-rank intermediate size
  (moe_intermediate / tp = 256) this is memory-bound up to ~100 tokens.
* ``gather``: for a handful of tokens, only the selected experts are gathered
  and applied with batched matmuls.

W13 keeps its per-output-row scales (contraction over hidden).  W2 is
requantized with per-input-column scales so the scale folds into the
activations and one GEMM can sum over (expert, intermediate).
"""

import torch
import torch.nn.functional as F

FP8_MAX = 240.0  # Gaudi2 e4m3 range


def _cast_fp8(x: torch.Tensor, inv_scale: torch.Tensor) -> torch.Tensor:
    return torch.ops.hpu.cast_to_fp8_v2(x, inv_scale, False, False, torch.float8_e4m3fn)[0]


def _dyn_quant_rows(x: torch.Tensor):
    s = (x.abs().amax(dim=-1, keepdim=True).float() + 1e-8) / FP8_MAX
    return _cast_fp8(x, 1.0 / s), s


class StackedFp8Experts:
    """Builds per-rank stacked expert weights and registers them as buffers
    on ``owner`` (so torch.compile treats them as graph inputs, letting
    identical layers share one compiled graph)."""

    def __init__(self, owner, experts_layer, swiglu_limit: float | None):
        w13 = experts_layer.w13_weight.data  # [E, 2I, H] fp8
        s13 = experts_layer.w13_weight_scale_inv.data  # [E, 2I]
        w2 = experts_layer.w2_weight.data  # [E, H, I] fp8
        s2 = experts_layer.w2_weight_scale_inv.data  # [E, H]
        E, I2, H = w13.shape
        I = I2 // 2
        self.E, self.I, self.H = E, I, H
        w13_flat = w13.reshape(E * I2, H)  # zero-copy view
        s13_flat = s13.reshape(E * I2).float().contiguous()
        # W2 -> [E*I, H] with per-(e, i) scale, built one expert chunk at a time.
        w2t = torch.empty((E, I, H), dtype=torch.float8_e4m3fn, device=w2.device)
        s2c = torch.empty((E, I), dtype=torch.float32, device=w2.device)
        for e0 in range(0, E, 16):
            deq = w2[e0:e0 + 16].float() * s2[e0:e0 + 16].float().unsqueeze(-1)  # [e, H, I]
            deq = deq.transpose(1, 2)  # [e, I, H]
            sc = deq.abs().amax(dim=-1).clamp_min(1e-12) / FP8_MAX  # [e, I]
            w2t[e0:e0 + 16] = _cast_fp8(deq.contiguous(), (1.0 / sc).unsqueeze(-1))
            s2c[e0:e0 + 16] = sc
        for name, t in (("moe_w13", w13_flat), ("moe_s13", s13_flat),
                        ("moe_w2t", w2t.reshape(E * I, H)), ("moe_s2c", s2c),
                        ("moe_s2c_bf16", s2c.to(torch.bfloat16)),
                        ("moe_ones_h", torch.ones(H, dtype=torch.float32, device=w2.device))):
            owner.register_buffer(name, t, persistent=False)
        owner.moe_E, owner.moe_I, owner.moe_H = E, I, H
        self.limit = swiglu_limit
        import os
        if os.environ.get("GLM53_MOE_FUSED_PREFILL", "0") == "1" and hasattr(experts_layer, "moe_op"):
            # Keep the original per-expert lists: the fused HPU MoE op computes
            # only routed experts and wins for large prefill batches.
            object.__setattr__(owner, "_experts_layer", experts_layer)
        else:
            # Release the original W2 storage (the fused op's per-expert lists
            # are views of it).
            experts_layer.w2_weight.data = torch.empty(0, dtype=w2.dtype, device=w2.device)
            experts_layer.w2_weight_scale_inv.data = torch.empty(0, device=w2.device)
            if hasattr(experts_layer, "moe_op"):
                del experts_layer.moe_op


def _act(gate: torch.Tensor, up: torch.Tensor, limit: float | None) -> torch.Tensor:
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def moe_dense(m, x: torch.Tensor, rw: torch.Tensor, limit: float | None) -> torch.Tensor:
    """x: [T, H] bf16; rw: [T, E] routing weights (0 = unselected)."""
    T, E, I = x.shape[0], m.moe_E, m.moe_I
    xq, xs = _dyn_quant_rows(x)
    gu = torch.ops.hpu.fp8_gemm_v2(xq, False, m.moe_w13, True, None, torch.bfloat16, xs,
                                   m.moe_s13, None, False)
    # Epilogue stays in bf16 (as HF): [T, E*2I] is large at high batch and
    # fp32 passes over it were TPC-bound.
    gu = gu.view(T, E, 2, I)
    h = _act(gu[:, :, 0], gu[:, :, 1], limit)
    h = h * (rw.to(torch.bfloat16).unsqueeze(-1) * m.moe_s2c_bf16.unsqueeze(0))
    hq, hs = _dyn_quant_rows(h.reshape(T, E * I))
    return torch.ops.hpu.fp8_gemm_v2(hq, False, m.moe_w2t, False, None, torch.bfloat16, hs,
                                     m.moe_ones_h, None, False)


def moe_gather(m, x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, limit: float | None) -> torch.Tensor:
    """x: [T, H]; ids/wts: [T, K].  Applies only the selected experts."""
    T, K = ids.shape
    E, I, H = m.moe_E, m.moe_I, m.moe_H
    f = ids.reshape(-1)
    w13 = m.moe_w13.view(E, 2 * I, H).index_select(0, f).to(torch.bfloat16)
    s13 = m.moe_s13.view(E, 2 * I).index_select(0, f)
    xx = x.unsqueeze(1).expand(T, K, H).reshape(T * K, 1, H)
    gu = torch.bmm(xx, w13.transpose(1, 2)).squeeze(1).float() * s13  # [TK, 2I]
    h = _act(gu[:, :I], gu[:, I:], limit)
    h = h * m.moe_s2c.index_select(0, f) * wts.reshape(-1, 1).float()
    w2 = m.moe_w2t.view(E, I, H).index_select(0, f).to(torch.bfloat16)
    o = torch.bmm(h.to(torch.bfloat16).unsqueeze(1), w2).view(T, K, H)
    return o.float().sum(dim=1).to(x.dtype)


def route(x: torch.Tensor, gate_w32: torch.Tensor, bias: torch.Tensor, top_k: int,
          renormalize: bool, scale: float):
    """HF Glm5NextTextTopkRouter (n_group == 1): sigmoid scores, bias only for
    selection, renormalised weights times routed_scaling_factor."""
    logits = torch.matmul(x.float(), gate_w32.t())
    scores = torch.sigmoid(logits)
    ids = torch.topk(scores + bias, k=top_k, dim=-1).indices
    w = scores.gather(1, ids)
    if renormalize:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return ids, w * scale
