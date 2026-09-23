# SPDX-License-Identifier: Apache-2.0
"""Opt-in per-output-channel FP8 for GLM-5.3 BF16 linears on Gaudi.

Decode is memory-bound, so halving the bytes of the BF16 attention/KDA
projections cuts per-token HBM traffic.  Weights are quantized once after
loading (per output channel, Gaudi2 e4m3 range), activations dynamically per
token, and the GEMM runs on the MME in FP8 via ``fp8_gemm_v2``.
"""

import torch

FP8_MAX = 240.0


class HpuFp8ChannelLinearMethod:
    """Stand-in ``quant_method`` for an already-loaded unquantized linear."""

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        s = (x2.abs().amax(dim=-1, keepdim=True).float() + 1e-8) / FP8_MAX
        xq = torch.ops.hpu.cast_to_fp8_v2(x2, 1.0 / s, False, False, torch.float8_e4m3fn)[0]
        out = torch.ops.hpu.fp8_gemm_v2(xq, False, layer.weight, True, None, x.dtype, s,
                                        layer.weight_scale_fp8, bias, False)
        return out.view(*shape[:-1], out.shape[-1])

    def process_weights_after_loading(self, layer) -> None:  # already done
        return


@torch.no_grad()
def quantize_linear_fp8_(layer) -> None:
    w = layer.weight.data
    if w.dtype == torch.float8_e4m3fn:
        return
    s = (w.float().abs().amax(dim=1).clamp_min(1e-12) / FP8_MAX)
    q = torch.ops.hpu.cast_to_fp8_v2(w.float(), (1.0 / s).unsqueeze(1), False, False,
                                     torch.float8_e4m3fn)[0]
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.weight_scale_fp8 = s.contiguous()
    layer.quant_method = HpuFp8ChannelLinearMethod()
