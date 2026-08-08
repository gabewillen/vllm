# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum, is_deep_gemm_supported


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def _dequant_wo_a_bf16(wo_a: nn.Module, n_groups: int) -> torch.Tensor:
    """Dequantized BF16 view of the block-FP8 ``wo_a`` weight, cached on the module.

    Weight is [n_groups * o_lora_rank, group_in_dim] FP8 with 128x128 block
    scales in ``weight_scale_inv``; returns [n_groups, o_lora_rank, group_in_dim].
    """
    cached = getattr(wo_a, "_bf16_o_proj_weight", None)
    if cached is None:
        w = wo_a.weight
        s = wo_a.weight_scale_inv.to(torch.float32)
        s = s.repeat_interleave(128, dim=0)[: w.shape[0]]
        s = s.repeat_interleave(128, dim=1)[:, : w.shape[1]]
        cached = (
            (w.to(torch.float32) * s)
            .to(torch.bfloat16)
            .view(n_groups, -1, w.shape[1])
        )
        wo_a._bf16_o_proj_weight = cached
    return cached


def _inverse_rope(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Eager inverse RoPE on the trailing ``rope_dim`` dims of each head.

    Matches the fused triton kernel: interleaved (even, odd) pairs, cache row
    layout cos[rope_dim//2] || sin[rope_dim//2], conjugate rotation.
    """
    half = rope_dim // 2
    r = o[..., -rope_dim:].to(torch.float32).unflatten(-1, (half, 2))
    cs = cos_sin_cache[positions].to(torch.float32)
    cos = cs[:, :half].view(-1, 1, half)
    sin = cs[:, half:].view(-1, 1, half)
    a, b = r.unbind(-1)
    out = o.clone()
    out[..., -rope_dim:] = (
        torch.stack((a * cos + b * sin, b * cos - a * sin), dim=-1)
        .flatten(-2)
        .to(o.dtype)
    )
    return out


def bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    rope_dim: int,
) -> torch.Tensor:
    """O projection without DeepGEMM: inverse RoPE + BF16 grouped einsum + wo_b.

    Fallback for GPUs without DeepGEMM support (e.g. SM 8.9). Skips the FP8
    round-trip entirely, matching the reference implementation's BF16 path.
    """
    o = _inverse_rope(o, positions, cos_sin_cache, rope_dim)
    o = o.view(o.shape[0], n_groups, -1).to(torch.bfloat16)
    w = _dequant_wo_a_bf16(wo_a, n_groups)
    z = torch.einsum("bgr,gdr->bgd", o, w)
    return wo_b(z.flatten(1))


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    Falls back to :func:`bf16_o_proj` when DeepGEMM is unsupported.
    """
    if not is_deep_gemm_supported():
        return bf16_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            wo_b,
            n_groups=n_groups,
            rope_dim=rope_dim,
        )
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (wo_a.weight, wo_a.weight_scale_inv),
        z,
        recipe=einsum_recipe,
    )
    return wo_b(z.flatten(1))
