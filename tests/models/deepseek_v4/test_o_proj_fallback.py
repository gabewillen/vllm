# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SM89 (no-DeepGEMM) BF16 o_proj fallback."""

import pytest
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    _dequant_wo_a_bf16,
    bf16_o_proj,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

T, HEADS, HEAD_DIM, ROPE_DIM = 5, 8, 512, 64
N_GROUPS, O_LORA_RANK = 4, 32
GROUP_IN = HEADS // N_GROUPS * HEAD_DIM


class _FakeWoA(nn.Module):
    def __init__(self, device):
        super().__init__()
        w = torch.randn(N_GROUPS * O_LORA_RANK, GROUP_IN, device=device) * 0.05
        self.weight = w.to(torch.float8_e4m3fn)
        self.weight_scale_inv = (
            torch.rand(
                (N_GROUPS * O_LORA_RANK + 127) // 128,
                (GROUP_IN + 127) // 128,
                device=device,
            )
            + 0.5
        )


class _FakeWoB(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.w = torch.randn(
            64, N_GROUPS * O_LORA_RANK, device=device, dtype=torch.bfloat16
        )

    def forward(self, x):
        return x @ self.w.t()


def _ref_inverse_rope(o, positions, cos_sin_cache):
    half = ROPE_DIM // 2
    o = o.clone().to(torch.float32)
    r = o[..., -ROPE_DIM:].unflatten(-1, (half, 2))
    cs = cos_sin_cache[positions]
    cos, sin = cs[:, :half].view(-1, 1, half), cs[:, half:].view(-1, 1, half)
    a, b = r.unbind(-1)
    rot = torch.stack((a * cos + b * sin, b * cos - a * sin), dim=-1).flatten(-2)
    o[..., -ROPE_DIM:] = rot
    return o


@requires_cuda
def test_bf16_o_proj_matches_reference():
    torch.manual_seed(0)
    device = "cuda"
    o = torch.randn(T, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, 1024, (T,), device=device)
    cos_sin = torch.randn(1024, ROPE_DIM, device=device, dtype=torch.float32)
    wo_a, wo_b = _FakeWoA(device), _FakeWoB(device)

    out = bf16_o_proj(
        o,
        positions,
        cos_sin,
        wo_a,
        wo_b,
        n_groups=N_GROUPS,
        rope_dim=ROPE_DIM,
    )

    s = wo_a.weight_scale_inv.repeat_interleave(128, 0)[: N_GROUPS * O_LORA_RANK]
    s = s.repeat_interleave(128, 1)[:, :GROUP_IN]
    w_ref = (wo_a.weight.to(torch.float32) * s).view(N_GROUPS, O_LORA_RANK, GROUP_IN)
    o_ref = _ref_inverse_rope(o, positions, cos_sin).view(T, N_GROUPS, GROUP_IN)
    z_ref = torch.einsum("bgr,gdr->bgd", o_ref, w_ref)
    ref = z_ref.flatten(1).to(torch.bfloat16) @ wo_b.w.t().to(torch.bfloat16)

    ref32 = ref.to(torch.float32)
    torch.testing.assert_close(
        out.to(torch.float32), ref32, rtol=5e-2, atol=0.03 * ref32.std().item()
    )


@requires_cuda
def test_dequant_cache_reused():
    wo_a = _FakeWoA("cuda")
    first = _dequant_wo_a_bf16(wo_a, N_GROUPS)
    assert _dequant_wo_a_bf16(wo_a, N_GROUPS) is first
    assert first.shape == (N_GROUPS, O_LORA_RANK, GROUP_IN)


def test_dispatch_uses_fallback_on_this_gpu():
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cap = torch.cuda.get_device_capability()
    if cap in ((9, 0), (10, 0)):
        pytest.skip("DeepGEMM-capable GPU; fallback not selected here")
    assert not is_deep_gemm_supported()
