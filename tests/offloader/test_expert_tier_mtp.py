# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MTP (DSpark) support in the expert-tier cache path.

Covers the binding-side manifest filter (MTP layers are dropped from pool
sizing when speculative decoding is off) and the pure helpers of the
Flash MTP drafter: checkpoint weight-name remapping, fp8 block dequant,
TP shard slicing, and the windowed draft attention math. The drafter
module is loaded from its file path because the deepseek_v4 package
imports GPU-only kernels at package-import time.
"""

import importlib.util
import math
import os

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_tier_binding import (
    ExpertTierManifest,
    compute_slot_counts,
    drop_mtp_layers,
)
from vllm.model_executor.offloader.expert_tier import ExpertTensorSpec

_MTP_FLASH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "vllm",
    "models",
    "deepseek_v4",
    "nvidia",
    "mtp_flash.py",
)


@pytest.fixture(scope="module")
def mtp_flash():
    spec = importlib.util.spec_from_file_location("_mtp_flash_test", _MTP_FLASH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#
# Binding-side manifest filter
#


def _manifest(layer_ids, meta):
    specs = {
        layer_id: [ExpertTensorSpec("w", (1024,), torch.uint8)]
        for layer_id in layer_ids
    }
    files = {(layer_id, "w"): f"l{layer_id}.raw" for layer_id in layer_ids}
    return ExpertTierManifest(
        path="<test>", num_experts=8, layer_specs=specs, files=files, meta=meta
    )


def test_drop_mtp_layers():
    manifest = _manifest([0, 1, 43, 44], {"mtp_layer_ids": [43, 44]})
    filtered = drop_mtp_layers(manifest)
    assert sorted(filtered.layer_specs) == [0, 1]
    assert sorted(k[0] for k in filtered.files) == [0, 1]
    assert filtered.num_experts == manifest.num_experts

    # Without the meta key the manifest passes through untouched.
    legacy = _manifest([0, 1], {})
    assert drop_mtp_layers(legacy) is legacy


def test_drop_mtp_layers_restores_slot_counts():
    """Pool sizing over a filtered manifest matches a cache converted
    without --mtp, so non-speculative serving is unaffected."""
    legacy = _manifest([0, 1], {})
    upgraded = _manifest([0, 1, 43], {"mtp_layer_ids": [43]})
    # 7 legacy slots (1 KiB rows, staged): stay under the num_experts
    # clamp so the extra-layer shrinkage is observable.
    budget = dict(gpu_bytes=7 * 4096, pinned_bytes=2**20)
    legacy_slots = compute_slot_counts(legacy, **budget)
    filtered_slots = compute_slot_counts(drop_mtp_layers(upgraded), **budget)
    assert filtered_slots == legacy_slots
    # Unfiltered, the extra layer shrinks the per-layer pools.
    gpu_all, _ = compute_slot_counts(upgraded, **budget)
    assert gpu_all[0] < legacy_slots[0][0]


#
# Drafter weight-name remapping
#


def test_remap_flash_mtp_weight_name(mtp_flash):
    remap = mtp_flash.remap_flash_mtp_weight_name
    base = "model.layers.43."
    cases = {
        "mtp.0.main_proj.weight": base + "main_proj.weight",
        "mtp.0.main_proj.scale": base + "main_proj.scale",
        "mtp.0.main_norm.weight": base + "main_norm.weight",
        "mtp.0.attn.wq_a.weight": base + "blocks.0.attn.wq_a.weight",
        "mtp.1.attn.wq_b.scale": base + "blocks.1.attn.wq_b.scale",
        "mtp.1.attn.attn_sink": base + "blocks.1.attn.attn_sink",
        "mtp.2.attn.q_norm.weight": base + "blocks.2.attn.q_norm.weight",
        "mtp.0.attn_norm.weight": base + "blocks.0.attn_norm.weight",
        "mtp.2.ffn_norm.weight": base + "blocks.2.ffn_norm.weight",
        "mtp.1.hc_attn_fn": base + "blocks.1.hc_attn_fn",
        "mtp.1.hc_ffn_scale": base + "blocks.1.hc_ffn_scale",
        "mtp.0.ffn.gate.weight": base + "blocks.0.ffn.gate.weight",
        "mtp.0.ffn.gate.bias": base + "blocks.0.ffn.gate.bias",
        "mtp.0.ffn.experts.7.w1.weight": base + "blocks.0.ffn.experts.7.w1.weight",
        "mtp.2.ffn.shared_experts.w2.scale": (
            base + "blocks.2.ffn.shared_experts.w2.scale"
        ),
        # Final-block heads and norm hoist to the spec layer.
        "mtp.2.norm.weight": base + "shared_head.norm.weight",
        "mtp.2.hc_head_fn": base + "hc_head_fn",
        "mtp.2.hc_head_base": base + "hc_head_base",
        "mtp.2.hc_head_scale": base + "hc_head_scale",
        # Unused heads and non-MTP tensors are skipped.
        "mtp.2.markov_head.markov_w1.weight": None,
        "mtp.2.confidence_head.proj.weight": None,
        "layers.0.attn.wq_a.weight": None,
        "embed.weight": None,
        "hc_head_fn": None,
    }
    for name, expected in cases.items():
        assert remap(name, 43) == expected, name


#
# fp8 block dequant + TP shard helpers
#


def test_e8m0_to_float(mtp_flash):
    raw = torch.tensor([127, 126, 130], dtype=torch.uint8)
    out = mtp_flash.e8m0_to_float(raw)
    assert torch.equal(out, torch.tensor([1.0, 0.5, 8.0]))


def test_dequant_fp8_block(mtp_flash):
    torch.manual_seed(0)
    weight = torch.randn(6, 8).to(torch.float8_e4m3fn)
    # Blocks of 4: scales [2, 2] with exponents 2^1 and 2^-1 etc.
    scale = torch.tensor([[128, 126], [127, 129]], dtype=torch.uint8)
    out = mtp_flash.dequant_fp8_block(weight, scale, block=4)
    assert out.dtype == torch.bfloat16
    assert out.shape == weight.shape
    w = weight.to(torch.float32)
    expected = torch.empty_like(w)
    factors = [[2.0, 0.5], [1.0, 4.0]]
    for bi in range(2):
        for bj in range(2):
            rows = slice(bi * 4, min((bi + 1) * 4, 6))
            cols = slice(bj * 4, (bj + 1) * 4)
            expected[rows, cols] = w[rows, cols] * factors[bi][bj]
    torch.testing.assert_close(out.to(torch.float32), expected, atol=0.02, rtol=0.02)


def test_tp_shard(mtp_flash):
    t = torch.arange(16).reshape(4, 4)
    assert torch.equal(mtp_flash.tp_shard(t, None, 1, 2), t)
    assert torch.equal(mtp_flash.tp_shard(t, 0, 1, 2), t[2:4])
    assert torch.equal(mtp_flash.tp_shard(t, 1, 0, 2), t[:, :2])
    # Shards partition the tensor.
    parts = [mtp_flash.tp_shard(t, 0, r, 4) for r in range(4)]
    assert torch.equal(torch.cat(parts, dim=0), t)


#
# Windowed draft attention
#


def _naive_dspark_attention(
    q, window_kv, window_pos, draft_kv, positions, sink, scale, window
):
    """Per-row brute-force softmax over visible keys + self key + sink."""
    num_tokens, heads, head_dim = q.shape
    out = torch.zeros_like(q, dtype=torch.float32)
    for t in range(num_tokens):
        pos = positions[t].item()
        keys = []
        for s in range(window):
            p = window_pos[s].item()
            if 0 <= p <= pos and p > pos - window:
                keys.append(window_kv[s])
        keys.append(draft_kv[t])
        kmat = torch.stack(keys).to(torch.float32)
        for h in range(heads):
            logits = kmat @ q[t, h].to(torch.float32) * scale
            logits = torch.cat([logits, sink[h : h + 1].to(torch.float32)])
            probs = torch.softmax(logits, dim=-1)
            out[t, h] = probs[:-1] @ kmat
    return out


def test_dspark_window_attention_matches_naive(mtp_flash):
    torch.manual_seed(5)
    num_tokens, heads, head_dim, window = 3, 2, 8, 4
    q = torch.randn(num_tokens, heads, head_dim)
    window_kv = torch.randn(window, head_dim)
    window_pos = torch.tensor([8, 9, 6, 7])  # slot s holds position p=s (mod 4)
    draft_kv = torch.randn(num_tokens, head_dim)
    positions = torch.tensor([7, 8, 9])
    sink = torch.randn(heads)
    out = mtp_flash.dspark_window_attention(
        q, window_kv, window_pos, draft_kv, positions, sink, 0.3, window
    )
    naive = _naive_dspark_attention(
        q, window_kv, window_pos, draft_kv, positions, sink, 0.3, window
    )
    torch.testing.assert_close(out.to(torch.float32), naive, atol=1e-5, rtol=1e-5)


def test_dspark_window_attention_empty_window(mtp_flash):
    """With no valid window slots, output = softmax over self key + sink."""
    q = torch.ones(1, 1, 4)
    window_kv = torch.zeros(3, 4)
    window_pos = torch.full((3,), -1, dtype=torch.int64)
    draft_kv = torch.full((1, 4), 2.0)
    positions = torch.tensor([0])
    sink = torch.zeros(1)
    out = mtp_flash.dspark_window_attention(
        q, window_kv, window_pos, draft_kv, positions, sink, 1.0, 3
    )
    # logit(self) = q . k = 8, logit(sink) = 0 -> weight ~ sigmoid(8).
    weight = math.exp(8.0) / (math.exp(8.0) + 1.0)
    torch.testing.assert_close(
        out, torch.full((1, 1, 4), 2.0 * weight), atol=1e-4, rtol=1e-4
    )
