# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton FP8 MQA logits fallback of the sparse attn indexer."""

import random

import pytest
import torch

from vllm.model_executor.layers.sparse_attn_indexer_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

COMPRESS_RATIO = 4
NUM_HEADS = 64
HEAD_DIM = 128

requires_cuda = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")


def _quant_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token FP8 quantization: returns ([N, D] fp8, [N] fp32 scales)."""
    amax = x.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    sf = amax / 448.0
    x_fp8 = (x.float() / sf).to(torch.float8_e4m3fn)
    return x_fp8, sf.squeeze(-1)


def _ref_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantized fp64 einsum reference; returns (logits, valid mask)."""
    q = q_fp8.to(torch.float64)
    k = kv_fp8.to(torch.float64) * kv_scales.to(torch.float64)[:, None]
    score = torch.einsum("mhd,nd->mhn", q, k)
    logits = (score.clamp(min=0) * weights.to(torch.float64)[:, :, None]).sum(1)
    n = torch.arange(kv_fp8.shape[0], device=q.device)
    mask = (n[None, :] >= ks[:, None]) & (n[None, :] < ke[:, None])
    return logits.float(), mask


def _make_prefill_inputs(device: torch.device):
    """Multi-request chunk with causal masking at 4:1 compression."""
    query_lens = [33, 64, 5]
    seq_lens = [61, 128, 5]
    ks_list, ke_list = [], []
    seq_start = 0
    for q_len, seq_len in zip(query_lens, seq_lens):
        start_pos = seq_len - q_len
        n_kv = seq_len // COMPRESS_RATIO
        for i in range(q_len):
            ks_list.append(seq_start)
            ke_list.append(seq_start + (start_pos + 1 + i) // COMPRESS_RATIO)
        seq_start += n_kv
    m = sum(query_lens)
    n = seq_start
    ks = torch.tensor(ks_list, dtype=torch.int32, device=device)
    ke = torch.tensor(ke_list, dtype=torch.int32, device=device)

    q_fp8 = (torch.randn(m, NUM_HEADS, HEAD_DIM, device=device) * 3.0).to(
        torch.float8_e4m3fn
    )
    # Non-uniform per-token magnitudes to exercise the kv scales.
    kv = torch.randn(n, HEAD_DIM, device=device)
    kv *= torch.exp2(torch.empty(n, 1, device=device).uniform_(-8.0, 8.0))
    kv_fp8, kv_scales = _quant_per_token(kv)
    weights = torch.randn(m, NUM_HEADS, device=device, dtype=torch.float32)
    return q_fp8, kv_fp8, kv_scales, weights, ks, ke


@requires_cuda
@pytest.mark.parametrize("clean_logits", [False, True])
def test_triton_mqa_logits_prefill(clean_logits: bool):
    torch.manual_seed(0)
    device = torch.device("cuda")
    q_fp8, kv_fp8, kv_scales, weights, ks, ke = _make_prefill_inputs(device)

    logits = fp8_mqa_logits_triton(
        q_fp8, kv_fp8, kv_scales, weights, ks, ke, clean_logits=clean_logits
    )
    ref_logits, mask = _ref_mqa_logits(q_fp8, kv_fp8, kv_scales, weights, ks, ke)
    assert logits.shape == ref_logits.shape
    assert logits.dtype == torch.float32

    if clean_logits:
        # Exact -inf everywhere outside each row's [ks, ke) window,
        # including fully-empty rows (ke == ks).
        assert torch.equal(logits == float("-inf"), ~mask)
    torch.testing.assert_close(
        logits.masked_fill(~mask, 0),
        ref_logits.masked_fill(~mask, 0),
        rtol=1e-3,
        atol=1e-3,
    )


@requires_cuda
@pytest.mark.parametrize("block_size", [64, 256])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("clean_logits", [False, True])
def test_triton_paged_mqa_logits_decode(
    block_size: int, next_n: int, clean_logits: bool
):
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda")
    batch_size, max_model_len = 4, 1024

    context_lens = torch.tensor(
        [random.randint(next_n + 1, max_model_len - 1) for _ in range(batch_size)],
        dtype=torch.int32,
        device=device,
    )
    # 2D per-token context lens, as built by the indexer metadata builder.
    context_lens_2d = (
        context_lens[:, None]
        - next_n
        + 1
        + torch.arange(next_n, dtype=torch.int32, device=device)[None, :]
    ).contiguous()

    # Paged cache: per block, block_size * D fp8 values then block_size
    # fp32 scales.
    num_blocks = sum(cdiv(int(c), block_size) for c in context_lens.tolist()) + 3
    kv = torch.randn(num_blocks * block_size, HEAD_DIM, device=device)
    kv *= torch.exp2(
        torch.empty(num_blocks * block_size, 1, device=device).uniform_(-8.0, 8.0)
    )
    kv_fp8, kv_scales = _quant_per_token(kv)
    cache = torch.empty(
        num_blocks, block_size * (HEAD_DIM + 4), dtype=torch.uint8, device=device
    )
    cache[:, : block_size * HEAD_DIM] = kv_fp8.view(
        num_blocks, block_size * HEAD_DIM
    ).view(torch.uint8)
    cache[:, block_size * HEAD_DIM :] = kv_scales.view(num_blocks, block_size).view(
        torch.uint8
    )
    kv_cache = cache.view(torch.float8_e4m3fn).view(
        num_blocks, block_size, 1, HEAD_DIM + 4
    )

    # Nontrivial block table: shuffled physical blocks.
    max_blocks = cdiv(max_model_len, block_size)
    block_tables = torch.zeros(
        (batch_size, max_blocks), dtype=torch.int32, device=device
    )
    pool = list(range(num_blocks))
    random.shuffle(pool)
    counter = 0
    for b in range(batch_size):
        for i in range(cdiv(int(context_lens[b]), block_size)):
            block_tables[b, i] = pool[counter]
            counter += 1

    q_fp8 = (
        torch.randn(batch_size, next_n, NUM_HEADS, HEAD_DIM, device=device) * 3.0
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        batch_size * next_n, NUM_HEADS, device=device, dtype=torch.float32
    )

    logits = fp8_paged_mqa_logits_triton(
        q_fp8,
        kv_cache,
        weights,
        context_lens_2d,
        block_tables,
        max_model_len,
        clean_logits=clean_logits,
    )
    assert logits.shape == (batch_size * next_n, max_model_len)
    assert logits.dtype == torch.float32

    values_all = (
        cache[:, : block_size * HEAD_DIM]
        .view(torch.float8_e4m3fn)
        .view(num_blocks, block_size, HEAD_DIM)
    )
    scales_all = (
        cache[:, block_size * HEAD_DIM :]
        .view(torch.float32)
        .view(num_blocks, block_size)
    )
    for b in range(batch_size):
        for j in range(next_n):
            r = b * next_n + j
            ctx = int(context_lens_2d[b, j])
            t = torch.arange(ctx, device=device)
            pb = block_tables[b, t // block_size].long()
            pos = t % block_size
            k = values_all[pb, pos].to(torch.float64)
            k *= scales_all[pb, pos].to(torch.float64)[:, None]
            qr = q_fp8[b, j].to(torch.float64)
            score = torch.einsum("hd,nd->hn", qr, k)
            ref_row = (
                (score.clamp(min=0) * weights[r].to(torch.float64)[:, None])
                .sum(0)
                .float()
            )
            torch.testing.assert_close(logits[r, :ctx], ref_row, rtol=1e-3, atol=1e-3)
            if clean_logits:
                assert torch.all(logits[r, ctx:] == float("-inf"))


@requires_cuda
def test_triton_dispatch_gate(monkeypatch: pytest.MonkeyPatch):
    from vllm.model_executor.layers import sparse_attn_indexer as sai

    try:
        monkeypatch.setattr(sai, "is_deep_gemm_supported", lambda: False)
        sai._use_triton_indexer_logits.cache_clear()
        assert sai._use_triton_indexer_logits() is True
        monkeypatch.setattr(sai, "is_deep_gemm_supported", lambda: True)
        sai._use_triton_indexer_logits.cache_clear()
        assert sai._use_triton_indexer_logits() is False
    finally:
        sai._use_triton_indexer_logits.cache_clear()
