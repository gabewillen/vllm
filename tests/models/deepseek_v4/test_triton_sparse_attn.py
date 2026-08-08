# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DeepSeek-V4 Triton sparse-MLA attention backend (SM 8.0+).

The Triton kernel is compared against a pure-torch eager reference adapted
from the official DeepSeek-V4 ``sparse_attn`` semantics: per query token,
gather the latent KV rows named by the (-1 padded) index row, compute
scaled-dot-product logits, and softmax with the learned per-head sink as an
always-present, unscaled extra logit that contributes no value.
"""

import types

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    dequantize_and_gather_k_cache_triton,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.common.ops.sparse_attn_triton import (
    dsv4_sparse_attn_fp8,
)
from vllm.models.deepseek_v4.nvidia.triton_sparse import (
    DeepseekV4TritonSparseBackend,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
from vllm.platforms.interface import DeviceCapability

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = 448


def _make_cache(
    num_tokens: int,
    block_size: int,
    device: str,
    stride_pad_bytes: int = 0,
    seed: int = 0,
) -> torch.Tensor:
    """Random bf16 latent rows quantized into a paged fp8_ds_mla cache.

    Uses the canonical in-tree writer (``quantize_and_insert_k_cache``) so the
    kernel under test is checked against the production byte layout. An
    optional per-block stride padding emulates page-aligned allocations.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    rows = torch.randn(
        num_tokens, HEAD_DIM, generator=g, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    row_bytes = block_size * 584 + stride_pad_bytes
    flat = torch.zeros(num_blocks, row_bytes, dtype=torch.uint8, device=device)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
    quantize_and_insert_k_cache(rows, flat, slots, block_size=block_size)
    return torch.as_strided(flat, (num_blocks, block_size, 584), (row_bytes, 584, 1))


def _torch_decode(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Independent pure-torch decoder of the 584-byte fp8_ds_mla slot layout.

    Rounds the dequantized NoPE part to bf16 to match the kernel's compute
    dtype; returns fp32 rows.
    """
    _, block_size, _ = cache.shape
    out = torch.zeros(slots.numel(), HEAD_DIM, dtype=torch.float32, device=cache.device)
    for i, s in enumerate(slots.tolist()):
        blk, pos = divmod(s, block_size)
        data = cache[blk].reshape(-1)
        tok = data[pos * 576 : (pos + 1) * 576]
        scales = data[block_size * 576 + pos * 8 : block_size * 576 + pos * 8 + 8]
        nope = tok[:NOPE_DIM].view(torch.float8_e4m3fn).float()
        nope = nope * torch.pow(2.0, scales[:7].float() - 127).repeat_interleave(64)
        rope = tok[NOPE_DIM:576].view(torch.bfloat16).float()
        out[i] = torch.cat([nope.to(torch.bfloat16).float(), rope])
    return out


def _ref_sparse_attn(
    q: torch.Tensor,  # [T, H, 512] bf16
    swa_cache: torch.Tensor,
    comp_cache: torch.Tensor | None,
    indices: torch.Tensor,  # [T, width] int32
    lens: torch.Tensor,  # [T] int32
    sink: torch.Tensor,  # [H] fp32
    scale: float,
    window: int,
) -> torch.Tensor:
    """Eager fp32 reference (official sparse_attn semantics + sink)."""
    T, H, _ = q.shape
    out = torch.zeros(T, H, HEAD_DIM, dtype=torch.float32, device=q.device)
    for t in range(T):
        bound = int(lens[t])
        row = indices[t, :bound]
        swa_ids = row[:window]
        comp_ids = row[window:]
        rows = [_torch_decode(swa_cache, swa_ids[swa_ids >= 0])]
        if comp_cache is not None:
            rows.append(_torch_decode(comp_cache, comp_ids[comp_ids >= 0]))
        kv = torch.cat(rows)
        if kv.shape[0] == 0:
            continue
        logits = (q[t].float() @ kv.T) * scale
        m = torch.maximum(logits.max(-1).values, sink)
        p = torch.exp(logits - m[:, None])
        den = p.sum(-1) + torch.exp(sink - m)
        out[t] = (p @ kv) / den[:, None]
    return out


def _run_and_compare(
    q, swa_cache, comp_cache, indices, lens, sink, scale, window
) -> torch.Tensor:
    out = torch.empty_like(q)
    dsv4_sparse_attn_fp8(
        q, swa_cache, comp_cache, indices, lens, sink, scale, window, out
    )
    ref = _ref_sparse_attn(q, swa_cache, comp_cache, indices, lens, sink, scale, window)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    return out


def _rand_q(T: int, H: int, device: str, seed: int = 7) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        T, H, HEAD_DIM, generator=g, device=device, dtype=torch.float32
    ).to(torch.bfloat16)


@requires_cuda
def test_torch_decoder_matches_canonical_reader():
    """The test's own layout decoder must agree with the in-tree reader."""
    device = "cuda"
    n, bs = 96, 64
    cache = _make_cache(n, bs, device)
    ref = torch.zeros(1, n, HEAD_DIM, dtype=torch.bfloat16, device=device)
    dequantize_and_gather_k_cache_triton(
        ref,
        cache,
        seq_lens=torch.tensor([n], dtype=torch.int32, device=device),
        gather_lens=None,
        block_table=torch.arange(
            cache.shape[0], dtype=torch.int32, device=device
        ).unsqueeze(0),
        block_size=bs,
        offset=0,
    )
    mine = _torch_decode(cache, torch.arange(n))
    torch.testing.assert_close(mine.to(torch.bfloat16), ref[0])


@requires_cuda
def test_decode_c4a_two_caches_full_shape():
    """C4A decode: 64 heads, window-128 SWA cache + top-512 compressed cache."""
    device = "cuda"
    T, H, window, topk = 4, 64, 128, 512
    swa_cache = _make_cache(240, 64, device, seed=1)
    comp_cache = _make_cache(600, 64, device, seed=2)
    q = _rand_q(T, H, device)
    g = torch.Generator().manual_seed(3)
    sink = torch.randn(H, generator=g).to(device)

    width = window + topk
    indices = torch.full((T, width), -1, dtype=torch.int32, device=device)
    lens = torch.zeros(T, dtype=torch.int32, device=device)
    for t in range(T):
        swa_len = int(torch.randint(1, window + 1, (1,), generator=g))
        clen = int(torch.randint(0, topk + 1, (1,), generator=g))
        indices[t, :swa_len] = (
            torch.randperm(240, generator=g)[:swa_len].to(device).int()
        )
        indices[t, window : window + clen] = (
            torch.randperm(600, generator=g)[:clen].to(device).int()
        )
        # Builder semantics: lens = window + compressed_len; the SWA segment
        # may hold -1 holes below the window boundary.
        lens[t] = window + clen
    _run_and_compare(
        q, swa_cache, comp_cache, indices, lens, sink, HEAD_DIM**-0.5, window
    )


@requires_cuda
def test_decode_c128a_small_blocks_padded_stride():
    """C128A: 2-token compressed blocks and page-padded block stride."""
    device = "cuda"
    T, H, window = 2, 64, 128
    n_comp = 40
    swa_cache = _make_cache(200, 64, device, stride_pad_bytes=64, seed=4)
    comp_cache = _make_cache(n_comp, 2, device, stride_pad_bytes=16, seed=5)
    q = _rand_q(T, H, device, seed=8)
    g = torch.Generator().manual_seed(6)
    sink = torch.randn(H, generator=g).to(device)

    comp_width = 128  # aligned topk width with -1 tail
    indices = torch.full((T, window + comp_width), -1, dtype=torch.int32, device=device)
    lens = torch.zeros(T, dtype=torch.int32, device=device)
    for t in range(T):
        swa_len = 100 + t
        clen = n_comp - t
        indices[t, :swa_len] = torch.arange(swa_len, device=device).int()
        indices[t, window : window + clen] = torch.arange(clen, device=device).int()
        lens[t] = window + clen
    _run_and_compare(
        q, swa_cache, comp_cache, indices, lens, sink, HEAD_DIM**-0.5, window
    )


@requires_cuda
def test_swa_only_and_pad_token():
    """SWA-only layer rows (no compressed cache); lens=0 row yields zeros."""
    device = "cuda"
    T, H, window = 3, 64, 128
    swa_cache = _make_cache(160, 64, device, seed=9)
    q = _rand_q(T, H, device, seed=10)
    g = torch.Generator().manual_seed(11)
    sink = torch.randn(H, generator=g).to(device)

    indices = torch.full((T, window), -1, dtype=torch.int32, device=device)
    lens = torch.zeros(T, dtype=torch.int32, device=device)
    for t in range(T - 1):
        swa_len = 1 if t == 0 else window
        indices[t, :swa_len] = torch.arange(swa_len, device=device).int()
        lens[t] = window
    # Last row: pad token (lens 0) must produce exact zeros.
    out = _run_and_compare(
        q, swa_cache, None, indices, lens, sink, HEAD_DIM**-0.5, window
    )
    assert torch.all(out[-1] == 0)


@requires_cuda
def test_sink_dominates_small_logits():
    """A large sink shrinks outputs toward zero; must match the reference."""
    device = "cuda"
    T, H, window = 2, 64, 128
    swa_cache = _make_cache(64, 64, device, seed=12)
    q = _rand_q(T, H, device, seed=13) * 0.05
    sink = torch.full((H,), 8.0, device=device)

    indices = torch.full((T, window), -1, dtype=torch.int32, device=device)
    indices[:, :32] = torch.arange(32, device=device).int()
    lens = torch.full((T,), window, dtype=torch.int32, device=device)
    out = _run_and_compare(
        q, swa_cache, None, indices, lens, sink, HEAD_DIM**-0.5, window
    )
    # With sink logit 8 vs ~0 logits, softmax mass on values is ~exp(-8).
    assert out.float().abs().max() < 0.05


@requires_cuda
def test_small_head_count():
    """Head tiles are masked, so odd local head counts (TP) work."""
    device = "cuda"
    T, H, window = 2, 2, 128
    swa_cache = _make_cache(96, 64, device, seed=14)
    q = _rand_q(T, H, device, seed=15)
    g = torch.Generator().manual_seed(16)
    sink = torch.randn(H, generator=g).to(device)

    indices = torch.full((T, window), -1, dtype=torch.int32, device=device)
    indices[:, :96] = torch.arange(96, device=device).int()
    lens = torch.full((T,), window, dtype=torch.int32, device=device)
    _run_and_compare(q, swa_cache, None, indices, lens, sink, HEAD_DIM**-0.5, window)


def test_backend_capability_gates():
    sm89 = DeviceCapability(8, 9)
    assert DeepseekV4TritonSparseBackend.supports_compute_capability(sm89)
    assert DeepseekV4TritonSparseBackend.supports_compute_capability(
        DeviceCapability(9, 0)
    )
    assert not DeepseekV4TritonSparseBackend.supports_compute_capability(
        DeviceCapability(7, 5)
    )
    # The FlashMLA gate must remain untouched.
    assert not DeepseekV4FlashMLABackend.supports_compute_capability(sm89)
    assert DeepseekV4FlashMLABackend.supports_compute_capability(DeviceCapability(9, 0))


@requires_cuda
def test_attn_cls_selection():
    """On GPUs without FlashMLA sparse support the Triton class is selected."""
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls
    from vllm.models.deepseek_v4.nvidia.triton_sparse import (
        DeepseekV4TritonSparseAttention,
    )
    from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    cfg = types.SimpleNamespace(attention_config=types.SimpleNamespace(backend=None))
    selected = _select_dsv4_attn_cls(cfg)
    if is_flashmla_sparse_supported()[0]:
        from vllm.models.deepseek_v4.nvidia.flashmla import (
            DeepseekV4FlashMLAAttention,
        )

        assert selected is DeepseekV4FlashMLAAttention
    else:
        assert selected is DeepseekV4TritonSparseAttention

    from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferMLAAttention,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    cfg = types.SimpleNamespace(
        attention_config=types.SimpleNamespace(
            backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4
        )
    )
    assert _select_dsv4_attn_cls(cfg) is DeepseekV4FlashInferMLAAttention
