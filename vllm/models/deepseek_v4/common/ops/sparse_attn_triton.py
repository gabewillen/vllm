# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arch-generic Triton sparse-MLA attention over the DSV4 fp8_ds_mla cache.

One kernel serves decode and prefill for all three DeepseekV4 layer types
(SWA-only / C4A / C128A). Each query token carries one row of
``sparse_indices`` (built by ``build_flashinfer_mixed_sparse_indices``): the
first ``window_size`` columns are global slot ids into the paged SWA cache,
the remaining columns are global slot ids into the paged compressed cache,
with ``-1`` marking invalid entries. ``sparse_topk_lens`` upper-bounds the
column range that can contain valid entries.

The kernel gathers each referenced 584-byte ``fp8_ds_mla`` token slot
(448 UE8M0 block-scaled fp8 NoPE bytes + 64 bf16 RoPE values + 8 scale
bytes), dequantizes it inline to bf16, and runs sink-augmented online-softmax
attention (fp32 accumulation): the learned per-head sink joins the softmax
denominator as an always-present, unscaled logit.
"""

import torch

from vllm.triton_utils import LOG2E, tl, triton

# fp8_ds_mla per-token slot layout inside a cache block of B tokens:
#   [0, B*576):        per-token data, 448 fp8 NoPE bytes + 64 bf16 RoPE
#   [B*576, B*584):    per-token scales, 8 UE8M0 bytes (7 real + 1 pad)
_TOKEN_DATA_SIZE = tl.constexpr(576)
_NOPE_DIM = tl.constexpr(448)
_ROPE_DIM = tl.constexpr(64)
_HEAD_DIM = tl.constexpr(512)
_SCALE_DIM = tl.constexpr(8)


@triton.jit
def _attend_block(
    q_nope,  # [BLOCK_H, 512] bf16, cols >= 448 zeroed
    q_rope,  # [BLOCK_H, 64] bf16
    idx,  # [BLOCK_N] int32 global slot ids, -1 invalid
    cache_ptr,  # uint8 base pointer of the paged fp8_ds_mla cache
    cache_block_size,
    cache_block_stride,
    e_max,  # [BLOCK_H] fp32 running max (base-2 logit domain)
    e_sum,  # [BLOCK_H] fp32 running denominator
    acc,  # [BLOCK_H, 512] fp32 (cols >= 448 stay zero)
    acc_rope,  # [BLOCK_H, 64] fp32
    sm_scale_log2,
    BLOCK_N: tl.constexpr,
):
    valid = idx >= 0
    idx64 = tl.where(valid, idx, 0).to(tl.int64)
    blk = idx64 // cache_block_size
    pos = idx64 % cache_block_size
    row_base = blk * cache_block_stride + pos * _TOKEN_DATA_SIZE
    scale_base = (
        blk * cache_block_stride
        + cache_block_size.to(tl.int64) * _TOKEN_DATA_SIZE
        + pos * _SCALE_DIM
    )

    # Dequantize the NoPE fp8 bytes: scale_i = 2^(byte_i - 127) per 64-elem
    # quant block. Columns >= 448 are masked to zero.
    d = tl.arange(0, _HEAD_DIM)
    nope_mask = d < _NOPE_DIM
    load_mask = valid[:, None] & nope_mask[None, :]
    b_u8 = tl.load(cache_ptr + row_base[:, None] + d[None, :], mask=load_mask, other=0)
    f = b_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    s_u8 = tl.load(
        cache_ptr + scale_base[:, None] + (d // 64)[None, :], mask=load_mask, other=127
    )
    k_nope = (f * tl.exp2(s_u8.to(tl.float32) - 127.0)).to(tl.bfloat16)

    # RoPE bf16 values, reassembled from byte pairs (little-endian) so the
    # kernel needs no dtype-punned view of the uint8 cache.
    j = tl.arange(0, _ROPE_DIM)
    rope_off = row_base[:, None] + _NOPE_DIM + 2 * j[None, :]
    lo = tl.load(cache_ptr + rope_off, mask=valid[:, None], other=0)
    hi = tl.load(cache_ptr + rope_off + 1, mask=valid[:, None], other=0)
    k_rope = ((hi.to(tl.uint16) << 8) | lo.to(tl.uint16)).to(tl.bfloat16, bitcast=True)

    qk = tl.dot(q_nope, tl.trans(k_nope))
    qk = tl.dot(q_rope, tl.trans(k_rope), qk)
    qk = qk * sm_scale_log2
    qk = tl.where(valid[None, :], qk, -float("inf"))

    n_max = tl.maximum(e_max, tl.max(qk, 1))
    dead = n_max == -float("inf")  # no valid key seen yet
    rescale = tl.where(dead, 1.0, tl.exp2(e_max - n_max))
    p = tl.exp2(qk - tl.where(dead, 0.0, n_max)[:, None])
    p = tl.where(valid[None, :] & (~dead[:, None]), p, 0.0)
    p_bf16 = p.to(tl.bfloat16)
    acc = acc * rescale[:, None] + tl.dot(p_bf16, k_nope)
    acc_rope = acc_rope * rescale[:, None] + tl.dot(p_bf16, k_rope)
    e_sum = e_sum * rescale + tl.sum(p, 1)
    return n_max, e_sum, acc, acc_rope


@triton.jit
def _dsv4_sparse_attn_fp8_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    out_ptr,
    out_stride0,
    out_stride1,
    swa_cache_ptr,
    swa_block_size,
    swa_block_stride,
    comp_cache_ptr,
    comp_block_size,
    comp_block_stride,
    indices_ptr,
    indices_stride0,
    lens_ptr,
    sink_ptr,
    total_width,
    num_heads,
    sm_scale_log2,
    log2e,
    WINDOW: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h < num_heads

    d = tl.arange(0, _HEAD_DIM)
    nope_mask = d < _NOPE_DIM
    q_row = q_ptr + t * q_stride0 + h[:, None] * q_stride1
    q_full = tl.load(q_row + d[None, :], mask=mask_h[:, None], other=0.0)
    q_nope = tl.where(nope_mask[None, :], q_full, 0.0)
    j = tl.arange(0, _ROPE_DIM)
    q_rope = tl.load(q_row + _NOPE_DIM + j[None, :], mask=mask_h[:, None], other=0.0)

    e_max = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, _HEAD_DIM], dtype=tl.float32)
    acc_rope = tl.zeros([BLOCK_H, _ROPE_DIM], dtype=tl.float32)

    # Valid entries never appear at columns >= lens (rows are packed
    # valid-first within each of the SWA / compressed segments), so lens is a
    # safe per-token loop bound; -1 masking inside the loop is authoritative.
    bound = tl.minimum(tl.load(lens_ptr + t), total_width)
    idx_row = indices_ptr + t * indices_stride0
    for start in range(0, bound, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        idx = tl.load(idx_row + cols, mask=cols < total_width, other=-1)
        if start < WINDOW:
            # BLOCK_N divides WINDOW, so a block never straddles the boundary.
            e_max, e_sum, acc, acc_rope = _attend_block(
                q_nope,
                q_rope,
                idx,
                swa_cache_ptr,
                swa_block_size,
                swa_block_stride,
                e_max,
                e_sum,
                acc,
                acc_rope,
                sm_scale_log2,
                BLOCK_N=BLOCK_N,
            )
        else:
            e_max, e_sum, acc, acc_rope = _attend_block(
                q_nope,
                q_rope,
                idx,
                comp_cache_ptr,
                comp_block_size,
                comp_block_stride,
                e_max,
                e_sum,
                acc,
                acc_rope,
                sm_scale_log2,
                BLOCK_N=BLOCK_N,
            )

    # The sink is an always-present unscaled logit: it joins the softmax
    # denominator but contributes no value.
    sink = tl.load(sink_ptr + h, mask=mask_h, other=-float("inf")) * log2e
    n_max = tl.maximum(e_max, sink)
    dead = n_max == -float("inf")
    r = tl.where(dead, 0.0, tl.exp2(e_max - n_max))
    s = tl.where(dead, 0.0, tl.exp2(sink - n_max))
    denom = e_sum * r + s
    denom = tl.where(denom == 0.0, 1.0, denom)
    w = (r / denom)[:, None]

    out_row = out_ptr + t * out_stride0 + h[:, None] * out_stride1
    tl.store(
        out_row + d[None, :],
        (acc * w).to(tl.bfloat16),
        mask=mask_h[:, None] & nope_mask[None, :],
    )
    tl.store(
        out_row + _NOPE_DIM + j[None, :],
        (acc_rope * w).to(tl.bfloat16),
        mask=mask_h[:, None],
    )


def dsv4_sparse_attn_fp8(
    q: torch.Tensor,  # [num_tokens, num_heads, 512] bf16
    swa_kv_cache: torch.Tensor,  # [num_blocks, swa_block_size, 584] uint8
    comp_kv_cache: torch.Tensor | None,  # [num_blocks, comp_block_size, 584]
    sparse_indices: torch.Tensor,  # [num_tokens, window + padded_topk] int32
    sparse_topk_lens: torch.Tensor,  # [num_tokens] int32
    attn_sink: torch.Tensor,  # [num_heads] fp32
    softmax_scale: float,
    window_size: int,
    out: torch.Tensor,  # [num_tokens, num_heads, 512] bf16
) -> None:
    """Sink-augmented sparse-MLA attention over paged fp8_ds_mla caches.

    Args:
        q: Query tokens, RoPE already applied to the last 64 dims.
        swa_kv_cache: Paged SWA cache addressed by the first ``window_size``
            index columns.
        comp_kv_cache: Paged compressed cache addressed by the remaining
            columns; may be ``None`` for SWA-only layers (rows then have
            exactly ``window_size`` columns).
        sparse_indices: Per-token global slot ids, ``-1`` invalid.
        sparse_topk_lens: Per-token upper bound of the valid column range.
        attn_sink: Per-head learned sink logit (fp32).
        out: Written in place with the attention output (RoPE dims still
            rotated; the caller inverse-rotates them in the o-projection).
    """
    num_tokens, num_heads, head_dim = q.shape
    assert head_dim == _HEAD_DIM.value
    assert q.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert swa_kv_cache.dtype == torch.uint8
    assert sparse_indices.dtype == torch.int32
    total_width = sparse_indices.shape[1]

    # Best-measured config on SM 8.9 (L4); BLOCK_N must divide window_size.
    BLOCK_H = 32
    BLOCK_N = 32
    assert window_size % BLOCK_N == 0
    if comp_kv_cache is None:
        assert total_width <= window_size
        comp_kv_cache = swa_kv_cache  # never dereferenced

    if num_tokens == 0:
        return

    grid = (num_tokens, triton.cdiv(num_heads, BLOCK_H))
    _dsv4_sparse_attn_fp8_kernel[grid](
        q,
        q.stride(0),
        q.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        swa_kv_cache,
        swa_kv_cache.shape[1],
        swa_kv_cache.stride(0),
        comp_kv_cache,
        comp_kv_cache.shape[1],
        comp_kv_cache.stride(0),
        sparse_indices,
        sparse_indices.stride(0),
        sparse_topk_lens,
        attn_sink,
        total_width,
        num_heads,
        softmax_scale * LOG2E,
        LOG2E,
        WINDOW=window_size,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
