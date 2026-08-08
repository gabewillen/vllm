# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fallback kernels for the DSA lightning indexer FP8 MQA logits.

Drop-in equivalents of DeepGEMM's ``fp8_fp4_mqa_logits`` (unpaged prefill)
and ``fp8_fp4_paged_mqa_logits`` (paged decode) for the FP8 indexer-cache
configuration, used on CUDA devices where DeepGEMM has no kernels (e.g.
SM 8.9). The FP4 (MXFP4) Q/cache path is Blackwell-only and not supported
here.

Semantics (matching DeepGEMM):
  ``logits[m, n] = kv_scale[n] * sum_h(relu(q[m, h] . k[n]) * weights[m, h])``
with fp32 accumulation. FP8 products are computed exactly (fp8 values are
exact in fp16; tensor-core fp16 dots accumulate exact products in fp32).
Positions outside each row's valid window are written only when
``clean_logits`` is set (to ``-inf``); otherwise they are left
uninitialized, exactly like DeepGEMM with ``clean_logits=False``.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv


@triton.jit
def _mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    kv_scales_ptr,
    weights_ptr,
    ks_ptr,
    ke_ptr,
    out_ptr,
    n_kv,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_wm,
    stride_om,
    H: tl.constexpr,
    D: tl.constexpr,
    HB: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    n0 = tl.program_id(1) * BLOCK_N

    ks = tl.load(ks_ptr + m)
    ke = tl.load(ke_ptr + m)
    if n0 >= ke:
        return
    if n0 + BLOCK_N <= ks:
        return

    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HB)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H
    n_mask = offs_n < n_kv

    q = tl.load(
        q_ptr
        + m.to(tl.int64) * stride_qm
        + offs_h[:, None] * stride_qh
        + offs_d[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.float16)
    k = tl.load(
        kv_ptr + offs_n[:, None].to(tl.int64) * stride_kn + offs_d[None, :],
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float16)

    # [HB, BLOCK_N] fp32 = exact fp8 products accumulated in fp32.
    s = tl.dot(q, tl.trans(k))
    s = tl.maximum(s, 0.0)

    w = tl.load(
        weights_ptr + m.to(tl.int64) * stride_wm + offs_h,
        mask=h_mask,
        other=0.0,
    )
    logits = tl.sum(s * w[:, None], axis=0)
    kv_scales = tl.load(kv_scales_ptr + offs_n, mask=n_mask, other=0.0)
    logits *= kv_scales

    out_mask = (offs_n >= ks) & (offs_n < ke) & n_mask
    tl.store(out_ptr + m.to(tl.int64) * stride_om + offs_n, logits, mask=out_mask)


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Compute unpaged FP8 MQA logits (DeepGEMM ``fp8_fp4_mqa_logits``).

    Args:
        q: [M, H, D] float8_e4m3fn queries (per-token Q scale is folded
            into ``weights``).
        kv: [N, D] float8_e4m3fn keys.
        kv_scales: [N] float32 per-key dequant scales.
        weights: [M, H] float32 per-query-per-head weights.
        cu_seqlen_ks: [M] int32 inclusive valid-K start per query.
        cu_seqlen_ke: [M] int32 exclusive valid-K end per query.
        clean_logits: Fill positions outside [ks, ke) with ``-inf``.

    Returns:
        [M, N] float32 logits; positions outside each row's window are
        ``-inf`` when ``clean_logits`` else uninitialized.
    """
    M, H, D = q.shape
    N = kv.shape[0]
    if clean_logits:
        out = torch.full((M, N), float("-inf"), dtype=torch.float32, device=q.device)
    else:
        out = torch.empty((M, N), dtype=torch.float32, device=q.device)
    if M == 0 or N == 0:
        return out

    BLOCK_N = 128
    _mqa_logits_kernel[(M, cdiv(N, BLOCK_N))](
        q,
        kv,
        kv_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        N,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        weights.stride(0),
        out.stride(0),
        H=H,
        D=D,
        HB=max(16, triton.next_power_of_2(H)),
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr,
    kv_values_ptr,
    kv_scales_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    out_ptr,
    stride_qr,
    stride_qh,
    stride_cl,
    stride_bt,
    stride_wr,
    stride_or,
    NEXT_N: tl.constexpr,
    BS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HB: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    r = tl.program_id(0)
    b = r // NEXT_N
    j = r % NEXT_N
    n0 = tl.program_id(1) * BLOCK_N

    ctx = tl.load(context_lens_ptr + b * stride_cl + j)
    if n0 >= ctx:
        return

    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HB)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H
    n_mask = offs_n < ctx

    # One cache block per tile (BS % BLOCK_N == 0). Block layout: BS * D
    # fp8 values, then BS float32 scales (see indexer_k_quant_and_cache).
    pb = tl.load(block_table_ptr + b * stride_bt + n0 // BS).to(tl.int64)
    p = offs_n - (n0 // BS) * BS

    q = tl.load(
        q_ptr
        + r.to(tl.int64) * stride_qr
        + offs_h[:, None] * stride_qh
        + offs_d[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.float16)
    k = tl.load(
        kv_values_ptr + pb * (BS * (D + 4)) + p[:, None] * D + offs_d[None, :],
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float16)

    s = tl.dot(q, tl.trans(k))
    s = tl.maximum(s, 0.0)

    w = tl.load(
        weights_ptr + r.to(tl.int64) * stride_wr + offs_h,
        mask=h_mask,
        other=0.0,
    )
    logits = tl.sum(s * w[:, None], axis=0)
    kv_scales = tl.load(
        kv_scales_ptr + pb * (BS * (D + 4) // 4) + BS * (D // 4) + p,
        mask=n_mask,
        other=0.0,
    )
    logits *= kv_scales

    tl.store(out_ptr + r.to(tl.int64) * stride_or + offs_n, logits, mask=n_mask)


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Compute paged FP8 MQA logits (DeepGEMM ``fp8_fp4_paged_mqa_logits``).

    Args:
        q: [B, next_n, H, D] float8_e4m3fn queries (per-token Q scale is
            folded into ``weights``).
        kv_cache: [num_blocks, block_size, 1, D + 4] paged FP8 indexer
            cache. Per block the physical layout is ``block_size * D`` fp8
            values followed by ``block_size`` float32 scales.
        weights: [B * next_n, H] float32 per-query-per-head weights.
        context_lens: [B, next_n] int32 effective context length per token.
        block_tables: [B, max_blocks] int32 logical-to-physical block map.
        max_model_len: Number of columns of the logits output.
        clean_logits: Fill positions at or beyond each row's context length
            with ``-inf``.

    Returns:
        [B * next_n, max_model_len] float32 logits; columns at or beyond
        each row's context length are ``-inf`` when ``clean_logits`` else
        uninitialized.
    """
    B, next_n, H, D = q.shape
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    assert kv_cache.shape[-1] == D + 4, "FP4 indexer cache is not supported"
    assert context_lens.ndim == 2 and context_lens.shape[0] == B
    assert context_lens.shape[1] == next_n

    kv_bytes = kv_cache.reshape(num_blocks, -1).view(torch.uint8)
    kv_values = kv_bytes.view(torch.float8_e4m3fn)
    kv_scales = kv_bytes.view(torch.float32)

    if clean_logits:
        out = torch.full(
            (B * next_n, max_model_len),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        out = torch.empty(
            (B * next_n, max_model_len), dtype=torch.float32, device=q.device
        )
    if B == 0:
        return out

    BLOCK_N = min(64, block_size)
    assert block_size % BLOCK_N == 0
    q = q.reshape(B * next_n, H, D)
    _paged_mqa_logits_kernel[(B * next_n, cdiv(max_model_len, BLOCK_N))](
        q,
        kv_values,
        kv_scales,
        weights,
        context_lens,
        block_tables,
        out,
        q.stride(0),
        q.stride(1),
        context_lens.stride(0),
        block_tables.stride(0),
        weights.stride(0),
        out.stride(0),
        NEXT_N=next_n,
        BS=block_size,
        H=H,
        D=D,
        HB=max(16, triton.next_power_of_2(H)),
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out
