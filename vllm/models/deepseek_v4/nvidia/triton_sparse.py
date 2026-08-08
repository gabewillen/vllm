# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse-MLA attention backend for DeepSeek V4 (SM 8.0+).

Plan / structure:

- FlashMLA's decode/prefill kernels only ship sm90a/sm100 cubins, so on
  Ada (SM 8.9) and Ampere the model cannot build an attention backend. This
  backend fills that gap with an arch-generic Triton kernel.
- It reuses the FlashMLA V4 pipeline end to end: the same ``fp8_ds_mla``
  paged KV layout (584 B/token slot), the same SWA cache + compressor +
  indexer machinery, and the same metadata builders. Only the attention
  forward differs.
- Index preparation reuses ``build_flashinfer_mixed_sparse_indices`` (via
  ``DeepseekV4FlashInferMLAAttention._build_sparse_index_metadata``): one
  per-token row of global slot ids — ``window_size`` SWA columns followed by
  top-k / compressed columns, ``-1`` invalid — covering decode and prefill
  for all three layer types (SWA-only / C4A / C128A) and MTP multi-token
  decodes alike.
- The single Triton kernel (``dsv4_sparse_attn_fp8``) gathers the referenced
  slots from the two paged caches, dequantizes fp8_ds_mla inline, and runs
  sink-augmented online-softmax attention with fp32 accumulation.
- The o-projection inherits ``deep_gemm_fp8_o_proj``, which already falls
  back to the BF16 path when DeepGEMM is unsupported (e.g. SM 8.9).
"""

from typing import cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops.sparse_attn_triton import (
    dsv4_sparse_attn_fp8,
)
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLAAttention,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV4TritonSparseBackend(DeepseekV4FlashMLABackend):
    """FlashMLA-compatible V4 backend served by the Triton sparse kernel.

    Inherits the FlashMLA metadata builder and ``fp8_ds_mla`` KV-cache layout;
    only the compute-capability gate differs (Triton needs bf16, so SM 8.0+).
    """

    @staticmethod
    def get_name() -> str:
        return "TRITON_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


class DeepseekV4TritonSparseAttention(DeepseekV4FlashInferMLAAttention):
    """Triton sparse-MLA attention layer for DeepSeek V4 (SM 8.0+).

    Subclasses the FlashInfer variant only to reuse its sparse-index
    preparation (``_build_sparse_index_metadata``); the KV-cache layout is
    FlashMLA's packed ``fp8_ds_mla`` block format and the attention forward
    is the Triton kernel.
    """

    backend_cls = DeepseekV4TritonSparseBackend
    use_flashmla_fp8_layout = True

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # The Triton kernel masks head tiles, so no padding is needed.
        return num_heads

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert q.dtype == torch.bfloat16, (
            f"Triton sparse DSV4 requires bf16 q, got {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # Warmup dummy run: the kernel reads the caches directly and
            # needs no workspace.
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            DeepseekSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        num_tokens = swa_metadata.num_decode_tokens + swa_metadata.num_prefill_tokens
        if num_tokens == 0:
            return

        swa_only = self.compress_ratio <= 1
        # SWA-only layers don't allocate their own compressed KV cache.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        (
            comp_kv_cache,
            _seq_lens,
            sparse_indices,
            sparse_topk_lens,
        ) = self._build_sparse_index_metadata(
            kv_cache=self_kv_cache,
            swa_k_cache=swa_kv_cache,
            swa_metadata=swa_metadata,
            attn_metadata=flashmla_metadata,
            swa_only=swa_only,
        )

        # CUDA graph execution can pad q/output past the scheduled token
        # count; restrict to the real tokens. One launch covers decode and
        # prefill: causality and top-k selection are baked into the per-token
        # index rows.
        dsv4_sparse_attn_fp8(
            q=q[:num_tokens],
            swa_kv_cache=swa_kv_cache,
            comp_kv_cache=None if swa_only else comp_kv_cache,
            sparse_indices=sparse_indices[:num_tokens],
            sparse_topk_lens=sparse_topk_lens[:num_tokens],
            attn_sink=self.attn_sink,
            softmax_scale=self.scale,
            window_size=self.window_size,
            out=output[:num_tokens],
        )
