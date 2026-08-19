# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.draft_lm_head import maybe_quantize_shared_lm_head

logger = init_logger(__name__)


def _should_share(eagle: nn.Module, flag: str, draft, target) -> bool:
    """Share when the draft has no own copy, or its copy matches the target."""

    if not getattr(eagle, flag, False) or draft is None:
        return True
    if target is None:
        return False
    # torch.equal on GPU allocates a bool mask the size of the input.
    # Use the faster GPU path when there is plenty of headroom;
    # otherwise compare on CPU.
    w = draft.weight
    if w.is_cuda and torch.accelerator.get_memory_info(w.device)[0] < w.numel() * 2:
        return torch.equal(w.cpu(), target.weight.cpu())
    return torch.equal(w, target.weight)


def get_target_lm_head(target_model: nn.Module, target_language_model: nn.Module):
    """The target's lm_head — from get_language_model() for
    *ForConditionalGeneration targets, else the top-level module."""
    return getattr(target_language_model, "lm_head", None) or getattr(
        target_model, "lm_head", None
    )


def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    if speculative_config.kv_cache_dtype is not None:
        vllm_config = replace(
            vllm_config,
            cache_config=replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            ),
        )
    with set_model_tag("eagle_head"):
        eagle_model = get_model(
            vllm_config=vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = eagle_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        # If the target's embedding is LoRA-wrapped, share the underlying base
        # layer. The draft is not part of the LoRA adapter; sharing the wrapper
        # would make the draft run the LoRA embedding kernel with the target's
        # punica metadata (sized for the target's token count), causing an
        # out-of-bounds GPU access during multi-step draft decode.
        if isinstance(target_embed, BaseLayerWithLoRA):
            target_embed = target_embed.base_layer
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            eagle_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(eagle_model, "lm_head", None)
    share_lm_head = target_lm_head is not None and _should_share(
        eagle_model, "has_own_lm_head", draft_lm_head, target_lm_head
    )
    spec_cfg = vllm_config.speculative_config
    if (
        not share_lm_head
        and spec_cfg is not None
        and spec_cfg.draft_lm_head_dtype != "auto"
    ):
        logger.warning(
            "draft_lm_head_dtype=%s only applies to a shared target lm_head; "
            "the drafter keeps its own head.",
            spec_cfg.draft_lm_head_dtype,
        )
    if share_lm_head:
        if draft_lm_head is not None:
            del eagle_model.lm_head
        eagle_model.lm_head = target_lm_head
        if spec_cfg is not None:
            maybe_quantize_shared_lm_head(
                draft_model=eagle_model,
                target_lm_head=target_lm_head,
                dtype=spec_cfg.draft_lm_head_dtype,
            )

        # MTP layers route logits through layer.shared_head.head, not
        # eagle_model.lm_head, so the per-layer copies need fixing up too.
        layers = getattr(draft_inner, "layers", None)
        if layers is not None:
            items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
            for layer in items:
                sh = getattr(layer, "shared_head", None)
                if sh is not None and hasattr(sh, "head"):
                    del sh.head
                    sh.head = eagle_model.lm_head

    # MTP shares topk_indices_buffer with the target model. We update
    # every module in the draft that holds a buffer reference so that
    # the per-layer indexer and sparse-attention backends all point to
    # the target's buffer.
    if hasattr(target_inner, "topk_indices_buffer"):
        target_buffer = target_inner.topk_indices_buffer
        if target_buffer is not None:
            for _, module in draft_inner.named_modules():
                if hasattr(module, "topk_indices_buffer"):
                    module.topk_indices_buffer = target_buffer

    return eagle_model
