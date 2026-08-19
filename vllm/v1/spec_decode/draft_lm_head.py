# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantized copies of a shared target lm_head for speculative drafters.

MTP / EAGLE drafters that share the target's lm_head re-read the whole
(vocab / tp) x hidden shard on every draft step for one argmax. On
memory-bound GPUs that read dominates the draft step, so the drafter gets
its own fp8 (per-channel) or int4 (Marlin, group-128) copy. The target still
verifies with its unquantized head, so the output distribution is unchanged.
"""

import torch
import torch.nn as nn
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    check_marlin_supported,
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_pack_factor,
    quantize_weights,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_fp8_supported,
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

MARLIN_GROUP_SIZE = 128
MARLIN_WEIGHT_TYPE = scalar_types.uint4b8


class _Fp8DraftHeadMethod:
    """quant_method shim: dynamic per-token FP8 activations x per-channel FP8
    weights through the CUTLASS scaled_mm (SM89+)."""

    def apply(
        self,
        layer: "QuantizedDraftLMHead",
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.shape[-1])
        x_q, x_s = ops.scaled_fp8_quant(x2d, use_per_token_if_dynamic=True)
        out = ops.cutlass_scaled_mm(
            x_q,
            layer.weight_fp8_t,
            scale_a=x_s,
            scale_b=layer.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )
        return out.view(*x.shape[:-1], out.shape[-1])


class _Int4DraftHeadMethod:
    """quant_method shim: bf16 activations x group-128 INT4 weights through the
    Marlin W4A16 kernel."""

    def apply(
        self,
        layer: "QuantizedDraftLMHead",
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.marlin_qweight,
            weight_scale=layer.marlin_scales,
            weight_zp=layer.marlin_zp,
            g_idx=layer.marlin_g_idx,
            g_idx_sort_indices=layer.marlin_sort_indices,
            workspace=layer.marlin_workspace,
            wtype=MARLIN_WEIGHT_TYPE,
            output_size_per_partition=layer.output_size,
            input_size_per_partition=layer.input_size,
            is_k_full=True,
            bias=bias,
        )


def _pack_rows_gptq(q_w: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack unpacked [K, N] integer weights into GPTQ int32 rows [K/pack, N]."""
    pack_factor = get_pack_factor(num_bits)
    size_k, size_n = q_w.shape
    if size_k % pack_factor:
        raise ValueError(f"K={size_k} is not a multiple of {pack_factor}")
    q_w = q_w.to(torch.int32)
    packed = torch.zeros(
        (size_k // pack_factor, size_n), dtype=torch.int32, device=q_w.device
    )
    for i in range(pack_factor):
        packed |= q_w[i::pack_factor, :] << (num_bits * i)
    return packed


class QuantizedDraftLMHead(nn.Module):
    """Quantized (fp8 per-channel or int4 group-128) copy of the target lm_head
    shard, used only by the drafter. Everything except the weights delegates
    to the target head, so vocab-parallel argmax and logits code work
    unchanged."""

    def __init__(self, target_head: nn.Module, dtype: str = "fp8"):
        super().__init__()
        object.__setattr__(self, "_target_head", target_head)
        w = target_head.weight.data
        self.output_size, self.input_size = w.shape
        if dtype == "fp8":
            if not cutlass_fp8_supported():
                raise ValueError(
                    "draft_lm_head_dtype='fp8' needs CUTLASS FP8 scaled_mm "
                    "support (SM89+) on this device."
                )
            self._init_fp8(w)
            self.quant_method = _Fp8DraftHeadMethod()
        elif dtype == "int4":
            if not check_marlin_supported(
                quant_type=MARLIN_WEIGHT_TYPE, group_size=MARLIN_GROUP_SIZE
            ):
                raise ValueError(
                    "draft_lm_head_dtype='int4' needs Marlin uint4b8 group-128 "
                    "support on this device."
                )
            self._init_int4(w)
            self.quant_method = _Int4DraftHeadMethod()
        else:
            raise ValueError(f"unsupported draft_lm_head_dtype {dtype!r}")

    def _init_fp8(self, w: torch.Tensor) -> None:
        finfo = torch.finfo(torch.float8_e4m3fn)
        amax = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12)
        weight_scale = amax / finfo.max
        w_fp8 = (
            (w.float() / weight_scale)
            .clamp(finfo.min, finfo.max)
            .to(torch.float8_e4m3fn)
        )
        # cutlass_scaled_mm wants B as [K, N] column-major = [N, K].t()
        self.weight_fp8_t = w_fp8.contiguous().t()
        self.weight_scale = weight_scale.contiguous()

    def _init_int4(self, w: torch.Tensor) -> None:
        # Round-to-nearest symmetric group quantization ([K, N] layout), then
        # the same GPTQ pack -> Marlin repack path the GPTQ-Marlin loader uses.
        size_n, size_k = w.shape
        _, q_w, scales, _ = quantize_weights(
            w=w.t().contiguous().float(),
            quant_type=MARLIN_WEIGHT_TYPE,
            group_size=MARLIN_GROUP_SIZE,
        )
        empty = torch.empty(0, dtype=torch.int32, device=w.device)
        self.marlin_qweight = ops.gptq_marlin_repack(
            b_q_weight=_pack_rows_gptq(q_w=q_w, num_bits=MARLIN_WEIGHT_TYPE.size_bits),
            perm=empty,
            size_k=size_k,
            size_n=size_n,
            num_bits=MARLIN_WEIGHT_TYPE.size_bits,
        )
        self.marlin_scales = marlin_permute_scales(
            s=scales.to(w.dtype),
            size_k=size_k,
            size_n=size_n,
            group_size=MARLIN_GROUP_SIZE,
        )
        self.marlin_zp = empty
        self.marlin_g_idx = empty
        self.marlin_sort_indices = empty
        self.marlin_workspace = marlin_make_workspace_new(w.device)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_target_head"), name)


def maybe_quantize_shared_lm_head(
    draft_model: nn.Module, target_lm_head: nn.Module, dtype: str
) -> None:
    """Replace ``draft_model.lm_head`` (shared with the target) by a quantized
    copy when ``dtype`` asks for one. No-op for ``auto``. Callers that
    re-point per-layer ``shared_head.head`` copies must do so after this call
    and point them at ``draft_model.lm_head``."""
    if dtype == "auto":
        return
    draft_model.lm_head = QuantizedDraftLMHead(target_head=target_lm_head, dtype=dtype)
    logger.info(
        "Draft lm_head: %s copy of the target lm_head shard %s for the "
        "drafter; the target keeps %s.",
        dtype,
        tuple(target_lm_head.weight.shape),
        target_lm_head.weight.dtype,
    )
