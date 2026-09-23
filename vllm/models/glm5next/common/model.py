# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import ClassVar, Literal

import torch
from torch import nn

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mhc import (
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
)

try:
    # Added after the 0.26 Gaudi base image was cut.
    from vllm.model_executor.layers.mhc import hc_contract, hc_expand
except ImportError:
    def hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
        return x.unsqueeze(1).expand(-1, n, -1).contiguous()

    def hc_contract(x: torch.Tensor, n: int) -> torch.Tensor:
        return x.mean(dim=1)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import _get_moe_router_dtype
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from .attention import Glm5NextMLAAttention
from .kda import Glm5NextLinearAttention
from .multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
    Glm5NextVisionTransformer,
)

logger = init_logger(__name__)

import os as _os

_DBG_DIR = _os.environ.get("GLM_DUMP_DIR")
_dbg_done: set = set()


def _dbg(name: str, t: torch.Tensor | None) -> None:
    """Debug: save the first occurrence of a tensor on TP rank 0."""
    if not _DBG_DIR or t is None:
        return
    if get_tensor_model_parallel_rank() != 0:
        return

    _os.makedirs(_DBG_DIR, exist_ok=True)
    torch.save(t.detach().float().cpu(), f"{_DBG_DIR}/{name}.pt")


def _ensure_glm5_config_compat(config) -> None:
    """Bridge the Transformers 5.16 schema to vLLM's GLM5.3 field names.

    The Gaudi image's vLLM model code predates the final Transformers config
    naming.  Keep the aliases on the runtime object so all model components
    (including MTP) see one consistent configuration without modifying the
    checkpoint JSON or relaxing Transformers' strict validation.
    """

    aliases = {
        "mhc_num_residual_streams": ("hc_mult", 4),
        "mhc_sinkhorn_iterations": ("hc_sinkhorn_iters", 20),
        "mhc_post_mult_value": ("hc_post_mult_value", 2.0),
        "mhc_tau": ("mhc_temperature", 0.05),
        "mla_nope": ("mla_use_nope", True),
        "moe_renormalize": ("norm_topk_prob", True),
        "num_experts_per_token": ("num_experts_per_tok", 8),
        "logit_scale": ("logit_scale", 1.0),
    }
    for target, (source, default) in aliases.items():
        try:
            getattr(config, target)
        except AttributeError:
            value = getattr(config, source, default)
            object.__setattr__(config, target, value)

    try:
        getattr(config, "is_moe")
    except AttributeError:
        object.__setattr__(
            config,
            "is_moe",
            getattr(config, "n_routed_experts", None) is not None,
        )

    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        try:
            getattr(config, "is_linear_attn")
        except AttributeError:
            object.__setattr__(
                config,
                "is_linear_attn",
                any(t == "linear_attention" for t in layer_types),
            )
        try:
            getattr(config, "is_kda_layer")
        except AttributeError:
            object.__setattr__(
                config,
                "is_kda_layer",
                lambda layer_idx: (
                    layer_idx < len(config.layer_types)
                    and config.layer_types[layer_idx] == "linear_attention"
                ),
            )

# vLLM 0.26 exposes the MoE implementation directly.  Upstream GLM5.3
# switched to the newer factory name; the constructor signature is otherwise
# compatible for this model, so keep the model source usable on the Gaudi
# vLLM 0.26 base image.
FusedMoEFactory = FusedMoE


def _mhc_mix(
    x_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    n: int,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC mapping (HF Glm5NextTextHyperConnection): pre/post/comb mixes.

    x_flat: [T, n*D] fp32 residual streams.  Returns pre [T, n], post [T, n],
    comb [T, n, n] (Sinkhorn-projected, comb[i, j] maps stream i -> j).
    """
    mixes = torch.matmul(x_flat, fn.t())
    mixes = mixes * torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n]) + pre_eps
    post = torch.sigmoid(mixes[:, n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n]) * post_mult
    comb = mixes[:, 2 * n :].reshape(-1, n, n) * hc_scale[2] + hc_base[2 * n :].reshape(1, n, n)
    comb = torch.softmax(comb, dim=-1) + sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    return pre, post, comb


def _mhc_pre_hpu(residual, fn, hc_scale, hc_base, norm_weight, norm_eps, n, rms_eps,
                 pre_eps, sinkhorn_eps, post_mult, sinkhorn_iters):
    """hc_pre + the sublayer RMSNorm.  residual: [T, n, D] (bf16)."""
    T, _, D = residual.shape
    res32 = residual.float()
    pre, post, comb = _mhc_mix(res32.reshape(T, n * D), fn, hc_scale, hc_base, n, rms_eps,
                               pre_eps, sinkhorn_eps, post_mult, sinkhorn_iters)
    collapsed = (pre.unsqueeze(-1) * res32).sum(dim=1).to(residual.dtype).float()
    normed = collapsed * torch.rsqrt(collapsed.square().mean(dim=-1, keepdim=True) + norm_eps)
    layer_input = norm_weight * normed.to(residual.dtype)
    return post, comb, layer_input


def _mhc_post_hpu(x, residual, post, comb):
    """new_stream_j = post_j * x + sum_i comb[i, j] * residual_i  (fp32 accumulate)."""
    mixed = torch.matmul(comb.transpose(-1, -2), residual.float())
    return (mixed + post.unsqueeze(-1) * x.float().unsqueeze(-2)).to(residual.dtype)


class Glm5NextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )

        self.swiglu_limit = swiglu_limit
        if self.swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit=self.swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextMoE(nn.Module):
    def __init__(
        self,
        config: Glm5NextConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        swiglu_limit = config.swiglu_limit
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=swiglu_limit,
            )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=None,
            router_logits_dtype=self.gate.out_dtype,
            swiglu_limit=swiglu_limit,
        )

        # HPU: stacked-FP8 expert path (see moe_hpu.py), built after the
        # quant method has prepared the expert weights.
        self.top_k = config.num_experts_per_token
        self.renormalize = config.moe_renormalize
        self.swiglu_limit = swiglu_limit
        self._hpu_experts = None
        if current_platform.device_type == "hpu":
            import weakref
            owner = weakref.ref(self)

            def _post(layer, owner=owner):
                owner()._hpu_build(layer)

            for mod in (self.experts, getattr(self.experts, "routed_experts", None)):
                if mod is not None:
                    object.__setattr__(mod, "_post_hpu_moe_prepare", _post)

    def _hpu_build(self, experts_layer) -> None:
        from .moe_hpu import StackedFp8Experts
        StackedFp8Experts(self, experts_layer, self.swiglu_limit)
        self.register_buffer("gate_w32", self.gate.weight.data.float().contiguous(), persistent=False)
        self.register_buffer("gate_bias32", self.gate.e_score_correction_bias.data.float().contiguous(),
                             persistent=False)
        self._hpu_experts = True

    # tokens * top_k at or below which only the selected experts are gathered
    # The dense path quantizes activations to FP8 per row, which GLM's outlier
    # channels make lossy (flips greedy tokens); the bf16 gather path is exact.
    # 64 slots covers decode up to bs8 and the 8-token spec verify at bs1.
    _GATHER_MAX_SLOTS = int(_os.environ.get("GLM53_MOE_GATHER_MAX_SLOTS", "64"))
    # token count above which the fused (sparse) HPU MoE op is used
    _DENSE_MAX_T = int(_os.environ.get("GLM53_MOE_DENSE_MAX_T", "192"))

    def _forward_hpu(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_reduce
        from .moe_hpu import moe_dense, moe_gather, route
        T = x.shape[0]
        ids, w = route(x, self.gate_w32, self.gate_bias32, self.top_k,
                       self.renormalize, self.routed_scaling_factor)
        fused = getattr(self, "_experts_layer", None)
        if T * self.top_k <= self._GATHER_MAX_SLOTS:
            out = moe_gather(self, x, ids, w, self.swiglu_limit)
        elif fused is not None and T > self._DENSE_MAX_T:
            # Large (prefill) batches: sparse fused HPU op. NOTE: it has no
            # swiglu clamp (see GLM53_MOE_DENSE_MAX_T to force dense).
            out = fused.moe_op(x, ids.to(torch.int64), w.to(x.dtype),
                               permuted_weights=True, activation="silu")
            if positions is not None:
                # The fused op has no swiglu clamp; the attention-sink token
                # (position 0) carries most clamp activations, so recompute
                # that row exactly with the clamped gather path.
                row0 = moe_gather(self, x[:1], ids[:1], w[:1], self.swiglu_limit)
                is0 = (positions.reshape(-1)[:1] == 0).view(1, 1)
                out = torch.cat([torch.where(is0, row0, out[:1]), out[1:]], dim=0)
        else:
            rw = torch.zeros((T, self.moe_E), dtype=torch.float32, device=x.device)
            rw = rw.scatter(1, ids, w)
            out = moe_dense(self, x, rw, self.swiglu_limit)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        if self._hpu_experts is not None and not self.is_sequence_parallel:
            return self._forward_hpu(hidden_states, positions)

        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # MoERunner holds the gate (passed to FusedMoEFactory) and computes
        # the router logits itself, so nothing is precomputed here (matches
        # DeepseekV2MoE; `router_logits` is a placeholder).
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(num_tokens, hidden_dim)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        is_mtp_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        # Booleans (not the int index) are read in forward so torch.compile
        # guards don't specialise every layer separately.
        self._is_first = layer_idx == 0
        self._is_last = layer_idx == config.num_hidden_layers - 1
        self._tap = False
        # Transformers 5.16's GLM5.3 text config does not expose the
        # convenience ``is_moe`` field used by upstream vLLM.  The presence
        # of routed experts is the equivalent signal for this checkpoint.
        self.is_moe = getattr(
            config, "is_moe", getattr(config, "n_routed_experts", None) is not None
        )
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts
        self.is_mtp_layer = is_mtp_layer
        self.mhc = config.mhc
        # Transformers 5.16 keeps the per-layer schedule as data but no
        # longer exposes vLLM's helper method on the strict config class.
        layer_types = getattr(config, "layer_types", None)
        is_kda_layer = (
            not is_mtp_layer
            and layer_types is not None
            and layer_idx < len(layer_types)
            and layer_types[layer_idx] == "linear_attention"
        )
        self.layer_kind = "kda" if is_kda_layer else "mla"
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if is_kda_layer:
            self.self_attn = Glm5NextLinearAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            # MLA layers require the latent head dims, which are guaranteed set
            # on MLA configs; narrow away the `int | None`.
            assert config.v_head_dim is not None
            assert config.kv_lora_rank is not None
            self.self_attn = Glm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                quant_config=None,  # MLA projections are BF16 in checkpoint
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                skip_rope=config.mla_nope,
            )

        # MTP layers sit past the base model's hidden layers (layer_idx >=
        # num_hidden_layers), so they're outside mlp_layer_types; default them
        # to the last base layer's MLP type (sparse/MoE for these checkpoints).
        mlp_layer_types = config.mlp_layer_types
        mlp_type = (
            mlp_layer_types[layer_idx]
            if layer_idx < len(mlp_layer_types)
            else (mlp_layer_types[-1] if mlp_layer_types else "sparse")
        )
        if self.is_moe and self.num_experts is not None and mlp_type == "sparse":
            self.mlp = Glm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Cached for the hot forward path (isinstance per layer per step).
        self._mlp_is_moe = isinstance(self.mlp, Glm5NextMoE)
        # In SP, the attention output projection leaves a partial sum; the
        # decoder-layer reduce_scatter after attention completes it (DSv4 pattern).
        # MTP layers use the non-mHC path which has no sp_reduce_scatter, so
        # their o_proj must still reduce normally.
        if self.is_sequence_parallel and not is_mtp_layer:
            self.self_attn.o_proj.reduce_results = False
        # TIMING EXPERIMENT ONLY (numerically wrong): skip the attention
        # all-reduce to measure what the collective costs per step.
        if _os.environ.get("GLM53_EXP_SKIP_ATTN_AR") == "1":
            self.self_attn.o_proj.reduce_results = False
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc and not is_mtp_layer:
            # mhc config
            self.mhc_num_residual_streams = config.mhc_num_residual_streams
            self.mhc_tau = config.mhc_tau
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            n = config.mhc_num_residual_streams
            d_model = n * self.hidden_size
            mix_hc = (2 + n) * n

            self.n = n

            # attn hc
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            # ffn hc
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            self.mhc_pre_op = MHCPreOp()
            self.mhc_post_op = MHCPostOp()
            self.mhc_fused_post_pre_op = MHCFusedPostPreOp()

            if getattr(vllm_config.kernel_config, "enable_jit_warmup", False):
                from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                    _HC_PRENORM_GEMM_TILELANG_KERNEL,
                    _MHC_FUSED_TILELANG_KERNEL,
                    _MHC_POST_TILELANG_KERNEL,
                    _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
                )
                from vllm.utils.deep_gemm import is_deep_gemm_supported

                include_pre_gemm_splits = is_deep_gemm_supported()
                _MHC_PRE_BIG_FUSE_TILELANG_KERNEL.register_warmup(
                    vllm_config,
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                    use_norm_weight=True,
                    include_pre_gemm_splits=include_pre_gemm_splits,
                    include_broadcast_splits=False,
                    rms_eps=self.rms_norm_eps,
                    hc_pre_eps=self.hc_eps,
                    hc_sinkhorn_eps=self.hc_eps,
                    hc_post_mult_value=self.mhc_post_mult_value,
                    sinkhorn_repeat=self.mhc_sinkhorn_iterations,
                    norm_eps=(
                        self.input_layernorm.variance_epsilon,
                        self.post_attention_layernorm.variance_epsilon,
                    ),
                )
                if not include_pre_gemm_splits:
                    _HC_PRENORM_GEMM_TILELANG_KERNEL.register_warmup(
                        vllm_config,
                        hidden_size=self.hidden_size,
                        hc_mult=self.n,
                        n_out=self.n * (2 + self.n),
                    )
                _MHC_POST_TILELANG_KERNEL.register_warmup(
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                )
                _MHC_FUSED_TILELANG_KERNEL.register_warmup(
                    vllm_config,
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # The HPU runner can retain singleton batch/stream dimensions during
        # decode (e.g. [1, 1, 1, hidden]).  GLM's mHC operators use the vLLM
        # token-major [tokens, hidden] contract, so flatten those dimensions at
        # the model boundary.
        if current_platform.device_type == "hpu":
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        # 70B or MTP layers: KDA + MoE without HC.
        if not self.mhc or self.is_mtp_layer:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            attn_output = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
            hidden_states, residual = self.post_attention_layernorm(
                attn_output, residual=residual
            )
            hidden_states = self.mlp(hidden_states)
            if self.is_mtp_layer:
                # Return the unsummed pair: the MTP caller feeds it straight
                # into shared_head's fused_add_rms_norm (one kernel instead of
                # a separate residual-add + norm). The sum itself is unchanged
                # (fp32-accumulated inside the fused kernel).
                return hidden_states, residual, None, None
            hidden_states = residual + hidden_states
            return hidden_states, residual, None, None

        # mHC start. `post`/`comb` carry the previous layer's deferred
        # hc_post inputs (its ffn-pre outputs); when present, fuse that
        # hc_post with this layer's attn hc_pre into one kernel (inter-layer
        # fusion). Layer 0 has no incoming state -> standalone hc_pre.
        x = hidden_states
        if post is None:
            if self._is_first:
                x = hc_expand(x, self.n)
            residual = x
            post, comb, x = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
        else:
            residual, post, comb, x = self.hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )

        # Attention needs the full token sequence; mHC above ran on the SP
        # shard. Gather for attention, scatter back afterward (DSv4 pattern).
        if self.is_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        if _DBG_DIR:
            _dbg(f"l{self.layer_idx}_attn_in", x)
        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )
        if _DBG_DIR:
            _dbg(f"l{self.layer_idx}_attn_out", x)

        if self.is_sequence_parallel:
            x = sp_reduce_scatter(x)

        # Fuse post-attn hc_post + pre-FFN hc_pre (+ RMSNorm) into one kernel.
        residual, post, comb, x = self.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=self.post_attention_layernorm.weight.data,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )

        if _DBG_DIR:
            _dbg(f"l{self.layer_idx}_mlp_in", x)
        # Fully Connected
        if self._mlp_is_moe:
            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel,
                         positions=positions)
        else:
            x = self.mlp(x)
        if _DBG_DIR:
            _dbg(f"l{self.layer_idx}_mlp_out", x)
        if _DBG_DIR and self.layer_idx != self.num_hidden_layers - 1:
            _dbg(f"l{self.layer_idx}_stream", self.hc_post(x, residual, post, comb))

        # mHC end. The last mHC layer materializes its final hc_post (nothing
        # to fuse with) then contracts; every other layer defers its hc_post to
        # the next layer's fused pre, returning the state.
        if self._is_last:
            x = self.hc_post(x, residual, post, comb)
            x = hc_contract(x, self.n)
            return x, None, None, None

        if self._tap:
            # Stream-mean of this layer's output streams (the DFlash2 drafter
            # tap): mean_j(post_j x + sum_i comb[i, j] res_i).
            tap = (post.mean(-1, keepdim=True) * x.float()
                   + (comb.mean(-1).unsqueeze(-1) * residual.float()).sum(1)).to(x.dtype)
            return x, residual, post, comb, tap
        return x, residual, post, comb

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        if current_platform.device_type == "hpu":
            return _mhc_pre_hpu(
                x, hc_fn, hc_scale, hc_base, norm_weight, norm_eps, self.n,
                self.rms_norm_eps, self.hc_eps, self.hc_eps,
                self.mhc_post_mult_value, self.mhc_sinkhorn_iterations,
            )
        post_mix, res_mix, layer_input = self.mhc_pre_op(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        return post_mix, res_mix, layer_input

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        if current_platform.device_type == "hpu":
            return _mhc_post_hpu(x, residual, post, comb)
        return self.mhc_post_op(x, residual, post, comb)

    def hc_fused_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        if current_platform.device_type == "hpu":
            residual = _mhc_post_hpu(x, residual, post, comb)
            post, comb, layer_input = _mhc_pre_hpu(
                residual, hc_fn, hc_scale, hc_base, norm_weight, norm_eps, self.n,
                self.rms_norm_eps, self.hc_eps, self.hc_eps,
                self.mhc_post_mult_value, self.mhc_sinkhorn_iterations,
            )
            return residual, post, comb, layer_input
        return self.mhc_fused_post_pre_op(
            x=x,
            residual=residual,
            post_layer_mix=post,
            comb_res_mix=comb,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            n_splits=1,
            tile_n=1,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )


_SEG = {"n": 0, "acc": None, "pending": None}


def _seg_events_begin():
    """GLM53_SEG_TIMING=1: device time per layer group (rank 0, eager region)."""
    if _os.environ.get("GLM53_SEG_TIMING") != "1" or get_tensor_model_parallel_rank() != 0:
        return None
    if torch.compiler.is_compiling():
        return None
    e = torch.hpu.Event(enable_timing=True)
    e.record()
    return [e]


def _seg_event(evs):
    if evs is not None:
        e = torch.hpu.Event(enable_timing=True)
        e.record()
        evs.append(e)


def _seg_events_end(evs):
    if evs is None:
        return
    p = _SEG["pending"]
    if p is not None and p[-1].query():
        d = [p[i].elapsed_time(p[i + 1]) for i in range(len(p) - 1)]
        _SEG["acc"] = d if _SEG["acc"] is None else [a + b for a, b in zip(_SEG["acc"], d)]
        _SEG["n"] += 1
        if _SEG["n"] % 100 == 0:
            avg = [a / 100 for a in _SEG["acc"]]
            logger.info("GLM seg device ms per group: %s | total %.2f",
                        " ".join(f"{x:.2f}" for x in avg), sum(avg))
            _SEG["acc"] = None
        _SEG["pending"] = None
    if _SEG["pending"] is None:
        _SEG["pending"] = evs


class Glm5NextLayerGroup(nn.Module):
    """A run of consecutive decoder layers compiled as one region on HPU.

    GLM-5.3's layer pattern repeats every 4 layers (3x KDA + 1x MLA), so
    groups aligned to it are structurally identical and share one compiled
    graph, while cutting per-step compiled calls ~4x vs per-layer regions.
    """

    def __init__(self, layers):
        super().__init__()
        # plain list: the layers stay registered under Glm5NextModel.layers
        object.__setattr__(self, "_layers", list(layers))

    def forward(self, positions, hidden_states, residual, post, comb):
        for layer in self._layers:
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb)
        return hidden_states, residual, post, comb


class Glm5NextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        _ensure_glm5_config_compat(config)
        self.config = config

        # Gaudi's MLA kernel currently has no kpool sparse-indexer backend.
        # Use the same dense-MLA fallback as the Gaudi GLM5.2 path: this keeps
        # the full checkpoint semantics while avoiding CUDA/DeepGEMM-only
        # indexer kernels.  The indexer weights are harmlessly ignored.
        if current_platform.device_type == "hpu" and config.index_topk is not None:
            logger.warning(
                "GLM5.3 kpool sparse indexer is unavailable on Gaudi; "
                "using dense MLA fallback"
            )
            # transformers 5.16's strict dataclass validation rejects None for
            # this optional-in-practice field; bypass validation for the
            # runtime-only dense fallback.
            object.__setattr__(config, "index_topk", None)

        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type

        self.is_v32 = config.index_topk is not None
        if self.is_v32:
            topk_tokens = config.index_topk
            assert topk_tokens is not None
            # Reserve room for the incomplete pool tail.
            kpool = config.index_kpool
            assert kpool is not None
            buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
            # Sparse MLA tiles top-k in 128 columns; padded slots remain masked.
            sparse_topk_block_n = 128
            buffer_width = (
                (buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n
            ) * sparse_topk_block_n
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                buffer_width,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # Full-MLA config (no kpool sparse indexer): no topk buffer.
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # The active slice is fixed after construction; cache it so forward
        # doesn't rebuild the slice (a fresh list) every step.
        self._active_layers = self.layers[self.start_layer : self.end_layer]
        self._aux_layers: tuple[int, ...] = ()
        # HPU: compile layers in pattern-aligned groups (GLM53_LAYER_GROUP,
        # default 4; 0 disables).  Group 0 holds the dense-MLP prefix.
        self.layer_groups = None
        gsz = int(_os.environ.get("GLM53_LAYER_GROUP", "0"))
        if current_platform.device_type == "hpu" and gsz > 1:
            layers = list(self._active_layers)
            groups = [Glm5NextLayerGroup(layers[i:i + gsz]) for i in range(0, len(layers), gsz)]
            self.layer_groups = nn.ModuleList(groups)

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.is_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            post = None
            comb = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            # post/comb (deferred mHC hc_post state) are not propagated across
            # PP ranks; the receiving rank's first mHC layer uses standalone pre.
            post = None
            comb = None

        # The HPU runner feeds [batch, seq] token ids and indexes the output as
        # [batch, seq, hidden]; the layers run token-major, so remember the
        # batch shape and restore it at the end.
        out_shape = None
        if self.device == "hpu" and hidden_states.dim() > 2:
            out_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            positions = positions.reshape(-1)
        full_num_tokens = positions.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)

        if self.layer_groups is not None:
            _evs = _seg_events_begin()
            for group in self.layer_groups:
                hidden_states, residual, post, comb = group(
                    positions, hidden_states, residual, post, comb)
                _seg_event(_evs)
            _seg_events_end(_evs)
        elif self._aux_layers:
            taps = []
            for layer in self._active_layers:
                out = layer(positions, hidden_states, residual, post, comb)
                hidden_states, residual, post, comb = out[:4]
                if len(out) == 5:
                    taps.append(out[4])
        else:
            for layer in self._active_layers:
                hidden_states, residual, post, comb = layer(
                    positions, hidden_states, residual, post, comb
                )

        if not get_pp_group().is_last_rank:
            # PP is gated off for GLM-5.3-Flash (no make_empty_intermediate_tensors),
            # so this branch is not exercised. post/comb are the deferred
            # hc_post state of this rank's last mHC layer; a future PP path
            # would need to propagate them, but for now they are dropped (the
            # receiving rank's first layer would fall back to standalone pre).
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.is_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        hidden_states = self.norm(hidden_states)
        if out_shape is not None:
            hidden_states = hidden_states.view(out_shape)
        if self._aux_layers:
            # [T, taps * H], layer-major concat (the drafter's fc input order)
            return hidden_states, torch.cat(taps, dim=-1)
        return hidden_states

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Emit stream-mean taps at the outputs of ``layers`` (DFlash2)."""
        assert self.layer_groups is None, "aux taps need per-layer regions"
        self._aux_layers = tuple(layers)
        for layer in self._active_layers:
            layer._tap = layer.layer_idx in self._aux_layers
            assert not (layer._tap and layer._is_last)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # MLA: fuse q_a_proj and kv_a_proj_with_mqa
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            # Indexer: fuse wk and weights_proj
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
            # KDA: merge q, k, v, b, f_a, g_a projections into one GEMM
            (".in_proj_qkvbfg_a", ".q_proj", 0),
            (".in_proj_qkvbfg_a", ".k_proj", 1),
            (".in_proj_qkvbfg_a", ".v_proj", 2),
            (".in_proj_qkvbfg_a", ".b_proj", 3),
            (".in_proj_qkvbfg_a", ".f_a_proj", 4),
            (".in_proj_qkvbfg_a", ".g_a_proj", 5),
        ]
        if getattr(
            self.config,
            "is_moe",
            getattr(self.config, "n_routed_experts", None) is not None,
        ):
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            # EPLB: the mapping enumerates physical experts, so it must cover
            # the redundant replicas or their slots are never loaded.
            num_redundant_experts = next(
                (
                    layer.mlp.n_redundant_experts
                    for layer in self.layers
                    if isinstance(layer, Glm5NextDecoderLayer)
                    and isinstance(layer.mlp, Glm5NextMoE)
                ),
                0,
            )
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
                num_redundant_experts=num_redundant_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # GLM-5.3-Flash NoPE checkpoints omit the RoPE rows from
        # ``kv_a_proj_with_mqa``; pad them with zeros for the model shape.
        kv_a_pad_size = 0
        if self.config.mla_nope and self.config.qk_rope_head_dim > 0:
            kv_a_pad_size = self.config.qk_rope_head_dim

        _pending_wk_fp8: dict = {}

        for args in weights:
            name, loaded_weight = args[:2]
            kwargs: dict = args[2] if len(args) > 2 else {}
            # GLM5.3 checkpoints always carry the sparse kpool indexer
            # tensors.  Gaudi currently uses the dense MLA fallback and does
            # not instantiate an indexer module, so those tensors have no
            # destination parameter and must be ignored by the loader.
            if ".indexer." in name:
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            # Handle FP8 indexer WK: dequantize to BF16 for fusion with
            # weights_proj into wk_weights_proj.
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
            ):
                continue

            # FP8 checkpoint: dequantize BF16-kept MLA projections
            # (q_a_proj / kv_a_proj_with_mqa / o_proj) to BF16.
            if _try_load_fp8_attn_proj(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                kv_a_pad_size,
            ):
                continue

            # Pad kv_a_proj_with_mqa for NoPE models
            if kv_a_pad_size > 0 and ".kv_a_proj_with_mqa." in name:
                pad = torch.zeros(
                    kv_a_pad_size,
                    *loaded_weight.shape[1:],
                    dtype=loaded_weight.dtype,
                    device=loaded_weight.device,
                )
                loaded_weight = torch.cat([loaded_weight, pad], dim=0)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # QKV fusion: skip if fused module doesn't exist in model
                if param_name == ".fused_qkv_a_proj" and name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    # A checkpoint expert may map to several physical replicas
                    # under EPLB; keep `name` intact and try the next entry
                    # when this physical expert is not local to the rank.
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not (
                            getattr(self.config, "layer_types", None)
                            and any(
                                t == "linear_attention"
                                for t in self.config.layer_types
                            )
                        )
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        # HPU KDA: assemble the merged conv weight once the weights are loaded
        # (outside any compiled region).
        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if hasattr(attn, "_build_conv_w_t"):
                attn._build_conv_w_t()
        if current_platform.device_type == "hpu":
            self._maybe_fp8_bf16_linears()
        return loaded_params

    def _maybe_fp8_bf16_linears(self) -> None:
        """Opt-in FP8 for the BF16 projections (GLM53_FP8_MLA / GLM53_FP8_KDA)."""
        from .fp8_hpu import quantize_linear_fp8_
        do_mla = _os.environ.get("GLM53_FP8_MLA", "0") == "1"
        do_kda = _os.environ.get("GLM53_FP8_KDA", "0") == "1"
        n = 0
        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            if do_kda and hasattr(attn, "in_proj_qkvbfg_a"):
                names = ("in_proj_qkvbfg_a", "f_b_proj", "g_b_proj", "o_proj")
            elif do_mla and hasattr(attn, "q_b_proj"):
                names = ("fused_qkv_a_proj", "q_b_proj", "o_proj")
            else:
                continue
            for nm in names:
                mod = getattr(attn, nm, None)
                if mod is not None and getattr(mod, "weight", None) is not None:
                    quantize_linear_fp8_(mod)
                    n += 1
        if n:
            logger.info("GLM5.3 HPU: quantized %d BF16 projections to FP8", n)


class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = Glm5NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=self.config.logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        _ensure_glm5_config_compat(hf_config)
        # The mamba-state hook receives the multimodal wrapper config, while
        # the KDA dimensions live on its text sub-config.
        hf_config = getattr(hf_config, "text_config", hf_config)
        _ensure_glm5_config_compat(hf_config)
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_num_heads,
            hf_config.linear_head_dim,
            conv_kernel_size=hf_config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[
        MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc
    ]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, MixtureOfExperts
):
    # The text model (KDA + dense-MLA + MoE) is a hybrid mamba model. The
    # multimodal wrapper must declare the same interfaces so vLLM treats it as
    # hybrid (auto-aligns mamba/attention block sizes, sizes the mamba state
    # cache); the mamba-state classmethods delegate to the text model.
    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True

    # GLM-5.3-Flash stores the dense-MLP gate/up as separate tensors (like
    # ``Glm4vMoeForConditionalGeneration``, ``glm4_moe`` and ``deepseek_v2``),
    # so the fused ``gate_up_proj`` must expand to its real shard names for
    # per-layer quant-scheme resolution. The identity ``gate_up_proj`` entry
    # inherited from ``Glm4vForConditionalGeneration`` (pre-fused gate_up_proj)
    # would otherwise route the module to ``global_quant_config`` and mismatch
    # at load for mixed-precision Quark checkpoints.
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # NOTE: weight-prefix mapping is inherited from Glm4vForConditionalGeneration
    # (``model.visual.`` -> ``visual.``, ``model.language_model.`` ->
    # ``language_model.model.``, ``lm_head.`` -> ``language_model.lm_head.``),
    # matching the GLM-OCR / GLM-4V serialization convention. If the real
    # checkpoint's safetensors keys differ (e.g. ``language_model.model.`` with
    # no outer ``model.``), override ``hf_to_vllm_mapper`` accordingly.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm4vForConditionalGeneration, self).__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        _ensure_glm5_config_compat(config.text_config)

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                # Read eps from the VISION sub-config, not the top-level
                # `config.rms_norm_eps`: Glm5NextConfig.__getattribute__ mirrors
                # the latter onto text_config (1e-5), silently ignoring the
                # vision tower's own (1e-6) rms_norm_eps.
                norm_eps=config.vision_config.rms_norm_eps,
                # Vision tower ships BF16 weights in this fp8 checkpoint (no
                # weight_scale_inv for visual.*), so it must NOT inherit the
                # global fp8 quant_config -- doing so incorrectly quantizes
                # the tower
                # and yields NaN image features. Mirrors the MLA/KDA proj
                # pattern (quant_config=None for BF16 submodules).
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )

        self.set_moe_parameters()

        # Glm5NextForCausalLM does not implement make_empty_intermediate_tensors,
        # so pipeline parallelism is gated off (consistent with the text-only
        # model) and we intentionally do not alias it here.

    def set_moe_parameters(self) -> None:
        self.moe_mlp_layers = [
            layer.mlp
            for layer in self.language_model.model.layers
            if isinstance(layer, Glm5NextDecoderLayer)
            and isinstance(layer.mlp, Glm5NextMoE)
        ]
        self.moe_layers = [moe.experts for moe in self.moe_mlp_layers]
        self.num_moe_layers = len(self.moe_layers)
        if not self.num_moe_layers:
            return
        example_moe = self.moe_mlp_layers[0]
        self.num_expert_groups = self.config.text_config.n_group
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if not self.num_moe_layers:
            return
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def get_encoder_cudagraph_config(self):
        # This vision tower does not produce the absolute position embedding
        # buffer used by GLM4V.
        config = super().get_encoder_cudagraph_config()
        config.buffer_keys = [k for k in config.buffer_keys if k != "pos_embeds"]
        return config


def get_spec_layer_idx_from_weight_name(
    config: Glm5NextConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(
                f"model.layers.{layer_idx + i}."
            ) or weight_name.startswith(f"layers.{layer_idx + i}."):
                return layer_idx + i
    return None


def _try_load_fp8_indexer_wk(name, tensor, buf, params_dict, loaded_params):
    if "indexer.wk." not in name or "wk_weights" in name:
        return False
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False
    layer_prefix = name.rsplit(".wk.", 1)[0]
    entry = buf.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    del buf[layer_prefix]
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = scaled_dequantize(
        weight_fp8,
        scale_inv,
        group_shape=GroupShape(block_size, block_size),
        out_dtype=torch.bfloat16,
    )

    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True


def _dequant_fp8_block(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize a block-FP8 (e4m3) weight with per-block scale to BF16.

    Unlike ``scaled_dequantize`` this tolerates a non-divisible (partial last
    block) shape by zero-padding to a multiple of ``block_size`` before the
    scale broadcast and trimming back afterwards (e.g. kv_a_proj_with_mqa is
    576 rows = 4*128 + 64).
    """
    out_dim, in_dim = weight_fp8.shape
    pad_out = (-out_dim) % block_size
    pad_in = (-in_dim) % block_size
    w = weight_fp8
    if pad_out or pad_in:
        w = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))
    # scale_inv is (ceil(out/block), ceil(in/block)); broadcast to (out, in).
    s = scale_inv.to(torch.float32)
    s_full = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    out = (w.to(torch.float32) * s_full).to(torch.bfloat16)
    return out[:out_dim, :in_dim].contiguous()


# FP8 checkpoint projections that the MODEL keeps in BF16, so the block-FP8
# (weight + weight_scale_inv) must be dequantized to BF16 on load.
# Maps checkpoint proj-suffix -> (buffer key, model target base, fused shard id
# or None for a direct projection, whether NoPE rope-padding applies).
_FP8_ATTN_PROJS = {
    ".q_a_proj.": ("q_a", "fused_qkv_a_proj", 0, False),
    ".kv_a_proj_with_mqa.": ("kv_a", "fused_qkv_a_proj", 1, True),
    ".q_b_proj.": ("q_b", "q_b_proj", None, False),
    ".o_proj.": ("o_proj", "o_proj", None, False),
}


def _try_load_fp8_attn_proj(
    name,
    tensor,
    buf,
    params_dict,
    loaded_params,
    kv_a_pad_size: int,
) -> bool:
    """Dequantize FP8 q_a_proj / kv_a_proj_with_mqa / o_proj to BF16 on load.

    The FP8 checkpoint stores these as block-FP8 (weight + weight_scale_inv),
    but the model holds them in BF16 (``fused_qkv_a_proj`` is always BF16 via
    DeepSeekV2FusedQkvAProjLinear; ``o_proj`` is excluded by
    modules_to_not_convert). When the model target is BF16 (no
    ``weight_scale_inv`` param) we dequantize; otherwise we return False so the
    normal stacked/direct path loads the FP8 tensor as-is.
    """
    matched = None
    for suffix, info in _FP8_ATTN_PROJS.items():
        if suffix in name:
            matched = (suffix, info)
            break
    if matched is None:
        return False
    suffix, (key, target_base, shard_id, is_kva) = matched
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    # Need to accept both the DeepSeek-native ``weight_scale_inv`` and the Quark
    # ``weight_scale`` names before feeding the shared block dequant below.
    is_scale = "weight_scale_inv" in name or name.endswith(".weight_scale")
    if not is_weight and not is_scale:
        return False

    layer_prefix = name.rsplit(suffix, 1)[0]
    target_w = f"{layer_prefix}.{target_base}.weight"
    target_s = f"{layer_prefix}.{target_base}.weight_scale_inv"
    # If the model actually kept this projection in FP8, let the normal path
    # handle it (it has a weight_scale_inv param).
    if target_s in params_dict:
        return False

    entry = buf.setdefault(layer_prefix, {}).setdefault(key, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    buf[layer_prefix].pop(key, None)
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)
    # NoPE: pad kv_a rope portion (kv_lora_rank -> kv_lora_rank + qk_rope_head_dim).
    if is_kva and kv_a_pad_size > 0:
        pad = torch.zeros(
            kv_a_pad_size,
            weight_bf16.shape[1],
            dtype=weight_bf16.dtype,
            device=weight_bf16.device,
        )
        weight_bf16 = torch.cat([weight_bf16, pad], dim=0)

    param = params_dict[target_w]
    if shard_id is None:
        param.weight_loader(param, weight_bf16)
    else:
        param.weight_loader(param, weight_bf16, shard_id)
    loaded_params.add(target_w)
    return True
