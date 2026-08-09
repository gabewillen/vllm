# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark MTP draft model for DeepSeek-V4-Flash checkpoints.

Flash checkpoints (``dspark_*`` config fields) replace the V3-style
e_proj/h_proj MTP module with a "DSpark" stack stored under ``mtp.{i}.*``:

  * ``mtp.0.main_proj`` projects the concatenation of the target model's
    ``dspark_target_layer_ids`` hidden states (mean over hyper-connection
    streams) to one conditioning stream ``main_x``;
  * each of the three DSpark blocks is a full V4 decoder block (hyper
    connections + 256-expert MoE) whose attention reads a rolling window
    of ``wkv(main_x)`` keys plus the draft token's own key — there is no
    indexer/compressor and no paged KV cache;
  * ``mtp.2`` carries the final norm and hc_head for logits (the LM head
    and embedding are shared with the target model).

This module implements the sequential ``method="mtp"`` drafting path for
``num_speculative_tokens=1``: one draft token per verified step, exact
reference semantics for the sampled row. The reference block-diffusion
mode (noise-token blocks, markov logit bias, confidence gating) needs
parallel-drafting proposer support and is not wired here.

Heavy platform imports are deferred so the pure helpers stay importable
in CPU-only test environments.
"""

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

_NUM_DSPARK_BLOCKS = 3
_FP8_BLOCK = 128

# Attention/main_proj tensors kept as plain bf16 parameters (dequantized
# from the checkpoint's fp8 block-quant format at load time): the DSpark
# attention math is done in plain torch, outside the vLLM linear stack.
# Value = TP shard dim of the weight (None = replicated).
_RAW_DEQUANT_SHARD_DIMS: dict[str, int | None] = {
    "attn.wq_a": None,
    "attn.wq_b": 0,
    "attn.wkv": None,
    "attn.wo_a": 0,
    "attn.wo_b": 1,
    "main_proj": None,
}

_RAW_DEQUANT_RE = re.compile(
    r"\.(attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)|main_proj)\.(weight|scale)$"
)


def remap_flash_mtp_weight_name(
    name: str,
    num_hidden_layers: int,
    num_blocks: int = _NUM_DSPARK_BLOCKS,
) -> str | None:
    """Map a ``mtp.{i}.*`` checkpoint tensor to its drafter param path.

    Args:
        name: Checkpoint tensor name.
        num_hidden_layers: Target model layer count (spec layer id base).
        num_blocks: DSpark blocks in the checkpoint.

    Returns:
        The drafter parameter path, or None when the tensor is not part
        of the drafter (non-MTP tensors, markov/confidence heads).
    """
    m = re.match(r"mtp\.(\d+)\.", name)
    if m is None:
        return None
    block = int(m.group(1))
    if block >= num_blocks:
        return None
    rest = name[m.end() :]
    if rest.startswith(("markov_head.", "confidence_head.")):
        return None
    base = f"model.layers.{num_hidden_layers}."
    if block == num_blocks - 1:
        if rest == "norm.weight":
            return base + "shared_head.norm.weight"
        if rest.startswith("hc_head_"):
            return base + rest
    if block == 0 and rest.split(".")[0] in ("main_proj", "main_norm"):
        return base + rest
    return f"{base}blocks.{block}.{rest}"


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Decode raw E8M0 exponent bytes to float32 scales (2^(e-127))."""
    return torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0)


def dequant_fp8_block(
    weight: torch.Tensor, scale: torch.Tensor, block: int = _FP8_BLOCK
) -> torch.Tensor:
    """Dequantize a block-quantized fp8 weight to bf16.

    Args:
        weight: fp8-e4m3 tensor ``[rows, cols]``.
        scale: E8M0 block scales ``[ceil(rows/block), ceil(cols/block)]``.
        block: Quantization block size.

    Returns:
        bf16 tensor of ``weight``'s shape.
    """
    w = weight.view(torch.float8_e4m3fn).to(torch.float32)
    s = e8m0_to_float(scale)
    s = s.repeat_interleave(block, 0)[: w.shape[0]]
    s = s.repeat_interleave(block, 1)[:, : w.shape[1]]
    return (w * s).to(torch.bfloat16)


def tp_shard(
    tensor: torch.Tensor, dim: int | None, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """The rank's contiguous shard of ``tensor`` along ``dim``."""
    if dim is None:
        return tensor
    length = tensor.shape[dim] // tp_size
    return tensor.narrow(dim, tp_rank * length, length)


def dspark_window_attention(
    q: torch.Tensor,
    window_kv: torch.Tensor,
    window_pos: torch.Tensor,
    draft_kv: torch.Tensor,
    positions: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
    window: int,
) -> torch.Tensor:
    """Reference DSpark attention for a block size of one draft token.

    Row ``t`` (real position ``p_t``) attends the rolling window of main
    keys at positions ``(p_t - window, p_t]`` plus its own draft key,
    with a per-head sink logit contributing to the denominator only.
    MQA-absorbed: values are the 512-d kv latents themselves.

    Args:
        q: ``[T, heads, head_dim]`` queries.
        window_kv: ``[window, head_dim]`` rolling main-kv buffer.
        window_pos: ``[window]`` absolute position per slot (-1 = empty).
        draft_kv: ``[T, head_dim]`` the draft tokens' own keys.
        positions: ``[T]`` absolute positions.
        attn_sink: ``[heads]`` sink logits.
        scale: Softmax scale.
        window: Window size.

    Returns:
        ``[T, heads, head_dim]`` attention output in ``q``'s dtype.
    """
    qf = q.to(torch.float32)
    scores_win = torch.einsum("thd,wd->thw", qf, window_kv.to(torch.float32))
    scores_win = scores_win * scale
    valid = (
        (window_pos >= 0)
        & (window_pos.unsqueeze(0) <= positions.unsqueeze(1))
        & (window_pos.unsqueeze(0) > (positions - window).unsqueeze(1))
    )
    scores_win = scores_win.masked_fill(~valid.unsqueeze(1), float("-inf"))
    scores_self = torch.einsum("thd,td->th", qf, draft_kv.to(torch.float32)) * scale
    sink = attn_sink.to(torch.float32).view(1, -1, 1)
    logits = torch.cat(
        [scores_win, scores_self.unsqueeze(-1), sink.expand(q.shape[0], -1, -1)],
        dim=-1,
    )
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("thw,wd->thd", probs[..., :window], window_kv.to(torch.float32))
    out = out + probs[..., window : window + 1] * draft_kv.to(torch.float32).unsqueeze(
        1
    )
    return out.to(q.dtype)


class DSparkAttention(nn.Module):
    """DSpark draft attention: window of main-kv keys, torch math.

    Keeps its own per-sequence rolling window buffer instead of a paged
    KV cache (requires ``max_num_seqs == 1``, enforced by the predictor).
    All projections are plain bf16 parameters dequantized at load time.
    """

    def __init__(self, config, tp_rank: int, tp_size: int) -> None:
        super().__init__()
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope

        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // tp_size
        self.n_local_groups = config.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.window = config.sliding_window
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        def _param(*shape: int) -> nn.Parameter:
            return nn.Parameter(
                torch.empty(*shape, dtype=torch.bfloat16), requires_grad=False
            )

        self.wq_a = _param(self.q_lora_rank, self.hidden_size)
        self.wq_b = _param(self.n_local_heads * self.head_dim, self.q_lora_rank)
        self.wkv = _param(self.head_dim, self.hidden_size)
        self.wo_a = _param(
            self.n_local_groups * self.o_lora_rank,
            self.n_heads * self.head_dim // config.o_groups,
        )
        self.wo_b = _param(self.hidden_size, self.n_local_groups * self.o_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.attn_sink = nn.Parameter(
            torch.full((self.n_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=1,
        )
        self.register_buffer(
            "kv_window",
            torch.zeros(self.window, self.head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "kv_pos",
            torch.full((self.window,), -1, dtype=torch.int64),
            persistent=False,
        )

    def _rope(
        self, x: torch.Tensor, positions: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """GPT-J-style rotation of the trailing rope dims of ``x``."""
        rd = self.rope_head_dim
        cos_sin = self.rotary_emb.cos_sin_cache[positions].to(torch.float32)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if inverse:
            sin = -sin
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        rot = x[..., -rd:].to(torch.float32).unflatten(-1, (-1, 2))
        x1, x2 = rot.unbind(-1)
        rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(
            -2
        )
        out = x.clone()
        out[..., -rd:] = rot.to(x.dtype)
        return out

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        main_x: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_reduce

        num_tokens = x.shape[0]
        main_kv = self._rope(self.kv_norm(F.linear(main_x, self.wkv)), positions)
        draft_kv = self._rope(self.kv_norm(F.linear(x, self.wkv)), positions + 1)
        # Update the rolling window with the last <=window rows (positions
        # are strictly increasing for the single tracked sequence, so the
        # surviving rows map to distinct slots).
        sel = positions > positions.max() - self.window
        slots = positions[sel] % self.window
        self.kv_window[slots] = main_kv[sel]
        self.kv_pos[slots] = positions[sel]

        q = F.linear(self.q_norm(F.linear(x, self.wq_a)), self.wq_b)
        q = q.view(num_tokens, self.n_local_heads, self.head_dim)
        q = q * torch.rsqrt(
            q.to(torch.float32).square().mean(-1, keepdim=True) + self.eps
        ).to(q.dtype)
        q = self._rope(q, positions + 1)

        o = dspark_window_attention(
            q,
            self.kv_window,
            self.kv_pos,
            draft_kv,
            positions,
            self.attn_sink,
            self.scale,
            self.window,
        )
        o = self._rope(o, positions + 1, inverse=True)
        o = o.view(num_tokens, self.n_local_groups, -1)
        wo_a = self.wo_a.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("tgd,grd->tgr", o.to(torch.float32), wo_a.to(torch.float32))
        o = F.linear(o.flatten(1).to(self.wo_b.dtype), self.wo_b)
        if self.tp_size > 1:
            o = tensor_model_parallel_all_reduce(o)
        return o


class DSparkDecoderLayer(nn.Module):
    """One DSpark block: hyper-connected attention + MoE sublayers."""

    def __init__(self, vllm_config, layer_idx: int, prefix: str) -> None:
        super().__init__()
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MoE

        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0

        self.attn = DSparkAttention(
            config,
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=get_tensor_model_parallel_world_size(),
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size

        def _hc(*shape: int) -> nn.Parameter:
            return nn.Parameter(
                torch.empty(*shape, dtype=torch.float32), requires_grad=False
            )

        self.hc_attn_fn = _hc(mix_hc, hc_dim)
        self.hc_ffn_fn = _hc(mix_hc, hc_dim)
        self.hc_attn_base = _hc(mix_hc)
        self.hc_ffn_base = _hc(mix_hc)
        self.hc_attn_scale = _hc(3)
        self.hc_ffn_scale = _hc(3)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        main_x: torch.Tensor,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm.model_executor.kernels.mhc.tilelang import (
            mhc_fused_post_pre_tilelang,
            mhc_pre_tilelang,
        )

        if residual is None:
            residual = x
            post_mix, res_mix, x = mhc_pre_tilelang(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                norm_weight=self.attn_norm.weight.data,
                norm_eps=self.attn_norm.variance_epsilon,
            )
        else:
            residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=self.attn_norm.weight.data,
                norm_eps=self.attn_norm.variance_epsilon,
            )

        x = self.attn(positions, x, main_x)

        residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            n_splits=1,
            tile_n=1,
            norm_weight=self.ffn_norm.weight.data,
            norm_eps=self.ffn_norm.variance_epsilon,
        )

        x = self.ffn(x, None)
        return x, residual, post_mix, res_mix


class DeepSeekV4FlashMTPLayer(nn.Module):
    """The single spec layer: main_proj conditioning + 3 DSpark blocks."""

    def __init__(self, vllm_config, prefix: str) -> None:
        super().__init__()
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.models.deepseek_mtp import SharedHead

        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        num_targets = len(config.dspark_target_layer_ids)

        self.main_proj = nn.Parameter(
            torch.empty(
                config.hidden_size,
                num_targets * config.hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.main_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        spec_layer = config.num_hidden_layers
        self.blocks = nn.ModuleList(
            DSparkDecoderLayer(
                vllm_config,
                layer_idx=spec_layer + i,
                prefix=f"model.layers.{spec_layer + i}",
            )
            for i in range(_NUM_DSPARK_BLOCKS)
        )
        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=vllm_config.quant_config
        )
        self.hc_head_fn = nn.Parameter(
            torch.empty(
                self.hc_mult,
                self.hc_mult * config.hidden_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang

        num_targets = len(self.config.dspark_target_layer_ids)
        main_hidden = previous_hidden_states[:, : num_targets * self.config.hidden_size]
        main_x = self.main_norm(
            F.linear(main_hidden.to(self.main_proj.dtype), self.main_proj)
        )
        x = inputs_embeds.unsqueeze(1).repeat(1, self.hc_mult, 1)
        residual = post_mix = res_mix = None
        for block in self.blocks:
            x, residual, post_mix, res_mix = block(
                x, positions, main_x, post_mix, res_mix, residual
            )
        x = mhc_post_tilelang(x, residual, post_mix, res_mix)
        # Pre-hc_head residual, flat; hc_head is deferred to compute_logits.
        return x.flatten(1)


class DeepSeekV4FlashMultiTokenPredictor(nn.Module):
    """DSpark predictor: drop-in ``model`` for :class:`DeepSeekV4MTP`."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )
        from vllm.model_executor.models.utils import maybe_prefix

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        config = spec_config.draft_model_config.hf_config
        if spec_config.num_speculative_tokens != 1:
            raise NotImplementedError(
                "DeepSeek-V4-Flash DSpark MTP supports num_speculative_tokens=1 "
                "only (block drafting needs parallel-drafting support)"
            )
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise NotImplementedError(
                "DeepSeek-V4-Flash DSpark MTP keeps a per-sequence rolling "
                "kv window; run with --max-num-seqs 1"
            )
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = 1
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx): DeepSeekV4FlashMTPLayer(
                    vllm_config,
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx}",
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    @property
    def _layer(self) -> DeepSeekV4FlashMTPLayer:
        return self.layers[str(self.mtp_start_layer_idx)]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return self._layer(positions, previous_hidden_states, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        from vllm.model_executor.kernels.mhc.tilelang import (
            hc_head_fused_kernel_tilelang,
        )
        from vllm.models.deepseek_v4.common.ops import mtp_shared_head_rmsnorm

        layer = self._layer
        hidden_states = hidden_states.view(-1, layer.hc_mult, layer.config.hidden_size)
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            layer.hc_head_fn,
            layer.hc_head_scale,
            layer.hc_head_base,
            layer.rms_norm_eps,
            layer.hc_eps,
        )
        hidden_states = mtp_shared_head_rmsnorm(
            hidden_states,
            layer.shared_head.norm.weight.data,
            layer.shared_head.norm.variance_epsilon,
        )
        return self.logits_processor(layer.shared_head.head, hidden_states)


def load_flash_mtp_weights(model: nn.Module, weights) -> set[str]:
    """Load ``mtp.{i}.*`` checkpoint tensors into a Flash MTP draft model.

    Args:
        model: The :class:`DeepSeekV4MTP` root whose ``model`` attribute
            is a :class:`DeepSeekV4FlashMultiTokenPredictor`.
        weights: Iterable of (checkpoint name, tensor).

    Returns:
        Root-relative names of the loaded parameters.
    """
    from vllm.model_executor.layers.fused_moe import (
        fused_moe_make_expert_params_mapping,
    )
    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader,
    )

    predictor = model.model
    config = predictor.config
    layer = predictor._layer
    params_dict = dict(model.named_parameters())
    loaded_params: set[str] = set()
    attn = layer.blocks[0].attn
    tp_rank, tp_size = attn.tp_rank, attn.tp_size
    head_start = attn.n_local_heads * tp_rank
    head_end = attn.n_local_heads * (tp_rank + 1)
    spec_layer = config.num_hidden_layers

    shared_mapping = {
        "embed.weight": "model.embed_tokens.weight",
        "head.weight": f"model.layers.{spec_layer}.shared_head.head.weight",
    }
    stacked_params_mapping = [
        ("gate_up_proj", "w1", 0),
        ("gate_up_proj", "w3", 1),
    ]
    expert_mapping = fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=config.n_routed_experts,
    )
    # (param path, "weight"|"scale") -> raw tensor, paired for dequant.
    pending_raw: dict[tuple[str, str], torch.Tensor] = {}

    for name, loaded_weight in weights:
        if name in shared_mapping:
            name = shared_mapping[name]
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            continue
        name = remap_flash_mtp_weight_name(name, spec_layer)
        if name is None:
            continue
        raw = _RAW_DEQUANT_RE.search(name)
        if raw is not None:
            pending_raw[(name[: raw.start(2) - 1], raw.group(2))] = loaded_weight
            continue
        if ".experts." in name:
            if name.endswith(".scale") and loaded_weight.dtype == (
                torch.float8_e8m0fnu
            ):
                # Preserve raw E8M0 exponent bytes (numeric copy_ zeroes
                # them); mirrors the main DeepseekV4 loader.
                loaded_weight = loaded_weight.view(torch.uint8)
            if name.endswith(".scale"):
                name = name.removesuffix(".scale") + ".weight_scale"
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                param = params_dict[name_mapped]
                success = param.weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded_params.add(name_mapped)
                    break
            continue
        if name.endswith(".scale"):
            name = name.removesuffix(".scale") + ".weight_scale_inv"
        if ".shared_experts.w2" in name:
            name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
        if name.endswith(".ffn.gate.bias"):
            name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
        if name.endswith("attn.attn_sink"):
            params_dict[name].copy_(loaded_weight[head_start:head_end])
            loaded_params.add(name)
            continue
        handled = False
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            param.weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(name)
            handled = True
            break
        if handled:
            continue
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        loaded_params.add(name)

    for base in sorted({base for base, _ in pending_raw}):
        weight = pending_raw[(base, "weight")]
        scale = pending_raw[(base, "scale")]
        key = next(k for k in _RAW_DEQUANT_SHARD_DIMS if base.endswith(k))
        dim = _RAW_DEQUANT_SHARD_DIMS[key]
        weight = tp_shard(weight, dim, tp_rank, tp_size)
        scale = tp_shard(scale, dim, tp_rank, tp_size)
        param = params_dict[base]
        param.copy_(dequant_fp8_block(weight, scale).to(param.dtype))
        loaded_params.add(base)

    for block in layer.blocks:
        block.ffn.finalize_mega_moe_weights()
    logger.info_once(
        "Flash DSpark MTP draft model loaded: %d params", len(loaded_params)
    )
    return loaded_params
