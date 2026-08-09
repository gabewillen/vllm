# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glue between :class:`ExpertTierCache` and the fused-MoE runtime.

When expert-tier offloading is enabled, routed-expert weights live in a
tiered cache (GPU slot pools <- pinned RAM <- mmap'd cold files) instead
of resident VRAM:

- The cold files hold the *post-process_weights_after_loading* kernel
  format (Marlin MXFP4), expert-major, one raw file per (layer, tensor),
  described by a per-rank ``manifest.json`` written by the checkpoint
  converter.
- At layer construction the cache's stable GPU slot pools are registered
  as the layer's weight parameters, so the quant config and MoE kernel
  reference pool tensors directly. Checkpoint expert-weight loading and
  Marlin repacking are skipped.
- Each forward step the runner calls :meth:`ExpertTierLayerBinding.ensure`
  with the routed expert ids and the cache's ``slot_of`` tensor serves as
  the layer's ``expert_map`` (global id -> GPU slot, -1 = absent), so the
  Marlin kernel computes exactly the routed experts from their slots.

Baseline mode is on-demand: ensure() per layer per step, no prediction.
It must run outside cudagraph capture; config validation forces eager.

Large prefill chunks can route more distinct experts per layer than the
GPU slot pool holds; :meth:`ExpertTierLayerBinding.plan_sub_batches`
splits such token batches into contiguous sub-batches that each fit, and
the runner applies the kernel per sub-batch (ensure -> apply -> release).
"""

import atexit
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.offloader.expert_prefetch import ExpertPrefetcher
from vllm.model_executor.offloader.expert_tier import (
    _STAGED_H2D_MAX_ROW_BYTES,
    ExpertTensorSpec,
    ExpertTierCache,
    MmapColdBacking,
)
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.config.offload import ExpertTierOffloadConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

MANIFEST_FILENAME = "manifest.json"

# GPU slots kept out of a sub-batch's distinct-expert bound: headroom so
# ensure() never runs the pool to exactly zero evictable slots and hot
# LRU entries can survive across sub-batches.
SUB_BATCH_SLOT_MARGIN = 4


def plan_expert_sub_batches(
    topk_ids: torch.Tensor, max_distinct: int
) -> list[tuple[int, int]] | None:
    """Split a token batch into contiguous sub-batches by distinct experts.

    Greedy scan over the routing ``topk_ids`` ``[num_tokens, top_k]``:
    tokens are appended to the current sub-batch until adding the next
    token would push its distinct-expert count past ``max_distinct``.

    Returns:
        ``[start, end)`` row ranges partitioning ``[0, num_tokens)``, or
        None when the whole batch already fits ``max_distinct``. A single
        token routing more than ``max_distinct`` experts still gets its
        own sub-batch (the cache's ensure() enforces the hard limit).
    """
    num_tokens = topk_ids.shape[0]
    if num_tokens <= 1:
        return None
    if topk_ids.device.type != "cpu":
        topk_ids = topk_ids.cpu()
    if int(torch.unique(topk_ids).numel()) <= max_distinct:
        return None
    bounds: list[tuple[int, int]] = []
    start = 0
    seen: set[int] = set()
    for i, row in enumerate(topk_ids.tolist()):
        new = set(row) - seen
        if i > start and len(seen) + len(new) > max_distinct:
            bounds.append((start, i))
            start = i
            seen = set(row)
        else:
            seen |= new
    bounds.append((start, num_tokens))
    return bounds


@dataclass(frozen=True)
class ExpertTierManifest:
    """Parsed per-rank cold-cache manifest.

    Attributes:
        path: The manifest.json path this was loaded from.
        num_experts: Global routed expert count per layer.
        layer_specs: layer_id -> per-expert tensor specs.
        files: (layer_id, tensor_name) -> absolute cold file path.
        meta: Free-form converter metadata.
    """

    path: str
    num_experts: int
    layer_specs: dict[int, list[ExpertTensorSpec]]
    files: dict[tuple[int, str], str]
    meta: dict


def _parse_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"manifest dtype {name!r} is not a torch dtype")
    return dtype


def load_manifest(cache_dir: str, rank: int) -> ExpertTierManifest:
    """Load and validate ``<cache_dir>/rank<rank>/manifest.json``.

    Relative ``file`` entries are resolved against the manifest directory.

    Raises:
        FileNotFoundError: Manifest missing.
        ValueError: Schema, dtype, or row_bytes mismatch.
    """
    manifest_dir = os.path.join(cache_dir, f"rank{rank}")
    path = os.path.join(manifest_dir, MANIFEST_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"expert-tier manifest not found: {path}. Run the expert cache "
            "converter for this model/TP layout first."
        )
    with open(path) as f:
        data = json.load(f)

    num_experts = int(data["num_experts"])
    layer_specs: dict[int, list[ExpertTensorSpec]] = {}
    files: dict[tuple[int, str], str] = {}
    for layer_key, tensors in data["layers"].items():
        layer_id = int(layer_key)
        specs = []
        for name, entry in tensors.items():
            spec = ExpertTensorSpec(
                name=name,
                shape=tuple(int(d) for d in entry["per_expert_shape"]),
                dtype=_parse_dtype(entry["dtype"]),
            )
            row_bytes = int(entry["row_bytes"])
            if row_bytes != spec.row_bytes:
                raise ValueError(
                    f"manifest layer {layer_id} tensor {name}: row_bytes "
                    f"{row_bytes} != shape/dtype-derived {spec.row_bytes}"
                )
            file_path = entry["file"]
            if not os.path.isabs(file_path):
                file_path = os.path.join(manifest_dir, file_path)
            specs.append(spec)
            files[(layer_id, name)] = file_path
        if not specs:
            raise ValueError(f"manifest layer {layer_id} has no tensors")
        layer_specs[layer_id] = specs
    if not layer_specs:
        raise ValueError(f"manifest {path} has no layers")
    return ExpertTierManifest(
        path=path,
        num_experts=num_experts,
        layer_specs=layer_specs,
        files=files,
        meta=data.get("meta", {}),
    )


def _small_row_bytes(specs: list[ExpertTensorSpec]) -> int:
    """Bytes/expert of tensors that use the cache's staged-H2D path."""
    return sum(s.row_bytes for s in specs if s.row_bytes <= _STAGED_H2D_MAX_ROW_BYTES)


def drop_mtp_layers(manifest: ExpertTierManifest) -> ExpertTierManifest:
    """Manifest without the MTP draft layers listed in ``meta.mtp_layer_ids``.

    The converter records MTP (DSpark) draft-block experts as extra cache
    layers; without speculative decoding they are never bound, so they are
    dropped before pool sizing to keep slot counts identical to a cache
    converted without ``--mtp``.
    """
    mtp_ids = set(manifest.meta.get("mtp_layer_ids", []))
    if not mtp_ids:
        return manifest
    return ExpertTierManifest(
        path=manifest.path,
        num_experts=manifest.num_experts,
        layer_specs={
            layer_id: specs
            for layer_id, specs in manifest.layer_specs.items()
            if layer_id not in mtp_ids
        },
        files={
            key: path for key, path in manifest.files.items() if key[0] not in mtp_ids
        },
        meta=manifest.meta,
    )


def _byte_specs(
    layer_specs: dict[int, list[ExpertTensorSpec]],
) -> dict[int, list[ExpertTensorSpec]]:
    """Rewrite specs as flat uint8 byte rows for the cache.

    The cache only moves bytes; storing rows as uint8 keeps its staged
    index_copy/index_select path independent of dtypes CUDA kernels do
    not cover (e.g. float8_e8m0fnu Marlin scales). The binding exposes
    typed views of the pools to the kernel.
    """
    return {
        layer_id: [ExpertTensorSpec(s.name, (s.row_bytes,), torch.uint8) for s in specs]
        for layer_id, specs in layer_specs.items()
    }


def compute_slot_counts(
    manifest: ExpertTierManifest,
    gpu_bytes: int,
    pinned_bytes: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """Uniform slots-per-layer pool sizing from per-rank byte budgets.

    Per layer ``l`` with ``B_l`` bytes/expert and ``S_l`` bytes/expert of
    small (staged) tensors, the cache allocates per GPU slot ``B_l + S_l``
    device bytes (slot pool + staging), and per pinned slot ``B_l`` pinned
    bytes plus a fixed ``gpu_slots * S_l`` pinned staging buffer. So:

        gpu_slots    = gpu_bytes // sum_l (B_l + S_l)
        pinned_slots = (pinned_bytes - gpu_slots * sum_l S_l) // sum_l B_l

    Both are clamped to ``num_experts``; ``pinned_slots`` is raised to
    ``gpu_slots`` (the cache requires the pinned pool to hold a full
    ensure() cold working set).

    Raises:
        ValueError: If the GPU budget cannot hold a single slot per layer.
    """
    bytes_all = sum(
        sum(s.row_bytes for s in specs) for specs in manifest.layer_specs.values()
    )
    small_all = sum(_small_row_bytes(specs) for specs in manifest.layer_specs.values())

    gpu_slots = gpu_bytes // (bytes_all + small_all)
    if gpu_slots < 1:
        raise ValueError(
            f"expert-tier gpu budget {gpu_bytes} bytes cannot hold one GPU "
            f"slot per layer ({bytes_all + small_all} bytes needed); raise "
            "--expert-tier-gpu-gb"
        )
    gpu_slots = min(int(gpu_slots), manifest.num_experts)

    pinned_slots = (pinned_bytes - gpu_slots * small_all) // bytes_all
    if pinned_slots < gpu_slots:
        logger.warning(
            "expert-tier pinned budget %d bytes gives %d slots/layer, below "
            "the %d GPU slots/layer; raising to %d (the pinned pool must "
            "hold a full ensure() working set). Actual pinned use will "
            "exceed the budget.",
            pinned_bytes,
            pinned_slots,
            gpu_slots,
            gpu_slots,
        )
        pinned_slots = gpu_slots
    pinned_slots = min(int(pinned_slots), manifest.num_experts)

    layer_ids = manifest.layer_specs.keys()
    return (
        {layer_id: gpu_slots for layer_id in layer_ids},
        {layer_id: pinned_slots for layer_id in layer_ids},
    )


class ExpertTierLayerBinding:
    """Per-layer handle used by the MoE runner and quant method."""

    def __init__(
        self,
        cache: ExpertTierCache,
        layer_id: int,
        num_experts: int,
        specs: list[ExpertTensorSpec],
        gpu_slots: int,
        prefetcher: ExpertPrefetcher | None = None,
    ):
        self.cache = cache
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.specs = specs
        self.gpu_slots = gpu_slots
        self.prefetcher = prefetcher
        # An empty ensure() exposes the layer's stable pool tensors and
        # live slot_of view without any IO or pinning.
        result = cache.ensure(layer_id, [])
        cache.release(layer_id)
        # The cache shuttles opaque byte rows (see _byte_specs); expose
        # zero-copy views with each tensor's real dtype and shape.
        self.gpu_pools = {}
        for spec in specs:
            byte_pool = result.gpu_tensors[spec.name]
            self.gpu_pools[spec.name] = byte_pool.view(spec.dtype).view(
                byte_pool.shape[0], *spec.shape
            )
        # Live int32 [num_experts] device tensor: global id -> GPU slot,
        # -1 when absent. Served as the layer's expert_map.
        self.slot_of = result.slot_of

    def register_pool_parameters(
        self,
        layer: torch.nn.Module,
        expected_names: list[str],
        extra_weight_attrs: dict,
    ) -> None:
        """Register the GPU slot pools as the layer's weight parameters.

        Replaces the quant method's normal parameter allocation, so no
        full-size expert tensors are ever created. The registered
        parameters share storage with the cache-owned pools.

        Raises:
            ValueError: If the manifest tensor names do not match the
                names the quant method would have created.
        """
        if set(expected_names) != set(self.gpu_pools):
            raise ValueError(
                f"layer {self.layer_id}: manifest tensors "
                f"{sorted(self.gpu_pools)} do not match the quant method's "
                f"expert tensors {sorted(expected_names)}"
            )
        for name in expected_names:
            pool = self.gpu_pools[name]
            param = torch.nn.Parameter(pool, requires_grad=False)
            assert param.data_ptr() == pool.data_ptr()
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def plan_sub_batches(self, topk_ids: torch.Tensor) -> list[tuple[int, int]] | None:
        """Contiguous token sub-batches whose distinct-expert count fits
        the layer's GPU slot pool; None when the whole batch fits and no
        split is needed. Split pieces target ``gpu_slots`` minus a small
        margin so the pool keeps eviction headroom across sub-batches.
        Each returned range is served by its own ensure -> apply ->
        release cycle in the runner.
        """
        if topk_ids.shape[0] <= 1:
            return None
        if topk_ids.device.type != "cpu":
            topk_ids = topk_ids.cpu()
        if int(torch.unique(topk_ids).numel()) <= self.gpu_slots:
            return None
        bound = max(topk_ids.shape[-1], self.gpu_slots - SUB_BATCH_SLOT_MARGIN)
        return plan_expert_sub_batches(topk_ids, bound)

    def ensure(self, topk_ids: torch.Tensor) -> None:
        """Make the routed experts resident (on-demand, blocking on ids).

        Ids are moved to the CPU first (the host sync on the routed ids
        is inherent to the on-demand tier mode); the same CPU ids feed
        the predictive prefetcher's bookkeeping at zero extra sync cost.
        """
        if topk_ids.device.type != "cpu":
            topk_ids = topk_ids.cpu()
        self.cache.ensure(self.layer_id, topk_ids)
        if self.prefetcher is not None:
            self.prefetcher.observe(self.layer_id, topk_ids)

    def release(self) -> None:
        self.cache.release(self.layer_id)

    def register_predictor_sources(
        self,
        hash_table: torch.Tensor | None,
        gate: torch.nn.Module | None,
    ) -> None:
        """Give the prefetcher this layer's hash table / router gate."""
        if self.prefetcher is None:
            return
        if hash_table is not None:
            self.prefetcher.note_hash_table(self.layer_id, hash_table)
        if gate is not None:
            self.prefetcher.note_gate(self.layer_id, gate)

    def observe_hidden(self, hidden_states: torch.Tensor) -> None:
        """Feed the gate-lookahead predictor (no-op unless enabled)."""
        if self.prefetcher is not None:
            self.prefetcher.observe_hidden(self.layer_id, hidden_states)


class ExpertTierState:
    """Per-rank cache state shared by all bound MoE layers."""

    def __init__(
        self,
        config: "ExpertTierOffloadConfig",
        device: torch.device,
        rank: int,
        max_routed_experts: int | None,
        include_mtp_layers: bool = False,
    ):
        assert config.cache_dir is not None
        self.config = config
        self.device = device
        self.rank = rank
        self.manifest = load_manifest(config.cache_dir, rank)
        if not include_mtp_layers:
            self.manifest = drop_mtp_layers(self.manifest)
        meta_backend = self.manifest.meta.get("mxfp4_backend")
        if meta_backend is not None and meta_backend != "MARLIN":
            raise ValueError(
                f"expert cache {self.manifest.path} was converted for "
                f"backend {meta_backend!r}; only MARLIN is supported"
            )
        gpu_slots, pinned_slots = compute_slot_counts(
            self.manifest,
            gpu_bytes=int(config.gpu_gb * 2**30),
            pinned_bytes=int(config.pinned_gb * 2**30),
        )
        self.gpu_slots_per_layer = gpu_slots
        self.pinned_slots_per_layer = pinned_slots
        slots = next(iter(gpu_slots.values()))
        if max_routed_experts is not None:
            worst_case = min(self.manifest.num_experts, max_routed_experts)
            if slots < worst_case:
                logger.info(
                    "expert-tier GPU pools hold %d slots/layer but a batch "
                    "can route up to %d distinct experts per layer; such "
                    "batches are applied in contiguous token sub-batches "
                    "that fit the pool.",
                    slots,
                    worst_case,
                )
        logger.info(
            "expert-tier cache: rank %d, %d layers, %d experts/layer, "
            "%d GPU slots/layer, %d pinned slots/layer, %d io threads",
            rank,
            len(self.manifest.layer_specs),
            self.manifest.num_experts,
            slots,
            next(iter(pinned_slots.values())),
            config.io_threads,
        )
        # Model init runs inside a `with target_device:` context; the
        # cache's host-side bookkeeping tensors rely on the default
        # device being CPU, while its pools pass devices explicitly.
        with torch.device("cpu"):
            self.cache = ExpertTierCache(
                device=device,
                gpu_slots_per_layer=gpu_slots,
                pinned_slots_per_layer=pinned_slots,
                layer_specs=_byte_specs(self.manifest.layer_specs),
                num_experts=self.manifest.num_experts,
                cold=MmapColdBacking(self.manifest.files),
                io_threads=config.io_threads,
            )
        self.prefetcher: ExpertPrefetcher | None = None
        predictors = config.predictor_set
        if predictors:
            self.prefetcher = ExpertPrefetcher(
                cache=self.cache,
                layer_ids=sorted(self.manifest.layer_specs),
                num_experts=self.manifest.num_experts,
                predictors=predictors,
                lookahead=config.prefetch_lookahead,
                popular_k=config.prefetch_popular,
                stats_fn=lambda: self.cache.stats()["total"],
                stats_interval_s=20.0,
                name=f"-rank{rank}",
            )
            logger.info(
                "expert-tier prefetch: rank %d, predictors %s, lookahead "
                "%d, popular top-%d",
                rank,
                sorted(predictors),
                config.prefetch_lookahead,
                config.prefetch_popular,
            )
        atexit.register(self._log_final_stats)

    def _log_final_stats(self) -> None:
        try:
            stats = json.dumps(self.cache.stats()["total"])
            prefetch = json.dumps(self.prefetcher.stats()) if self.prefetcher else "{}"
            logger.info(
                "expert-tier final stats: rank %d cache=%s prefetch=%s",
                self.rank,
                stats,
                prefetch,
            )
        except Exception:
            pass

    def bind_layer(self, layer_id: int) -> ExpertTierLayerBinding:
        if layer_id not in self.manifest.layer_specs:
            raise ValueError(
                f"expert-tier manifest {self.manifest.path} has no layer "
                f"{layer_id}; layers: {sorted(self.manifest.layer_specs)}"
            )
        return ExpertTierLayerBinding(
            cache=self.cache,
            layer_id=layer_id,
            num_experts=self.manifest.num_experts,
            specs=self.manifest.layer_specs[layer_id],
            gpu_slots=self.gpu_slots_per_layer[layer_id],
            prefetcher=self.prefetcher,
        )


_state: ExpertTierState | None = None


def reset_expert_tier_state() -> None:
    """Drop the process-wide cache state (tests only)."""
    global _state
    if _state is not None:
        if _state.prefetcher is not None:
            _state.prefetcher.close()
        _state.cache.close()
    _state = None


def notify_sampled_token_ids(token_ids: torch.Tensor) -> None:
    """Feed next-step token ids to the hash predictor, if active.

    Called from the model runner right after sampling; the ids may still
    be on the GPU (async D2H is handled by the prefetcher). No-op when
    expert-tier offloading or the hash predictor is disabled.
    """
    state = _state
    if state is not None and state.prefetcher is not None:
        state.prefetcher.observe_sampled_tokens(token_ids)


def _get_or_create_state(
    config: "ExpertTierOffloadConfig",
    device: torch.device,
    rank: int,
    max_routed_experts: int | None,
    include_mtp_layers: bool = False,
) -> ExpertTierState:
    global _state
    if _state is None:
        _state = ExpertTierState(
            config, device, rank, max_routed_experts, include_mtp_layers
        )
    elif _state.config != config or _state.rank != rank:
        raise ValueError(
            "expert-tier cache already initialized with a different "
            "config/rank in this process"
        )
    return _state


def _validate_layer(layer: "RoutedExperts") -> None:
    """Check the layer is servable by the baseline tier integration."""
    quant_method = layer.quant_method
    backend = getattr(quant_method, "mxfp4_backend", None)
    if backend is None or backend.name != "MARLIN":
        raise ValueError(
            "expert-tier offloading requires the Marlin MXFP4 MoE backend "
            f"(got {type(quant_method).__name__} / {backend}); run with "
            '--kernel-config \'{"moe_backend": "marlin"}\' on an MXFP4 '
            "expert checkpoint"
        )
    if quant_method.is_monolithic:
        raise ValueError(
            "expert-tier offloading requires a modular MoE kernel (routed "
            "ids must be visible to the runner before the kernel runs)"
        )
    parallel = layer.moe_config.moe_parallel_config
    if parallel.use_ep or parallel.dp_size > 1 or parallel.enable_eplb:
        raise ValueError(
            "expert-tier offloading supports plain TP only "
            f"(got ep={parallel.use_ep}, dp_size={parallel.dp_size}, "
            f"eplb={parallel.enable_eplb})"
        )


def maybe_init_expert_tier(
    layer: "RoutedExperts",
) -> ExpertTierLayerBinding | None:
    """Bind a RoutedExperts layer to the tier cache when the mode is on.

    Called from ``RoutedExperts.__init__`` after the quant method exists
    and before ``create_weights``. Returns None (and does nothing) when
    expert-tier offloading is disabled.
    """
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    offload_config = vllm_config.offload_config
    if offload_config is None or not offload_config.expert_tier.enabled:
        return None

    _validate_layer(layer)
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise ValueError(
            "expert-tier offloading does not support pipeline parallelism "
            "yet (pools are sized for all manifest layers per rank)"
        )

    # Delayed import to avoid a circular dependency.
    from vllm.model_executor.models.utils import extract_layer_index

    layer_id = extract_layer_index(layer.layer_name)
    max_routed = None
    scheduler_config = vllm_config.scheduler_config
    if scheduler_config is not None and scheduler_config.max_num_batched_tokens:
        max_routed = (
            scheduler_config.max_num_batched_tokens * layer.moe_config.experts_per_token
        )
    state = _get_or_create_state(
        offload_config.expert_tier,
        device=torch.get_default_device(),
        rank=layer.moe_config.tp_rank,
        max_routed_experts=max_routed,
        include_mtp_layers=vllm_config.speculative_config is not None,
    )
    if state.manifest.num_experts != layer.global_num_experts:
        raise ValueError(
            f"expert-tier manifest has {state.manifest.num_experts} experts "
            f"but layer {layer.layer_name} has {layer.global_num_experts}"
        )
    return state.bind_layer(layer_id)
