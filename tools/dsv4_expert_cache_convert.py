# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline DeepSeek-V4-Flash MXFP4 expert cache converter (Marlin backend).

Converts the checkpoint's routed-expert weights into per-rank, per-layer,
per-tensor raw cache files holding exactly the bytes vLLM's Marlin MXFP4
MoE path produces after ``process_weights_after_loading``. The output is
the cold tier consumed by ``ExpertTierCache`` / ``MmapColdBacking``
(vllm/model_executor/offloader/expert_tier.py): one raw file per
(layer, tensor), all experts contiguous expert-major.

Byte-identity is by construction: each layer is converted by building the
real ``RoutedExperts`` module with ``Mxfp4MoEMethod`` in a faked TP-rank
context, loading checkpoint tensors through the module's own
``weight_loader``, and running ``process_weights_after_loading``.

Example:
    python tools/dsv4_expert_cache_convert.py \\
        --checkpoint-dir <snapshot_dir> \\
        --out-dir /shared/vllm/vllm-cache/dsv4-marlin-tp4 \\
        --tp-size 4 --ranks 0,1,2,3 --layers 0-42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import time
from typing import Any

import numpy as np
import torch

CACHE_TENSOR_NAMES = (
    "w13_weight",
    "w13_weight_scale",
    "w2_weight",
    "w2_weight_scale",
)

MARLIN_TILE = 16
MXFP4_GROUP = 32
CONVERTER_VERSION = 1

_TORCH_DTYPES = {
    "torch.int32": torch.int32,
    "torch.uint8": torch.uint8,
    "torch.float8_e8m0fnu": torch.float8_e8m0fnu,
}


def parse_int_list(spec: str) -> list[int]:
    """Parse an int list spec like ``"0-42"`` or ``"0,2,5-7"``.

    Args:
        spec: Comma-separated ints and inclusive ranges.

    Returns:
        Sorted, de-duplicated list of ints.

    Raises:
        ValueError: On malformed input or descending ranges.
    """
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"descending range {part!r} in {spec!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"empty int list spec {spec!r}")
    return sorted(out)


def expected_marlin_specs(
    tp_size: int,
    hidden_size: int = 4096,
    intermediate_size: int = 2048,
) -> dict[str, dict[str, Any]]:
    """Expected per-expert Marlin tensor specs for one TP rank.

    Mirrors ``Mxfp4MoEMethod`` create_weights + Marlin repack shapes:
    ``gptq_marlin_repack`` maps (size_n, size_k//2) uint8 FP4 pairs to
    (size_k//16, size_n*2) int32, and scale permutation yields
    (size_k//group, size_n) e8m0. w13 uses size_n=2*n, size_k=hidden;
    w2 uses size_n=hidden, size_k=n (n = intermediate per partition).

    Args:
        tp_size: Tensor-parallel world size.
        hidden_size: Model hidden size.
        intermediate_size: Full (unsharded) MoE intermediate size.

    Returns:
        Mapping of tensor name to {"per_expert_shape", "dtype",
        "row_bytes"}.

    Raises:
        ValueError: If sizes violate Marlin alignment (would trigger
            padding, which this converter intentionally does not model).
    """
    if intermediate_size % tp_size != 0:
        raise ValueError(
            f"intermediate_size={intermediate_size} not divisible by "
            f"tp_size={tp_size}"
        )
    n = intermediate_size // tp_size
    if n % 128 != 0 or hidden_size % 256 != 0:
        raise ValueError(
            f"marlin would pad sizes (n={n}, hidden={hidden_size}); "
            "converter requires pre-aligned sizes"
        )
    e8m0 = "torch.float8_e8m0fnu"
    specs = {
        "w13_weight": ((hidden_size // MARLIN_TILE, 2 * n * 2), "torch.int32"),
        "w13_weight_scale": ((hidden_size // MXFP4_GROUP, 2 * n), e8m0),
        "w2_weight": ((n // MARLIN_TILE, hidden_size * 2), "torch.int32"),
        "w2_weight_scale": ((n // MXFP4_GROUP, hidden_size), e8m0),
    }
    out = {}
    for name, (shape, dtype) in specs.items():
        itemsize = torch.empty(0, dtype=_TORCH_DTYPES[dtype]).element_size()
        row_bytes = int(np.prod(shape)) * itemsize
        out[name] = {
            "per_expert_shape": list(shape),
            "dtype": dtype,
            "row_bytes": row_bytes,
        }
    return out


def ckpt_shard_slice(
    shard_id: str, ckpt_shape: tuple[int, ...], tp_rank: int, tp_size: int
) -> tuple[int, int, int]:
    """Checkpoint slice one TP rank loads from an expert tensor.

    Mirrors ``RoutedExperts._load_w13`` / ``_load_w2``: w1/w3 shard the
    leading (intermediate) dim, w2 shards the trailing dim. Applies to
    both packed weights and block scales (same dims, different widths).

    Args:
        shard_id: One of "w1", "w2", "w3".
        ckpt_shape: 2D shape of the checkpoint tensor for one expert.
        tp_rank: TP rank.
        tp_size: TP world size.

    Returns:
        Tuple ``(dim, start, length)`` to ``narrow`` the tensor with.
    """
    if shard_id in ("w1", "w3"):
        dim = 0
    elif shard_id == "w2":
        dim = 1
    else:
        raise ValueError(f"bad shard_id {shard_id!r}")
    per_rank = ckpt_shape[dim] // tp_size
    return dim, per_rank * tp_rank, per_rank


def cache_file_name(layer_id: int, tensor_name: str) -> str:
    """Cache file basename for one (layer, tensor)."""
    return f"layer{layer_id:03d}.{tensor_name}.raw"


def file_complete(path: str, expected_bytes: int) -> bool:
    """Whether ``path`` exists with exactly ``expected_bytes`` bytes."""
    try:
        return os.path.getsize(path) == expected_bytes
    except OSError:
        return False


def build_layer_manifest(
    layer_id: int, specs: dict[str, dict[str, Any]], num_experts: int
) -> dict[str, Any]:
    """Manifest entries for one layer's cache files.

    Args:
        layer_id: Model layer index.
        specs: Output of :func:`expected_marlin_specs`.
        num_experts: Global routed expert count.

    Returns:
        Mapping of tensor name to its manifest entry.
    """
    entries = {}
    for name, spec in specs.items():
        entries[name] = {
            "file": cache_file_name(layer_id, name),
            "dtype": spec["dtype"],
            "per_expert_shape": spec["per_expert_shape"],
            "row_bytes": spec["row_bytes"],
            "nbytes": spec["row_bytes"] * num_experts,
        }
    return entries


def update_manifest(
    manifest_path: str,
    meta: dict[str, Any],
    num_experts: int,
    layer_id: int,
    layer_entries: dict[str, Any],
) -> None:
    """Merge one layer's entries into the rank manifest atomically."""
    manifest: dict[str, Any] = {
        "meta": meta,
        "num_experts": num_experts,
        "layers": {},
    }
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["meta"] = meta
        manifest["num_experts"] = num_experts
    manifest.setdefault("layers", {})[str(layer_id)] = layer_entries
    manifest["layers"] = {
        k: manifest["layers"][k]
        for k in sorted(manifest["layers"], key=int)
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, manifest_path)


def safetensors_expected_size(path: str) -> int:
    """Total on-disk size a safetensors shard should have.

    Computed from the header only, so it detects truncated downloads
    without reading tensor data.
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_end = max(
        (v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"),
        default=0,
    )
    return 8 + header_len + data_end


def resolve_snapshot(checkpoint_dir: str) -> str:
    """Resolve a checkpoint dir or HF hub cache root to a snapshot dir."""
    if os.path.isfile(os.path.join(checkpoint_dir, "model.safetensors.index.json")):
        return checkpoint_dir
    snap_root = os.path.join(checkpoint_dir, "snapshots")
    if os.path.isdir(snap_root):
        snaps = sorted(os.listdir(snap_root))
        if len(snaps) == 1:
            return os.path.join(snap_root, snaps[0])
        raise ValueError(f"expected exactly one snapshot in {snap_root}: {snaps}")
    raise ValueError(f"no model.safetensors.index.json under {checkpoint_dir}")


def wait_for_complete_shards(
    snapshot: str, shard_files: list[str], timeout_s: float, poll_s: float = 30.0
) -> None:
    """Block until all shards exist with their full size.

    Raises:
        TimeoutError: If shards are still incomplete after ``timeout_s``.
    """
    deadline = time.time() + timeout_s
    while True:
        incomplete = []
        for shard in shard_files:
            path = os.path.join(snapshot, shard)
            try:
                expected = safetensors_expected_size(path)
                if os.path.getsize(path) != expected:
                    incomplete.append(shard)
            except (OSError, json.JSONDecodeError, struct.error):
                incomplete.append(shard)
        if not incomplete:
            return
        if time.time() > deadline:
            raise TimeoutError(
                f"{len(incomplete)} shards incomplete after {timeout_s}s: "
                f"{incomplete[:4]}..."
            )
        _log(f"waiting for {len(incomplete)} incomplete shards...")
        time.sleep(poll_s)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class LayerTensorLoader:
    """Loads one MoE layer's expert tensors from safetensors shards.

    Scales are returned as raw uint8 views (matching the model's
    load_weights, which views float8_e8m0fnu checkpoints as uint8 before
    the MoE weight_loader).
    """

    def __init__(self, snapshot: str, num_experts: int):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.snapshot = snapshot
        self.num_experts = num_experts
        with open(os.path.join(snapshot, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles: dict[str, Any] = {}

    def shard_files(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    def has_layer(self, layer_id: int) -> bool:
        return f"layers.{layer_id}.ffn.experts.0.w1.weight" in self.weight_map

    def _get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = self._safe_open(
                os.path.join(self.snapshot, shard), framework="pt", device="cpu"
            )
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def load_layer(
        self, layer_id: int, experts: list[int] | None = None
    ) -> dict[tuple[int, str, str], torch.Tensor]:
        """Load (expert, shard_id, kind) -> tensor for one layer.

        Args:
            layer_id: Model layer index.
            experts: Expert ids to load; all when None.

        Returns:
            Dict keyed by ``(expert_id, shard_id, kind)`` where kind is
            "weight" or "scale".
        """
        expert_ids = range(self.num_experts) if experts is None else experts
        out: dict[tuple[int, str, str], torch.Tensor] = {}
        for e in expert_ids:
            for w in ("w1", "w2", "w3"):
                base = f"layers.{layer_id}.ffn.experts.{e}.{w}"
                weight = self._get(f"{base}.weight")
                scale = self._get(f"{base}.scale")
                if scale.dtype == torch.float8_e8m0fnu:
                    scale = scale.view(torch.uint8)
                out[(e, w, "weight")] = weight
                out[(e, w, "scale")] = scale
        return out


def _hf_config(snapshot: str) -> dict[str, Any]:
    with open(os.path.join(snapshot, "config.json")) as f:
        return json.load(f)


def build_routed_experts(
    hf_config: dict[str, Any], tp_rank: int, tp_size: int, device: str
):
    """Build a standalone DeepSeek-V4 RoutedExperts module for one rank.

    Reproduces the FusedMoE construction path (fused_moe/layer.py) with a
    faked TP-rank parallel config, so ``create_weights``, the
    ``weight_loader`` TP slicing, and ``process_weights_after_loading``
    behave exactly as they do at serve time with tp_size ranks.
    """
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        MoEActivation,
        get_routing_method_type,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        ExpertMapManager,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    num_experts = hf_config["n_routed_experts"]
    top_k = hf_config["num_experts_per_tok"]
    mpc = FusedMoEParallelConfig(
        tp_size=tp_size,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=tp_rank,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    emm = ExpertMapManager(
        max_num_batched_tokens=8192,
        top_k=top_k,
        global_num_experts=num_experts,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=mpc,
        placement_strategy="linear",
        enable_eplb=False,
    )
    moe = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hf_config["hidden_size"],
        intermediate_size=hf_config["moe_intermediate_size"],
        num_local_experts=emm.local_num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.from_str(hf_config["hidden_act"]),
        device=device,
        routing_method=get_routing_method_type(
            scoring_func=hf_config["scoring_func"],
            top_k=top_k,
            renormalize=hf_config["norm_topk_prob"],
            num_expert_group=None,
            has_e_score_bias=True,
            routed_scaling_factor=hf_config.get("routed_scaling_factor", 1.0),
        ),
        moe_parallel_config=mpc,
        in_dtype=torch.bfloat16,
        router_logits_dtype=torch.float32,
        moe_backend="auto",
        max_num_tokens=8192,
        has_bias=False,
        swiglu_limit=hf_config.get("swiglu_limit"),
    )
    with torch.device(device):
        layer = RoutedExperts(
            layer_name=f"dsv4_convert_rank{tp_rank}",
            params_dtype=torch.bfloat16,
            moe_config=moe,
            quant_config=_make_shim_quant_config(),
            expert_map_manager=emm,
        )
    method = type(layer.quant_method).__name__
    backend = layer.quant_method.mxfp4_backend.value
    if method != "Mxfp4MoEMethod" or backend != "MARLIN":
        raise RuntimeError(
            f"unexpected quant path {method}/{backend}; this converter "
            "targets the Marlin MXFP4 MoE backend"
        )
    return layer


def _make_shim_quant_config():
    """Minimal quant config returning the exact DeepSeek-V4 fp4 method.

    ``DeepseekV4FP8Config.get_quant_method`` returns
    ``Mxfp4MoEMethod(layer.moe_config)`` for fp4 RoutedExperts; this shim
    does the same without needing a full VllmConfig/hf_config context.
    """
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    class _ShimConfig(QuantizationConfig):
        @classmethod
        def get_name(cls):
            return "deepseek_v4_fp8"

        @classmethod
        def get_supported_act_dtypes(cls):
            return [torch.bfloat16]

        @classmethod
        def get_min_capability(cls):
            return 80

        @classmethod
        def get_config_filenames(cls):
            return []

        @classmethod
        def from_config(cls, config):
            return cls()

        def get_quant_method(self, layer, prefix):
            if isinstance(layer, RoutedExperts):
                return Mxfp4MoEMethod(layer.moe_config)
            return None

    return _ShimConfig()


def load_into_layer(
    layer, tensors: dict[tuple[int, str, str], torch.Tensor]
) -> None:
    """Feed checkpoint tensors through the layer's own weight_loader."""
    for (expert_id, shard_id, kind), tensor in tensors.items():
        pname = "w13" if shard_id in ("w1", "w3") else "w2"
        pname += "_weight" if kind == "weight" else "_weight_scale"
        layer.weight_loader(
            getattr(layer, pname),
            tensor,
            pname,
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=False,
        )


def check_loaded_slices(
    layer,
    tensors: dict[tuple[int, str, str], torch.Tensor],
    expert_ids: list[int],
    tp_rank: int,
    tp_size: int,
) -> None:
    """Cross-check weight_loader TP slicing against ckpt_shard_slice.

    Raises:
        RuntimeError: On any slicing mismatch (guards silent divergence
            between this tool's slicing model and vLLM's loader).
    """
    for e in expert_ids:
        for shard_id in ("w1", "w2", "w3"):
            for kind in ("weight", "scale"):
                src = tensors[(e, shard_id, kind)]
                dim, start, length = ckpt_shard_slice(
                    shard_id, tuple(src.shape), tp_rank, tp_size
                )
                expected = (
                    src.narrow(dim, start, length).contiguous().view(torch.uint8)
                )
                pname = "w13" if shard_id in ("w1", "w3") else "w2"
                pname += "_weight" if kind == "weight" else "_weight_scale"
                param = getattr(layer, pname).data[e]
                if shard_id == "w1":
                    got = param.narrow(0, 0, length)
                elif shard_id == "w3":
                    got = param.narrow(0, param.shape[0] // 2, length)
                else:
                    got = param
                got = got.contiguous().view(torch.uint8).cpu()
                if not torch.equal(got, expected):
                    raise RuntimeError(
                        f"TP slicing mismatch: expert {e} {shard_id} {kind} "
                        f"rank {tp_rank}/{tp_size}"
                    )


def _build_meta(
    hf_config: dict[str, Any],
    snapshot: str,
    tp_size: int,
    tp_rank: int,
    layer,
) -> dict[str, Any]:
    import vllm
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        get_marlin_input_dtype,
    )

    input_dtype = get_marlin_input_dtype()
    return {
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "model_sha": os.path.basename(snapshot.rstrip("/")),
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "num_experts": hf_config["n_routed_experts"],
        "hidden_size": hf_config["hidden_size"],
        "intermediate_size": hf_config["moe_intermediate_size"],
        "quant_method": type(layer.quant_method).__name__,
        "mxfp4_backend": layer.quant_method.mxfp4_backend.value,
        "experts_cls": layer.quant_method.experts_cls.__name__,
        "marlin_input_dtype": str(input_dtype) if input_dtype else None,
        "params_dtype": "torch.bfloat16",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "converter_version": CONVERTER_VERSION,
    }


def _dump_param(param: torch.Tensor, path: str) -> None:
    data = param.detach().cpu().contiguous().view(torch.uint8)
    tmp = path + ".tmp"
    data.numpy().tofile(tmp)
    os.replace(tmp, path)


def convert(args: argparse.Namespace) -> None:
    """Run the conversion loop: layer-major, ranks inner, resumable."""
    from vllm.config import VllmConfig, set_current_vllm_config

    snapshot = resolve_snapshot(args.checkpoint_dir)
    hf_config = _hf_config(snapshot)
    num_experts = hf_config["n_routed_experts"]
    layers = parse_int_list(args.layers)
    ranks = parse_int_list(args.ranks)
    specs = expected_marlin_specs(
        args.tp_size,
        hf_config["hidden_size"],
        hf_config["moe_intermediate_size"],
    )
    loader = LayerTensorLoader(snapshot, num_experts)
    for layer_id in layers:
        if not loader.has_layer(layer_id):
            raise ValueError(f"layer {layer_id} has no routed experts")
    wait_for_complete_shards(snapshot, loader.shard_files(), args.wait_timeout)

    rank_dirs = {}
    for r in ranks:
        rank_dirs[r] = os.path.join(args.out_dir, f"rank{r}")
        os.makedirs(rank_dirs[r], exist_ok=True)

    layer_times: list[float] = []
    slicing_checked: set[int] = set()
    with set_current_vllm_config(VllmConfig()):
        for i, layer_id in enumerate(layers):
            t_layer = time.time()
            todo: dict[int, list[str]] = {}
            for r in ranks:
                missing = [
                    name
                    for name in CACHE_TENSOR_NAMES
                    if not file_complete(
                        os.path.join(rank_dirs[r], cache_file_name(layer_id, name)),
                        specs[name]["row_bytes"] * num_experts,
                    )
                ]
                if missing:
                    todo[r] = missing
            layer_entries = build_layer_manifest(layer_id, specs, num_experts)
            if not todo:
                for r in ranks:
                    _finish_rank_layer(
                        args, hf_config, snapshot, rank_dirs[r], r, layer_id,
                        layer_entries, layer=None,
                    )
                _log(f"layer {layer_id}: complete, skipped")
                continue

            t0 = time.time()
            tensors = loader.load_layer(layer_id)
            load_s = time.time() - t0
            for r in ranks:
                if r not in todo:
                    _finish_rank_layer(
                        args, hf_config, snapshot, rank_dirs[r], r, layer_id,
                        layer_entries, layer=None,
                    )
                    continue
                t0 = time.time()
                layer = build_routed_experts(hf_config, r, args.tp_size, args.device)
                load_into_layer(layer, tensors)
                if r not in slicing_checked:
                    check_loaded_slices(layer, tensors, [0], r, args.tp_size)
                    slicing_checked.add(r)
                layer.quant_method.process_weights_after_loading(layer)
                torch.cuda.synchronize()
                proc_s = time.time() - t0
                t0 = time.time()
                for name in todo[r]:
                    param = getattr(layer, name)
                    shape = tuple(param.shape)
                    expected = (num_experts, *specs[name]["per_expert_shape"])
                    if shape != expected:
                        raise RuntimeError(
                            f"{name} shape {shape} != expected {expected}"
                        )
                    out_path = os.path.join(
                        rank_dirs[r], cache_file_name(layer_id, name)
                    )
                    _dump_param(param, out_path)
                dump_s = time.time() - t0
                _finish_rank_layer(
                    args, hf_config, snapshot, rank_dirs[r], r, layer_id,
                    layer_entries, layer=layer,
                )
                del layer
                torch.cuda.empty_cache()
                _log(
                    f"layer {layer_id} rank {r}: proc {proc_s:.1f}s "
                    f"dump {dump_s:.1f}s ({len(todo[r])} tensors)"
                )
            del tensors
            layer_times.append(time.time() - t_layer)
            done = i + 1
            avg = sum(layer_times) / len(layer_times)
            eta_min = avg * (len(layers) - done) / 60
            _log(
                f"layer {layer_id} done in {layer_times[-1]:.1f}s "
                f"(load {load_s:.1f}s) [{done}/{len(layers)}] "
                f"ETA {eta_min:.1f} min"
            )
    _log("conversion complete")


_META_CACHE: dict[int, dict[str, Any]] = {}


def _finish_rank_layer(
    args, hf_config, snapshot, rank_dir, tp_rank, layer_id, layer_entries, layer
) -> None:
    """Update the rank manifest for one completed (or skipped) layer."""
    meta = _META_CACHE.get(tp_rank)
    if meta is None:
        if layer is None:
            layer = build_routed_experts(
                hf_config, tp_rank, args.tp_size, args.device
            )
            meta = _build_meta(hf_config, snapshot, args.tp_size, tp_rank, layer)
            del layer
            torch.cuda.empty_cache()
        else:
            meta = _build_meta(hf_config, snapshot, args.tp_size, tp_rank, layer)
        _META_CACHE[tp_rank] = meta
    update_manifest(
        os.path.join(rank_dir, "manifest.json"),
        meta,
        hf_config["n_routed_experts"],
        layer_id,
        layer_entries,
    )


def verify(args: argparse.Namespace) -> None:
    """Recompute N random (layer, expert) rows and byte-compare files."""
    from vllm.config import VllmConfig, set_current_vllm_config

    snapshot = resolve_snapshot(args.checkpoint_dir)
    hf_config = _hf_config(snapshot)
    num_experts = hf_config["n_routed_experts"]
    layers = parse_int_list(args.layers)
    ranks = parse_int_list(args.ranks)
    specs = expected_marlin_specs(
        args.tp_size,
        hf_config["hidden_size"],
        hf_config["moe_intermediate_size"],
    )
    rng = random.Random(args.seed)
    samples = [
        (rng.choice(layers), rng.randrange(num_experts)) for _ in range(args.verify)
    ]
    by_layer: dict[int, list[int]] = {}
    for layer_id, e in samples:
        by_layer.setdefault(layer_id, []).append(e)

    loader = LayerTensorLoader(snapshot, num_experts)
    failures = 0
    checked = 0
    with set_current_vllm_config(VllmConfig()):
        for layer_id, experts in sorted(by_layer.items()):
            experts = sorted(set(experts))
            tensors = loader.load_layer(layer_id, experts)
            for r in ranks:
                rank_dir = os.path.join(args.out_dir, f"rank{r}")
                layer = build_routed_experts(hf_config, r, args.tp_size, args.device)
                load_into_layer(layer, tensors)
                check_loaded_slices(layer, tensors, experts, r, args.tp_size)
                layer.quant_method.process_weights_after_loading(layer)
                torch.cuda.synchronize()
                for name in CACHE_TENSOR_NAMES:
                    row_bytes = specs[name]["row_bytes"]
                    path = os.path.join(rank_dir, cache_file_name(layer_id, name))
                    if not file_complete(path, row_bytes * num_experts):
                        raise FileNotFoundError(
                            f"cache file missing or wrong size: {path}"
                        )
                    mm = np.memmap(path, dtype=np.uint8, mode="r")
                    param = getattr(layer, name)
                    for e in experts:
                        got = mm[e * row_bytes : (e + 1) * row_bytes]
                        want = (
                            param.data[e]
                            .cpu()
                            .contiguous()
                            .view(torch.uint8)
                            .numpy()
                            .reshape(-1)
                        )
                        checked += 1
                        if not np.array_equal(got, want):
                            failures += 1
                            _log(
                                f"MISMATCH layer {layer_id} expert {e} rank {r} "
                                f"{name}"
                            )
                del layer
                torch.cuda.empty_cache()
            _log(f"verified layer {layer_id} experts {experts} on ranks {ranks}")
    _log(f"verify: {checked - failures}/{checked} rows byte-identical")
    if failures:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--ranks", default="0,1,2,3")
    parser.add_argument("--layers", default="0-42")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="verify N random (layer, expert) rows instead of converting",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=6 * 3600,
        help="max seconds to wait for the checkpoint download to finish",
    )
    args = parser.parse_args()
    for r in parse_int_list(args.ranks):
        if not 0 <= r < args.tp_size:
            raise SystemExit(f"rank {r} out of range for tp_size {args.tp_size}")
    if args.verify > 0:
        verify(args)
    else:
        convert(args)


if __name__ == "__main__":
    main()
