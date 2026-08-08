# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone tiered expert-weight cache for MoE offloading.

Expert weights live in three tiers per rank:

- GPU tier: fixed slot pools allocated once at init. One slot holds one
  expert's row across all of a layer's expert tensors.
- Pinned-CPU tier: larger slot pools in pinned host memory, the DMA
  source for H2D copies on a dedicated copy stream owned by the cache.
- Cold tier: read-only backing (e.g. mmap'd files on NVMe) accessed by
  a small IO thread pool that only fills pinned memory (no CUDA calls).

Integration contract: register per-layer expert tensor groups at load
time, call :meth:`ExpertTierCache.ensure` with the routed expert ids for
a layer each forward step (paired with :meth:`ExpertTierCache.release`),
and optionally call :meth:`ExpertTierCache.prefetch` from a predictor
with expert ids for future layers/steps.
"""

import os
import threading
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as _futures_wait
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import get_dtype_size

logger = init_logger(__name__)

__all__ = [
    "ColdBacking",
    "EnsureResult",
    "ExpertTensorSpec",
    "ExpertTierCache",
    "MmapColdBacking",
]

# Rows at or below this size go through a pinned staging gather plus a
# single H2D copy and one index_copy_ scatter; larger rows use one
# non_blocking DMA per row directly from the pinned pool. See module
# tests/benchmarks for the crossover measurement.
_STAGED_H2D_MAX_ROW_BYTES = 65536

_STAT_KEYS = (
    "ensure_calls",
    "prefetch_calls",
    "preload_calls",
    "gpu_hits",
    "pinned_hits",
    "cold_misses",
    "inflight_waits",
    "bytes_cold_to_pinned",
    "bytes_pinned_to_gpu",
    "gpu_evictions",
    "pinned_evictions",
    "pinned_stalls",
    "prefetch_drops",
    "io_errors",
)


@dataclass(frozen=True)
class ExpertTensorSpec:
    """Per-expert tensor metadata for one layer.

    Attributes:
        name: Tensor name, e.g. "w13_weight_packed".
        shape: Per-expert shape (leading expert dim removed).
        dtype: Tensor dtype.
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def row_bytes(self) -> int:
        """Bytes of one expert's row of this tensor."""
        numel = 1
        for dim in self.shape:
            numel *= dim
        return numel * get_dtype_size(self.dtype)


@runtime_checkable
class ColdBacking(Protocol):
    """Read-only cold-tier source of expert weights."""

    def read_into(
        self,
        layer_id: int,
        expert_id: int,
        name: str,
        dst_cpu_tensor: torch.Tensor,
    ) -> None:
        """Fill ``dst_cpu_tensor`` with the (layer, expert, tensor) data.

        Must be thread-safe (called concurrently from IO threads).
        ``dst_cpu_tensor`` is a contiguous CPU tensor with the spec's
        per-expert shape and dtype; implementations must write it fully.

        Raises:
            Exception: Any error; it fails the load of that expert.
        """
        ...


class MmapColdBacking:
    """Cold tier backed by one raw file per (layer, tensor).

    Each file holds the expert-major array ``[num_experts, *spec.shape]``
    as raw row-major bytes. Files are mmap'd lazily on first read;
    nothing is read at construction time.

    Args:
        manifest: Maps ``(layer_id, tensor_name)`` to the file path.

    Raises:
        FileNotFoundError: If a manifest path does not exist.
    """

    def __init__(self, manifest: dict[tuple[int, str], str]):
        for key, path in manifest.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"cold backing file missing for {key}: {path}")
        self._manifest = dict(manifest)
        self._maps: dict[tuple[int, str], np.memmap] = {}
        self._lock = threading.Lock()

    def _get_map(self, key: tuple[int, str]) -> np.memmap:
        with self._lock:
            mm = self._maps.get(key)
            if mm is None:
                path = self._manifest.get(key)
                if path is None:
                    raise KeyError(f"no cold backing registered for {key}")
                mm = np.memmap(path, dtype=np.uint8, mode="r")
                self._maps[key] = mm
            return mm

    def read_into(
        self,
        layer_id: int,
        expert_id: int,
        name: str,
        dst_cpu_tensor: torch.Tensor,
    ) -> None:
        """Copy one expert row from the mmap'd file into a CPU tensor."""
        mm = self._get_map((layer_id, name))
        nbytes = dst_cpu_tensor.numel() * dst_cpu_tensor.element_size()
        start = expert_id * nbytes
        end = start + nbytes
        if end > mm.shape[0]:
            raise ValueError(
                f"cold backing for ({layer_id}, {name}) too small: expert "
                f"{expert_id} needs bytes [{start}, {end}), file has "
                f"{mm.shape[0]}"
            )
        dst_np = dst_cpu_tensor.view(torch.uint8).view(-1).numpy()
        np.copyto(dst_np, mm[start:end])


@dataclass
class EnsureResult:
    """Result of :meth:`ExpertTierCache.ensure` for one layer.

    Attributes:
        gpu_tensors: The layer's stable GPU slot pools keyed by tensor
            name, each shaped ``[gpu_slots, *spec.shape]``. Owned by the
            cache; treat as read-only. The tensor objects never change
            across calls, so kernels may hold references.
        slot_of: int32 tensor ``[num_experts]`` on the cache device
            mapping global expert id -> GPU slot (-1 when not resident).
            A live view owned by the cache, updated by later ensure/
            prefetch calls for this layer.
        event: CUDA event recorded on the copy stream after the last
            enqueued copy for this layer; the caller's compute stream
            already waits on it. None on CPU devices or when nothing was
            ever copied. Reused (re-recorded) by later calls.
    """

    gpu_tensors: dict[str, torch.Tensor]
    slot_of: torch.Tensor
    event: torch.cuda.Event | None


@dataclass
class _LoadJob:
    """An in-flight cold->pinned load owning one pinned slot."""

    slot: int
    futures: list[Future] = field(default_factory=list)
    wants_gpu: bool = False


class _TierState:
    """LRU slot bookkeeping for one tier of one layer."""

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        # expert_id -> slot; iteration order is LRU (oldest first).
        self.map: OrderedDict[int, int] = OrderedDict()
        self.free: list[int] = list(range(num_slots - 1, -1, -1))

    def get(self, expert_id: int) -> int | None:
        return self.map.get(expert_id)

    def touch(self, expert_id: int) -> None:
        self.map.move_to_end(expert_id)

    def insert(self, expert_id: int, slot: int) -> None:
        self.map[expert_id] = slot
        self.map.move_to_end(expert_id)

    def take_slot(self, protected: set[int]) -> tuple[int, int | None] | None:
        """Return (slot, evicted_expert_or_None), or None if exhausted."""
        if self.free:
            return self.free.pop(), None
        victim = next((e for e in self.map if e not in protected), None)
        if victim is None:
            return None
        return self.map.pop(victim), victim


class _LayerState:
    """All per-layer pools, maps, events, and counters."""

    def __init__(
        self,
        layer_id: int,
        specs: list[ExpertTensorSpec],
        gpu_slots: int,
        pinned_slots: int,
        num_experts: int,
        device: torch.device,
        is_cuda: bool,
    ):
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"layer {layer_id}: duplicate tensor names {names}")
        if not specs:
            raise ValueError(f"layer {layer_id}: empty tensor spec list")
        if gpu_slots < 1 or pinned_slots < 1:
            raise ValueError(
                f"layer {layer_id}: slot counts must be >= 1, got "
                f"gpu={gpu_slots} pinned={pinned_slots}"
            )
        self.layer_id = layer_id
        self.specs = list(specs)
        self.names = names
        self.gpu = _TierState(gpu_slots)
        self.pinned = _TierState(pinned_slots)
        self.gpu_pool: dict[str, torch.Tensor] = {}
        self.pinned_pool: dict[str, torch.Tensor] = {}
        self.small_names: list[str] = []
        self.large_names: list[str] = []
        self.stage_pin: dict[str, torch.Tensor] = {}
        self.stage_dev: dict[str, torch.Tensor] = {}
        self.bytes_per_expert = 0
        for spec in specs:
            self.bytes_per_expert += spec.row_bytes
            self.gpu_pool[spec.name] = torch.empty(
                (gpu_slots, *spec.shape), dtype=spec.dtype, device=device
            )
            self.pinned_pool[spec.name] = torch.empty(
                (pinned_slots, *spec.shape),
                dtype=spec.dtype,
                device="cpu",
                pin_memory=is_cuda,
            )
            if is_cuda and spec.row_bytes <= _STAGED_H2D_MAX_ROW_BYTES:
                self.small_names.append(spec.name)
                self.stage_pin[spec.name] = torch.empty(
                    (gpu_slots, *spec.shape),
                    dtype=spec.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                self.stage_dev[spec.name] = torch.empty(
                    (gpu_slots, *spec.shape), dtype=spec.dtype, device=device
                )
            else:
                self.large_names.append(spec.name)
        self.slot_of_cpu = torch.full(
            (num_experts,), -1, dtype=torch.int32, pin_memory=is_cuda
        )
        self.slot_of_dev = self.slot_of_cpu.to(device) if is_cuda else self.slot_of_cpu
        self.idx_pin: torch.Tensor | None = None
        self.idx_dev: torch.Tensor | None = None
        if self.small_names:
            self.idx_pin = torch.empty(gpu_slots, dtype=torch.int64, pin_memory=True)
            self.idx_dev = torch.empty(gpu_slots, dtype=torch.int64, device=device)
        self.stage_busy = False
        # Expert ids pinned by unreleased ensure() calls (unevictable).
        self.pins: set[int] = set()
        # expert_id -> in-flight cold->pinned load.
        self.loading: dict[int, _LoadJob] = {}
        # Pinned slots that were H2D sources since the last verified-
        # complete copy-stream sync; must not be overwritten until then.
        self.pinned_srcs: set[int] = set()
        self.fill_event = torch.cuda.Event() if is_cuda else None
        self.fill_recorded = False
        self.release_event = torch.cuda.Event() if is_cuda else None
        self.release_recorded = False
        self.compute_stream: torch.cuda.Stream | None = None
        self.stats: dict[str, int] = dict.fromkeys(_STAT_KEYS, 0)


class ExpertTierCache:
    """Tiered (GPU / pinned-CPU / cold) expert-weight cache.

    All pools are allocated once at construction; no per-step tensor
    allocation happens on the hot path. Eviction is LRU per tier;
    experts pinned between :meth:`ensure` and :meth:`release` are never
    evicted from the GPU tier, and their pinned-tier entries are also
    protected.

    Concurrency contract: every public method is thread-safe behind one
    internal lock. IO threads never issue CUDA calls; all H2D copies go
    on the single copy stream owned by the cache.

    Caller preconditions:
        - ``gpu_slots_per_layer[l]`` must be >= the maximum concurrent
          (ensure..release) working set for layer ``l``; ensure() raises
          RuntimeError otherwise.
        - ``pinned_slots_per_layer[l]`` must be >= the per-call cold
          working set (>= gpu slots recommended).
        - Expert ids must lie in ``[0, num_experts)``.

    Args:
        device: GPU-tier device ("cuda", "cuda:N", or "cpu" for tests;
            on "cpu" all copies are synchronous and events are None).
        gpu_slots_per_layer: layer_id -> GPU slot-pool size.
        pinned_slots_per_layer: layer_id -> pinned slot-pool size.
        layer_specs: layer_id -> per-expert tensor specs for that layer.
        num_experts: Global routed expert count per layer.
        cold: Cold-tier backing (source of truth for expert weights).
        io_threads: Worker threads for cold->pinned reads.
    """

    def __init__(
        self,
        device: torch.device | str,
        gpu_slots_per_layer: dict[int, int],
        pinned_slots_per_layer: dict[int, int],
        layer_specs: dict[int, list[ExpertTensorSpec]],
        num_experts: int,
        cold: ColdBacking,
        io_threads: int = 4,
    ):
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        layer_ids = set(layer_specs)
        if set(gpu_slots_per_layer) != layer_ids or (
            set(pinned_slots_per_layer) != layer_ids
        ):
            raise ValueError(
                "gpu_slots_per_layer, pinned_slots_per_layer, and "
                "layer_specs must cover the same layer ids"
            )
        self._device = torch.device(device)
        self._is_cuda = self._device.type == "cuda"
        self._num_experts = num_experts
        self._cold = cold
        self._lock = threading.Lock()
        self._copy_stream = (
            torch.cuda.Stream(device=self._device) if self._is_cuda else None
        )
        self._io = ThreadPoolExecutor(
            max_workers=io_threads, thread_name_prefix="expert_tier_io"
        )
        self._layers = {
            layer_id: _LayerState(
                layer_id=layer_id,
                specs=layer_specs[layer_id],
                gpu_slots=gpu_slots_per_layer[layer_id],
                pinned_slots=pinned_slots_per_layer[layer_id],
                num_experts=num_experts,
                device=self._device,
                is_cuda=self._is_cuda,
            )
            for layer_id in sorted(layer_ids)
        }
        gpu_bytes = sum(
            st.bytes_per_expert * st.gpu.num_slots for st in self._layers.values()
        )
        pinned_bytes = sum(
            st.bytes_per_expert * st.pinned.num_slots for st in self._layers.values()
        )
        logger.debug(
            "[ExpertTierCache] %d layers, %d experts/layer: GPU pools "
            "%.3f GiB, pinned pools %.3f GiB",
            len(self._layers),
            num_experts,
            gpu_bytes / 2**30,
            pinned_bytes / 2**30,
        )

    def preload(
        self,
        layer_id: int,
        expert_ids: torch.Tensor | Iterable[int],
        to_gpu: bool = False,
    ) -> None:
        """Synchronously bulk-load experts cold->pinned (and to GPU).

        Intended for startup. Blocks until all reads (and, with
        ``to_gpu``, all H2D copies) complete.

        Raises:
            RuntimeError: If the requested set does not fit the pinned
                pool (or the GPU pool when ``to_gpu`` is set).
            Exception: Propagated cold-read failure.
        """
        ids = self._normalize_ids(expert_ids)
        with self._lock:
            st = self._get_layer(layer_id)
            st.stats["preload_calls"] += 1
            self._sweep_ready(st)
            protected = st.pins | set(ids)
            for expert_id in ids:
                if expert_id in st.loading or st.pinned.get(expert_id) is not None:
                    continue
                slot = self._take_pinned_slot_blocking(st, protected)
                if slot is None:
                    raise RuntimeError(
                        f"layer {layer_id}: pinned pool "
                        f"({st.pinned.num_slots} slots) cannot hold preload "
                        f"set of {len(ids)} experts"
                    )
                self._start_load(st, expert_id, slot, wants_gpu=False)
            moves: list[tuple[int, int, int]] = []
            for expert_id in ids:
                job = st.loading.get(expert_id)
                if job is not None:
                    self._finish_load(st, expert_id, job)
                st.pinned.touch(expert_id)
                if to_gpu and st.gpu.get(expert_id) is None:
                    dst = self._try_take_gpu_slot(st, protected=protected)
                    if dst is None:
                        raise RuntimeError(
                            f"layer {layer_id}: gpu pool "
                            f"({st.gpu.num_slots} slots) cannot hold preload "
                            f"set of {len(ids)} experts"
                        )
                    st.gpu.insert(expert_id, dst)
                    moves.append((expert_id, st.pinned.map[expert_id], dst))
            self._h2d_batch(st, moves)
            if self._is_cuda:
                assert self._copy_stream is not None
                self._copy_stream.synchronize()

    def ensure(
        self,
        layer_id: int,
        expert_ids: torch.Tensor | Iterable[int],
        compute_stream: "torch.cuda.Stream | None" = None,
    ) -> EnsureResult:
        """Make the given experts resident in the layer's GPU slot pools.

        The requested ids become pinned (unevictable) until
        :meth:`release` is called for the layer; repeated ensure()
        calls before release() pin the union. All requested experts are
        guaranteed resident once ``compute_stream`` passes the recorded
        copy event (waited via ``event.wait(compute_stream)``, no host
        block). The only host block is waiting out a cold->pinned read
        that is still in flight (counted in stats).

        Args:
            layer_id: Layer whose pools to fill.
            expert_ids: Routed expert ids (duplicates allowed; a CPU
                tensor avoids a device sync).
            compute_stream: Stream that will consume the weights;
                defaults to the current CUDA stream. Ignored on CPU.

        Returns:
            EnsureResult with the layer's GPU pools, the id->slot remap,
            and the recorded copy event.

        Raises:
            RuntimeError: If the pinned working set would exceed the GPU
                slot pool, or the pinned pool cannot hold the cold set.
            Exception: Propagated cold-read failure.
        """
        ids = self._normalize_ids(expert_ids)
        with self._lock:
            st = self._get_layer(layer_id)
            st.stats["ensure_calls"] += 1
            self._sweep_ready(st)
            new_pins = [e for e in ids if e not in st.pins]
            if len(st.pins) + len(new_pins) > st.gpu.num_slots:
                raise RuntimeError(
                    f"layer {layer_id}: pinned working set "
                    f"{len(st.pins) + len(new_pins)} exceeds gpu slot pool "
                    f"{st.gpu.num_slots}; size gpu_slots_per_layer[layer] >= "
                    "the max concurrent (ensure..release) working set"
                )
            st.pins.update(new_pins)
            if self._is_cuda:
                if compute_stream is None:
                    compute_stream = torch.cuda.current_stream(self._device)
                st.compute_stream = compute_stream

            gpu_miss: list[int] = []
            for expert_id in ids:
                if st.gpu.get(expert_id) is not None:
                    st.gpu.touch(expert_id)
                    if st.pinned.get(expert_id) is not None:
                        st.pinned.touch(expert_id)
                    st.stats["gpu_hits"] += 1
                else:
                    gpu_miss.append(expert_id)

            cold_started: set[int] = set()
            for expert_id in gpu_miss:
                if expert_id in st.loading or (st.pinned.get(expert_id) is not None):
                    continue
                slot = self._take_pinned_slot_blocking(st, protected=st.pins)
                if slot is None:
                    raise RuntimeError(
                        f"layer {layer_id}: pinned pool "
                        f"({st.pinned.num_slots} slots) cannot hold the cold "
                        f"working set of this ensure() call"
                    )
                self._start_load(st, expert_id, slot, wants_gpu=False)
                cold_started.add(expert_id)
                st.stats["cold_misses"] += 1

            moves: list[tuple[int, int, int]] = []
            for expert_id in gpu_miss:
                job = st.loading.get(expert_id)
                if job is not None:
                    if expert_id in cold_started:
                        pass  # already counted as cold_misses
                    elif all(f.done() for f in job.futures):
                        st.stats["pinned_hits"] += 1
                    else:
                        st.stats["inflight_waits"] += 1
                    self._finish_load(st, expert_id, job)
                else:
                    st.stats["pinned_hits"] += 1
                st.pinned.touch(expert_id)
                dst = self._take_gpu_slot(st)
                st.gpu.insert(expert_id, dst)
                moves.append((expert_id, st.pinned.map[expert_id], dst))
            self._h2d_batch(st, moves)

            event = None
            if self._is_cuda and st.fill_recorded:
                assert st.fill_event is not None and compute_stream is not None
                event = st.fill_event
                compute_stream.wait_event(event)
            return EnsureResult(
                gpu_tensors=st.gpu_pool, slot_of=st.slot_of_dev, event=event
            )

    def release(self, layer_id: int) -> None:
        """Unpin the layer's ensure() working set, allowing eviction.

        Call after the layer's forward work is enqueued on the compute
        stream; slot reuse is fenced against that stream via an event
        recorded here (no host block).
        """
        with self._lock:
            st = self._get_layer(layer_id)
            st.pins.clear()
            if self._is_cuda and st.compute_stream is not None:
                assert st.release_event is not None
                st.release_event.record(st.compute_stream)
                st.release_recorded = True

    def prefetch(
        self,
        layer_id: int,
        expert_ids: torch.Tensor | Iterable[int],
        to_gpu: bool = True,
    ) -> None:
        """Best-effort async promotion of experts toward the GPU tier.

        cold->pinned reads run on the IO threads; pinned->GPU copies are
        enqueued on the copy stream (from the calling thread, never from
        IO threads; loads finishing in the background are promoted by
        the next ensure/prefetch/preload call for the layer). Never
        evicts entries pinned by an unreleased ensure() or in-flight
        loads; when no slot is free the request is dropped and counted.

        May be called from a different thread than ensure().

        Raises:
            ValueError: If an expert id is out of range.
        """
        ids = self._normalize_ids(expert_ids)
        with self._lock:
            st = self._get_layer(layer_id)
            st.stats["prefetch_calls"] += 1
            self._sweep_ready(st)
            moves: list[tuple[int, int, int]] = []
            protected = st.pins | set(ids)
            for expert_id in ids:
                if st.gpu.get(expert_id) is not None:
                    continue
                job = st.loading.get(expert_id)
                if job is not None:
                    job.wants_gpu = job.wants_gpu or to_gpu
                    continue
                src = st.pinned.get(expert_id)
                if src is not None:
                    if not to_gpu:
                        continue
                    dst = self._try_take_gpu_slot(st, protected=protected)
                    if dst is None:
                        st.stats["prefetch_drops"] += 1
                        continue
                    st.gpu.insert(expert_id, dst)
                    moves.append((expert_id, src, dst))
                    continue
                slot = self._take_pinned_slot(st, protected)
                if slot is None:
                    st.stats["prefetch_drops"] += 1
                    continue
                self._start_load(st, expert_id, slot, wants_gpu=to_gpu)
            self._h2d_batch(st, moves)

    def stats(self) -> dict:
        """Per-layer and total counters (hits, misses, bytes, evictions)."""
        with self._lock:
            per_layer = {
                layer_id: dict(st.stats) for layer_id, st in self._layers.items()
            }
        total = {
            key: sum(layer[key] for layer in per_layer.values()) for key in _STAT_KEYS
        }
        return {"per_layer": per_layer, "total": total}

    def wait_io(self, timeout: float | None = None) -> None:
        """Block until currently in-flight cold->pinned reads finish.

        Loaded entries are promoted lazily by the next ensure/prefetch/
        preload call for their layer.
        """
        with self._lock:
            futures = [
                future
                for st in self._layers.values()
                for job in st.loading.values()
                for future in job.futures
            ]
        _futures_wait(futures, timeout=timeout)

    def close(self) -> None:
        """Stop IO threads and drain the copy stream (terminal)."""
        self._io.shutdown(wait=True)
        if self._is_cuda:
            assert self._copy_stream is not None
            self._copy_stream.synchronize()

    def _get_layer(self, layer_id: int) -> _LayerState:
        st = self._layers.get(layer_id)
        if st is None:
            raise KeyError(f"unknown layer id {layer_id}")
        return st

    def _normalize_ids(self, expert_ids: torch.Tensor | Iterable[int]) -> list[int]:
        if isinstance(expert_ids, torch.Tensor):
            ids_t = expert_ids.detach()
            if ids_t.device.type != "cpu":
                ids_t = ids_t.cpu()
            ids = [int(e) for e in torch.unique(ids_t).tolist()]
        else:
            ids = sorted({int(e) for e in expert_ids})
        for expert_id in ids:
            if not 0 <= expert_id < self._num_experts:
                raise ValueError(
                    f"expert id {expert_id} out of range [0, {self._num_experts})"
                )
        return ids

    def _start_load(
        self, st: _LayerState, expert_id: int, slot: int, wants_gpu: bool
    ) -> None:
        future = self._io.submit(self._read_expert, st, expert_id, slot)
        st.loading[expert_id] = _LoadJob(
            slot=slot, futures=[future], wants_gpu=wants_gpu
        )

    def _read_expert(self, st: _LayerState, expert_id: int, slot: int) -> None:
        """Fill one pinned slot from cold storage (runs on an IO thread)."""
        for spec in st.specs:
            dst = st.pinned_pool[spec.name][slot]
            self._cold.read_into(st.layer_id, expert_id, spec.name, dst)

    def _finish_load(self, st: _LayerState, expert_id: int, job: _LoadJob) -> None:
        """Wait out a load and publish it to the pinned tier (lock held)."""
        try:
            for future in job.futures:
                future.result()
        except Exception:
            del st.loading[expert_id]
            st.pinned.free.append(job.slot)
            st.stats["io_errors"] += 1
            raise
        del st.loading[expert_id]
        st.pinned.insert(expert_id, job.slot)
        st.stats["bytes_cold_to_pinned"] += st.bytes_per_expert

    def _sweep_ready(self, st: _LayerState) -> None:
        """Publish finished background loads; promote wants_gpu entries."""
        if not st.loading:
            return
        ready = [
            expert_id
            for expert_id, job in st.loading.items()
            if all(f.done() for f in job.futures)
        ]
        moves: list[tuple[int, int, int]] = []
        promoted: set[int] = set()
        for expert_id in ready:
            job = st.loading[expert_id]
            try:
                self._finish_load(st, expert_id, job)
            except Exception:
                logger.warning(
                    "[ExpertTierCache] background load failed: layer %d expert %d",
                    st.layer_id,
                    expert_id,
                    exc_info=True,
                )
                continue
            if job.wants_gpu and st.gpu.get(expert_id) is None:
                dst = self._try_take_gpu_slot(st, protected=st.pins | promoted)
                if dst is None:
                    st.stats["prefetch_drops"] += 1
                else:
                    st.gpu.insert(expert_id, dst)
                    promoted.add(expert_id)
                    moves.append((expert_id, job.slot, dst))
        self._h2d_batch(st, moves)

    def _take_pinned_slot(self, st: _LayerState, protected: set[int]) -> int | None:
        """Allocate a pinned slot, evicting LRU if needed (lock held).

        Host-syncs the copy stream only when the reclaimed slot was a
        recent H2D source that may still be read by an in-flight copy.
        """
        took = st.pinned.take_slot(protected)
        if took is None:
            return None
        slot, victim = took
        if victim is not None:
            st.stats["pinned_evictions"] += 1
        if slot in st.pinned_srcs:
            if self._is_cuda and st.fill_recorded:
                assert st.fill_event is not None
                st.fill_event.synchronize()
            st.pinned_srcs.clear()
            st.stage_busy = False
        return slot

    def _take_pinned_slot_blocking(
        self, st: _LayerState, protected: set[int]
    ) -> int | None:
        """Like :meth:`_take_pinned_slot`, but waits out transient
        exhaustion caused by in-flight loads (which own slots without
        being evictable). Returns None only when the pool is genuinely
        too small for the protected set. Lock held.
        """
        while True:
            slot = self._take_pinned_slot(st, protected)
            if slot is not None:
                return slot
            if not st.loading:
                return None
            st.stats["pinned_stalls"] += 1
            futures = [f for job in st.loading.values() for f in job.futures]
            _futures_wait(futures, return_when=FIRST_COMPLETED)
            self._sweep_ready(st)

    def _try_take_gpu_slot(self, st: _LayerState, protected: set[int]) -> int | None:
        took = st.gpu.take_slot(protected)
        if took is None:
            return None
        slot, victim = took
        if victim is not None:
            st.slot_of_cpu[victim] = -1
            st.stats["gpu_evictions"] += 1
        return slot

    def _take_gpu_slot(self, st: _LayerState) -> int:
        slot = self._try_take_gpu_slot(st, protected=st.pins)
        if slot is None:
            raise RuntimeError(
                f"layer {st.layer_id}: no evictable gpu slot (pins "
                f"{len(st.pins)} of {st.gpu.num_slots})"
            )
        return slot

    def _h2d_batch(self, st: _LayerState, moves: list[tuple[int, int, int]]) -> None:
        """Enqueue pinned->GPU copies for (expert, src, dst) moves.

        Large tensors use one non_blocking DMA per row straight from the
        pinned pool; small tensors are gathered into a pinned staging
        buffer, copied H2D once, and scattered with index_copy_. Records
        the layer fill event after the last copy. Lock held.
        """
        if not moves:
            return
        num = len(moves)
        src = [move[1] for move in moves]
        dst = [move[2] for move in moves]
        for expert_id, _, dst_slot in moves:
            assert st.gpu.get(expert_id) == dst_slot
            st.slot_of_cpu[expert_id] = dst_slot
        st.stats["bytes_pinned_to_gpu"] += num * st.bytes_per_expert
        if not self._is_cuda:
            for name in st.names:
                gpu_pool = st.gpu_pool[name]
                pinned_pool = st.pinned_pool[name]
                for src_slot, dst_slot in zip(src, dst):
                    gpu_pool[dst_slot].copy_(pinned_pool[src_slot])
            if st.slot_of_dev is not st.slot_of_cpu:
                st.slot_of_dev.copy_(st.slot_of_cpu)
            return
        assert self._copy_stream is not None and st.fill_event is not None
        if st.small_names:
            assert st.idx_pin is not None and st.idx_dev is not None
            if st.stage_busy:
                st.fill_event.synchronize()
                st.pinned_srcs.clear()
            src_idx = torch.tensor(src, dtype=torch.int64)
            for name in st.small_names:
                torch.index_select(
                    st.pinned_pool[name], 0, src_idx, out=st.stage_pin[name][:num]
                )
            st.idx_pin[:num] = torch.tensor(dst, dtype=torch.int64)
            st.stage_busy = True
        copy_stream = self._copy_stream
        if st.release_recorded:
            assert st.release_event is not None
            copy_stream.wait_event(st.release_event)
        with torch.cuda.stream(copy_stream):
            for name in st.large_names:
                gpu_pool = st.gpu_pool[name]
                pinned_pool = st.pinned_pool[name]
                for src_slot, dst_slot in zip(src, dst):
                    gpu_pool[dst_slot].copy_(pinned_pool[src_slot], non_blocking=True)
            if st.small_names:
                assert st.idx_pin is not None and st.idx_dev is not None
                st.idx_dev[:num].copy_(st.idx_pin[:num], non_blocking=True)
                for name in st.small_names:
                    st.stage_dev[name][:num].copy_(
                        st.stage_pin[name][:num], non_blocking=True
                    )
                    st.gpu_pool[name].index_copy_(
                        0, st.idx_dev[:num], st.stage_dev[name][:num]
                    )
            st.slot_of_dev.copy_(st.slot_of_cpu, non_blocking=True)
        st.pinned_srcs.update(src)
        st.fill_event.record(copy_stream)
        st.fill_recorded = True
