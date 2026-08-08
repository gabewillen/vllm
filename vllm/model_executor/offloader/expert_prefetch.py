# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predictive expert prefetching for :class:`ExpertTierCache`.

Predicts which routed experts upcoming layers/steps will need and promotes
them toward the GPU tier ahead of the blocking ``ensure()`` call, hiding
cold-read and H2D latency behind compute. All prediction bookkeeping and
``cache.prefetch()`` issuance runs on one background worker thread; the
forward thread only enqueues small observation records (and, for the hash
and gate predictors, a non-blocking D2H copy fenced by a CUDA event).

Predictors (independently selectable):

- ``temporal``: per layer, re-request the experts the layer routed to in
  the previous decode step for the next ``lookahead`` layers.
- ``popular``: per layer, keep an EMA popularity count of routed experts
  and prefetch the top ``popular_k`` for the next ``lookahead`` layers.
- ``hash``: DeepSeek-V4 hash layers route purely by token id via a
  ``tid2eid`` table; once next-step token ids are known on the sampling
  path, their hash-layer experts are exact and prefetched a step early.
- ``gate``: apply layer ``l+1``'s router gate to layer ``l``'s hidden
  state to predict ``l+1``'s experts one layer early (approximate).
"""

import threading
import time
from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

PREDICTOR_NAMES = ("temporal", "popular", "hash", "gate")

_RING_SLOTS = 4
_RING_CAPACITY = 8192
_GATE_TOPK = 6
_GATE_MAX_TOKENS = 2

_PF_STAT_KEYS = (
    "observes",
    "prefetch_calls",
    "prefetch_ids",
    "token_batches",
    "gate_batches",
    "ring_drops",
)


class _Ring:
    """Small pinned-buffer ring for async D2H id handoff to the worker."""

    def __init__(self, num_slots: int, capacity: int):
        self.bufs = [
            torch.empty(capacity, dtype=torch.int64, pin_memory=True)
            for _ in range(num_slots)
        ]
        self.events = [torch.cuda.Event() for _ in range(num_slots)]
        self.busy = [False] * num_slots

    def acquire(self) -> int | None:
        for slot, busy in enumerate(self.busy):
            if not busy:
                self.busy[slot] = True
                return slot
        return None


class ExpertPrefetcher:
    """Background predictor feeding ``cache.prefetch()``.

    Args:
        cache: Object exposing ``prefetch(layer_id, expert_ids, to_gpu)``.
        layer_ids: MoE layer ids in forward order (lookahead wraps).
        num_experts: Global routed expert count per layer.
        predictors: Enabled predictor names (subset of PREDICTOR_NAMES).
        lookahead: How many layers ahead temporal/popular prefetch.
        popular_k: Top-k popularity experts prefetched per target layer.
        ema_decay: Per-step decay of the popularity counts.
        stats_fn: Optional callable returning a dict logged periodically.
        stats_interval_s: Minimum seconds between periodic stats logs
            (0 disables).
        name: Tag used in log lines.
    """

    def __init__(
        self,
        cache,
        layer_ids: Sequence[int],
        num_experts: int,
        predictors: Iterable[str],
        lookahead: int = 4,
        popular_k: int = 8,
        ema_decay: float = 0.98,
        stats_fn=None,
        stats_interval_s: float = 0.0,
        name: str = "",
    ):
        self._cache = cache
        self._order = list(layer_ids)
        self._pos = {layer_id: i for i, layer_id in enumerate(self._order)}
        self._num_experts = num_experts
        self._predictors = frozenset(predictors)
        unknown = self._predictors - set(PREDICTOR_NAMES)
        if unknown:
            raise ValueError(f"unknown predictors {sorted(unknown)}")
        self._lookahead = lookahead
        self._popular_k = popular_k
        self._ema_decay = ema_decay
        self._stats_fn = stats_fn
        self._stats_interval_s = stats_interval_s
        self._name = name

        self._counts: dict[int, np.ndarray] = {}
        self._last: dict[int, list[int]] = {}
        # layer_id -> device tid2eid tensor (CPU copy made lazily on the
        # worker, after checkpoint weights have been loaded in place).
        self._hash_tables: dict[int, torch.Tensor] = {}
        self._hash_np: dict[int, np.ndarray] = {}
        self._gates: dict[int, torch.nn.Module] = {}

        self._cv = threading.Condition()
        self._obs: list[tuple[int, list[int]]] = []
        # ("hash"|"gate", target_layer_or_None, ring_slot_or_ids, count)
        self._jobs: list[tuple[str, int | None, int | list[int], int]] = []
        self._ring: _Ring | None = None
        self._closed = False
        self._idle = True
        self.counters: dict[str, int] = dict.fromkeys(_PF_STAT_KEYS, 0)
        self._last_stats_log = time.monotonic()
        self._thread = threading.Thread(
            target=self._worker, name=f"expert_prefetch{name}", daemon=True
        )
        self._thread.start()

    @property
    def predictors(self) -> frozenset:
        return self._predictors

    def note_hash_table(self, layer_id: int, table: torch.Tensor) -> None:
        """Register a hash layer's token-id -> expert-ids table."""
        self._hash_tables[layer_id] = table

    def note_gate(self, layer_id: int, gate: torch.nn.Module) -> None:
        """Register a layer's router gate module for gate lookahead."""
        self._gates[layer_id] = gate

    def observe(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Record a layer's routed CPU ids; triggers lookahead prefetch."""
        if not (self._predictors & {"temporal", "popular"}):
            return
        ids = topk_ids.reshape(-1).tolist()
        with self._cv:
            self._obs.append((layer_id, ids))
            self._idle = False
            self._cv.notify()

    def observe_sampled_tokens(self, token_ids: torch.Tensor) -> None:
        """Hand next-step token ids to the hash predictor.

        Accepts the device tensor produced by sampling; enqueues a
        non-blocking D2H copy fenced by a CUDA event, never a host sync.
        """
        if "hash" not in self._predictors or not self._hash_tables:
            return
        self._submit_ids_async("hash", None, token_ids)

    def observe_hidden(self, layer_id: int, hidden_states: torch.Tensor) -> None:
        """Predict the next layer's experts from this layer's MoE input.

        Decode-only: prefill chunks route too many experts per layer for
        speculative loading to pay off (measured TTFT regression).
        """
        if "gate" not in self._predictors:
            return
        if hidden_states.shape[0] > _GATE_MAX_TOKENS:
            return
        pos = self._pos.get(layer_id)
        if pos is None or pos + 1 >= len(self._order):
            return
        target = self._order[pos + 1]
        if target in self._hash_tables:
            return
        gate = self._gates.get(target)
        if gate is None:
            return
        weight = gate.weight
        with torch.no_grad():
            logits = F.linear(hidden_states.to(weight.dtype), weight).float()
            scores = torch.sqrt(F.softplus(logits))
            bias = getattr(gate, "e_score_correction_bias", None)
            if bias is not None:
                scores = scores + bias
            k = min(_GATE_TOPK, scores.shape[-1])
            pred = torch.topk(scores, k, dim=-1).indices
        self._submit_ids_async("gate", target, pred)

    def _submit_ids_async(
        self, kind: str, target: int | None, ids: torch.Tensor
    ) -> None:
        ids = ids.reshape(-1)
        if ids.device.type == "cpu":
            with self._cv:
                self._jobs.append((kind, target, ids.tolist(), ids.numel()))
                self._idle = False
                self._cv.notify()
            return
        count = ids.numel()
        if count > _RING_CAPACITY:
            return
        with self._cv:
            if self._ring is None:
                self._ring = _Ring(_RING_SLOTS, _RING_CAPACITY)
            slot = self._ring.acquire()
            if slot is None:
                self.counters["ring_drops"] += 1
                return
        self._ring.bufs[slot][:count].copy_(ids, non_blocking=True)
        self._ring.events[slot].record(torch.cuda.current_stream(ids.device))
        with self._cv:
            self._jobs.append((kind, target, slot, count))
            self._idle = False
            self._cv.notify()

    def flush(self, timeout: float | None = None) -> None:
        """Block until all queued work has been issued (tests)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while self._obs or self._jobs or not self._idle:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("prefetcher flush timed out")
                self._cv.wait(remaining)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join(timeout=5.0)

    def stats(self) -> dict[str, int]:
        with self._cv:
            return dict(self.counters)

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not (self._obs or self._jobs or self._closed):
                    self._idle = True
                    self._cv.notify_all()
                    self._cv.wait()
                if self._closed:
                    self._idle = True
                    self._cv.notify_all()
                    return
                obs, self._obs = self._obs, []
                jobs, self._jobs = self._jobs, []
            for kind, target, payload, count in jobs:
                try:
                    self._run_job(kind, target, payload, count)
                except Exception:
                    logger.warning(
                        "[ExpertPrefetcher%s] %s job failed",
                        self._name,
                        kind,
                        exc_info=True,
                    )
            for layer_id, ids in obs:
                try:
                    self._run_observation(layer_id, ids)
                except Exception:
                    logger.warning(
                        "[ExpertPrefetcher%s] observation failed",
                        self._name,
                        exc_info=True,
                    )
            self._maybe_log_stats()

    def _prefetch(self, layer_id: int, ids: list[int], to_gpu: bool = True) -> None:
        if not ids:
            return
        self._cache.prefetch(layer_id, ids, to_gpu=to_gpu)
        with self._cv:
            self.counters["prefetch_calls"] += 1
            self.counters["prefetch_ids"] += len(ids)

    def _run_observation(self, layer_id: int, raw_ids: list[int]) -> None:
        ids = sorted({i for i in raw_ids if 0 <= i < self._num_experts})
        with self._cv:
            self.counters["observes"] += 1
        if "popular" in self._predictors:
            counts = self._counts.get(layer_id)
            if counts is None:
                counts = np.zeros(self._num_experts, dtype=np.float32)
                self._counts[layer_id] = counts
            counts *= self._ema_decay
            counts[ids] += 1.0
        self._last[layer_id] = ids
        pos = self._pos.get(layer_id)
        if pos is None:
            return
        # Target only the layer `lookahead` ahead: each layer is then
        # requested exactly once per step (by its k-th predecessor) with
        # the full lookahead lead time, instead of redundantly by every
        # closer predecessor (measured to add cache-lock contention).
        target = self._order[(pos + self._lookahead) % len(self._order)]
        wanted: set[int] = set()
        if "temporal" in self._predictors:
            wanted.update(self._last.get(target, ()))
        if "popular" in self._predictors:
            wanted.update(self._popular(target))
        self._prefetch(target, sorted(wanted))

    def _popular(self, layer_id: int) -> list[int]:
        counts = self._counts.get(layer_id)
        if counts is None or self._popular_k <= 0:
            return []
        k = min(self._popular_k, self._num_experts)
        top = np.argpartition(counts, -k)[-k:]
        return [int(i) for i in top if counts[i] > 0.0]

    def _job_ids(self, payload: int | list[int], count: int) -> list[int]:
        if isinstance(payload, list):
            return payload
        assert self._ring is not None
        slot = payload
        self._ring.events[slot].synchronize()
        ids = self._ring.bufs[slot][:count].tolist()
        with self._cv:
            self._ring.busy[slot] = False
        return ids

    def _run_job(
        self, kind: str, target: int | None, payload: int | list[int], count: int
    ) -> None:
        ids = self._job_ids(payload, count)
        if kind == "gate":
            assert target is not None
            with self._cv:
                self.counters["gate_batches"] += 1
            valid = sorted({i for i in ids if 0 <= i < self._num_experts})
            # Pinned tier only: mispredicts must not evict GPU-resident
            # experts or spend H2D bandwidth; correct predictions still
            # turn ensure() cold misses into cheap pinned hits.
            self._prefetch(target, valid, to_gpu=False)
            return
        with self._cv:
            self.counters["token_batches"] += 1
        for layer_id, table in self._hash_tables.items():
            table_np = self._hash_np.get(layer_id)
            if table_np is None:
                table_np = table.detach().cpu().numpy()
                self._hash_np[layer_id] = table_np
            vocab = table_np.shape[0]
            tokens = [t for t in ids if 0 <= t < vocab]
            if not tokens:
                continue
            experts = np.unique(table_np[tokens].ravel())
            self._prefetch(layer_id, [int(e) for e in experts])

    def _maybe_log_stats(self) -> None:
        if self._stats_fn is None or self._stats_interval_s <= 0:
            return
        now = time.monotonic()
        if now - self._last_stats_log < self._stats_interval_s:
            return
        self._last_stats_log = now
        try:
            logger.info(
                "[ExpertPrefetcher%s] cache=%s prefetch=%s",
                self._name,
                self._stats_fn(),
                self.stats(),
            )
        except Exception:
            logger.debug("stats logging failed", exc_info=True)
