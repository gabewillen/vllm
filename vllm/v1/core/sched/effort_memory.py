# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online memory of pooled prefill states, keyed by cosine, valued by thinking.

The v3 dynamic-effort signal (docs/dynamic-reasoning.claude.md §13,
docs/effort-hidden-probe.md): the last prompt token's final hidden state - the
vector `lm_head` consumes, which the prefill produced anyway - is looked up
against a memory the server filled from its own finished requests. The value is
how many reasoning tokens that request actually spent. Nothing is fitted: the
estimate is a cosine-weighted mean of the neighbours' observed lengths and the
only constants are percentile ranks of running digests, the same
self-calibration §11.0 already blesses for entropy and margin.

Entries are inserted at request finish. A request that was force-closed or
soft-closed is right-censored - the length it *would* have spent is unknown -
so it is stored as a key with no value: it counts for novelty and can never
pull the kNN average down.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm.config.reasoning import HiddenEffortConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.effort_quantiles import TDigest
from vllm.v1.sample.soft_limit import CLOSE_NATURAL

logger = init_logger(__name__)

MEMORY_VERSION = 1
"""Bumped whenever the stored layout or the value semantics change."""

SESSION_CAP_DIVISOR = 64
"""At most `memory_size / SESSION_CAP_DIVISOR` entries per session id, so one
long conversation cannot evict the memory."""


@dataclass(frozen=True)
class MemoryQuery:
    """What the memory can say about one pooled vector."""

    estimate: float | None
    """Cosine-weighted mean of `log1p(reasoning_tokens)` over the valued
    neighbours, or `None` when none of the `k` neighbours carries a value."""
    novelty: float
    """`(1 - max cos) / 2`: how far the nearest entry is."""
    spread: float | None
    """Weighted stdev of the valued neighbours' values; `None` with < 2."""
    max_cos: float
    n_entries: int
    n_valued: int
    """Valued entries among the `k` neighbours."""


@dataclass(frozen=True)
class StartRungDecision:
    """The asymmetric map's verdict for one request (§13.5)."""

    rung: int
    reason: str
    estimate_rank: float | None = None
    novelty_rank: float | None = None
    spread_rank: float | None = None


def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-9)


class EffortMemory:
    """Ring of pooled prefill states with a cosine kNN over them.

    Scheduler-side, one per engine core. Keys are L2-normalised float32 rows
    (a plain BLAS `sgemv` per query); the persisted copy is fp16, which is the
    10 KB/entry the design costs on disk.
    """

    def __init__(
        self,
        hidden_size: int,
        cfg: HiddenEffortConfig,
        model: str = "",
        ladder: tuple[int, ...] = (),
    ) -> None:
        self.hidden_size = hidden_size
        self.cfg = cfg
        self.model = model
        self.ladder = tuple(int(x) for x in ladder)
        size = int(cfg.memory_size)
        self._keys = np.zeros((size, hidden_size), dtype=np.float32)
        self._values = np.full(size, np.nan, dtype=np.float32)
        self._sessions: list[str | None] = [None] * size
        self._start_rungs = np.zeros(size, dtype=np.int8)
        self._final_rungs = np.zeros(size, dtype=np.int8)
        self._escalations = np.zeros(size, dtype=np.int8)
        self._by_session: dict[str, deque[int]] = {}
        self._next = 0
        self._n = 0
        self._inserts = 0
        self._since_flush = 0
        self._queries = 0
        self._hits = 0
        self.per_session_cap = max(1, math.ceil(size / SESSION_CAP_DIVISOR))
        self._est_digest = TDigest(compression=cfg.digest_compression)
        self._novelty_digest = TDigest(compression=cfg.digest_compression)
        self._spread_digest = TDigest(compression=cfg.digest_compression)

    # ---------------------------------------------------------------- state

    @property
    def n_entries(self) -> int:
        return self._n

    @property
    def n_valued(self) -> int:
        return int(np.count_nonzero(~np.isnan(self._values[: self._n])))

    @property
    def hit_rate(self) -> float:
        return self._hits / self._queries if self._queries else 0.0

    @property
    def ready(self) -> bool:
        """The memory may decide a starting rung."""
        return self._n >= self.cfg.min_entries

    # --------------------------------------------------------------- insert

    def insert(
        self,
        vec: np.ndarray,
        reasoning_tokens: int | None,
        close_kind: str,
        session_id: str | None = None,
        start_rung: int = 0,
        final_rung: int = 0,
        escalations: int = 0,
    ) -> None:
        """Record one finished request.

        Args:
            vec: the pooled prefill state (any scale; stored L2-normalised).
            reasoning_tokens: think tokens the request spent, or `None`.
            close_kind: `natural` / `soft` / `forced`. Only a natural close
                carries a value - the other two are right-censored, so they
                enter as keys with no value.
            session_id: conversation key for the per-session eviction cap.
            start_rung: rung the prefill decision chose.
            final_rung: rung the request ended at.
            escalations: mid-generation escalations it took.
        """
        key = _unit(vec)
        if key.shape[0] != self.hidden_size:
            raise ValueError(
                f"effort memory expects {self.hidden_size}-wide vectors, "
                f"got {key.shape[0]}"
            )
        slot = self._claim_slot(session_id)
        self._keys[slot] = key
        valued = close_kind == CLOSE_NATURAL and reasoning_tokens is not None
        self._values[slot] = (
            math.log1p(max(int(reasoning_tokens or 0), 0)) if valued else np.nan
        )
        self._sessions[slot] = session_id
        self._start_rungs[slot] = np.int8(max(-128, min(127, start_rung)))
        self._final_rungs[slot] = np.int8(max(-128, min(127, final_rung)))
        self._escalations[slot] = np.int8(max(-128, min(127, escalations)))
        if session_id is not None:
            self._by_session.setdefault(session_id, deque()).append(slot)
        self._inserts += 1
        self._since_flush += 1
        self.maybe_save()

    def _claim_slot(self, session_id: str | None) -> int:
        """FIFO ring, with a per-session cap that reuses the session's oldest."""
        if session_id is not None:
            slots = self._by_session.get(session_id)
            if slots is not None and len(slots) >= self.per_session_cap:
                return slots.popleft()
        slot = self._next
        self._next = (self._next + 1) % self._keys.shape[0]
        victim = self._sessions[slot]
        if victim is not None and self._n > slot:
            queue = self._by_session.get(victim)
            if queue is not None:
                with contextlib.suppress(ValueError):
                    queue.remove(slot)
                if not queue:
                    del self._by_session[victim]
        if self._n < self._keys.shape[0]:
            self._n += 1
        return slot

    # ---------------------------------------------------------------- query

    def query(self, vec: np.ndarray) -> MemoryQuery | None:
        """kNN against the ring; `None` while there is nothing to compare to.

        The digests are fed here (not at insert), so the percentile ranks the
        map cuts on are ranks over the *decisions the server has faced*. That
        also warms them during the cold phase, where the result is computed and
        discarded.
        """
        k = int(self.cfg.k)
        if self._n < max(k, 1):
            return None
        key = _unit(vec)
        sims = self._keys[: self._n] @ key
        top = np.argpartition(-sims, k - 1)[:k] if self._n > k else np.arange(self._n)
        top = top[np.argsort(-sims[top])]
        best = float(sims[top[0]])
        novelty = (1.0 - best) / 2.0
        values = self._values[top]
        valued = ~np.isnan(values)
        estimate: float | None = None
        spread: float | None = None
        if valued.any():
            # The softmax reference is the nearest *valued* neighbour, not the
            # nearest entry: with censored keys in the top-k the global maximum
            # can be arbitrarily far from every entry being averaged, and the
            # weights would underflow to zero together.
            valued_sims = sims[top][valued]
            w = np.exp(
                (valued_sims - valued_sims.max()) / max(self.cfg.temperature, 1e-6)
            )
            w = w / w.sum()
            vals = values[valued].astype(np.float64)
            mean = float((w * vals).sum())
            estimate = mean
            if valued.sum() >= 2:
                spread = float(math.sqrt(max((w * (vals - mean) ** 2).sum(), 0.0)))
        result = MemoryQuery(
            estimate=estimate,
            novelty=novelty,
            spread=spread,
            max_cos=best,
            n_entries=self._n,
            n_valued=int(valued.sum()),
        )
        self._queries += 1
        if estimate is not None:
            self._hits += 1
        return result

    def ranks(self, q: MemoryQuery) -> tuple[float | None, float | None, float | None]:
        """Percentile ranks of the query's three statistics, then absorb them."""
        est_rank = None if q.estimate is None else self._est_digest.rank(q.estimate)
        nov_rank = self._novelty_digest.rank(q.novelty)
        spread_rank = None if q.spread is None else self._spread_digest.rank(q.spread)
        if q.estimate is not None:
            self._est_digest.add(q.estimate)
        self._novelty_digest.add(q.novelty)
        if q.spread is not None:
            self._spread_digest.add(q.spread)
        return est_rank, nov_rank, spread_rank

    # ----------------------------------------------------------- persistence

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": MEMORY_VERSION,
            "model": self.model,
            "hidden_size": self.hidden_size,
            "ladder": list(self.ladder),
            "memory_size": int(self._keys.shape[0]),
            "n": self._n,
            "next": self._next,
            "inserts": self._inserts,
            "sessions": self._sessions[: self._n],
            "est_digest": self._est_digest.to_dict(),
            "novelty_digest": self._novelty_digest.to_dict(),
            "spread_digest": self._spread_digest.to_dict(),
        }

    def save(self) -> None:
        path = self.cfg.memory_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # mkstemp's suffix keeps np.savez from appending one of its own.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(
                    handle,
                    meta=np.frombuffer(
                        json.dumps(self.state_dict()).encode(), dtype=np.uint8
                    ),
                    keys=self._keys[: self._n].astype(np.float16),
                    values=self._values[: self._n],
                    start_rungs=self._start_rungs[: self._n],
                    final_rungs=self._final_rungs[: self._n],
                    escalations=self._escalations[: self._n],
                )
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self._since_flush = 0

    def maybe_save(self) -> bool:
        every = int(self.cfg.flush_every)
        if not self.cfg.memory_path or every <= 0 or self._since_flush < every:
            return False
        self.save()
        return True

    def load(self) -> bool:
        """Warm from `memory_path`; drop a file that no longer matches."""
        path = self.cfg.memory_path
        if not path or not os.path.exists(path):
            return False
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode())
            if (
                meta.get("version") != MEMORY_VERSION
                or meta.get("hidden_size") != self.hidden_size
                or (self.model and meta.get("model") not in ("", self.model))
                or (self.ladder and tuple(meta.get("ladder", ())) != self.ladder)
            ):
                logger.warning(
                    "dynamic_effort: dropping %s (model/dim/ladder/version "
                    "mismatch: %s)",
                    path,
                    {k: meta.get(k) for k in ("version", "model", "hidden_size")},
                )
                return False
            keys = data["keys"].astype(np.float32)
            n = min(keys.shape[0], self._keys.shape[0])
            self._keys[:n] = keys[:n]
            self._values[:n] = data["values"][:n]
            self._start_rungs[:n] = data["start_rungs"][:n]
            self._final_rungs[:n] = data["final_rungs"][:n]
            self._escalations[:n] = data["escalations"][:n]
        sessions = list(meta.get("sessions") or [])[:n]
        sessions += [None] * (n - len(sessions))
        self._sessions[:n] = sessions
        self._by_session.clear()
        for slot, sess in enumerate(sessions):
            if sess is not None:
                self._by_session.setdefault(sess, deque()).append(slot)
        self._n = n
        self._next = n % self._keys.shape[0]
        self._inserts = int(meta.get("inserts", n))
        self._est_digest = TDigest.from_dict(meta["est_digest"])
        self._novelty_digest = TDigest.from_dict(meta["novelty_digest"])
        self._spread_digest = TDigest.from_dict(meta["spread_digest"])
        return True


def decide_start_rung(
    query: MemoryQuery | None,
    ranks: tuple[float | None, float | None, float | None],
    cfg: HiddenEffortConfig,
    top_rung: int,
) -> StartRungDecision:
    """The asymmetric quantile map (§13.5).

    Raise freely, lower only on confidence: one under-routed step can kill a
    trajectory, over-routing only costs tokens. The two confidence gates guard
    the *downward* band alone - `novelty` (nothing similar in the memory, so it
    cannot be trusted to say "easy") and `spread` (the neighbours disagree).

    Args:
        query: the kNN result, or `None` when the memory could not answer.
        ranks: `(estimate, novelty, spread)` percentile ranks.
        cfg: hidden-effort settings.
        top_rung: highest rung of the request's ladder.

    Returns:
        The starting rung and why it was chosen.
    """
    safe = min(1, top_rung)
    est_rank, nov_rank, spread_rank = ranks
    if query is None or est_rank is None:
        return StartRungDecision(safe, "no-estimate", est_rank, nov_rank, spread_rank)
    if est_rank >= cfg.q_high:
        return StartRungDecision(
            min(2, top_rung), "q>=q_high", est_rank, nov_rank, spread_rank
        )
    if est_rank >= cfg.q_mid:
        return StartRungDecision(safe, "q>=q_mid", est_rank, nov_rank, spread_rank)
    gates_pass = (
        nov_rank is not None
        and spread_rank is not None
        and nov_rank <= cfg.novelty_gate_q
        and spread_rank <= cfg.spread_gate_q
    )
    if gates_pass:
        return StartRungDecision(0, "low-band", est_rank, nov_rank, spread_rank)
    return StartRungDecision(safe, "gated", est_rank, nov_rank, spread_rank)
