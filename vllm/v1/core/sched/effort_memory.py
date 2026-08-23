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

The value the kNN averages is *difficulty*, not raw spend: a request's spend
is ranked within the running distribution of the effort level it was rendered
at. Spend depends on the level as much as on the task (the same prompt thinks
~10x longer under the top sentence than under the lowest), so averaging raw
lengths would only echo the server's own past decisions back at it and ratchet
every neighbourhood upward. Ranking within the lane cancels that offset: a
trivial task routed to the top level ranks low there and a hard one routed
low ranks high there, so each neighbourhood can move in both directions.
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
from vllm.v1.core.sched.effort_controller import CLOSE_NATURAL
from vllm.v1.core.sched.effort_quantiles import TDigest

logger = init_logger(__name__)

MEMORY_VERSION = 1
"""Bumped whenever the stored layout or the value semantics change."""

SESSION_CAP_DIVISOR = 64
"""At most `memory_size / SESSION_CAP_DIVISOR` entries per session id, so one
long conversation cannot evict the memory."""

CALIBRATION_BINS = 8
"""Novelty-rank bins the estimate is calibrated in. Resolution, not a cut."""


class _Calibration:
    """Bayesian least squares of realised difficulty on the kNN estimate.

    One per novelty bin plus a pooled one. `fit` returns `(a, b)` for
    `a + b * estimate` under a prior worth `weight` observations spread over
    the unit interval at the parent's line: a bin shrinks toward the pooled
    fit, the pooled fit toward the identity, and the first pair already moves
    the posterior. With no predictive value in the neighbours `b` goes to 0
    and every estimate collapses to the mean realised difficulty.
    """

    __slots__ = ("n", "sx", "sy", "sxx", "sxy")

    def __init__(self) -> None:
        self.n = 0
        self.sx = self.sy = self.sxx = self.sxy = 0.0

    def add(self, x: float, y: float) -> None:
        self.n += 1
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.sxy += x * y

    def fit(self, prior: tuple[float, float], weight: float) -> tuple[float, float]:
        a0, b0 = prior
        half = weight / 2.0
        n = self.n + weight
        sx = self.sx + half
        sy = self.sy + half * a0 + half * (a0 + b0)
        sxx = self.sxx + half
        sxy = self.sxy + half * (a0 + b0)
        mx, my = sx / n, sy / n
        var = sxx / n - mx * mx
        if var <= 1e-9:
            return my, 0.0
        slope = (sxy / n - mx * my) / var
        slope = min(max(slope, 0.0), 2.0)
        return my - slope * mx, slope

    def to_dict(self) -> dict[str, float]:
        return {"n": self.n, "sx": self.sx, "sy": self.sy, "sxx": self.sxx, "sxy": self.sxy}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "_Calibration":
        c = cls()
        c.n = int(d.get("n", 0))
        c.sx, c.sy = float(d.get("sx", 0.0)), float(d.get("sy", 0.0))
        c.sxx, c.sxy = float(d.get("sxx", 0.0)), float(d.get("sxy", 0.0))
        return c


@dataclass(frozen=True)
class MemoryQuery:
    """What the memory can say about one pooled vector."""

    estimate: float | None
    """Cosine-weighted mean of the neighbours' within-level spend ranks
    (0..1), or `None` when none of the `k` neighbours carries a value."""
    novelty: float
    """`(1 - max cos) / 2`: how far the nearest entry is."""
    spread: float | None
    """Weighted stdev of the valued neighbours' values; `None` with < 2."""
    max_cos: float
    n_entries: int
    n_valued: int
    """Valued entries among the `k` neighbours."""


@dataclass(frozen=True)
class LevelDecision:
    """The asymmetric map's verdict for one request (§13.5)."""

    level: int
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
        levels: int = 0,
    ) -> None:
        self.hidden_size = hidden_size
        self.cfg = cfg
        self.model = model
        self.levels = int(levels)
        size = int(cfg.memory_size)
        self._keys = np.zeros((size, hidden_size), dtype=np.float32)
        self._values = np.full(size, np.nan, dtype=np.float32)
        self._sessions: list[str | None] = [None] * size
        self._levels_used = np.zeros(size, dtype=np.int8)
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
        self._spend_digest = TDigest(compression=cfg.digest_compression)
        self._level_digests: dict[int, TDigest] = {}
        self._probe_clock = 0
        self._calibration = [_Calibration() for _ in range(CALIBRATION_BINS)]
        self._calibration_all = _Calibration()

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
        """The memory may decide an effort level."""
        need = self.cfg.min_entries
        return self._n >= (int(self.cfg.k) if need is None else int(need))

    # --------------------------------------------------------------- insert

    def insert(
        self,
        vec: np.ndarray,
        reasoning_tokens: int | None,
        close_kind: str,
        session_id: str | None = None,
        level: int = 0,
        estimate: float | None = None,
        novelty_rank: float | None = None,
    ) -> None:
        """Record one finished request.

        Args:
            vec: the pooled prefill state (any scale; stored L2-normalised).
            reasoning_tokens: think tokens the request spent, or `None`.
            close_kind: `natural` / `soft` / `forced`. Only a natural close
                carries a value - the other two are right-censored, so they
                enter as keys with no value.
            session_id: conversation key for the per-session eviction cap.
            level: effort level the prefill decision chose.
            estimate: the raw kNN estimate the decision saw, if any.
            novelty_rank: the novelty rank the decision saw, if any.
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
        self._levels_used[slot] = np.int8(max(-128, min(127, level)))
        if valued:
            log_spend = float(self._values[slot])
            if estimate is not None:
                realised = self._difficulty(int(level), log_spend)
                self._observe(estimate, novelty_rank, realised)
            self._absorb_spend(int(level), log_spend)
            self._loo_observe(slot)
        if session_id is not None:
            self._by_session.setdefault(session_id, deque()).append(slot)
        self._inserts += 1
        self._since_flush += 1
        self.maybe_save()

    def _absorb_spend(self, level: int, log_spend: float) -> None:
        self._spend_digest.add(log_spend)
        digest = self._level_digests.get(level)
        if digest is None:
            digest = self._level_digests[level] = TDigest(
                compression=self.cfg.digest_compression
            )
        digest.add(log_spend)

    def _difficulty(self, level: int, log_spend: float) -> float:
        """Spend ranked within its own level's lane; the global spend rank
        stands in until the lane has seen `k` natural closes."""
        digest = self._level_digests.get(int(level))
        if digest is not None and digest.count >= self.cfg.k:
            rank = digest.rank(log_spend)
            if rank is not None:
                return rank
        return self._spend_digest.rank(log_spend) or 0.0

    def take_probe(self) -> bool:
        """Every `probe_every`-th call answers True (0 disables)."""
        every = int(self.cfg.probe_every)
        if every <= 0:
            return False
        self._probe_clock = (self._probe_clock + 1) % every
        return self._probe_clock == 0

    @staticmethod
    def _bin(novelty_rank: float) -> int:
        return min(CALIBRATION_BINS - 1, max(0, int(novelty_rank * CALIBRATION_BINS)))

    def calibrate(self, estimate: float, novelty_rank: float | None) -> float:
        """Map a raw estimate through the calibration of its novelty bin.

        Hierarchical: the bin's posterior sits on the pooled posterior, which
        sits on the identity; each prior is worth one neighbourhood (`k`)."""
        k = float(self.cfg.k)
        pooled = self._calibration_all.fit((0.0, 1.0), k)
        line = pooled
        if novelty_rank is not None:
            line = self._calibration[self._bin(novelty_rank)].fit(pooled, k)
        a, b = line
        return min(max(a + b * estimate, 0.0), 1.0)

    def _observe(self, estimate: float, novelty_rank: float | None, realised: float) -> None:
        self._calibration_all.add(estimate, realised)
        if novelty_rank is not None:
            self._calibration[self._bin(novelty_rank)].add(estimate, realised)

    def _loo_observe(self, slot: int) -> None:
        """Leave-one-out estimates for `slot` and the neighbours it changed.

        The memory already holds the evidence a fresh pair gives: what it
        would predict for an entry it has, against what that entry spent. One
        batched matmul yields ~k+1 pairs per insert instead of one."""
        k = int(self.cfg.k)
        n = self._n
        if n <= k:
            return
        sims_new = self._keys[:n] @ self._keys[slot]
        sims_new[slot] = -np.inf
        near = np.argpartition(-sims_new, k - 1)[:k]
        slots = [slot] + [int(i) for i in near if not math.isnan(self._values[i])]
        sims = self._keys[slots] @ self._keys[:n].T
        for row, s_ in enumerate(slots):
            if math.isnan(self._values[s_]):
                continue
            sim = sims[row]
            sim[s_] = -np.inf
            top = np.argpartition(-sim, k - 1)[:k]
            vals = self._values[top]
            ok = ~np.isnan(vals)
            if not ok.any():
                continue
            vs = sim[top][ok]
            w = np.exp((vs - vs.max()) / max(self.cfg.temperature, 1e-6))
            w /= w.sum()
            diff = np.fromiter(
                (
                    self._difficulty(int(lv), float(r))
                    for lv, r in zip(self._levels_used[top][ok], vals[ok])
                ),
                dtype=np.float64,
                count=int(ok.sum()),
            )
            est = float((w * diff).sum())
            novelty = (1.0 - float(sim[top].max())) / 2.0
            realised = self._difficulty(int(self._levels_used[s_]), float(self._values[s_]))
            self._observe(est, self._novelty_digest.rank(novelty), realised)

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
            raw = values[valued].astype(np.float64)
            lvls = self._levels_used[top][valued]
            vals = np.fromiter(
                (self._difficulty(int(lv), float(r)) for lv, r in zip(lvls, raw)),
                dtype=np.float64,
                count=int(valued.sum()),
            )
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
        """`(calibrated estimate, novelty rank, spread rank)`, then absorb.

        The estimate is a difficulty percentile; it is passed through the
        calibration of its novelty bin, so neighbours that have not predicted
        anything at that distance pull it to the mean instead of deciding."""
        nov_rank = self._novelty_digest.rank(q.novelty)
        est_rank = (
            None if q.estimate is None else self.calibrate(q.estimate, nov_rank)
        )
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
            "levels": self.levels,
            "memory_size": int(self._keys.shape[0]),
            "n": self._n,
            "next": self._next,
            "inserts": self._inserts,
            "sessions": self._sessions[: self._n],
            "est_digest": self._est_digest.to_dict(),
            "novelty_digest": self._novelty_digest.to_dict(),
            "spread_digest": self._spread_digest.to_dict(),
            "spend_digest": self._spend_digest.to_dict(),
            "level_digests": {
                str(level): digest.to_dict()
                for level, digest in self._level_digests.items()
            },
            "probe_clock": self._probe_clock,
            "calibration": [c.to_dict() for c in self._calibration],
            "calibration_all": self._calibration_all.to_dict(),
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
                    levels_used=self._levels_used[: self._n],
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
                or (self.levels and meta.get("levels") != self.levels)
            ):
                logger.warning(
                    "dynamic_effort: dropping %s (model/dim/levels/version "
                    "mismatch: %s)",
                    path,
                    {k: meta.get(k) for k in ("version", "model", "hidden_size")},
                )
                return False
            keys = data["keys"].astype(np.float32)
            n = min(keys.shape[0], self._keys.shape[0])
            self._keys[:n] = keys[:n]
            self._values[:n] = data["values"][:n]
            self._levels_used[:n] = data["levels_used"][:n]
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
        self._novelty_digest = TDigest.from_dict(meta["novelty_digest"])
        self._spread_digest = TDigest.from_dict(meta["spread_digest"])
        self._probe_clock = int(meta.get("probe_clock", 0))
        cal = meta.get("calibration")
        if cal and len(cal) == CALIBRATION_BINS:
            self._calibration = [_Calibration.from_dict(c) for c in cal]
            self._calibration_all = _Calibration.from_dict(
                meta.get("calibration_all", {})
            )
        if "level_digests" in meta:
            self._est_digest = TDigest.from_dict(meta["est_digest"])
            self._spend_digest = TDigest.from_dict(meta["spend_digest"])
            self._level_digests = {
                int(level): TDigest.from_dict(d)
                for level, d in meta["level_digests"].items()
            }
        else:
            # A file written when the estimate was raw spend: rebuild the
            # lanes from the stored spend + level and restart the estimate
            # digest, whose units changed.
            self._est_digest = TDigest(compression=self.cfg.digest_compression)
            self._spend_digest = TDigest(compression=self.cfg.digest_compression)
            self._level_digests = {}
            for slot in range(n):
                value = float(self._values[slot])
                if not math.isnan(value):
                    self._absorb_spend(int(self._levels_used[slot]), value)
            logger.info(
                "dynamic_effort: migrated %s to within-level difficulty "
                "(%d valued entries, lanes %s)",
                path,
                self.n_valued,
                {k: int(d.count) for k, d in sorted(self._level_digests.items())},
            )
        return True


def decide_effort_level(
    query: MemoryQuery | None,
    ranks: tuple[float | None, float | None, float | None],
    cfg: HiddenEffortConfig,
    top_level: int,
    probe: bool = False,
) -> LevelDecision:
    """The quantile map, low-resting (§13.5).

    Low is the resting level; the memory has to earn anything above it. The
    estimate is the neighbours' within-level difficulty (0..1), calibrated
    against what requests at that novelty actually turned out to need, so the
    cuts apply to it directly: at or above `q_high` the top level, at or
    above `q_mid` the middle one, otherwise low. A prompt whose neighbours
    have never predicted anything is pulled to the mean by the calibration,
    not gated by a constant. A `probe` decision renders one level below the
    map's verdict, so every neighbourhood keeps receiving samples at the
    cheaper level and can be pulled down by them.

    Args:
        query: the kNN result, or `None` when the memory could not answer.
        ranks: `(estimate, novelty rank, spread rank)`.
        cfg: hidden-effort settings.
        top_level: highest effort level the request has.
        probe: render one level below the verdict.

    Returns:
        The effort level and why it was chosen.
    """
    default = min(cfg.default_level, top_level)
    estimate, nov_rank, spread_rank = ranks
    if query is None or estimate is None:
        return LevelDecision(default, "no-estimate", estimate, nov_rank, spread_rank)
    if estimate >= cfg.q_high:
        level, reason = min(2, top_level), "q>=q_high"
    elif estimate >= cfg.q_mid:
        level, reason = min(1, top_level), "q>=q_mid"
    else:
        level, reason = 0, "low"
    if probe and level > 0:
        level, reason = level - 1, f"probe/{reason}"
    return LevelDecision(level, reason, estimate, nov_rank, spread_rank)
