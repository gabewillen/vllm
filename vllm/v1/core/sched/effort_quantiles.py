# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Running quantile sketches for the dynamic-effort signals.

The controller is self-normalizing: instead of a per-model calibration table
of absolute means and standard deviations, the scheduler keeps one streaming
t-digest per signal, fed by every request that emits signals. Features are
then *percentile ranks* in that digest, so a threshold like "top quartile of
uncertainty" means the same thing on any model, quantization or sampler
setting.

The digests are persisted to a JSON file (``dynamic_effort.quantile_path``)
so a restart warms instantly; while a digest is cold (fewer than
``min_samples`` observations) it reports ``None`` and the controller never
escalates.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from typing import Any

SKETCH_VERSION = 1
SIGNAL_KEYS = ("entropy", "margin", "p_end", "acceptance")


class TDigest:
    """Small merging t-digest over ``(mean, weight)`` centroids.

    Supports both directions the controller needs: ``rank(value)`` for a
    percentile-rank feature and ``quantile(q)`` for resolving a policy
    threshold into an absolute cutoff the worker can compare against.
    """

    def __init__(self, compression: float = 100.0, buffer_size: int = 256):
        if compression < 10.0:
            raise ValueError("t-digest compression must be >= 10")
        self.compression = float(compression)
        self.buffer_size = int(buffer_size)
        self.means: list[float] = []
        self.weights: list[float] = []
        self.count: float = 0.0
        self.min: float = math.inf
        self.max: float = -math.inf
        self._buffer: list[float] = []

    # -- ingestion ---------------------------------------------------------

    def add(self, value: float, weight: float = 1.0) -> None:
        """Add one observation; non-finite values and weights are ignored."""
        if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0.0:
            return
        self.count += weight
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value
        if weight == 1.0:
            self._buffer.append(value)
            if len(self._buffer) >= self.buffer_size:
                self.compress()
        else:
            self.means.append(value)
            self.weights.append(weight)
            self.compress()

    def add_many(self, values: Iterable[float]) -> None:
        for value in values:
            self.add(value)

    def compress(self) -> None:
        """Merge the pending buffer into the centroid list."""
        if not self._buffer and len(self.means) <= self._max_centroids():
            return
        pairs = list(zip(self.means, self.weights))
        pairs.extend((v, 1.0) for v in self._buffer)
        self._buffer.clear()
        if not pairs:
            return
        pairs.sort(key=lambda p: p[0])
        total = sum(w for _, w in pairs)
        if total <= 0.0:
            return
        means: list[float] = []
        weights: list[float] = []
        cur_mean, cur_weight = pairs[0]
        so_far = 0.0
        q_limit = self._q_limit(self._k(so_far / total) + 1.0)
        for mean, weight in pairs[1:]:
            proposed = cur_weight + weight
            if (so_far + proposed) / total <= q_limit:
                cur_mean += (mean - cur_mean) * (weight / proposed)
                cur_weight = proposed
                continue
            means.append(cur_mean)
            weights.append(cur_weight)
            so_far += cur_weight
            q_limit = self._q_limit(self._k(so_far / total) + 1.0)
            cur_mean, cur_weight = mean, weight
        means.append(cur_mean)
        weights.append(cur_weight)
        self.means, self.weights = means, weights
        self.count = total

    def _max_centroids(self) -> int:
        return int(math.ceil(self.compression * 2))

    def _k(self, q: float) -> float:
        """Scale function k1 (asin), which spends detail on the tails."""
        q = min(max(q, 0.0), 1.0)
        return self.compression * math.asin(2.0 * q - 1.0) / (2.0 * math.pi)

    def _q_limit(self, k: float) -> float:
        return (math.sin(2.0 * math.pi * k / self.compression) + 1.0) / 2.0

    # -- queries -----------------------------------------------------------

    def _sorted(self) -> tuple[list[float], list[float], float]:
        if self._buffer:
            self.compress()
        return self.means, self.weights, self.count

    def rank(self, value: float) -> float | None:
        """Fraction of the observed mass below ``value``, in ``[0, 1]``."""
        means, weights, total = self._sorted()
        if not means or total <= 0.0 or not math.isfinite(value):
            return None
        if value <= self.min:
            return 0.0
        if value >= self.max:
            return 1.0
        below = 0.0
        for mean, weight in zip(means, weights):
            if value < mean:
                # Linear interpolation inside the straddling centroid.
                below += weight * 0.5
                break
            if value == mean:
                below += weight * 0.5
                break
            below += weight
        return min(max(below / total, 0.0), 1.0)

    def quantile(self, q: float) -> float | None:
        """Value at quantile ``q`` (``0 <= q <= 1``)."""
        means, weights, total = self._sorted()
        if not means or total <= 0.0:
            return None
        q = min(max(q, 0.0), 1.0)
        if q == 0.0:
            return self.min
        if q == 1.0:
            return self.max
        target = q * total
        seen = 0.0
        for i, (mean, weight) in enumerate(zip(means, weights)):
            centre = seen + weight * 0.5
            if target <= centre:
                if i == 0:
                    lo_x, lo_c = self.min, 0.0
                else:
                    lo_x, lo_c = means[i - 1], seen - weights[i - 1] * 0.5
                span = centre - lo_c
                if span <= 0.0:
                    return mean
                return lo_x + (mean - lo_x) * (target - lo_c) / span
            seen += weight
        return self.max

    def edges(self, num_edges: int) -> list[float] | None:
        """Monotone quantile grid ``[quantile(0), ..., quantile(1)]``.

        The worker converts a value to a rank by binary search in this grid,
        which is how the rank rule stays identical on both evaluation sites.
        """
        if num_edges < 2:
            raise ValueError("num_edges must be >= 2")
        means, _, total = self._sorted()
        if not means or total <= 0.0:
            return None
        out: list[float] = []
        prev = -math.inf
        for i in range(num_edges):
            value = self.quantile(i / (num_edges - 1))
            assert value is not None
            # Guarantee non-decreasing edges even under float noise.
            prev = value = max(value, prev)
            out.append(value)
        return out

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        self.compress()
        return {
            "compression": self.compression,
            "count": self.count,
            "min": self.min if math.isfinite(self.min) else None,
            "max": self.max if math.isfinite(self.max) else None,
            "means": self.means,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TDigest:
        digest = cls(compression=float(data.get("compression", 100.0)))
        means = [float(x) for x in data.get("means", [])]
        weights = [float(x) for x in data.get("weights", [])]
        if len(means) != len(weights):
            raise ValueError("t-digest means/weights length mismatch")
        digest.means = means
        digest.weights = weights
        digest.count = float(data.get("count", sum(weights)))
        lo, hi = data.get("min"), data.get("max")
        digest.min = float(lo) if lo is not None else (means[0] if means else math.inf)
        digest.max = float(hi) if hi is not None else (means[-1] if means else -math.inf)
        return digest


class SignalSketches:
    """One :class:`TDigest` per dynamic-effort signal, with JSON persistence.

    ``rank`` and ``quantile`` return ``None`` while a signal is cold (fewer
    than ``min_samples`` observations); the controller treats that as "never
    escalate", which is the documented cold-start behaviour.
    """

    def __init__(
        self,
        min_samples: int = 2048,
        compression: float = 100.0,
        path: str | None = None,
        flush_every: int = 5000,
        keys: Sequence[str] = SIGNAL_KEYS,
    ):
        self.min_samples = int(min_samples)
        self.compression = float(compression)
        self.path = path
        self.flush_every = int(flush_every)
        self.digests: dict[str, TDigest] = {
            key: TDigest(compression=compression) for key in keys
        }
        self._since_flush = 0
        self.model: str | None = None

    # -- ingestion ---------------------------------------------------------

    def observe(self, key: str, value: float, weight: float = 1.0) -> None:
        digest = self.digests.get(key)
        if digest is None:
            return
        digest.add(value, weight)
        self._since_flush += 1

    def warm(self, key: str) -> bool:
        digest = self.digests.get(key)
        return digest is not None and digest.count >= self.min_samples

    def count(self, key: str) -> float:
        digest = self.digests.get(key)
        return 0.0 if digest is None else digest.count

    # -- queries -----------------------------------------------------------

    def rank(self, key: str, value: float | None) -> float | None:
        if value is None or not self.warm(key):
            return None
        return self.digests[key].rank(value)

    def quantile(self, key: str, q: float) -> float | None:
        if not self.warm(key):
            return None
        return self.digests[key].quantile(q)

    def edges(self, key: str, num_edges: int) -> list[float] | None:
        if not self.warm(key):
            return None
        return self.digests[key].edges(num_edges)

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SKETCH_VERSION,
            "model": self.model,
            "min_samples": self.min_samples,
            "signals": {k: d.to_dict() for k, d in self.digests.items()},
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        if int(data.get("version", 0)) != SKETCH_VERSION:
            raise ValueError(
                f"effort sketch version {data.get('version')} != {SKETCH_VERSION}"
            )
        self.model = data.get("model")
        for key, blob in (data.get("signals") or {}).items():
            if key in self.digests:
                self.digests[key] = TDigest.from_dict(blob)

    def load(self) -> bool:
        """Warm the digests from ``path``; ``False`` when nothing was loaded."""
        if not self.path or not os.path.exists(self.path):
            return False
        with open(self.path, encoding="utf-8") as f:
            self.load_dict(json.load(f))
        return True

    def save(self) -> None:
        """Atomically write the digests to ``path``."""
        if not self.path:
            return
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self._since_flush = 0

    def maybe_save(self) -> bool:
        """Persist once ``flush_every`` observations have accumulated."""
        if not self.path or self.flush_every <= 0:
            return False
        if self._since_flush < self.flush_every:
            return False
        self.save()
        return True
