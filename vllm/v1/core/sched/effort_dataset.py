# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-disk examples for a future hidden-state effort classifier (§13.13).

Every dynamic-effort request that reached the §13.3 seam contributes one
example: the pooled last-prompt-token hidden state the memory keys on, the
level that was chosen and who chose it, and what the request then spent.
Examples are buffered by the scheduler and written by a background thread as
`shard-<pid>-<start_ts>-<seq>.npz` files, so the scheduler loop never waits on
the disk; a stalled or full disk drops examples with a warning.
"""

import contextlib
import glob
import math
import os
import sys
import tempfile
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

DECIDED_BY = ("default", "memory", "vote", "forced")
"""Who chose the level: the server default, the kNN memory, the level vote,
or the client (`dynamic_effort_level` / `force_off`)."""

ABORTED = "aborted"
"""`finish_reason` of an example whose request never finished: an abort, or a
request still pending at shutdown."""

TAG_MAX_LEN = 128
"""Longest `vllm_xargs.effort_tag` accepted."""

_SCALAR_FIELDS: dict[str, Any] = {
    "req_id": str,
    "ts": np.float64,
    "num_prompt_tokens": np.int32,
    "body_len": np.int32,
    "level": np.int16,
    "decided_by": str,
    "estimate": np.float32,
    "calibrated": np.float32,
    "novelty_rank": np.float32,
    "neighbours": np.int32,
    "reasoning_tokens": np.int32,
    "num_output_tokens": np.int32,
    "close_kind": str,
    "finish_reason": str,
    "tag": str,
}
_MATRIX_FIELDS = ("vector", "vote_probs", "level_votes")
FIELDS = tuple(_SCALAR_FIELDS) + _MATRIX_FIELDS


def _float(value: float | None) -> float:
    return math.nan if value is None else float(value)


def _pad_rows(rows: list[list[float]], width: int, fill: float, dtype) -> np.ndarray:
    out = np.full((len(rows), width), fill, dtype=dtype)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row[:width]
    return out


def example_nbytes(hidden_size: int, num_levels: int = 3) -> int:
    """Approximate on-disk bytes of one example (the vector dominates)."""
    return hidden_size * 2 + num_levels * 6 + 128


class EffortDatasetWriter:
    """Buffers examples per request and writes shards from a background thread.

    Args:
        directory: where the shards go; created on first use.
        hidden_size: width of the pooled vector; other widths are rejected.
        shard_size: examples per shard; the pending shard is also written at
            `close()`.
        max_buffered: examples held for the writer thread before new ones
            are dropped (the disk is slow or full).
    """

    def __init__(
        self,
        directory: str,
        hidden_size: int,
        shard_size: int = 4096,
        max_buffered: int | None = None,
    ) -> None:
        self.directory = directory
        self.hidden_size = int(hidden_size)
        self.shard_size = max(int(shard_size), 1)
        self.max_buffered = (
            4 * self.shard_size if max_buffered is None else int(max_buffered)
        )
        self._pending: dict[str, dict[str, Any]] = {}
        self._queue: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._seq = 0
        self._start_ts = int(time.time())
        self.num_written = 0
        self.num_dropped = 0
        self.num_shards = 0
        self._thread = threading.Thread(
            target=self._run, name="effort-dataset", daemon=True
        )
        self._thread.start()

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    def begin(
        self,
        req_id: str,
        vector: np.ndarray,
        num_prompt_tokens: int,
        body_len: int,
        tag: str | None = None,
    ) -> bool:
        """Open the example at the seam, with the vector the decision saw."""
        vec = np.asarray(vector).reshape(-1)
        if vec.shape[0] != self.hidden_size:
            logger.warning_once(
                "effort dataset: vector width %d != hidden_size %d; not recorded",
                vec.shape[0],
                self.hidden_size,
            )
            return False
        self._pending[req_id] = {
            "req_id": req_id,
            "ts": time.time(),
            "vector": vec.astype(np.float16),
            "num_prompt_tokens": int(num_prompt_tokens),
            "body_len": int(body_len),
            "tag": tag or "",
        }
        return True

    def finish(
        self,
        req_id: str,
        *,
        level: int,
        decided_by: str,
        vote_probs: list[float] | None,
        level_votes: list[int] | None,
        estimate: float | None,
        calibrated: float | None,
        novelty_rank: float | None,
        neighbours: int | None,
        reasoning_tokens: int,
        num_output_tokens: int,
        close_kind: str,
        finish_reason: str | None,
    ) -> bool:
        """Complete the example and hand it to the writer thread.

        Returns False when the request never reached `begin` or the buffer
        is full (the example is dropped and counted).
        """
        example = self._pending.pop(req_id, None)
        if example is None:
            return False
        assert decided_by in DECIDED_BY, decided_by
        example.update(
            level=int(level),
            decided_by=decided_by,
            vote_probs=[_float(p) for p in vote_probs] if vote_probs else [],
            level_votes=[int(v) for v in level_votes] if level_votes else [],
            estimate=_float(estimate),
            calibrated=_float(calibrated),
            novelty_rank=_float(novelty_rank),
            neighbours=-1 if neighbours is None else int(neighbours),
            reasoning_tokens=int(reasoning_tokens),
            num_output_tokens=int(num_output_tokens),
            close_kind=close_kind,
            finish_reason=finish_reason or ABORTED,
        )
        with self._lock:
            if len(self._queue) >= self.max_buffered:
                self.num_dropped += 1
                logger.warning_once(
                    "effort dataset: %d examples buffered and the writer has not "
                    "caught up; dropping (is %s slow or full?)",
                    len(self._queue),
                    self.directory,
                )
                return False
            self._queue.append(example)
            ready = len(self._queue) >= self.shard_size
        if ready:
            self._wake.set()
        return True

    def forget(self, req_id: str) -> None:
        """Drop a pending example without writing it."""
        self._pending.pop(req_id, None)

    def close(self, timeout: float = 30.0) -> None:
        """Write pending requests as aborted, flush, and stop the thread."""
        for req_id in list(self._pending):
            self.finish(
                req_id,
                level=-1,
                decided_by="default",
                vote_probs=None,
                level_votes=None,
                estimate=None,
                calibrated=None,
                novelty_rank=None,
                neighbours=None,
                reasoning_tokens=0,
                num_output_tokens=0,
                close_kind="",
                finish_reason=ABORTED,
            )
        self._stop = True
        self._wake.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("effort dataset: writer thread did not finish in time")

    def flush(self) -> None:
        """Write everything buffered now (test / shutdown helper)."""
        while True:
            with self._lock:
                if not self._queue:
                    return
                batch = [self._queue.popleft() for _ in range(len(self._queue))]
            self._write_shard(batch)

    def _run(self) -> None:
        while not self._stop:
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            self._drain(final=False)
        self._drain(final=True)

    def _drain(self, final: bool) -> None:
        while True:
            with self._lock:
                n = len(self._queue)
                if n == 0 or (n < self.shard_size and not final):
                    return
                batch = [self._queue.popleft() for _ in range(min(n, self.shard_size))]
            self._write_shard(batch)

    def _write_shard(self, batch: list[dict[str, Any]]) -> None:
        try:
            arrays = self._pack(batch)
            os.makedirs(self.directory, exist_ok=True)
            name = f"shard-{os.getpid()}-{self._start_ts}-{self._seq:05d}.npz"
            self._seq += 1
            fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".npz")
            try:
                with os.fdopen(fd, "wb") as f:
                    np.savez(f, **arrays)
                os.replace(tmp, os.path.join(self.directory, name))
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            self.num_written += len(batch)
            self.num_shards += 1
        except Exception as exc:  # noqa: BLE001
            self.num_dropped += len(batch)
            logger.warning(
                "effort dataset: dropped %d examples, write to %s failed: %s",
                len(batch),
                self.directory,
                exc,
            )

    def _pack(self, batch: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for name, dtype in _SCALAR_FIELDS.items():
            values = [ex[name] for ex in batch]
            arrays[name] = (
                np.asarray(values) if dtype is str else np.asarray(values, dtype=dtype)
            )
        arrays["vector"] = np.stack([ex["vector"] for ex in batch]).astype(np.float16)
        probs = [ex["vote_probs"] for ex in batch]
        votes = [ex["level_votes"] for ex in batch]
        arrays["vote_probs"] = _pad_rows(
            probs, max((len(p) for p in probs), default=0), math.nan, np.float32
        )
        arrays["level_votes"] = _pad_rows(
            votes, max((len(v) for v in votes), default=0), -1, np.int16
        )
        return arrays


def shard_paths(directory: str) -> list[str]:
    return sorted(glob.glob(os.path.join(directory, "shard-*.npz")))


def load_dataset(directory: str) -> dict[str, np.ndarray]:
    """Concatenate every shard under `directory` into one dict of arrays.

    Ragged `vote_probs` / `level_votes` widths are padded (NaN / -1) to the
    widest shard. An empty directory yields empty arrays.
    """
    parts: dict[str, list[np.ndarray]] = {name: [] for name in FIELDS}
    for path in shard_paths(directory):
        with np.load(path, allow_pickle=False) as npz:
            for name in FIELDS:
                parts[name].append(npz[name])
    out: dict[str, np.ndarray] = {}
    for name, chunks in parts.items():
        if not chunks:
            out[name] = np.zeros((0,), dtype=np.float32)
            continue
        if name in ("vote_probs", "level_votes"):
            width = max(c.shape[1] for c in chunks)
            fill = math.nan if name == "vote_probs" else -1
            chunks = [
                np.pad(c, ((0, 0), (0, width - c.shape[1])), constant_values=fill)
                for c in chunks
            ]
        out[name] = np.concatenate(chunks)
    return out


def _summary(directory: str) -> str:
    paths = shard_paths(directory)
    data = load_dataset(directory)
    n = len(data["req_id"])
    lines = [f"{directory}: {n} examples in {len(paths)} shards"]
    if paths:
        sizes = [os.path.getsize(p) for p in paths]
        lines.append(
            f"  shard bytes: min {min(sizes)} max {max(sizes)} total {sum(sizes)}"
            f"  ({sum(sizes) // max(n, 1)} per example)"
        )
    if n == 0:
        return "\n".join(lines)
    vec = data["vector"]
    lines.append(f"  vector: {vec.shape[1]} x {vec.dtype}")

    def counts(key: str) -> str:
        values, freq = np.unique(data[key], return_counts=True)
        return ", ".join(f"{v}={c}" for v, c in zip(values.tolist(), freq.tolist()))

    for key in ("level", "decided_by", "finish_reason", "close_kind"):
        lines.append(f"  {key}: {counts(key)}")
    tags = data["tag"]
    tagged = tags != ""
    lines.append(f"  tag: {int(tagged.sum())} tagged, {counts('tag')}")
    finished = data["finish_reason"] != ABORTED
    if finished.any():
        rt = data["reasoning_tokens"][finished]
        lines.append(
            "  reasoning_tokens (finished): "
            f"median {int(np.median(rt))} p90 {int(np.percentile(rt, 90))}"
        )
        for level in np.unique(data["level"][finished]).tolist():
            sel = finished & (data["level"] == level)
            lines.append(
                f"    level {level}: n={int(sel.sum())} median "
                f"{int(np.median(data['reasoning_tokens'][sel]))}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(
            "usage: python -m vllm.v1.core.sched.effort_dataset <dir>",
            file=sys.stderr,
        )
        return 2
    print(_summary(args[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
