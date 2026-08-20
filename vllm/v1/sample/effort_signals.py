# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Effort telemetry: canonical-stage entropy / top-2 margin per sampled row.

Pure torch/numpy helpers shared by the V1 and V2 samplers, the model runners
and the scheduler-side JSONL sink. Everything here runs on CPU tensors so the
semantics are unit-testable without a GPU.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from typing import IO, Any

import numpy as np
import torch

from vllm.sampling_params import SamplingParams

# Column layout of a reduced per-request signal row.
ENTROPY, MARGIN, NUM_ROWS = 0, 1, 2

# ``SamplingParams.extra_args`` keys that opt a request into the telemetry.
_OPT_IN_KEYS = ("effort_telemetry", "dynamic_effort")


def wants_effort_signals(sampling_params: SamplingParams | None) -> bool:
    """True when the request opted into effort telemetry via ``extra_args``."""
    if sampling_params is None:
        return False
    extra_args = sampling_params.extra_args
    if not extra_args:
        return False
    return any(bool(extra_args.get(key)) for key in _OPT_IN_KEYS)


def effort_row_signals(logits: torch.Tensor) -> torch.Tensor:
    """Per-row normalised entropy and top1-top2 margin.

    Args:
        logits: ``[rows, vocab]`` canonical-stage logits (any float dtype).

    Returns:
        ``[rows, 2]`` fp32: entropy / ``log(vocab)`` in ``[0, 1]`` and the
        top1-top2 margin in logit units.
    """
    z = logits.float()
    vocab = z.shape[-1]
    lse = torch.logsumexp(z, dim=-1)
    probs = torch.softmax(z, dim=-1)
    # ``0 * -inf`` is nan; masked entries carry zero mass and zero entropy.
    pz = torch.where(probs > 0, probs * z, torch.zeros_like(z))
    entropy = (lse - pz.sum(dim=-1)) / math.log(vocab)
    top2 = z.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    out = torch.stack((entropy, margin), dim=-1)
    return torch.nan_to_num(out, nan=0.0)


def effort_row_signals_scattered(
    logits: torch.Tensor,
    row_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Row signals for a subset of rows, zero elsewhere.

    Args:
        logits: ``[rows, vocab]`` logits.
        row_indices: int64 row indices to compute, or ``None`` for all rows.

    Returns:
        ``[rows, 2]`` fp32 with computed rows filled and other rows zero.
    """
    if row_indices is None:
        return effort_row_signals(logits)
    out = torch.zeros((logits.shape[0], 2), dtype=torch.float32, device=logits.device)
    if row_indices.numel() == 0:
        return out
    row_indices = row_indices.to(device=logits.device)
    out.index_copy_(0, row_indices, effort_row_signals(logits[row_indices]))
    return out


def reduce_committed(
    row_signals: torch.Tensor,
    cu_num_rows: torch.Tensor,
    num_committed: torch.Tensor,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-request means over the committed prefix of each request's rows.

    Rows of request ``i`` are ``cu_num_rows[i]:cu_num_rows[i+1]`` in commit
    order; only the first ``num_committed[i]`` rows (bonus / accepted
    positions) contribute. No host sync: all shapes come from
    ``row_signals.shape[0]`` and ``cu_num_rows.shape[0]``.

    Args:
        row_signals: ``[rows, 2]`` per-row signals.
        cu_num_rows: ``[num_reqs + 1]`` int cumulative row counts.
        num_committed: ``[num_reqs]`` int committed row count per request.
        row_mask: optional ``[rows]`` bool; rows with ``False`` are excluded.

    Returns:
        ``[num_reqs, 3]`` fp32: mean entropy, mean margin, committed row count.
    """
    num_reqs = cu_num_rows.shape[0] - 1
    num_rows = row_signals.shape[0]
    device = row_signals.device
    if num_reqs == 0 or num_rows == 0:
        return torch.zeros((num_reqs, 3), dtype=torch.float32, device=device)
    counts = (cu_num_rows[1:] - cu_num_rows[:-1]).long()
    req_of_row = torch.repeat_interleave(
        torch.arange(num_reqs, device=device), counts, output_size=num_rows
    )
    local_pos = torch.arange(num_rows, device=device) - cu_num_rows[req_of_row].long()
    committed = local_pos < num_committed.long()[req_of_row]
    if row_mask is not None:
        committed = committed & row_mask
    weight = committed.to(torch.float32)
    sums = torch.zeros((num_reqs, 2), dtype=torch.float32, device=device)
    sums.index_add_(0, req_of_row, row_signals.float() * weight[:, None])
    n_rows = torch.zeros(num_reqs, dtype=torch.float32, device=device)
    n_rows.index_add_(0, req_of_row, weight)
    means = sums / n_rows.clamp(min=1.0)[:, None]
    return torch.cat((means, n_rows[:, None]), dim=-1)


def commit_order_permutation(num_draft_tokens: Sequence[int]) -> np.ndarray:
    """Row permutation from ``[target rows..., bonus rows...]`` to commit order.

    V1 spec decode keeps target rows (all requests' draft positions) and bonus
    rows (one per request) in two blocks. Commit order per request is its draft
    positions followed by its bonus row.

    Args:
        num_draft_tokens: per-request draft counts.

    Returns:
        int64 index array of length ``sum(num_draft_tokens) + len(...)``.
    """
    d = np.asarray(num_draft_tokens, dtype=np.int64)
    num_reqs = d.shape[0]
    total_draft = int(d.sum())
    cu = np.zeros(num_reqs + 1, dtype=np.int64)
    np.cumsum(d, out=cu[1:])
    perm = np.empty(total_draft + num_reqs, dtype=np.int64)
    out_start = cu + np.arange(num_reqs + 1, dtype=np.int64)
    for i in range(num_reqs):
        perm[out_start[i] : out_start[i] + d[i]] = np.arange(cu[i], cu[i + 1])
        perm[out_start[i] + d[i]] = total_draft + i
    return perm


def flagged_row_indices(
    flags: np.ndarray, num_rows_per_req: np.ndarray | None = None
) -> np.ndarray:
    """Row indices belonging to flagged requests.

    Args:
        flags: ``[num_reqs]`` bool.
        num_rows_per_req: rows per request in row order; ``None`` means one.

    Returns:
        int64 row indices (empty when nothing is flagged).
    """
    if num_rows_per_req is None:
        return np.flatnonzero(flags).astype(np.int64)
    return np.flatnonzero(np.repeat(flags, num_rows_per_req)).astype(np.int64)


def signals_to_dict(
    req_ids: Sequence[str],
    signals: np.ndarray,
    flags: np.ndarray | None,
) -> dict[str, tuple[float, float, int]]:
    """Build ``ModelRunnerOutput.effort_signals`` for flagged requests only.

    Args:
        req_ids: batch-ordered request ids.
        signals: ``[num_reqs, 3]`` host array (entropy, margin, n_rows).
        flags: ``[num_reqs]`` bool opt-in mask; ``None`` means all flagged.

    Returns:
        req_id -> (mean entropy, mean margin, n committed rows).
    """
    out: dict[str, tuple[float, float, int]] = {}
    for i, req_id in enumerate(req_ids):
        if flags is not None and not flags[i]:
            continue
        row = signals[i]
        out[req_id] = (float(row[ENTROPY]), float(row[MARGIN]), int(row[NUM_ROWS]))
    return out


class ThinkTracker:
    """Incremental "inside a think block" detector over a growing token list.

    Same rule as ``ThinkingBudgetStateHolder``: the last start sequence after
    the last end sequence means thinking. Missing start/end ids -> unknown.
    """

    def __init__(
        self,
        start_ids: Sequence[int] | None,
        end_ids: Sequence[int] | None,
    ):
        self.start_ids = list(start_ids or [])
        self.end_ids = list(end_ids or [])
        self.enabled = bool(self.start_ids and self.end_ids)
        self._overlap = max(len(self.start_ids), len(self.end_ids)) - 1
        self._scanned = 0
        self._last_start = -1
        self._last_end = -1
        self._prompt_in_think = False

    @staticmethod
    def _last_index(tokens: Sequence[int], pattern: list[int], base: int) -> int:
        n = len(pattern)
        for i in range(len(tokens) - n, -1, -1):
            if list(tokens[i : i + n]) == pattern:
                return base + i
        return -1

    def seed_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        """Start in think mode when the prompt's tail opens a think block."""
        if not self.enabled:
            return
        tail = list(prompt_token_ids[-max(64, self._overlap + 1) :])
        s = self._last_index(tail, self.start_ids, 0)
        e = self._last_index(tail, self.end_ids, 0)
        if s > e:
            self._prompt_in_think = True

    def update(self, token_ids: Sequence[int]) -> bool | None:
        """Scan tokens appended since the last call; return in-think state."""
        if not self.enabled:
            return None
        n = len(token_ids)
        if n < self._scanned:
            self._scanned, self._last_start, self._last_end = 0, -1, -1
        base = max(0, self._scanned - self._overlap)
        window = token_ids[base:]
        s = self._last_index(window, self.start_ids, base)
        e = self._last_index(window, self.end_ids, base)
        if s > self._last_start:
            self._last_start = s
        if e > self._last_end:
            self._last_end = e
        self._scanned = n
        if self._last_start == -1 and self._last_end == -1:
            return self._prompt_in_think
        return self._last_start > self._last_end


def format_sink_record(
    req_id: str,
    step: int,
    num_output_tokens: int,
    signal: tuple[float, float, int],
    num_draft_tokens: int | None,
    num_accepted: int | None,
    in_think: bool | None,
) -> str:
    """One JSONL line for the effort telemetry sink."""
    rec: dict[str, Any] = {
        "req_id": req_id,
        "step": step,
        "num_output_tokens": num_output_tokens,
        "entropy": signal[ENTROPY],
        "margin": signal[MARGIN],
        "n_rows": signal[NUM_ROWS],
        "num_draft_tokens": num_draft_tokens,
        "num_accepted": num_accepted,
        "in_think": in_think,
    }
    return json.dumps(rec, separators=(",", ":"))


class EffortTelemetrySink:
    """Buffered JSONL writer keyed by request; flushes every N lines."""

    FLUSH_EVERY = 256

    def __init__(
        self,
        path: str,
        start_ids: Sequence[int] | None,
        end_ids: Sequence[int] | None,
        stream: IO[str] | None = None,
    ):
        self.path = path
        self._start_ids = start_ids
        self._end_ids = end_ids
        self._file = stream if stream is not None else open(path, "a")  # noqa: SIM115
        self._buffer: list[str] = []
        self._steps: dict[str, int] = {}
        self._trackers: dict[str, ThinkTracker] = {}

    @classmethod
    def from_env(
        cls,
        start_ids: Sequence[int] | None,
        end_ids: Sequence[int] | None,
    ) -> EffortTelemetrySink | None:
        path = os.environ.get("VLLM_EFFORT_TELEMETRY")
        if not path:
            return None
        return cls(path, start_ids, end_ids)

    def record(
        self,
        req_id: str,
        output_token_ids: Sequence[int],
        signal: tuple[float, float, int],
        num_draft_tokens: int | None,
        num_accepted: int | None,
        finished: bool,
        prompt_token_ids: Sequence[int] | None = None,
    ) -> None:
        tracker = self._trackers.get(req_id)
        if tracker is None:
            tracker = ThinkTracker(self._start_ids, self._end_ids)
            if prompt_token_ids:
                # Chat templates open the think block in the prompt
                # (e.g. "<think>\n"), so seed the state from the prompt tail.
                tracker.seed_from_prompt(prompt_token_ids)
            self._trackers[req_id] = tracker
        step = self._steps.get(req_id, 0) + 1
        self._steps[req_id] = step
        self._buffer.append(
            format_sink_record(
                req_id,
                step,
                len(output_token_ids),
                signal,
                num_draft_tokens,
                num_accepted,
                tracker.update(output_token_ids),
            )
        )
        if finished:
            self.forget(req_id)
            self.flush()
        elif len(self._buffer) >= self.FLUSH_EVERY:
            self.flush()

    def record_finish(self, req_id: str, payload: dict[str, Any]) -> None:
        """One `event: finish` line per request: what the client asked for,
        what was served, and the controller's closing report."""
        self._buffer.append(
            json.dumps({"req_id": req_id, "event": "finish", **payload})
        )
        self.flush()

    def forget(self, req_id: str) -> None:
        self._steps.pop(req_id, None)
        self._trackers.pop(req_id, None)

    def flush(self) -> None:
        if not self._buffer:
            return
        self._file.write("\n".join(self._buffer) + "\n")
        self._file.flush()
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self._file.close()
