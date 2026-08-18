# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic reasoning-effort controller: pure per-request policy.

The scheduler owns one `EffortState` per dynamic request and feeds it one
`EffortEvent` per engine step through `step_effort`. Nothing here touches the
scheduler, so a recorded event stream replays the same decisions.

Signals (docs/dynamic-reasoning.claude.md §3): think position (S1), mean
normalised entropy and top1-top2 margin of the committed rows (S2/S3) with
fast/slow EMAs (S4), MTP acceptance (S6, corroboration only), loop/marker
evidence (S7), max_tokens headroom (S8), batch-size rung cap (S11) and a
client deadline (S12). Decisions (§4): hard-stop clamp on a loop, and a
one-rung escalation at the `check_at` / `final_check_at` points of the cap.
"""

import math
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

from vllm.config.reasoning import DynamicEffortConfig


@dataclass
class EffortEvent:
    """Per-step observations for one request (all CPU scalars)."""

    new_token_ids: Sequence[int]
    """Committed output tokens of this step, in commit order."""
    entropy: float | None = None
    """Mean normalised entropy over the committed rows; `None` = missing."""
    margin: float | None = None
    """Mean top1-top2 logit margin over the committed rows; `None` = missing."""
    n_rows: int = 0
    """Committed rows the two means were taken over."""
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    batch_size: int = 1
    max_tokens: int = 0
    """The request's `max_tokens`."""
    now_ms: float | None = None
    """Monotonic clock in ms (only needed for `deadline_ms`)."""
    repetition_evidence: bool = False
    """The request's own `RepetitionDetectionParams` fired at a relaxed count."""
    acked_revision: int | None = None
    """Budget revision the worker reported applied this step."""


@dataclass
class EffortDecision:
    """What the scheduler must do after a step."""

    budget_update: tuple[int, int] | None = None
    """`(revision, absolute thinking budget)` to ship, or `None`."""
    escalation: tuple[int, int] | None = None
    """`(from_rung, to_rung)` when the step escalated."""
    stall_clamp: bool = False
    late: bool = False
    """The request left the think block with an unacked budget update."""
    checked: bool = False
    score: float | None = None
    vector: dict[str, Any] = field(default_factory=dict)
    """Signal vector at the check (replay/debug log)."""


@dataclass
class EffortState:
    """Controller state for one request."""

    request_id: str
    ladder: list[int]
    theta: list[float]
    start_ids: list[int]
    end_ids: list[int]
    marker_seqs: list[tuple[int, ...]] = field(default_factory=list)
    bias: float = 0.0
    deadline_ms: float | None = None
    start_ms: float | None = None
    max_tokens: int = 0

    rung: int = 0
    cap: int = 0
    revision: int = 0
    acked_revision: int = 0
    pending_is_escalation: bool = False

    in_think: bool = False
    think_count: int = 0
    reasoning_tokens: int = 0
    rung_entry_think: int = 0
    last_escalation_think: int = 0
    checked_primary: bool = False
    checked_final: bool = False

    h_fast: float | None = None
    h_slow: float | None = None
    margin_ema: float | None = None
    samples: int = 0
    acc_ema: float | None = None
    acc_base: float | None = None
    acc_base_draft: int = 0
    acc_base_accepted: int = 0

    loop_flag: bool = False
    churn: bool = False
    stalled: bool = False
    late: bool = False
    escalations: int = 0
    finished: bool = False

    _tail: deque[int] = field(default_factory=deque, repr=False)
    _window: deque[int] = field(default_factory=deque, repr=False)
    _recent: deque[int] = field(default_factory=deque, repr=False)
    _ngram_counts: Counter[int] = field(default_factory=Counter, repr=False)
    _hash_counts: Counter[int] = field(default_factory=Counter, repr=False)
    _marker_positions: deque[int] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self.cap = self.ladder[0]
        tail_len = max(
            len(self.start_ids),
            len(self.end_ids),
            max((len(s) for s in self.marker_seqs), default=1),
        )
        self._tail = deque(maxlen=tail_len)

    @property
    def top_rung(self) -> int:
        return len(self.ladder) - 1

    @property
    def report(self) -> dict[str, int]:
        return {
            "rung": self.rung,
            "escalations": self.escalations,
            "reasoning_tokens": self.reasoning_tokens,
            "late": int(self.late),
            "stall_clamps": int(self.stalled),
        }


def _rfind(seq: Sequence[int], sub: Sequence[int]) -> int:
    """Index of the last occurrence of `sub` in `seq`, or -1."""
    if not sub or len(sub) > len(seq):
        return -1
    if len(sub) == 1:
        rev = list(seq)[::-1]
        try:
            return len(seq) - 1 - rev.index(sub[0])
        except ValueError:
            return -1
    for i in range(len(seq) - len(sub), -1, -1):
        if list(seq[i : i + len(sub)]) == list(sub):
            return i
    return -1


def new_effort_state(
    request_id: str,
    cfg: DynamicEffortConfig,
    overrides: dict[str, Any],
    start_ids: list[int],
    end_ids: list[int],
    marker_seqs: list[tuple[int, ...]],
    prompt_token_ids: Sequence[int] | None,
    max_tokens: int,
    now_ms: float | None = None,
) -> EffortState:
    """Build the state for a request; a prompt ending mid-think starts in it."""
    ladder = [int(x) for x in overrides.get("ladder", cfg.ladder)]
    theta = [float(x) for x in overrides.get("theta", cfg.theta or [])]
    state = EffortState(
        request_id=request_id,
        ladder=ladder,
        theta=theta,
        start_ids=start_ids,
        end_ids=end_ids,
        marker_seqs=marker_seqs,
        bias=float(overrides.get("bias", 0.0)),
        deadline_ms=overrides.get("deadline_ms"),
        start_ms=now_ms,
        max_tokens=max_tokens,
    )
    if prompt_token_ids:
        last_start = _rfind(prompt_token_ids, start_ids)
        last_end = _rfind(prompt_token_ids, end_ids)
        if last_start >= 0 and last_start > last_end:
            _enter_think(state)
            state.think_count = len(prompt_token_ids) - last_start - len(start_ids)
    return state


def _enter_think(state: EffortState) -> None:
    state.in_think = True
    state.think_count = 0
    state.rung_entry_think = 0
    state.last_escalation_think = 0
    state.checked_primary = False
    state.checked_final = False
    state.h_fast = state.h_slow = state.margin_ema = None
    state.samples = 0
    state.loop_flag = False
    state.churn = False
    state._window.clear()
    state._recent.clear()
    state._ngram_counts.clear()
    state._hash_counts.clear()
    state._marker_positions.clear()


def _leave_think(state: EffortState) -> None:
    state.in_think = False
    # The end sequence itself is not reasoning content.
    state.think_count = max(state.think_count - len(state.end_ids), 0)
    state.reasoning_tokens += state.think_count
    if state.revision > state.acked_revision and state.pending_is_escalation:
        state.late = True


def _tail_endswith(tail: deque[int], seq: Sequence[int]) -> bool:
    n = len(seq)
    if n == 0 or len(tail) < n:
        return False
    if n == 1:
        return tail[-1] == seq[0]
    return list(tail)[-n:] == list(seq)


def _push_think_token(state: EffortState, cfg: DynamicEffortConfig, tok: int) -> None:
    """Loop and marker bookkeeping for one committed think token."""
    window = state._window
    recent = state._recent
    if recent.maxlen != cfg.hash_window:
        state._recent = recent = deque(recent, maxlen=cfg.hash_window)
    if len(window) >= cfg.loop_window:
        old = hash(tuple(islice(window, 0, cfg.loop_ngram)))
        state._ngram_counts[old] -= 1
        if state._ngram_counts[old] <= 0:
            del state._ngram_counts[old]
        window.popleft()
    window.append(tok)
    recent.append(tok)
    if len(recent) >= cfg.loop_ngram:
        gram = hash(tuple(list(recent)[-cfg.loop_ngram :]))
        state._ngram_counts[gram] += 1
        if state._ngram_counts[gram] >= cfg.loop_repeats:
            state.loop_flag = True
    if len(recent) >= cfg.hash_window:
        h = hash(tuple(recent))
        state._hash_counts[h] += 1
        if state._hash_counts[h] >= cfg.loop_repeats:
            state.loop_flag = True
    for seq in state.marker_seqs:
        if _tail_endswith(state._tail, seq):
            state._marker_positions.append(state.think_count)
            break
    while (
        state._marker_positions
        and state._marker_positions[0] < state.think_count - cfg.marker_window
    ):
        state._marker_positions.popleft()


def _ingest_tokens(
    state: EffortState, cfg: DynamicEffortConfig, tokens: Sequence[int]
) -> None:
    for tok in tokens:
        state._tail.append(tok)
        if not state.in_think:
            if _tail_endswith(state._tail, state.start_ids):
                _enter_think(state)
            continue
        state.think_count += 1
        if _tail_endswith(state._tail, state.end_ids):
            _leave_think(state)
            continue
        _push_think_token(state, cfg, tok)


def _ema(prev: float | None, sample: float, alpha: float, n: int) -> float:
    if prev is None:
        return sample
    w = 1.0 - (1.0 - alpha) ** max(n, 1)
    return (1.0 - w) * prev + w * sample


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def _ingest_signals(
    state: EffortState, cfg: DynamicEffortConfig, ev: EffortEvent
) -> None:
    if state.in_think and ev.n_rows > 0 and _finite(ev.entropy) and _finite(ev.margin):
        assert ev.entropy is not None and ev.margin is not None
        state.h_fast = _ema(state.h_fast, ev.entropy, cfg.ema_fast_alpha, ev.n_rows)
        state.h_slow = _ema(state.h_slow, ev.entropy, cfg.ema_slow_alpha, ev.n_rows)
        state.margin_ema = _ema(
            state.margin_ema, ev.margin, cfg.ema_fast_alpha, ev.n_rows
        )
        state.samples += ev.n_rows
    if ev.num_draft_tokens > 0:
        acc = ev.num_accepted_tokens / ev.num_draft_tokens
        if state.acc_base is None:
            state.acc_base_draft += ev.num_draft_tokens
            state.acc_base_accepted += ev.num_accepted_tokens
            if state.acc_base_draft >= cfg.acc_baseline_tokens:
                state.acc_base = state.acc_base_accepted / state.acc_base_draft
        else:
            state.acc_ema = _ema(state.acc_ema, acc, cfg.ema_fast_alpha, 1)
    if state.in_think and state.h_fast is not None and state.h_slow is not None:
        rate = len(state._marker_positions) / cfg.marker_window
        state.churn = rate > cfg.marker_max_rate and state.h_fast >= state.h_slow


def effective_cap(state: EffortState, cfg: DynamicEffortConfig, rung: int) -> int:
    """Rung cap after the S8 headroom rule (`max_tokens - answer_reserve`)."""
    cap = state.ladder[rung]
    if state.max_tokens > 0:
        cap = min(cap, state.max_tokens - cfg.answer_reserve_tokens)
    return cap


def escalation_score(state: EffortState, cfg: DynamicEffortConfig) -> float | None:
    """The §4 score, or `None` when a required signal is missing."""
    if state.h_fast is None or state.h_slow is None or state.margin_ema is None:
        return None
    mean_h, std_h = cfg.calibration["entropy"]
    mean_m, std_m = cfg.calibration["margin"]
    z_h = (state.h_fast - mean_h) / std_h
    z_m = (state.margin_ema - mean_m) / std_m
    trend = 1.0 if state.h_fast >= state.h_slow else 0.0
    acc_term = 0.0
    if state.acc_base is not None and state.acc_ema is not None:
        acc_term = state.acc_base - state.acc_ema
    score = (
        cfg.w_h * z_h
        + cfg.w_m * (-z_m)
        + cfg.w_t * trend
        + cfg.w_a * acc_term
        + state.bias
    )
    return score if math.isfinite(score) else None


def _deadline_blocks(state: EffortState, ev: EffortEvent, next_cap: int) -> bool:
    if state.deadline_ms is None or ev.now_ms is None or state.start_ms is None:
        return False
    elapsed = ev.now_ms - state.start_ms
    if elapsed <= 0 or state.think_count <= 0:
        return False
    rate = state.think_count / elapsed  # tokens per ms
    projected = elapsed + (next_cap - state.think_count) / rate
    return projected > state.deadline_ms


def _try_escalate(
    state: EffortState,
    cfg: DynamicEffortConfig,
    ev: EffortEvent,
    decision: EffortDecision,
) -> None:
    if not state.in_think or state.stalled or state.rung >= state.top_rung:
        return
    check_pt = cfg.check_at * state.cap
    final_pt = cfg.final_check_at * state.cap
    if not state.checked_primary and state.think_count >= check_pt:
        state.checked_primary = True
    elif not state.checked_final and state.think_count >= final_pt:
        state.checked_final = True
    else:
        return
    decision.checked = True
    score = escalation_score(state, cfg)
    next_rung = state.rung + 1
    next_cap = effective_cap(state, cfg, next_rung)
    vector = {
        "rung": state.rung,
        "think": state.think_count,
        "cap": state.cap,
        "h_fast": state.h_fast,
        "h_slow": state.h_slow,
        "margin": state.margin_ema,
        "acc_base": state.acc_base,
        "acc_ema": state.acc_ema,
        "samples": state.samples,
        "loop": state.loop_flag,
        "churn": state.churn,
        "batch": ev.batch_size,
        "next_cap": next_cap,
        "score": score,
        "theta": state.theta[state.rung],
    }
    decision.vector = vector
    decision.score = score
    if score is None or state.samples < cfg.min_samples:
        return
    if state.loop_flag or state.churn:
        return
    if state.think_count - state.rung_entry_think < cfg.dwell_tokens:
        return
    if (
        state.escalations
        and state.think_count - state.last_escalation_think < cfg.cooldown_tokens
    ):
        return
    if next_rung > cfg.max_rung_for_batch_size(ev.batch_size):
        return
    if next_cap <= state.cap:
        return
    if _deadline_blocks(state, ev, next_cap):
        return
    if score < state.theta[state.rung]:
        return
    decision.escalation = (state.rung, next_rung)
    state.rung = next_rung
    state.cap = next_cap
    state.escalations += 1
    state.rung_entry_think = state.think_count
    state.last_escalation_think = state.think_count
    state.checked_primary = False
    state.checked_final = False
    state.revision += 1
    state.pending_is_escalation = True
    decision.budget_update = (state.revision, state.cap)


def step_effort(
    state: EffortState, cfg: DynamicEffortConfig, ev: EffortEvent
) -> EffortDecision:
    """Advance one request by one step and return what to do.

    Deterministic in `(state, cfg, ev)`; mutates `state` in place.
    """
    decision = EffortDecision()
    if ev.acked_revision is not None and ev.acked_revision > state.acked_revision:
        state.acked_revision = ev.acked_revision
    if ev.max_tokens:
        state.max_tokens = ev.max_tokens
    was_late = state.late
    _ingest_tokens(state, cfg, ev.new_token_ids)
    _ingest_signals(state, cfg, ev)
    if ev.repetition_evidence and state.in_think:
        state.loop_flag = True
    if state.late and not was_late:
        decision.late = True
    if state.in_think and state.loop_flag and not state.stalled:
        state.stalled = True
        state.cap = state.think_count + cfg.hard_stop_margin
        state.revision += 1
        state.pending_is_escalation = False
        decision.stall_clamp = True
        decision.budget_update = (state.revision, state.cap)
        return decision
    _try_escalate(state, cfg, ev, decision)
    return decision


def finish_effort(state: EffortState) -> dict[str, int]:
    """Close the state at request finish and return the report."""
    if state.in_think:
        # Finished mid-think (length cap / abort): count what was thought.
        state.reasoning_tokens += state.think_count
        state.in_think = False
        if state.revision > state.acked_revision and state.pending_is_escalation:
            state.late = True
    state.finished = True
    return state.report


def resolve_marker_sequences(
    markers: Sequence[str], encode: Any
) -> list[tuple[int, ...]]:
    """Token sequences for the backtrack markers (bare, space and newline
    prefixed variants); `encode(text) -> list[int]`."""
    seqs: set[tuple[int, ...]] = set()
    for marker in markers:
        for variant in (marker, " " + marker, "\n" + marker):
            ids = tuple(encode(variant))
            if ids:
                seqs.add(ids)
    return sorted(seqs, key=len)
