# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic reasoning-effort controller: pure per-request policy.

The scheduler owns one `EffortState` per dynamic request and feeds it one
`EffortEvent` per engine step through `step_effort`. Nothing here touches the
scheduler, so a recorded event stream replays the same decisions.

Signals (docs/dynamic-reasoning.claude.md §3, §P6): think position (S1), mean
normalised entropy, top1-top2 margin and p(reasoning end) of the committed
rows (S2/S3/P6) with fast/slow EMAs (S4), MTP acceptance (S6, corroboration
only), loop / n-gram-novelty evidence (S7), max_tokens headroom (S8),
batch-size rung cap (S11) and a client deadline (S12).

Decisions (§11, §12): a hard-stop clamp on a degenerate loop; a one-rung
escalation at the `check_at` / `final_check_at` points of the cap whenever the
rule in `vllm/v1/sample/effort_policy.py` fires; and, near the cap, a one-shot
grace window when p(reasoning end) is rising so a model that is already
wrapping up is not cut off mid-sentence. The grace window is superseded by the
soft-limit ramp (`vllm/v1/sample/soft_limit.py`), which grants the same room
unconditionally; the scheduler zeroes `grace_tokens` while it is active.

This module is the V1-runner path and the reference for the V2 worker-side
evaluation in `vllm/v1/worker/gpu/sample/effort_escalation.py`.
"""

import math
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

from vllm.config.reasoning import DynamicEffortConfig
from vllm.v1.sample.effort_policy import (
    EffortPolicy,
    escalation_verdict,
    rank_from_edges,
    uncertainty_rank,
)
from vllm.v1.sample.soft_limit import (
    CLOSE_NATURAL,
    classify_close,
    soft_limit_from_config,
)


@dataclass
class EffortEvent:
    """Per-step observations for one request (all CPU scalars)."""

    new_token_ids: Sequence[int]
    """Committed output tokens of this step, in commit order."""
    entropy: float | None = None
    """Mean normalised entropy over the committed rows; `None` = missing."""
    margin: float | None = None
    """Mean top1-top2 logit margin over the committed rows; `None` = missing."""
    p_end: float | None = None
    """Mean p(reasoning end) over the committed rows; `None` = missing."""
    n_rows: int = 0
    """Committed rows the means were taken over."""
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
    grace: bool = False
    """A p(end) grace window was granted near the cap."""
    late: bool = False
    """The request left the think block with an unacked budget update."""
    checked: bool = False
    score: float | None = None
    """Uncertainty rank (`rule="rank"`) or the weighted score (`"score"`)."""
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
    soft_ramp: int = 0
    """Soft-limit ramp in force for this request; 0 when it is off."""

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
    p_end_fast: float | None = None
    p_end_slow: float | None = None
    samples: int = 0
    acc_ema: float | None = None
    acc_base: float | None = None
    acc_base_draft: int = 0
    acc_base_accepted: int = 0

    base_h: float | None = None
    base_margin: float | None = None
    base_ready: bool = False
    """The within-request baseline (first `baseline_tokens` think tokens)."""

    grace_used: bool = False
    grace_granted: int = 0
    close_kind: str = CLOSE_NATURAL
    """How the last think block ended: natural / soft / forced."""

    loop_flag: bool = False
    churn: bool = False
    novelty_rate: float | None = None
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
    _novelty_recent: deque[int] = field(default_factory=deque, repr=False)
    _novelty_flags: deque[bool] = field(default_factory=deque, repr=False)
    _novelty_seen: set[int] = field(default_factory=set, repr=False)

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
    def report(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "escalations": self.escalations,
            "reasoning_tokens": self.reasoning_tokens,
            "late": int(self.late),
            "stall_clamps": int(self.stalled),
            "grace_tokens": self.grace_granted,
            "close_kind": self.close_kind,
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
        soft_ramp=soft_limit_from_config(cfg.soft_limit).ramp,
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
    state.p_end_fast = state.p_end_slow = None
    state.base_h = state.base_margin = None
    state.base_ready = False
    state.samples = 0
    state.loop_flag = False
    state.churn = False
    state.novelty_rate = None
    state._window.clear()
    state._recent.clear()
    state._ngram_counts.clear()
    state._hash_counts.clear()
    state._marker_positions.clear()
    state._novelty_recent.clear()
    state._novelty_flags.clear()
    state._novelty_seen.clear()


def _leave_think(state: EffortState) -> None:
    state.in_think = False
    # The end sequence itself is not reasoning content.
    state.think_count = max(state.think_count - len(state.end_ids), 0)
    state.close_kind = classify_close(state.think_count, state.cap, state.soft_ramp)
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


def _push_novelty(state: EffortState, cfg: DynamicEffortConfig, tok: int) -> None:
    """N-gram novelty rate over the last `novelty_window` think n-grams.

    Language-agnostic churn evidence: a window in which few n-grams are new is
    a model going in circles, whatever the language.
    """
    recent = state._novelty_recent
    if recent.maxlen != cfg.novelty_ngram:
        state._novelty_recent = recent = deque(recent, maxlen=cfg.novelty_ngram)
    flags = state._novelty_flags
    if flags.maxlen != cfg.novelty_window:
        state._novelty_flags = flags = deque(flags, maxlen=cfg.novelty_window)
    recent.append(tok)
    if len(recent) < cfg.novelty_ngram:
        return
    gram = hash(tuple(recent))
    is_new = gram not in state._novelty_seen
    if is_new:
        state._novelty_seen.add(gram)
    flags.append(is_new)
    if len(flags) >= cfg.novelty_window:
        state.novelty_rate = sum(flags) / len(flags)


def _push_think_token(state: EffortState, cfg: DynamicEffortConfig, tok: int) -> None:
    """Loop, novelty and marker bookkeeping for one committed think token."""
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
    _push_novelty(state, cfg, tok)
    if cfg.backtrack_marker_weight > 0.0:
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
        if _finite(ev.p_end):
            assert ev.p_end is not None
            state.p_end_fast = _ema(
                state.p_end_fast, ev.p_end, cfg.ema_fast_alpha, ev.n_rows
            )
            state.p_end_slow = _ema(
                state.p_end_slow, ev.p_end, cfg.ema_slow_alpha, ev.n_rows
            )
        state.samples += ev.n_rows
        if not state.base_ready and state.think_count >= cfg.baseline_tokens:
            state.base_h = state.h_fast
            state.base_margin = state.margin_ema
            state.base_ready = True
    if ev.num_draft_tokens > 0:
        acc = ev.num_accepted_tokens / ev.num_draft_tokens
        state.acc_ema = _ema(state.acc_ema, acc, cfg.ema_fast_alpha, 1)
        if state.acc_base is None:
            state.acc_base_draft += ev.num_draft_tokens
            state.acc_base_accepted += ev.num_accepted_tokens
            if state.acc_base_draft >= cfg.acc_baseline_tokens:
                state.acc_base = state.acc_base_accepted / state.acc_base_draft
    if state.in_think:
        state.churn = _churn(state, cfg)


def _churn(state: EffortState, cfg: DynamicEffortConfig) -> bool:
    """Language-agnostic churn: a low-novelty window, or (opt-in) markers."""
    if state.novelty_rate is not None and state.novelty_rate < cfg.novelty_min_rate:
        return True
    if cfg.backtrack_marker_weight > 0.0 and state.h_fast is not None:
        rate = (
            cfg.backtrack_marker_weight
            * len(state._marker_positions)
            / cfg.marker_window
        )
        if rate > cfg.marker_max_rate and (
            state.h_slow is None or state.h_fast >= state.h_slow
        ):
            return True
    return False


def effective_cap(state: EffortState, cfg: DynamicEffortConfig, rung: int) -> int:
    """Rung cap after the S8 headroom rule (`max_tokens - answer_reserve`)."""
    return cap_limit(state, cfg, state.ladder[rung])


def cap_limit(state: EffortState, cfg: DynamicEffortConfig, cap: int) -> int:
    """Clamp a candidate cap to the request's `max_tokens` headroom (S8).

    The actuator forces at `cap + soft_ramp`, so the ramp comes out of the
    headroom too - otherwise a request could ramp straight through the answer
    reserve it was given to write its answer in.
    """
    if state.max_tokens > 0:
        cap = min(cap, state.max_tokens - cfg.answer_reserve_tokens - state.soft_ramp)
    return cap


def escalation_score(state: EffortState, cfg: DynamicEffortConfig) -> float | None:
    """The pre-P6 weighted z-score (`rule="score"`), or `None` if unusable."""
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


def request_ranks(
    state: EffortState, policy: EffortPolicy
) -> tuple[float | None, float | None, float | None]:
    """`(u_now, u_baseline, acceptance rank)` for the ordinal rule."""
    h_rank = rank_from_edges(policy.entropy_edges, state.h_fast)
    m_rank = rank_from_edges(policy.margin_edges, state.margin_ema)
    u_now = uncertainty_rank(h_rank, m_rank)
    u_base = None
    if state.base_ready:
        u_base = uncertainty_rank(
            rank_from_edges(policy.entropy_edges, state.base_h),
            rank_from_edges(policy.margin_edges, state.base_margin),
        )
    acc_rank = rank_from_edges(policy.acceptance_edges, state.acc_ema)
    return u_now, u_base, acc_rank


def is_converging(state: EffortState, policy: EffortPolicy) -> bool:
    """The request is wrapping up, so more budget would not be used.

    p(end) rising is the whole test under the default length rule; the entropy
    trend is an uncertainty feature and only joins when those are active.
    """
    if not p_end_rising(state, policy):
        return False
    if not policy.use_uncertainty:
        return True
    if state.h_fast is None or state.h_slow is None:
        return False
    return state.h_fast < state.h_slow - policy.h_trend_eps


def p_end_rising(state: EffortState, policy: EffortPolicy) -> bool:
    """The model is wrapping up: the fast p(end) EMA leads the slow one."""
    if state.p_end_fast is None or state.p_end_slow is None:
        return False
    return state.p_end_fast > state.p_end_slow + policy.p_end_rise_eps


def _deadline_blocks(state: EffortState, ev: EffortEvent, next_cap: int) -> bool:
    if state.deadline_ms is None or ev.now_ms is None or state.start_ms is None:
        return False
    elapsed = ev.now_ms - state.start_ms
    if elapsed <= 0 or state.think_count <= 0:
        return False
    rate = state.think_count / elapsed  # tokens per ms
    projected = elapsed + (next_cap - state.think_count) / rate
    return projected > state.deadline_ms


def _blocked(
    state: EffortState, cfg: DynamicEffortConfig, ev: EffortEvent, next_cap: int
) -> bool:
    """Vetoes that are independent of the uncertainty rule itself."""
    if state.loop_flag or state.churn:
        return True
    if state.samples < cfg.min_samples:
        return True
    if state.think_count - state.rung_entry_think < cfg.dwell_tokens:
        return True
    if (
        state.escalations
        and state.think_count - state.last_escalation_think < cfg.cooldown_tokens
    ):
        return True
    if next_cap <= state.cap:
        return True
    return _deadline_blocks(state, ev, next_cap)


def _try_escalate(
    state: EffortState,
    cfg: DynamicEffortConfig,
    ev: EffortEvent,
    policy: EffortPolicy,
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
    next_rung = state.rung + 1
    next_cap = effective_cap(state, cfg, next_rung)
    blocked = _blocked(state, cfg, ev, next_cap)
    if next_rung > cfg.max_rung_for_batch_size(ev.batch_size):
        blocked = True
    if next_rung > policy.max_rung:
        blocked = True

    u_now, u_base, acc_rank = request_ranks(state, policy)
    converging = is_converging(state, policy)
    if cfg.rule == "score":
        score = escalation_score(state, cfg)
        fire = (
            score is not None
            and not blocked
            and score >= state.theta[min(state.rung, len(state.theta) - 1)]
        )
    else:
        score = u_now
        fire = escalation_verdict(
            policy, state.rung, u_now, u_base, converging, blocked, acc_rank
        )

    decision.score = score
    decision.vector = {
        "rung": state.rung,
        "think": state.think_count,
        "cap": state.cap,
        "rule": cfg.rule,
        "h_fast": state.h_fast,
        "h_slow": state.h_slow,
        "margin": state.margin_ema,
        "p_end": state.p_end_fast,
        "u_now": u_now,
        "u_base": u_base,
        "acc_rank": acc_rank,
        "p_uncertain": policy.p_for_rung(state.rung),
        "converging": converging,
        "use_uncertainty": policy.use_uncertainty,
        "samples": state.samples,
        "loop": state.loop_flag,
        "churn": state.churn,
        "novelty": state.novelty_rate,
        "batch": ev.batch_size,
        "next_cap": next_cap,
        "warm": policy.warm,
    }
    if not fire:
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


def _try_grace(
    state: EffortState,
    cfg: DynamicEffortConfig,
    policy: EffortPolicy,
    decision: EffortDecision,
) -> None:
    """Grant one grace window when p(end) is rising close to the cap."""
    if not state.in_think or state.stalled or state.grace_used:
        return
    if policy.grace_tokens <= 0 or decision.escalation is not None:
        return
    if state.think_count < cfg.final_check_at * state.cap:
        return
    if not p_end_rising(state, policy):
        return
    new_cap = cap_limit(state, cfg, state.cap + policy.grace_tokens)
    if new_cap <= state.cap:
        state.grace_used = True
        return
    state.grace_used = True
    state.grace_granted = new_cap - state.cap
    state.cap = new_cap
    state.revision += 1
    state.pending_is_escalation = False
    decision.grace = True
    decision.budget_update = (state.revision, state.cap)


def step_effort(
    state: EffortState,
    cfg: DynamicEffortConfig,
    ev: EffortEvent,
    policy: EffortPolicy | None = None,
) -> EffortDecision:
    """Advance one request by one step and return what to do.

    Args:
        state: the request's controller state (mutated in place).
        cfg: server-side dynamic-effort config.
        ev: this step's observations.
        policy: the step's resolved policy (quantile grids and thresholds).
            `None` means "cold": no escalation, only the stall clamp.

    Returns:
        The scheduler's actions for this step. Deterministic in
        `(state, cfg, ev, policy)`.
    """
    decision = EffortDecision()
    if policy is None:
        policy = EffortPolicy(warm=False, max_rung=cfg.top_rung)
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
        # The clamp is a hard stop, so it aims the *force point*, not the cap:
        # the actuator forces at `cap + soft_ramp`, so the cap goes one ramp
        # back and the bias is already saturated on the way there. A cap cannot
        # be negative, so the close lands at
        # `max(think + hard_stop_margin, soft_ramp)` - only a loop detected
        # inside the first `soft_ramp` think tokens sees the second term, and
        # the request's own repetition_detection stop (which this never
        # weakens) is still the terminal guarantee.
        state.cap = max(state.think_count + cfg.hard_stop_margin - state.soft_ramp, 0)
        state.revision += 1
        state.pending_is_escalation = False
        decision.stall_clamp = True
        decision.budget_update = (state.revision, state.cap)
        return decision
    _try_escalate(state, cfg, ev, policy, decision)
    _try_grace(state, cfg, policy, decision)
    return decision


def finish_effort(state: EffortState) -> dict[str, Any]:
    """Close the state at request finish and return the report."""
    if state.in_think:
        # Finished mid-think (length cap / abort): count what was thought.
        state.close_kind = classify_close(state.think_count, state.cap, state.soft_ramp)
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
