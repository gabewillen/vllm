# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ordinal escalation rule, shared by the scheduler and the worker.

The scheduler owns the quantile sketches; once per step it resolves them into
an :class:`EffortPolicy` (monotone quantile grids plus the per-rung
``p_uncertain`` thresholds) and ships it in ``SchedulerOutput``. Both
evaluation sites - the scheduler-side controller (V1 fallback) and the
worker-side tensors (V2) - turn a raw signal into a rank with the same grid
and apply the same predicate, so the decision does not depend on where it is
made or on which TP rank makes it.

Rule (docs/dynamic-reasoning.claude.md §P6):

    u = max(rank(H_fast), 1 - rank(margin))
    escalate = (
        u >= p_uncertain[rung]                    # globally uncertain
        and u - u_baseline >= baseline_rise       # and rising for *this* request
        and (H_fast >= H_slow or p_end not rising)  # not converging
        and no loop / churn
        and rank(MTP acceptance) <= acc_veto_rank   # corroboration only
    )
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

DEFAULT_P_UNCERTAIN = (0.85, 0.92, 0.96)


@dataclass
class EffortPolicy:
    """Step-global policy resolved from the scheduler's quantile sketches.

    Everything here is absolute: the worker never sees a sketch, only the
    monotone grids that turn a value into a rank and the ordinal thresholds
    the rank is compared against.
    """

    p_uncertain: list[float] = field(default_factory=lambda: list(DEFAULT_P_UNCERTAIN))
    """Uncertainty-rank threshold per rung transition."""
    entropy_edges: list[float] | None = None
    """Monotone quantile grid of the normalised-entropy sketch."""
    margin_edges: list[float] | None = None
    """Monotone quantile grid of the top1-top2 margin sketch."""
    acceptance_edges: list[float] | None = None
    """Monotone quantile grid of the MTP acceptance sketch."""
    max_rung: int = 0
    """Highest rung the current batch size allows (S11)."""
    check_at: float = 0.75
    final_check_at: float = 0.9
    baseline_tokens: int = 128
    """Think tokens that form the within-request baseline."""
    baseline_rise: float = 0.10
    """Rank rise over the request's own baseline required to escalate."""
    min_signal_rows: int = 64
    """Committed signal rows required before a request may escalate."""
    dwell_tokens: int = 128
    grace_tokens: int = 256
    """Extra think tokens granted once when p(end) is rising near the cap."""
    p_end_rise_eps: float = 0.0
    h_trend_eps: float = 1e-4
    """Entropy EMA gap below which the fast/slow trend counts as flat (the two
    EMAs of a constant signal differ only by float noise)."""
    acc_veto_rank: float = 0.85
    """Acceptance rank above which escalation is vetoed (text is predictable)."""
    warm: bool = False
    """False while the sketches are cold; no request may escalate."""

    def p_for_rung(self, rung: int) -> float:
        if not self.p_uncertain:
            return 1.0
        idx = min(max(rung, 0), len(self.p_uncertain) - 1)
        return self.p_uncertain[idx]


def rank_from_edges(edges: list[float] | None, value: float | None) -> float | None:
    """Percentile rank of ``value`` in a monotone quantile grid.

    Args:
        edges: ``n`` non-decreasing values, ``edges[i]`` at quantile
            ``i / (n - 1)``.
        value: the observation.

    Returns:
        The rank in ``[0, 1]``, or ``None`` when the grid is missing.
    """
    if not edges or value is None or len(edges) < 2:
        return None
    n = len(edges)
    if value <= edges[0]:
        return 0.0
    if value >= edges[-1]:
        return 1.0
    i = bisect.bisect_right(edges, value) - 1
    i = min(max(i, 0), n - 2)
    lo, hi = edges[i], edges[i + 1]
    frac = 0.0 if hi <= lo else (value - lo) / (hi - lo)
    return (i + frac) / (n - 1)


def uncertainty_rank(
    entropy_rank: float | None, margin_rank: float | None
) -> float | None:
    """Combine the two ranks into one ordinal uncertainty feature."""
    margin_uncertainty = None if margin_rank is None else 1.0 - margin_rank
    parts = [r for r in (entropy_rank, margin_uncertainty) if r is not None]
    return max(parts) if parts else None


def escalation_verdict(
    policy: EffortPolicy,
    rung: int,
    u_now: float | None,
    u_base: float | None,
    converging: bool,
    blocked: bool,
    acceptance_rank: float | None,
) -> bool:
    """The ordinal escalation predicate.

    Args:
        policy: resolved policy for this step.
        rung: the request's current rung.
        u_now: current uncertainty rank, or ``None`` when unavailable.
        u_base: within-request baseline uncertainty rank.
        converging: ``H_fast < H_slow`` and p(end) is rising.
        blocked: a loop / churn / dwell / headroom veto fired.
        acceptance_rank: rank of the request's MTP acceptance EMA.

    Returns:
        Whether the request should climb one rung.
    """
    if not policy.warm or blocked or converging:
        return False
    if u_now is None or u_base is None:
        return False
    if u_now < policy.p_for_rung(rung):
        return False
    if u_now - u_base < policy.baseline_rise:
        return False
    return not (acceptance_rank is not None and acceptance_rank > policy.acc_veto_rank)
