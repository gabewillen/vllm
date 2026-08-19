# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side dynamic-effort escalation for Model Runner V2.

The pre-P6 controller decided in the scheduler and shipped the new cap in the
next ``SchedulerOutput``; the worker applied it one or two steps later, so a
request could hit the old cap first (``late``). Here the escalation rule is
evaluated in the same place the cap is applied - the sampler, right before
``ThinkingBudgetState`` forces the reasoning end sequence - so the decision and
its actuation are the same event and ``late`` is 0 by construction.

Every operation is elementwise over ``[max_num_reqs]`` staged/device tensors
and is deterministic in its inputs, so all TP ranks compute the same caps from
the same numbers - the same argument that already licenses the budget forcing.
The whole module is plain torch and runs unchanged on CPU, which makes it the
torch reference the CPU tests check against a Python model of the rule.

The scheduler still owns policy: it resolves its quantile sketches into an
:class:`~vllm.v1.sample.effort_policy.EffortPolicy` (monotone quantile grids +
per-rung ``p_uncertain``), keeps the loop / churn detector (a token-level
bookkeeping job) and ships per-step vetoes and stall clamps.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.effort_policy import EffortPolicy

MAX_RUNGS = 8
_INT32_MAX = int(np.iinfo(np.int32).max)


def rank_from_edges_torch(
    edges: torch.Tensor | None, values: torch.Tensor
) -> torch.Tensor | None:
    """Percentile rank of each value in a monotone quantile grid.

    The tensor twin of :func:`vllm.v1.sample.effort_policy.rank_from_edges`.

    Args:
        edges: ``[num_edges]`` non-decreasing grid, ``edges[i]`` at quantile
            ``i / (num_edges - 1)``.
        values: any shape.

    Returns:
        Ranks in ``[0, 1]`` with the shape of ``values``, or ``None`` when the
        grid is missing.
    """
    if edges is None or edges.numel() < 2:
        return None
    n = edges.numel()
    idx = torch.searchsorted(edges, values.contiguous(), right=True) - 1
    idx = idx.clamp(0, n - 2)
    lo = edges[idx]
    hi = edges[idx + 1]
    span = hi - lo
    frac = torch.where(span > 0, (values - lo) / span.clamp(min=1e-12), span * 0.0)
    return ((idx.to(values.dtype) + frac.clamp(0.0, 1.0)) / (n - 1)).clamp(0.0, 1.0)


class EffortEscalationState:
    """Per-slot escalation state and the elementwise rule over it."""

    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.enabled_np = np.zeros(max_num_reqs, dtype=bool)

        def _i32() -> torch.Tensor:
            return torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

        def _f32() -> torch.Tensor:
            return torch.zeros(max_num_reqs, dtype=torch.float32, device=device)

        # Staged (host-written) per-request configuration.
        self.ladder_np = np.zeros((max_num_reqs, MAX_RUNGS), dtype=np.int32)
        self.num_rungs_np = np.zeros(max_num_reqs, dtype=np.int32)
        self.cap_max_np = np.full(max_num_reqs, _INT32_MAX, dtype=np.int32)
        self.ladder = torch.zeros(
            (max_num_reqs, MAX_RUNGS), dtype=torch.int32, device=device
        )
        self.num_rungs = _i32()
        self.cap_max = torch.full(
            (max_num_reqs,), _INT32_MAX, dtype=torch.int32, device=device
        )
        self.enabled = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self._staged_dirty = False

        # Live per-request state.
        self.rung = _i32()
        self.cap = _i32()
        self.clamp = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=device)
        self.frozen = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.veto = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.checked_primary = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        self.checked_final = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.rung_entry_think = _i32()
        self.grace_used = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.grace_granted = _i32()
        self.escalations = _i32()
        self.samples = _f32()
        self.base_ready = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

        self.h_fast = _f32()
        self.h_slow = _f32()
        self.margin = _f32()
        self.p_end_fast = _f32()
        self.p_end_slow = _f32()
        self.base_h = _f32()
        self.base_margin = _f32()
        self.acc_ema = _f32()
        self.acc_valid = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.have_signal = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

        # Previous step's per-request signals, scattered by slot.
        self.last_signals = torch.zeros(
            (max_num_reqs, 4), dtype=torch.float32, device=device
        )
        self.last_acc = _f32()
        self.last_acc_valid = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

        self.policy: EffortPolicy | None = None
        self._entropy_edges: torch.Tensor | None = None
        self._margin_edges: torch.Tensor | None = None
        self._acceptance_edges: torch.Tensor | None = None
        self._p_uncertain: torch.Tensor | None = None
        self._ema_fast = 0.3
        self._ema_slow = 0.05
        self._reset_reqs: list[int] = []
        self._evaluated = False

    # -- request lifecycle -------------------------------------------------

    @property
    def any_enabled(self) -> bool:
        return bool(self.enabled_np.any())

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        """Arm (or disarm) a batch slot from the request's `dynamic_effort`."""
        overrides = None
        extra = sampling_params.extra_args
        if extra:
            candidate = extra.get("dynamic_effort")
            if isinstance(candidate, dict) and candidate.get("worker_eval"):
                overrides = candidate
        self.enabled_np[req_idx] = overrides is not None
        self.ladder_np[req_idx].fill(0)
        self.num_rungs_np[req_idx] = 0
        self.cap_max_np[req_idx] = _INT32_MAX
        self._reset_reqs.append(req_idx)
        self._staged_dirty = True
        if overrides is None:
            return
        ladder = [int(x) for x in overrides.get("ladder", ())][:MAX_RUNGS]
        if len(ladder) < 2:
            self.enabled_np[req_idx] = False
            return
        self.ladder_np[req_idx, : len(ladder)] = ladder
        self.num_rungs_np[req_idx] = len(ladder)
        cap_max = overrides.get("cap_max")
        if cap_max is not None and int(cap_max) > 0:
            self.cap_max_np[req_idx] = min(int(cap_max), _INT32_MAX)

    def apply_staged_writes(self) -> None:
        if not self._reset_reqs and not self._staged_dirty:
            return
        if self._reset_reqs:
            idx = torch.tensor(self._reset_reqs, dtype=torch.int64, device=self.device)
            for tensor in (
                self.rung,
                self.rung_entry_think,
                self.grace_granted,
                self.escalations,
            ):
                tensor.index_fill_(0, idx, 0)
            for tensor in (
                self.checked_primary,
                self.checked_final,
                self.grace_used,
                self.base_ready,
                self.frozen,
                self.veto,
                self.acc_valid,
                self.have_signal,
                self.last_acc_valid,
            ):
                tensor.index_fill_(0, idx, False)
            for tensor in (
                self.samples,
                self.h_fast,
                self.h_slow,
                self.margin,
                self.p_end_fast,
                self.p_end_slow,
                self.base_h,
                self.base_margin,
                self.acc_ema,
                self.last_acc,
            ):
                tensor.index_fill_(0, idx, 0.0)
            self.clamp.index_fill_(0, idx, -1)
            self.last_signals.index_fill_(0, idx, 0.0)
            self._reset_reqs.clear()
        if self._staged_dirty:
            self.ladder.copy_(torch.from_numpy(self.ladder_np))
            self.num_rungs.copy_(torch.from_numpy(self.num_rungs_np))
            self.cap_max.copy_(torch.from_numpy(self.cap_max_np))
            self.enabled.copy_(torch.from_numpy(self.enabled_np))
            # A freshly armed slot starts at rung 0's cap.
            first = torch.from_numpy(self.ladder_np[:, 0]).to(self.device)
            self.cap = torch.where(self.enabled & (self.cap == 0), first, self.cap)
            self._staged_dirty = False

    # -- scheduler -> worker -----------------------------------------------

    def set_policy(self, policy: EffortPolicy | None, ema_fast: float, ema_slow: float):
        """Install this step's resolved policy (grids + ordinal thresholds)."""
        self.policy = policy
        self._ema_fast = ema_fast
        self._ema_slow = ema_slow
        if policy is None:
            self._entropy_edges = self._margin_edges = None
            self._acceptance_edges = self._p_uncertain = None
            return
        self._entropy_edges = self._edges(policy.entropy_edges)
        self._margin_edges = self._edges(policy.margin_edges)
        self._acceptance_edges = self._edges(policy.acceptance_edges)
        self._p_uncertain = torch.tensor(
            policy.p_uncertain or [1.0], dtype=torch.float32, device=self.device
        )

    def _edges(self, values: Sequence[float] | None) -> torch.Tensor | None:
        if not values or len(values) < 2:
            return None
        return torch.tensor(list(values), dtype=torch.float32, device=self.device)

    def set_vetoes(
        self, req_ids: Iterable[str], req_id_to_index: dict[str, int]
    ) -> None:
        """One-step loop/churn veto for the named requests."""
        self.veto.zero_()
        idx = [
            req_id_to_index[r]
            for r in req_ids
            if r in req_id_to_index and self.enabled_np[req_id_to_index[r]]
        ]
        if idx:
            self.veto.index_fill_(
                0, torch.tensor(idx, dtype=torch.int64, device=self.device), True
            )

    def absorb_budget_updates(
        self,
        updates: dict[str, tuple[int, int]],
        req_id_to_index: dict[str, int],
    ) -> None:
        """A scheduler budget update on a worker-evaluated request is a stall
        clamp: it lowers the cap and freezes further escalation."""
        idx: list[int] = []
        caps: list[int] = []
        for req_id, (_, budget) in updates.items():
            req_idx = req_id_to_index.get(req_id)
            if req_idx is None or not self.enabled_np[req_idx]:
                continue
            idx.append(req_idx)
            caps.append(min(max(int(budget), 0), _INT32_MAX))
        if not idx:
            return
        index = torch.tensor(idx, dtype=torch.int64, device=self.device)
        values = torch.tensor(caps, dtype=torch.int32, device=self.device)
        self.clamp.index_copy_(0, index, values)
        self.frozen.index_fill_(0, index, True)

    # -- per-step ----------------------------------------------------------

    def begin(self) -> None:
        """Arm for one engine step (the rule is evaluated once per step)."""
        self._evaluated = False

    def evaluate(
        self,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        total_len: torch.Tensor,
        cached_last_start: torch.Tensor,
        cached_last_end: torch.Tensor,
        start_len: int,
    ) -> None:
        """Update the EMAs and apply the rule; writes ``cap`` in place.

        Args:
            idx_mapping: ``[num_reqs]`` batch position -> slot.
            idx_mapping_np: host copy of ``idx_mapping``.
            total_len: ``[max_num_reqs]`` committed token count per slot.
            cached_last_start: ``[max_num_reqs]`` last reasoning-start position.
            cached_last_end: ``[max_num_reqs]`` last natural reasoning end.
            start_len: length of the reasoning start sequence.
        """
        if self._evaluated or not np.any(self.enabled_np[idx_mapping_np]):
            return
        self._evaluated = True
        slots = idx_mapping.to(torch.int64)
        evaluate_escalation_torch(
            self,
            slots,
            total_len,
            cached_last_start,
            cached_last_end,
            start_len,
        )

    def effective_budget(self, staged_budget: torch.Tensor) -> torch.Tensor:
        """Budget the cap actuator must use this step."""
        if not self.any_enabled:
            return staged_budget
        cap = torch.where(
            self.clamp >= 0, torch.minimum(self.cap, self.clamp), self.cap
        )
        return torch.where(self.enabled, cap, staged_budget)

    def record_signals(
        self,
        effort_signals: torch.Tensor | None,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | None = None,
        num_rejected: torch.Tensor | None = None,
    ) -> None:
        """Stage this step's per-request signals for the next evaluation."""
        if not self.any_enabled:
            return
        slots = idx_mapping.to(torch.int64)
        if effort_signals is not None:
            self.last_signals.index_copy_(0, slots, effort_signals.float())
        else:
            self.last_signals.index_fill_(0, slots, 0.0)
        if num_sampled is None or num_rejected is None:
            self.last_acc_valid.index_fill_(0, slots, False)
            return
        accepted = (num_sampled.to(torch.float32) - 1.0).clamp(min=0.0)
        drafted = accepted + num_rejected.to(torch.float32)
        valid = drafted > 0
        acc = torch.where(valid, accepted / drafted.clamp(min=1.0), accepted * 0.0)
        self.last_acc.index_copy_(0, slots, acc)
        self.last_acc_valid.index_copy_(0, slots, valid)

    def reports(self, idx_mapping: torch.Tensor) -> torch.Tensor:
        """``[num_reqs, 4]`` int32 (rung, escalations, grace tokens, late=0)."""
        slots = idx_mapping.to(torch.int64)
        zeros = torch.zeros_like(self.rung)
        return torch.stack(
            (
                self.rung[slots],
                self.escalations[slots],
                self.grace_granted[slots],
                zeros[slots],
            ),
            dim=-1,
        )


def _ema(
    prev: torch.Tensor,
    sample: torch.Tensor,
    alpha: float,
    n: torch.Tensor,
    have: torch.Tensor,
) -> torch.Tensor:
    """Batched twin of the controller's `_ema` (n-sample catch-up weight)."""
    w = 1.0 - torch.pow(
        torch.tensor(1.0 - alpha, dtype=prev.dtype, device=prev.device),
        n.clamp(min=1.0),
    )
    merged = (1.0 - w) * prev + w * sample
    return torch.where(have, merged, sample)


def evaluate_escalation_torch(
    state: EffortEscalationState,
    slots: torch.Tensor,
    total_len: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    start_len: int,
) -> None:
    """The elementwise escalation rule (torch reference and implementation).

    Reads the previous step's signals from ``state.last_signals``, updates the
    per-request EMAs and baseline, then applies the ordinal rule of
    ``vllm/v1/sample/effort_policy.py`` plus the p(end) grace window, writing
    the new cap into ``state.cap``.
    """
    policy = state.policy
    device = state.cap.device
    enabled = state.enabled[slots]

    last = state.last_signals[slots]
    entropy, margin, p_end, n_rows = last[:, 0], last[:, 1], last[:, 2], last[:, 3]

    start = cached_last_start[slots].to(torch.int64)
    end = cached_last_end[slots].to(torch.int64)
    seq_len = total_len[slots].to(torch.int64)
    in_think = enabled & (start >= 0) & (start > end)
    think = (seq_len - (start + start_len)).clamp(min=0).to(torch.int32)

    # --- EMA update ---------------------------------------------------
    have = state.have_signal[slots]
    fresh = in_think & (n_rows > 0) & torch.isfinite(entropy) & torch.isfinite(margin)
    h_fast = _ema(state.h_fast[slots], entropy, state._ema_fast, n_rows, have)
    h_slow = _ema(state.h_slow[slots], entropy, state._ema_slow, n_rows, have)
    m_ema = _ema(state.margin[slots], margin, state._ema_fast, n_rows, have)
    pe_fast = _ema(state.p_end_fast[slots], p_end, state._ema_fast, n_rows, have)
    pe_slow = _ema(state.p_end_slow[slots], p_end, state._ema_slow, n_rows, have)
    samples = state.samples[slots] + torch.where(fresh, n_rows, n_rows * 0.0)

    state.h_fast[slots] = torch.where(fresh, h_fast, state.h_fast[slots])
    state.h_slow[slots] = torch.where(fresh, h_slow, state.h_slow[slots])
    state.margin[slots] = torch.where(fresh, m_ema, state.margin[slots])
    state.p_end_fast[slots] = torch.where(fresh, pe_fast, state.p_end_fast[slots])
    state.p_end_slow[slots] = torch.where(fresh, pe_slow, state.p_end_slow[slots])
    state.samples[slots] = samples
    state.have_signal[slots] = have | fresh

    acc_new = state.last_acc[slots]
    acc_ok = state.last_acc_valid[slots] & enabled
    acc_prev = state.acc_ema[slots]
    acc_have = state.acc_valid[slots]
    acc_ema = torch.where(
        acc_have,
        (1.0 - state._ema_fast) * acc_prev + state._ema_fast * acc_new,
        acc_new,
    )
    state.acc_ema[slots] = torch.where(acc_ok, acc_ema, acc_prev)
    state.acc_valid[slots] = acc_have | acc_ok

    h_fast = state.h_fast[slots]
    h_slow = state.h_slow[slots]
    m_ema = state.margin[slots]
    pe_fast = state.p_end_fast[slots]
    pe_slow = state.p_end_slow[slots]
    samples = state.samples[slots]

    if policy is None:
        return

    # --- within-request baseline --------------------------------------
    base_ready = state.base_ready[slots]
    take_base = in_think & ~base_ready & (think >= policy.baseline_tokens)
    state.base_h[slots] = torch.where(take_base, h_fast, state.base_h[slots])
    state.base_margin[slots] = torch.where(take_base, m_ema, state.base_margin[slots])
    state.base_ready[slots] = base_ready | take_base
    base_ready = state.base_ready[slots]
    base_h = state.base_h[slots]
    base_margin = state.base_margin[slots]

    # --- ranks ---------------------------------------------------------
    h_rank = rank_from_edges_torch(state._entropy_edges, h_fast)
    m_rank = rank_from_edges_torch(state._margin_edges, m_ema)
    bh_rank = rank_from_edges_torch(state._entropy_edges, base_h)
    bm_rank = rank_from_edges_torch(state._margin_edges, base_margin)
    zero = torch.zeros_like(h_fast)
    have_rank = torch.ones_like(h_fast, dtype=torch.bool)
    if h_rank is None and m_rank is None:
        have_rank = torch.zeros_like(h_fast, dtype=torch.bool)
    u_now = torch.maximum(
        h_rank if h_rank is not None else zero,
        (1.0 - m_rank) if m_rank is not None else zero,
    )
    u_base = torch.maximum(
        bh_rank if bh_rank is not None else zero,
        (1.0 - bm_rank) if bm_rank is not None else zero,
    )
    acc_rank = rank_from_edges_torch(state._acceptance_edges, state.acc_ema[slots])
    acc_veto = torch.zeros_like(u_now, dtype=torch.bool)
    if acc_rank is not None:
        acc_veto = state.acc_valid[slots] & (acc_rank > policy.acc_veto_rank)

    # --- check points ---------------------------------------------------
    cap = state.cap[slots]
    cap_f = cap.to(torch.float32)
    think_f = think.to(torch.float32)
    checked_primary = state.checked_primary[slots]
    checked_final = state.checked_final[slots]
    fire_primary = in_think & ~checked_primary & (think_f >= policy.check_at * cap_f)
    fire_final = (
        in_think
        & ~fire_primary
        & ~checked_final
        & (think_f >= policy.final_check_at * cap_f)
    )
    checked = fire_primary | fire_final
    state.checked_primary[slots] = checked_primary | fire_primary
    state.checked_final[slots] = checked_final | fire_final

    # --- rule ------------------------------------------------------------
    rung = state.rung[slots]
    num_rungs = state.num_rungs[slots]
    next_rung = (rung + 1).clamp(max=MAX_RUNGS - 1)
    ladder_next = state.ladder[slots].gather(1, next_rung.to(torch.int64)[:, None])
    next_cap = torch.minimum(ladder_next[:, 0], state.cap_max[slots])
    p_idx = next_rung.to(torch.int64) - 1
    if state._p_uncertain is not None:
        p_idx = p_idx.clamp(0, state._p_uncertain.numel() - 1)
        p_req = state._p_uncertain[p_idx]
    else:
        p_req = torch.ones_like(u_now)

    converging = (h_fast < h_slow - policy.h_trend_eps) & (
        pe_fast > pe_slow + policy.p_end_rise_eps
    )
    blocked = (
        state.veto[slots]
        | state.frozen[slots]
        | (samples < policy.min_signal_rows)
        | ((think - state.rung_entry_think[slots]) < policy.dwell_tokens)
        | (next_cap <= cap)
        | (rung >= num_rungs - 1)
        | (next_rung > policy.max_rung)
        | acc_veto
        | ~have_rank
        | ~base_ready
    )
    fire = (
        checked
        & in_think
        & torch.tensor(bool(policy.warm), device=device)
        & ~blocked
        & ~converging
        & (u_now >= p_req)
        & ((u_now - u_base) >= policy.baseline_rise)
    )

    new_cap = torch.where(fire, next_cap, cap)
    state.rung[slots] = torch.where(fire, next_rung, rung)
    state.escalations[slots] = state.escalations[slots] + fire.to(torch.int32)
    state.rung_entry_think[slots] = torch.where(
        fire, think, state.rung_entry_think[slots]
    )
    state.checked_primary[slots] = torch.where(
        fire, torch.zeros_like(fire), state.checked_primary[slots]
    )
    state.checked_final[slots] = torch.where(
        fire, torch.zeros_like(fire), state.checked_final[slots]
    )

    # --- p(end) grace window ---------------------------------------------
    grace_used = state.grace_used[slots]
    rising = pe_fast > pe_slow + policy.p_end_rise_eps
    want_grace = (
        in_think
        & ~fire
        & ~grace_used
        & ~state.frozen[slots]
        & rising
        & (think_f >= policy.final_check_at * new_cap.to(torch.float32))
        & (policy.grace_tokens > 0)
    )
    graced_cap = torch.minimum(new_cap + policy.grace_tokens, state.cap_max[slots])
    grants = want_grace & (graced_cap > new_cap)
    state.grace_used[slots] = grace_used | want_grace
    state.grace_granted[slots] = state.grace_granted[slots] + torch.where(
        grants, graced_cap - new_cap, torch.zeros_like(new_cap)
    )
    new_cap = torch.where(grants, graced_cap, new_cap)

    state.cap[slots] = torch.where(enabled & in_think, new_cap, cap)


def reports_to_dict(
    req_ids: Sequence[str], reports: np.ndarray
) -> dict[str, tuple[int, int, int, int]]:
    """Build ``ModelRunnerOutput.effort_reports`` from the host array.

    Args:
        req_ids: batch-ordered request ids.
        reports: ``[num_reqs, 4]`` (rung, escalations, grace tokens, late).

    Returns:
        req_id -> the four counters; requests that never escalated and were
        never graced are omitted so the dict stays empty on cold traffic.
    """
    out: dict[str, tuple[int, int, int, int]] = {}
    for i, req_id in enumerate(req_ids):
        if i >= reports.shape[0]:
            break
        row = reports[i]
        if not (row[0] or row[1] or row[2]):
            continue
        out[req_id] = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
    return out
