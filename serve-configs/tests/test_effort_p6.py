# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for P6 dynamic reasoning effort.

Covers the self-normalizing quantile sketches, the ordinal rank rule, the
language-agnostic novelty churn, the p(end) grace window, the worker-side
evaluation (checked against an independent Python model of the rule) and the
V1 fallback parity.
"""

import itertools
import json
import math

import numpy as np
import pytest
import torch

from vllm.config.reasoning import (
    QWEN_GRACEFUL_FORCE_END_STR,
    DynamicEffortConfig,
    ReasoningConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.effort_controller import (
    EffortEvent,
    new_effort_state,
    step_effort,
)
from vllm.v1.core.sched.effort_quantiles import SignalSketches, TDigest
from vllm.v1.sample.effort_policy import (
    EffortPolicy,
    escalation_verdict,
    rank_from_edges,
    uncertainty_rank,
)
from vllm.v1.worker.gpu.sample.effort_escalation import (
    EffortEscalationState,
    rank_from_edges_torch,
    reports_to_dict,
)
from vllm.v1.worker.gpu.sample.thinking_budget import (
    forced_end_tokens_torch,
    update_marker_cache_torch,
)

START, END = 1, 2
LADDER = [100, 400, 1600]
EDGES = [i / 32.0 for i in range(33)]
_UNIQUE = itertools.count(10_000)


def _tokens(n):
    """`n` never-repeating think tokens (no n-gram can loop)."""
    return [next(_UNIQUE) for _ in range(n)]


# --------------------------------------------------------------- t-digest


def test_tdigest_rank_and_quantile_are_monotone():
    digest = TDigest(compression=100.0)
    rng = np.random.default_rng(0)
    values = rng.normal(size=20_000)
    digest.add_many(values.tolist())
    assert digest.count == pytest.approx(20_000)
    ranks = [digest.rank(v) for v in (-3.0, -1.0, 0.0, 1.0, 3.0)]
    assert all(a <= b for a, b in zip(ranks, ranks[1:]))
    quantiles = [digest.quantile(q) for q in (0.01, 0.1, 0.5, 0.9, 0.99)]
    assert all(a <= b for a, b in zip(quantiles, quantiles[1:]))
    # The sketch is accurate enough to place the median and the quartiles.
    assert digest.quantile(0.5) == pytest.approx(0.0, abs=0.05)
    assert digest.quantile(0.25) == pytest.approx(-0.674, abs=0.08)
    assert digest.rank(0.0) == pytest.approx(0.5, abs=0.03)


def test_tdigest_edges_are_non_decreasing_and_bracket_the_data():
    digest = TDigest()
    digest.add_many([float(i) for i in range(1000)])
    edges = digest.edges(33)
    assert len(edges) == 33
    assert all(a <= b for a, b in zip(edges, edges[1:]))
    assert edges[0] == digest.min and edges[-1] == digest.max
    # rank_from_edges reproduces the digest's own ranks within grid resolution.
    for value in (10.0, 250.0, 500.0, 900.0):
        assert rank_from_edges(edges, value) == pytest.approx(
            digest.rank(value), abs=0.05
        )


def test_tdigest_json_roundtrip():
    digest = TDigest()
    digest.add_many([float(i) for i in range(500)])
    clone = TDigest.from_dict(json.loads(json.dumps(digest.to_dict())))
    assert clone.count == digest.count
    assert clone.quantile(0.5) == pytest.approx(digest.quantile(0.5))
    assert clone.rank(250.0) == pytest.approx(digest.rank(250.0))


def test_sketches_cold_then_warm_and_persist(tmp_path):
    path = tmp_path / "sketch.json"
    sketches = SignalSketches(min_samples=100, path=str(path), flush_every=50)
    assert sketches.rank("entropy", 0.5) is None  # cold: no escalation
    assert sketches.quantile("entropy", 0.9) is None
    assert sketches.edges("entropy", 33) is None
    for i in range(99):
        sketches.observe("entropy", i / 99.0)
    assert not sketches.warm("entropy")
    for i in range(99, 200):
        sketches.observe("entropy", i / 199.0)
    assert sketches.warm("entropy")
    assert 0.0 <= sketches.rank("entropy", 0.5) <= 1.0
    sketches.save()
    assert path.exists()

    warm = SignalSketches(min_samples=100, path=str(path))
    assert warm.load()
    assert warm.warm("entropy")
    assert warm.rank("entropy", 0.5) == pytest.approx(
        sketches.rank("entropy", 0.5), abs=1e-6
    )
    assert not warm.warm("margin")  # untouched signals stay cold


def test_sketches_maybe_save_respects_flush_every(tmp_path):
    path = tmp_path / "s.json"
    sketches = SignalSketches(min_samples=1, path=str(path), flush_every=10)
    for _ in range(9):
        sketches.observe("margin", 1.0)
    assert not sketches.maybe_save()
    sketches.observe("margin", 1.0)
    assert sketches.maybe_save()
    assert json.loads(path.read_text())["signals"]["margin"]["count"] == 10


# ------------------------------------------------------------ rank helpers


def test_rank_from_edges_matches_the_torch_twin():
    edges = [0.0, 0.1, 0.4, 0.4, 0.9, 2.0]
    values = [-1.0, 0.0, 0.05, 0.4, 0.65, 1.5, 2.0, 5.0]
    want = [rank_from_edges(edges, v) for v in values]
    got = rank_from_edges_torch(
        torch.tensor(edges), torch.tensor(values, dtype=torch.float32)
    )
    assert got.tolist() == pytest.approx(want, abs=1e-6)
    assert rank_from_edges_torch(None, torch.zeros(3)) is None
    assert rank_from_edges(None, 1.0) is None


def test_uncertainty_rank_takes_the_more_uncertain_half():
    assert uncertainty_rank(0.9, 0.9) == pytest.approx(0.9)  # entropy high
    assert uncertainty_rank(0.1, 0.05) == pytest.approx(0.95)  # margin low
    assert uncertainty_rank(None, None) is None


# --------------------------------------------------------------- rank rule


def _cfg(**kw) -> DynamicEffortConfig:
    base = dict(
        ladder=LADDER,
        p_uncertain=[0.75, 0.85],
        min_samples=8,
        dwell_tokens=0,
        baseline_tokens=16,
        novelty_window=32,
        novelty_ngram=4,
    )
    base.update(kw)
    return DynamicEffortConfig(**base)


def _policy(cfg: DynamicEffortConfig, **kw) -> EffortPolicy:
    base = dict(
        p_uncertain=list(cfg.p_uncertain),
        entropy_edges=list(EDGES),
        margin_edges=list(EDGES),
        max_rung=cfg.top_rung,
        check_at=cfg.check_at,
        final_check_at=cfg.final_check_at,
        baseline_tokens=cfg.baseline_tokens,
        baseline_rise=cfg.baseline_rise,
        min_signal_rows=cfg.min_samples,
        dwell_tokens=cfg.dwell_tokens,
        grace_tokens=cfg.grace_tokens,
        acc_veto_rank=cfg.acc_veto_rank,
        warm=True,
    )
    base.update(kw)
    return EffortPolicy(**base)


def _drive(cfg, policy, entropy_of, *, n=140, p_end_of=None, tokens=None, **kw):
    """Feed one think token per step; return (state, decisions)."""
    state = new_effort_state("r", cfg, {}, [START], [END], [], None, 100_000)
    stream = tokens if tokens is not None else [START] + _tokens(n)
    decisions = []
    for i, tok in enumerate(stream):
        ev = EffortEvent(
            new_token_ids=[tok],
            entropy=entropy_of(i),
            margin=0.5,
            p_end=0.001 if p_end_of is None else p_end_of(i),
            n_rows=1,
            max_tokens=100_000,
            **kw,
        )
        decisions.append(step_effort(state, cfg, ev, policy))
    return state, decisions


def test_rank_rule_escalates_on_a_relative_rise():
    cfg = _cfg()
    state, decisions = _drive(cfg, _policy(cfg), lambda i: 0.2 if i < 40 else 0.95)
    assert state.rung == 1 and state.escalations == 1
    fired = [d for d in decisions if d.escalation]
    assert fired and fired[0].escalation == (0, 1)
    assert fired[0].vector["u_now"] > fired[0].vector["u_base"]


def test_rank_rule_does_not_escalate_on_a_flat_high_absolute_level():
    cfg = _cfg()
    state, decisions = _drive(cfg, _policy(cfg), lambda i: 0.95)
    assert state.rung == 0 and state.escalations == 0
    checks = [d for d in decisions if d.checked]
    assert checks, "the check point must still fire"
    # Globally in the top rank, but flat against its own baseline.
    assert checks[0].vector["u_now"] >= 0.9
    assert checks[0].vector["u_now"] == pytest.approx(checks[0].vector["u_base"])


def test_rank_rule_is_cold_until_the_sketches_are_warm():
    cfg = _cfg()
    state, _ = _drive(cfg, _policy(cfg, warm=False), lambda i: 0.2 if i < 40 else 0.95)
    assert state.rung == 0 and state.escalations == 0
    state, _ = _drive(
        cfg,
        _policy(cfg, entropy_edges=None, margin_edges=None),
        lambda i: 0.2 if i < 40 else 0.95,
    )
    assert state.rung == 0


def test_rank_rule_low_margin_alone_is_enough_uncertainty():
    cfg = _cfg()
    policy = _policy(cfg)
    state = new_effort_state("r", cfg, {}, [START], [END], [], None, 100_000)
    for i, tok in enumerate([START] + _tokens(120)):
        margin = 0.9 if i < 40 else 0.02
        step_effort(
            state,
            cfg,
            EffortEvent(
                new_token_ids=[tok],
                entropy=0.1,
                margin=margin,
                p_end=0.001,
                n_rows=1,
                max_tokens=100_000,
            ),
            policy,
        )
    assert state.rung == 1


def test_mtp_acceptance_rank_vetoes_escalation():
    cfg = _cfg(acc_veto_rank=0.5)
    policy = _policy(cfg, acceptance_edges=[0.0, 0.5, 1.0])
    state = new_effort_state("r", cfg, {}, [START], [END], [], None, 100_000)
    for i, tok in enumerate([START] + _tokens(120)):
        step_effort(
            state,
            cfg,
            EffortEvent(
                new_token_ids=[tok],
                entropy=0.2 if i < 40 else 0.95,
                margin=0.5,
                p_end=0.001,
                n_rows=1,
                max_tokens=100_000,
                num_draft_tokens=4,
                num_accepted_tokens=4,  # the drafter predicts everything
            ),
            policy,
        )
    assert state.rung == 0
    assert state.acc_ema == pytest.approx(1.0)


# ------------------------------------------------------------ novelty churn


def test_novelty_churn_vetoes_escalation_without_any_marker_list():
    cfg = _cfg(novelty_min_rate=0.5, loop_repeats=1000)
    policy = _policy(cfg)
    # A 12-token phrase repeated forever: after the first pass every 4-gram is
    # old, so the novelty rate collapses. No language-specific marker needed.
    phrase = _tokens(12)
    stream = [START] + phrase * 14
    state, _ = _drive(cfg, policy, lambda i: 0.2 if i < 40 else 0.95, tokens=stream)
    assert state.novelty_rate is not None and state.novelty_rate < 0.5
    assert state.churn and state.rung == 0


def test_novel_text_keeps_a_high_novelty_rate():
    cfg = _cfg(novelty_min_rate=0.5)
    state, _ = _drive(cfg, _policy(cfg), lambda i: 0.2 if i < 40 else 0.95)
    assert state.novelty_rate == pytest.approx(1.0)
    assert not state.churn and state.rung == 1


def test_backtrack_markers_are_off_by_default_and_still_configurable():
    cfg = DynamicEffortConfig()
    assert cfg.backtrack_marker_weight == 0.0
    assert cfg.backtrack_markers  # the list survives for opt-in use
    marked = _cfg(backtrack_marker_weight=1.0, marker_window=64, marker_max_rate=0.05)
    state = new_effort_state("r", marked, {}, [START], [END], [(50,)], None, 100_000)
    for tok in [START] + [50 if i % 8 == 0 else 100 + i for i in range(100)]:
        step_effort(
            state,
            marked,
            EffortEvent(
                new_token_ids=[tok], entropy=0.9, margin=0.1, n_rows=1, p_end=0.0
            ),
            _policy(marked),
        )
    assert state.churn
    # The same stream with the P6 default weight is not churn.
    plain = _cfg(marker_window=64)
    state = new_effort_state("r", plain, {}, [START], [END], [(50,)], None, 100_000)
    for tok in [START] + [50 if i % 8 == 0 else 100 + i for i in range(100)]:
        step_effort(
            state,
            plain,
            EffortEvent(
                new_token_ids=[tok], entropy=0.9, margin=0.1, n_rows=1, p_end=0.0
            ),
            _policy(plain),
        )
    assert not state.churn


# ------------------------------------------------------------- p(end) grace


def test_p_end_grace_is_granted_when_the_model_is_wrapping_up():
    cfg = _cfg()
    policy = _policy(cfg, grace_tokens=256)
    state, decisions = _drive(
        cfg, policy, lambda i: 0.95, p_end_of=lambda i: 0.001 + i * 0.002
    )
    graced = [d for d in decisions if d.grace]
    assert len(graced) == 1
    assert state.grace_granted == 256 and state.cap == LADDER[0] + 256
    assert graced[0].budget_update == (1, LADDER[0] + 256)
    # One window only.
    assert state.grace_used


def test_p_end_flat_closes_at_the_cap_without_grace():
    cfg = _cfg()
    state, decisions = _drive(cfg, _policy(cfg), lambda i: 0.95, p_end_of=lambda i: 0.5)
    assert state.grace_granted == 0 and state.cap == LADDER[0]
    assert not any(d.grace for d in decisions)


def test_p_end_grace_is_skipped_when_the_step_escalated():
    cfg = _cfg()
    state, decisions = _drive(
        cfg,
        _policy(cfg),
        lambda i: 0.2 if i < 40 else 0.95,
        p_end_of=lambda i: 0.001 + i * 0.002,
    )
    assert state.rung == 1
    escalated = [d for d in decisions if d.escalation]
    assert escalated and not escalated[0].grace


def test_grace_is_disabled_by_zero_grace_tokens():
    cfg = _cfg(grace_tokens=0)
    state, _ = _drive(
        cfg,
        _policy(cfg, grace_tokens=0),
        lambda i: 0.95,
        p_end_of=lambda i: 0.001 + i * 0.002,
    )
    assert state.grace_granted == 0 and state.cap == LADDER[0]


# ------------------------------------------------- worker-side evaluation


def _worker(ladder=LADDER, cap_max=None, max_num_reqs=2):
    state = EffortEscalationState(max_num_reqs, torch.device("cpu"))
    overrides = {"ladder": list(ladder), "worker_eval": True}
    if cap_max is not None:
        overrides["cap_max"] = cap_max
    state.add_request(0, SamplingParams(extra_args={"dynamic_effort": overrides}))
    state.apply_staged_writes()
    return state


class RuleModel:
    """Independent Python model of the worker rule (no torch, no shared code)."""

    def __init__(self, ladder, policy, ema_fast=0.3, ema_slow=0.05):
        self.ladder = list(ladder)
        self.p = policy
        self.af, self.as_ = ema_fast, ema_slow
        self.rung = 0
        self.cap = self.ladder[0]
        self.h = self.hs = self.m = self.pf = self.ps = None
        self.samples = 0.0
        self.base = None
        self.entry = 0
        self.checked_primary = self.checked_final = False
        self.grace_used = False
        self.grace = 0
        self.escalations = 0

    @staticmethod
    def _ema(prev, x, a, n):
        if prev is None:
            return x
        w = 1.0 - (1.0 - a) ** max(n, 1)
        return (1.0 - w) * prev + w * x

    @staticmethod
    def _rank(edges, value):
        return rank_from_edges(edges, value)

    def step(self, think, entropy, margin, p_end, n_rows):
        if n_rows > 0:
            self.h = self._ema(self.h, entropy, self.af, n_rows)
            self.hs = self._ema(self.hs, entropy, self.as_, n_rows)
            self.m = self._ema(self.m, margin, self.af, n_rows)
            self.pf = self._ema(self.pf, p_end, self.af, n_rows)
            self.ps = self._ema(self.ps, p_end, self.as_, n_rows)
            self.samples += n_rows
        if self.base is None and think >= self.p.baseline_tokens:
            self.base = (self.h, self.m)
        fire_primary = not self.checked_primary and think >= self.p.check_at * self.cap
        fire_final = (
            not fire_primary
            and not self.checked_final
            and think >= self.p.final_check_at * self.cap
        )
        self.checked_primary |= fire_primary
        self.checked_final |= fire_final
        u_now = uncertainty_rank(
            self._rank(self.p.entropy_edges, self.h),
            self._rank(self.p.margin_edges, self.m),
        )
        u_base = None
        if self.base is not None:
            u_base = uncertainty_rank(
                self._rank(self.p.entropy_edges, self.base[0]),
                self._rank(self.p.margin_edges, self.base[1]),
            )
        rising = (
            self.pf is not None
            and self.ps is not None
            and self.pf > self.ps + self.p.p_end_rise_eps
        )
        converging = (
            self.h is not None
            and self.hs is not None
            and self.h < self.hs - self.p.h_trend_eps
            and rising
        )
        next_rung = self.rung + 1
        next_cap = self.ladder[min(next_rung, len(self.ladder) - 1)]
        blocked = (
            self.samples < self.p.min_signal_rows
            or (think - self.entry) < self.p.dwell_tokens
            or next_cap <= self.cap
            or self.rung >= len(self.ladder) - 1
            or next_rung > self.p.max_rung
            or self.base is None
        )
        fire = (fire_primary or fire_final) and escalation_verdict(
            self.p, self.rung, u_now, u_base, converging, blocked, None
        )
        if fire:
            self.rung = next_rung
            self.cap = next_cap
            self.escalations += 1
            self.entry = think
            self.checked_primary = self.checked_final = False
        if (
            not fire
            and not self.grace_used
            and rising
            and self.p.grace_tokens > 0
            and think >= self.p.final_check_at * self.cap
        ):
            self.grace_used = True
            self.grace += self.p.grace_tokens
            self.cap += self.p.grace_tokens
        return self.cap


def _run_worker(state, policy, n, entropy_of, margin_of, p_end_of, n_rows_of):
    """Drive the worker state over `n` committed think tokens."""
    idx = torch.tensor([0], dtype=torch.int32)
    idx_np = np.array([0])
    total_len = torch.zeros(state.max_num_reqs, dtype=torch.int32)
    last_start = torch.full((state.max_num_reqs,), -1, dtype=torch.int32)
    last_end = torch.full((state.max_num_reqs,), -1, dtype=torch.int32)
    last_start[0] = 0
    state.set_policy(policy, 0.3, 0.05)
    caps = []
    for t in range(1, n + 1):
        total_len[0] = t + 1
        state.last_signals[0] = torch.tensor(
            [entropy_of(t), margin_of(t), p_end_of(t), float(n_rows_of(t))]
        )
        state.begin()
        state.evaluate(idx, idx_np, total_len, last_start, last_end, 1)
        caps.append(int(state.cap[0]))
    return caps


@pytest.mark.parametrize("n_rows", [1, 7])
def test_worker_rule_matches_the_python_model(n_rows):
    cfg = _cfg()
    policy = _policy(cfg, grace_tokens=64)
    state = _worker()
    entropy_of = lambda t: 0.2 if t < 40 else 0.95
    margin_of = lambda t: 0.5
    p_end_of = lambda t: 0.001 + t * 0.0005
    caps = _run_worker(
        state, policy, 300, entropy_of, margin_of, p_end_of, lambda t: n_rows
    )
    model = RuleModel(LADDER, policy)
    want = [
        model.step(t, entropy_of(t), margin_of(t), p_end_of(t), n_rows)
        for t in range(1, 301)
    ]
    assert caps == want
    assert int(state.rung[0]) == model.rung
    assert int(state.escalations[0]) == model.escalations
    assert int(state.grace_granted[0]) == model.grace


def test_worker_rule_matches_the_python_model_on_a_flat_stream():
    cfg = _cfg()
    policy = _policy(cfg, grace_tokens=128)
    state = _worker()
    fns = (lambda t: 0.95, lambda t: 0.5, lambda t: 0.001 + t * 0.002, lambda t: 1)
    caps = _run_worker(state, policy, 200, *fns)
    model = RuleModel(LADDER, policy)
    want = [model.step(t, fns[0](t), fns[1](t), fns[2](t), 1) for t in range(1, 201)]
    assert caps == want
    assert int(state.grace_granted[0]) == 128


def test_worker_veto_and_stall_clamp_freeze_escalation():
    cfg = _cfg()
    policy = _policy(cfg)
    state = _worker()
    state.set_vetoes(["r"], {"r": 0})
    caps = _run_worker(
        state,
        policy,
        200,
        lambda t: 0.2 if t < 40 else 0.95,
        lambda t: 0.5,
        lambda t: 0.0,
        lambda t: 1,
    )
    assert set(caps) == {LADDER[0]}  # vetoed for the whole run

    state = _worker()
    state.absorb_budget_updates({"r": (1, 42)}, {"r": 0})
    caps = _run_worker(
        state,
        policy,
        200,
        lambda t: 0.2 if t < 40 else 0.95,
        lambda t: 0.5,
        lambda t: 0.0,
        lambda t: 1,
    )
    assert int(state.rung[0]) == 0
    budget = state.effective_budget(torch.full((2,), 999, dtype=torch.int32))
    assert int(budget[0]) == 42  # the stall clamp wins over the escalation cap
    assert int(budget[1]) == 999  # untouched slots keep the staged budget


def test_worker_respects_cap_max_headroom():
    policy = _policy(_cfg(), grace_tokens=1000)
    state = _worker(cap_max=140)
    _run_worker(
        state,
        policy,
        300,
        lambda t: 0.2 if t < 40 else 0.95,
        lambda t: 0.5,
        lambda t: 0.0,
        lambda t: 1,
    )
    assert int(state.cap[0]) == 140


def test_worker_reports_are_late_free_and_sparse():
    policy = _policy(_cfg())
    state = _worker()
    _run_worker(
        state,
        policy,
        200,
        lambda t: 0.2 if t < 40 else 0.95,
        lambda t: 0.5,
        lambda t: 0.0,
        lambda t: 1,
    )
    reports = state.reports(torch.tensor([0, 1], dtype=torch.int32))
    assert reports.shape == (2, 4)
    assert reports[0].tolist() == [1, 1, 0, 0]  # rung, escalations, grace, late
    as_dict = reports_to_dict(["a", "b"], reports.numpy())
    assert as_dict == {"a": (1, 1, 0, 0)}  # idle requests are omitted


# --------------------------------------- worker cap feeds the cap actuator


def _force_decisions(budget, committed, drafts, end_ids, start_ids=(START,)):
    """Forced (row, token) pairs for one request with a draft window."""
    max_len = max(len(start_ids), len(end_ids))
    all_tokens = torch.zeros((1, len(committed) + 8), dtype=torch.int32)
    all_tokens[0, : len(committed)] = torch.tensor(committed, dtype=torch.int32)
    total_len = torch.tensor([len(committed)], dtype=torch.int32)
    last_start = torch.tensor([-1], dtype=torch.int32)
    last_end = torch.tensor([-1], dtype=torch.int32)
    scan_pos = torch.tensor([0], dtype=torch.int32)
    start_t = torch.tensor(list(start_ids), dtype=torch.int32)
    end_t = torch.tensor(list(end_ids), dtype=torch.int32)
    budget_t = torch.tensor([budget], dtype=torch.int32)
    update_marker_cache_torch(
        torch.tensor([0]),
        budget_t,
        all_tokens,
        total_len,
        last_start,
        last_end,
        scan_pos,
        start_t,
        end_t,
    )
    num_rows = len(drafts) + 1
    input_ids = torch.tensor([committed[-1]] + list(drafts), dtype=torch.int32)
    rows, tokens = forced_end_tokens_torch(
        torch.zeros(num_rows, dtype=torch.int32),
        budget_t,
        all_tokens,
        total_len,
        input_ids,
        torch.arange(num_rows, dtype=torch.int32),
        last_start,
        last_end,
        start_t,
        end_t,
        end_t,
    )
    assert max_len >= 1
    return list(zip(rows, tokens))


def test_raised_cap_moves_the_force_point_with_k7_drafts_and_a_multi_token_end():
    end_ids = (7, 8, END)  # a three-token graceful close
    committed = [5, START] + list(range(100, 200))  # 100 think tokens
    drafts = list(range(900, 907))  # K = 7
    # At the rung-0 cap every row from the first over-budget position forces.
    at_cap = _force_decisions(100, committed, drafts, end_ids)
    assert at_cap and at_cap[0] == (0, 7)
    # The same state with the escalated cap forces nothing.
    assert _force_decisions(400, committed, drafts, end_ids) == []
    # And the raised cap is exactly what the worker rule produces.
    policy = _policy(_cfg(), grace_tokens=0)
    state = _worker()
    _run_worker(
        state,
        policy,
        100,
        lambda t: 0.2 if t < 40 else 0.95,
        lambda t: 0.5,
        lambda t: 0.0,
        lambda t: 1,
    )
    budget = state.effective_budget(torch.zeros(2, dtype=torch.int32))
    assert int(budget[0]) == 400
    assert _force_decisions(int(budget[0]), committed, drafts, end_ids) == []


def test_multi_token_end_sequence_is_forced_token_by_token():
    end_ids = (7, 8, END)
    committed = [5, START] + list(range(100, 200)) + [7]
    decisions = _force_decisions(100, committed, [], end_ids)
    assert decisions == [(0, 8)]  # the tail already holds the first end token
    committed2 = committed + [8]
    assert _force_decisions(100, committed2, [], end_ids) == [(0, END)]


# ------------------------------------------------------- V1/V2 parity


def test_scheduler_and_worker_agree_on_the_same_signal_stream():
    cfg = _cfg(grace_tokens=64)
    policy = _policy(cfg, grace_tokens=64)
    entropy_of = lambda t: 0.2 if t < 40 else 0.95  # noqa: E731
    p_end_of = lambda t: 0.001 + t * 0.0005  # noqa: E731

    worker = _worker()
    caps = _run_worker(
        worker, policy, 300, entropy_of, lambda t: 0.5, p_end_of, lambda t: 1
    )

    state = new_effort_state("r", cfg, {}, [START], [END], [], None, 100_000)
    sched_caps = []
    stream = [START] + _tokens(300)
    for i, tok in enumerate(stream):
        step_effort(
            state,
            cfg,
            EffortEvent(
                new_token_ids=[tok],
                entropy=entropy_of(i),
                margin=0.5,
                p_end=p_end_of(i),
                n_rows=1,
                max_tokens=100_000,
            ),
            policy,
        )
        if i:
            sched_caps.append(state.cap)
    assert state.rung == int(worker.rung[0])
    assert state.escalations == int(worker.escalations[0])
    assert state.grace_granted == int(worker.grace_granted[0])
    # The scheduler sees a step's signals one step later than the worker does
    # (the worker reads them from the previous step's reduction), so the caps
    # agree up to that single-step offset.
    assert sched_caps[-1] == caps[-1]
    assert sorted(set(sched_caps)) == sorted(set(caps))


# ------------------------------------------------------------- config


def test_p6_config_defaults():
    cfg = DynamicEffortConfig()
    assert cfg.rule == "length" and cfg.evaluation == "worker"
    assert cfg.ladder == [1024, 4096, 16384]
    assert cfg.p_uncertain == [0.85, 0.92]
    assert cfg.baseline_tokens == 128 and cfg.baseline_rise == 0.10
    assert cfg.grace_tokens == 256
    assert cfg.backtrack_marker_weight == 0.0
    assert cfg.graceful_force_end and cfg.force_end_str == QWEN_GRACEFUL_FORCE_END_STR
    assert cfg.quantile_path is None and cfg.quantile_min_samples == 2048


@pytest.mark.parametrize(
    "kw,match",
    [
        (dict(rule="magic"), "rule must be"),
        (dict(evaluation="gpu"), "evaluation must be"),
        (dict(p_uncertain=[0.5]), "one entry per ladder transition"),
        (dict(p_uncertain=[0.0, 0.5]), r"in \(0, 1\]"),
        (dict(p_uncertain=[0.5, 1.5]), r"in \(0, 1\]"),
        (dict(novelty_window=16, novelty_ngram=16), "novelty_window must exceed"),
        (dict(graceful_force_end=True, force_end_str=""), "non-empty force_end_str"),
    ],
)
def test_p6_config_rejects_bad_values(kw, match):
    with pytest.raises(ValueError, match=match):
        DynamicEffortConfig(**kw)


def test_p_uncertain_is_padded_to_longer_ladders():
    cfg = DynamicEffortConfig(ladder=[1, 2, 4, 8, 16])
    assert cfg.p_uncertain == [0.85, 0.92, 0.96, 0.96]


def test_force_end_str_splits_forcing_from_detection():
    class _Tok:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 97 for c in text]

    rc = ReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str="</think>",
        force_end_str="STOP</think>",
    )
    tok = _Tok()
    rc._reasoning_start_token_ids = tok.encode("<think>")
    rc._reasoning_end_token_ids = tok.encode("STOP</think>")
    rc._natural_reasoning_end_token_ids = tok.encode("</think>")
    assert rc.reasoning_end_token_ids != rc.natural_reasoning_end_token_ids
    assert (
        rc.reasoning_end_token_ids[-len(rc.natural_reasoning_end_token_ids) :]
        == rc.natural_reasoning_end_token_ids
    )


def test_graceful_force_end_string_ends_with_the_bare_marker():
    assert QWEN_GRACEFUL_FORCE_END_STR.strip().endswith("</think>")
    assert "limited time" in QWEN_GRACEFUL_FORCE_END_STR
    assert not math.isnan(len(QWEN_GRACEFUL_FORCE_END_STR))


# ------------------------------------------------------- calibration script


def _calibrate_module():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "effort_calibrate.py"
    spec = importlib.util.spec_from_file_location("effort_calibrate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_effort_calibrate_builds_a_warm_sketch_file(tmp_path):
    calibrate = _calibrate_module()
    sink = tmp_path / "telemetry.jsonl"
    with sink.open("w") as f:
        for i in range(400):
            f.write(
                json.dumps(
                    {
                        "req_id": f"r{i % 4}",
                        "step": i,
                        "num_output_tokens": i,
                        "entropy": (i % 100) / 100.0,
                        "margin": float(i % 17),
                        "p_end": (i % 50) / 100.0,
                        "n_rows": 2,
                        "num_draft_tokens": 4,
                        "num_accepted": i % 5,
                        "in_think": i % 5 != 0,
                    }
                )
                + "\n"
            )
    out = tmp_path / "sketch.json"
    counts = calibrate.build_sketches(
        [str(sink)], str(out), "test-model", min_samples=100, compression=100.0
    )
    assert counts["entropy"] == counts["margin"] == 640.0  # 320 in-think x n_rows
    assert counts["p_end"] == 640.0
    assert counts["acceptance"] == 320.0

    sketches = SignalSketches(min_samples=100, path=str(out))
    assert sketches.load()
    assert sketches.model == "test-model"
    assert all(sketches.warm(k) for k in ("entropy", "margin", "p_end"))
    assert 0.0 <= sketches.quantile("entropy", 0.85) <= 1.0
    assert "entropy" in calibrate.summarise(str(out))


def test_effort_calibrate_skips_rows_outside_the_think_block(tmp_path):
    calibrate = _calibrate_module()
    sink = tmp_path / "t.jsonl"
    sink.write_text(
        json.dumps({"entropy": 1.0, "margin": 1.0, "n_rows": 1, "in_think": False})
        + "\n"
        + json.dumps({"entropy": 2.0, "margin": 2.0, "n_rows": 1, "in_think": True})
        + "\n"
    )
    out = tmp_path / "s.json"
    counts = calibrate.build_sketches([str(sink)], str(out), None, 1, 100.0)
    assert counts["entropy"] == 1.0
    counts = calibrate.build_sketches(
        [str(sink)], str(out), None, 1, 100.0, include_all_rows=True
    )
    assert counts["entropy"] == 2.0


def test_worker_slot_reuse_resets_the_cap_and_state():
    policy = _policy(_cfg())
    state = _worker()
    _run_worker(
        state,
        policy,
        200,
        lambda t: 0.2 if t < 40 else 0.95,
        lambda t: 0.5,
        lambda t: 0.0,
        lambda t: 1,
    )
    assert int(state.cap[0]) == LADDER[1] and int(state.rung[0]) == 1
    # The slot is recycled by a request with a different ladder.
    state.add_request(
        0,
        SamplingParams(
            extra_args={"dynamic_effort": {"ladder": [64, 256], "worker_eval": True}}
        ),
    )
    state.apply_staged_writes()
    assert int(state.cap[0]) == 64
    assert int(state.rung[0]) == 0 and int(state.escalations[0]) == 0
    assert not bool(state.base_ready[0]) and float(state.h_fast[0]) == 0.0

    # And a slot recycled by a non-dynamic request is disarmed entirely.
    state.add_request(1, SamplingParams())
    state.apply_staged_writes()
    assert not state.enabled_np[1]
    budget = state.effective_budget(torch.full((2,), 777, dtype=torch.int32))
    assert int(budget[1]) == 777


# ------------------------------------------- P7: length rule + AUC evidence gate


def test_default_rule_is_length_based_and_needs_no_calibration():
    cfg = DynamicEffortConfig()
    assert cfg.rule == "length"
    assert cfg.uncertainty_min_auc == 0.60
    # No AUC in the file means no evidence, so the rank features stay out.
    active, reason = cfg.uncertainty_features(None)
    assert not active and "no discriminative AUC" in reason


@pytest.mark.parametrize(
    "rule,auc,active",
    [
        ("length", None, False),
        ("length", 0.41, False),
        ("length", 0.599, False),
        ("length", 0.60, True),
        ("length", 0.83, True),
        ("rank", None, True),
        ("rank", 0.41, True),
        ("score", None, True),
    ],
)
def test_the_rule_chooses_its_features_by_measured_auc(rule, auc, active):
    cfg = DynamicEffortConfig(rule=rule)
    got, reason = cfg.uncertainty_features(auc)
    assert got is active
    assert reason


def test_uncertainty_min_auc_is_configurable():
    cfg = DynamicEffortConfig(uncertainty_min_auc=0.75)
    assert not cfg.uncertainty_features(0.70)[0]
    assert cfg.uncertainty_features(0.80)[0]


def test_length_rule_escalates_without_any_uncertainty_evidence():
    """Flat, low uncertainty: the P6 rank rule refuses, the length rule fires."""
    cfg = _cfg()
    flat = lambda i: 0.02  # noqa: E731 - bottom of the entropy distribution
    ranked = _policy(cfg)
    state, _ = _drive(cfg, ranked, flat, p_end_of=lambda i: 0.001)
    assert state.rung == 0  # no rise over its own baseline

    length = _policy(cfg, use_uncertainty=False)
    state, decisions = _drive(cfg, length, flat, p_end_of=lambda i: 0.001)
    assert state.rung == 1 and state.escalations == 1
    fired = [d for d in decisions if d.escalation]
    assert fired[0].vector["use_uncertainty"] is False


def test_length_rule_needs_no_warm_sketches_at_all():
    cfg = _cfg()
    policy = _policy(cfg, use_uncertainty=False, entropy_edges=None, margin_edges=None)
    state, _ = _drive(cfg, policy, lambda i: 0.5, p_end_of=lambda i: 0.001)
    assert state.rung == 1


def test_length_rule_does_not_escalate_a_request_that_is_wrapping_up():
    cfg = _cfg()
    policy = _policy(cfg, use_uncertainty=False)
    # p(end) climbing = converging, whatever the entropy trend does.
    state, _ = _drive(cfg, policy, lambda i: 0.95, p_end_of=lambda i: 0.001 + i * 0.002)
    assert state.rung == 0 and state.escalations == 0


def test_length_rule_still_respects_loop_churn_and_the_mtp_veto():
    churny = _cfg(novelty_min_rate=0.5, loop_repeats=1000)
    policy = _policy(churny, use_uncertainty=False)
    phrase = _tokens(12)
    state, _ = _drive(
        churny,
        policy,
        lambda i: 0.5,
        p_end_of=lambda i: 0.001,
        tokens=[START] + phrase * 14,
    )
    assert state.churn and state.rung == 0

    vetoed = _cfg(acc_veto_rank=0.5)
    policy = _policy(vetoed, use_uncertainty=False, acceptance_edges=[0.0, 0.5, 1.0])
    state = new_effort_state("r", vetoed, {}, [START], [END], [], None, 100_000)
    for tok in [START] + _tokens(120):
        step_effort(
            state,
            vetoed,
            EffortEvent(
                new_token_ids=[tok],
                entropy=0.5,
                margin=0.5,
                p_end=0.001,
                n_rows=1,
                max_tokens=100_000,
                num_draft_tokens=4,
                num_accepted_tokens=4,
            ),
            policy,
        )
    assert state.rung == 0


def test_escalation_verdict_skips_the_rank_terms_only_when_gated():
    ranked = EffortPolicy(p_uncertain=[0.9], warm=True, baseline_rise=0.1)
    assert not escalation_verdict(ranked, 0, 0.2, 0.1, False, False, None)
    assert not escalation_verdict(ranked, 0, None, None, False, False, None)
    length = EffortPolicy(
        p_uncertain=[0.9], warm=True, baseline_rise=0.1, use_uncertainty=False
    )
    assert escalation_verdict(length, 0, 0.2, 0.1, False, False, None)
    assert escalation_verdict(length, 0, None, None, False, False, None)
    # The non-uncertainty vetoes still bind.
    assert not escalation_verdict(length, 0, None, None, True, False, None)
    assert not escalation_verdict(length, 0, None, None, False, True, None)
    assert not escalation_verdict(length, 0, None, None, False, False, 0.99)


def test_worker_matches_the_scheduler_under_the_length_rule():
    cfg = _cfg()
    policy = _policy(cfg, use_uncertainty=False, grace_tokens=0)
    worker = _worker()
    caps = _run_worker(
        worker,
        policy,
        300,
        lambda t: 0.02,
        lambda t: 0.9,
        lambda t: 0.001,
        lambda t: 1,
    )
    state = new_effort_state("r", cfg, {}, [START], [END], [], None, 100_000)
    sched_caps = []
    for i, tok in enumerate([START] + _tokens(300)):
        step_effort(
            state,
            cfg,
            EffortEvent(
                new_token_ids=[tok],
                entropy=0.02,
                margin=0.9,
                p_end=0.001,
                n_rows=1,
                max_tokens=100_000,
            ),
            policy,
        )
        if i:
            sched_caps.append(state.cap)
    assert int(worker.rung[0]) == state.rung == 2
    assert int(worker.escalations[0]) == state.escalations
    assert sched_caps[-1] == caps[-1] == LADDER[2]


def test_worker_length_rule_stops_at_a_rising_p_end():
    cfg = _cfg()
    policy = _policy(cfg, use_uncertainty=False, grace_tokens=0)
    worker = _worker()
    _run_worker(
        worker,
        policy,
        300,
        lambda t: 0.02,
        lambda t: 0.9,
        lambda t: 0.001 + t * 0.002,
        lambda t: 1,
    )
    assert int(worker.rung[0]) == 0


# ------------------------------------------------ AUC in the calibration file


def _auc_sink(tmp_path, *, positives=30, negatives=30, rising=True):
    """A telemetry sink whose long requests do (or do not) look uncertain."""
    sink = tmp_path / "auc.jsonl"
    with sink.open("w") as f:
        for r in range(negatives):
            for _ in range(40):  # 400 think tokens: closed under the 1024 cap
                f.write(
                    json.dumps(
                        {
                            "req_id": f"n{r}",
                            "entropy": 0.05,
                            "margin": 5.0,
                            "n_rows": 10,
                            "in_think": True,
                        }
                    )
                    + "\n"
                )
        for r in range(positives):
            for step in range(200):  # 2000 think tokens: escalated and finished
                entropy = 0.05 + step * 0.001 if rising else 0.05
                f.write(
                    json.dumps(
                        {
                            "req_id": f"p{r}",
                            "entropy": entropy,
                            "margin": 5.0,
                            "n_rows": 10,
                            "in_think": True,
                        }
                    )
                    + "\n"
                )
    return sink


def test_effort_calibrate_measures_and_stores_the_uncertainty_auc(tmp_path):
    calibrate = _calibrate_module()
    sink = _auc_sink(tmp_path)
    out = tmp_path / "sketch.json"
    calibrate.build_sketches([str(sink)], str(out), "m", 10, 100.0)

    sketches = SignalSketches(min_samples=10, path=str(out))
    assert sketches.load()
    assert sketches.uncertainty_auc is not None
    assert sketches.uncertainty_auc > 0.9  # entropy rises only on the long group
    assert sketches.auc["n_positive"] == 30 and sketches.auc["n_negative"] == 30
    assert set(sketches.auc["features"]) == {
        "entropy_first",
        "entropy_last",
        "entropy_rise",
        "margin_first",
        "margin_last",
        "margin_drop",
    }
    assert "uncertainty AUC" in calibrate.format_auc(sketches.auc)
    assert "uncertainty AUC" in calibrate.summarise(str(out))


def test_uncertainty_auc_is_at_chance_when_the_signal_is_flat(tmp_path):
    calibrate = _calibrate_module()
    out = tmp_path / "sketch.json"
    calibrate.build_sketches(
        [str(_auc_sink(tmp_path, rising=False))], str(out), "m", 10, 100.0
    )
    sketches = SignalSketches(min_samples=10, path=str(out))
    sketches.load()
    # Identical distributions: every directional AUC sits at 0.5, which is
    # below the 0.60 gate, so the features stay off.
    assert sketches.uncertainty_auc == pytest.approx(0.5, abs=0.01)
    assert not DynamicEffortConfig().uncertainty_features(sketches.uncertainty_auc)[0]


def test_uncertainty_auc_is_inconclusive_without_enough_requests(tmp_path):
    calibrate = _calibrate_module()
    out = tmp_path / "sketch.json"
    calibrate.build_sketches(
        [str(_auc_sink(tmp_path, positives=3, negatives=3))], str(out), "m", 10, 100.0
    )
    sketches = SignalSketches(min_samples=10, path=str(out))
    sketches.load()
    assert sketches.uncertainty_auc is None
    assert sketches.auc["inconclusive"]
    assert "inconclusive" in calibrate.format_auc(sketches.auc)


def test_requests_that_landed_on_a_higher_cap_are_not_positives(tmp_path):
    calibrate = _calibrate_module()
    sink = tmp_path / "cap.jsonl"
    with sink.open("w") as f:
        for _ in range(410):  # 4100 think tokens: a hair over the 4096 rung
            f.write(
                json.dumps(
                    {
                        "req_id": "c",
                        "entropy": 0.5,
                        "margin": 1.0,
                        "n_rows": 10,
                        "in_think": True,
                    }
                )
                + "\n"
            )
        for _ in range(409):  # 4090: a cap landing, so neither label
            f.write(
                json.dumps(
                    {
                        "req_id": "d",
                        "entropy": 0.5,
                        "margin": 1.0,
                        "n_rows": 10,
                        "in_think": True,
                    }
                )
                + "\n"
            )
    auc = calibrate.compute_uncertainty_auc([str(sink)], [1024, 4096, 16384])
    assert auc["n_positive"] == 1  # "c" only; "d" landed within the slack
    assert auc["n_negative"] == 0
    assert auc["uncertainty_auc"] is None


def test_sketch_file_without_an_auc_block_still_loads(tmp_path):
    path = tmp_path / "old.json"
    sketches = SignalSketches(min_samples=1, path=str(path))
    sketches.observe("entropy", 0.5)
    sketches.save()
    blob = json.loads(path.read_text())
    assert "auc" not in blob  # additive: nothing is written when nothing is known
    warm = SignalSketches(min_samples=1, path=str(path))
    assert warm.load()
    assert warm.uncertainty_auc is None


def test_the_server_preserves_a_loaded_auc_when_it_reflushes(tmp_path):
    path = tmp_path / "s.json"
    first = SignalSketches(min_samples=1, path=str(path))
    first.auc = {"uncertainty_auc": 0.71, "features": {}}
    first.observe("entropy", 0.5)
    first.save()
    live = SignalSketches(min_samples=1, path=str(path))
    live.load()
    live.observe("entropy", 0.7)
    live.save()
    reloaded = SignalSketches(min_samples=1, path=str(path))
    reloaded.load()
    assert reloaded.uncertainty_auc == pytest.approx(0.71)


# --------------------------------------- policy resolution in the scheduler


def _policy_scheduler(cfg, sketches, use_uncertainty):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.sample.soft_limit import soft_limit_from_config

    sched = Scheduler.__new__(Scheduler)
    sched._effort_cfg = cfg
    sched._effort_sketches = sketches
    sched._effort_policy = None
    sched._effort_policy_age = 0
    sched._effort_use_uncertainty = use_uncertainty
    sched._effort_soft_limit = soft_limit_from_config(cfg.soft_limit)
    return sched


def test_resolved_policy_is_warm_and_grace_free_under_the_length_rule():
    cfg = _cfg()
    cold = SignalSketches(min_samples=10**9)
    sched = _policy_scheduler(cfg, cold, use_uncertainty=False)
    policy = sched._effort_resolve_policy(1)
    assert policy.warm  # nothing to warm when the features are off
    assert not policy.use_uncertainty
    # The soft-limit ramp already grants room past the cap, so the p(end)
    # grace window is switched off rather than stacked on top of it.
    assert policy.grace_tokens == 0

    ranked = _policy_scheduler(cfg, cold, use_uncertainty=True)
    assert not ranked._effort_resolve_policy(1).warm


def test_grace_window_survives_when_the_soft_limit_is_disabled():
    cfg = _cfg(soft_limit={"enabled": False}, grace_tokens=128)
    sched = _policy_scheduler(cfg, SignalSketches(min_samples=1), False)
    assert sched._effort_resolve_policy(1).grace_tokens == 128
