# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the soft-limit close.

The cap stops being a cliff: from `cap` onward the first token of the *natural*
reasoning end marker gets a bias rising to `max_bias` over `ramp_tokens`, and
only at `cap + ramp_tokens` does the hard force of the (possibly graceful) end
sequence take over. Covered here: the ramp arithmetic, both actuators writing
the same biases on the same drafts (V1 holder vs the V2 torch reference,
including a K=7 window), a scripted model that closes on its own inside the
ramp so nothing is ever forced, the hard force at the far end of the ramp, and
the untouched behaviour of a static `thinking_token_budget` when the soft limit
is off.
"""

import pytest
import torch
from test_v2_thinking_budget import (  # noqa: E402 - sibling test harness
    END,
    FORCE,
    START,
    V1Sim,
    V2Sim,
    _good_drafts,
    _no_pinned_memory,  # noqa: F401 - autouse fixture
    _Paired,
    _think_len,
    _verify,
)

from vllm.config.reasoning import DynamicEffortConfig, SoftLimitConfig
from vllm.v1.core.sched.effort_controller import (
    EffortEvent,
    new_effort_state,
    step_effort,
)
from vllm.v1.sample.effort_policy import EffortPolicy
from vllm.v1.sample.soft_limit import (
    SoftLimit,
    classify_close,
    soft_limit_bias,
    soft_limit_from_config,
    soft_limit_from_reasoning_config,
)

SOFT = SoftLimit(enabled=True, ramp_tokens=8, max_bias=10.0)


# ------------------------------------------------------------- ramp values


def test_ramp_is_zero_at_the_cap_and_max_one_ramp_later():
    assert soft_limit_bias(100, 100, 8, 10.0) == 0.0
    assert soft_limit_bias(104, 100, 8, 10.0) == pytest.approx(5.0)
    assert soft_limit_bias(107, 100, 8, 10.0) == pytest.approx(8.75)
    assert soft_limit_bias(108, 100, 8, 10.0) == pytest.approx(10.0)
    # Clamped on both sides: before the cap nothing, past the ramp no more.
    assert soft_limit_bias(99, 100, 8, 10.0) == 0.0
    assert soft_limit_bias(400, 100, 8, 10.0) == pytest.approx(10.0)


def test_ramp_curve_shapes_the_rise():
    linear = [soft_limit_bias(100 + i, 100, 8, 10.0, 1.0) for i in range(9)]
    late = [soft_limit_bias(100 + i, 100, 8, 10.0, 2.0) for i in range(9)]
    early = [soft_limit_bias(100 + i, 100, 8, 10.0, 0.5) for i in range(9)]
    assert linear[0] == late[0] == early[0] == 0.0
    assert linear[-1] == late[-1] == early[-1] == pytest.approx(10.0)
    # curve > 1 keeps the bias small until late; curve < 1 pushes it early.
    assert all(a >= b for a, b in zip(linear, late))
    assert all(a <= b for a, b in zip(linear, early))
    assert late[4] == pytest.approx(2.5) and early[4] == pytest.approx(10.0 * 0.5**0.5)


def test_zero_ramp_or_zero_bias_is_inactive():
    assert not SoftLimit(enabled=True, ramp_tokens=0).active
    assert not SoftLimit(enabled=True, max_bias=0.0).active
    assert not SoftLimit(enabled=False).active
    assert SoftLimit(enabled=True, ramp_tokens=0).ramp == 0
    assert soft_limit_bias(200, 100, 0, 10.0) == 0.0


@pytest.mark.parametrize(
    "think,cap,ramp,want",
    [
        (10, 100, 8, "natural"),
        (100, 100, 8, "natural"),  # the bias is exactly 0 at the cap
        (101, 100, 8, "soft"),
        (107, 100, 8, "soft"),
        (108, 100, 8, "forced"),
        (400, 100, 8, "forced"),
        (99, 100, 0, "natural"),  # soft limit off: the cap is the cliff again
        (100, 100, 0, "forced"),
    ],
)
def test_close_kind_classification(think, cap, ramp, want):
    assert classify_close(think, cap, ramp) == want


# ------------------------------------------------------------ config wiring


def test_soft_limit_defaults_and_config_resolution():
    cfg = DynamicEffortConfig()
    assert cfg.soft_limit.enabled
    assert cfg.soft_limit.ramp_tokens == 256
    assert cfg.soft_limit.max_bias == 10.0
    assert cfg.soft_limit.curve == 1.0
    resolved = soft_limit_from_config(cfg.soft_limit)
    assert resolved.active and resolved.ramp == 256
    assert resolved.bias(1024 + 128, 1024) == pytest.approx(5.0)


def test_soft_limit_is_absent_without_a_dynamic_effort_block():
    assert not soft_limit_from_reasoning_config(None).active
    assert not soft_limit_from_reasoning_config(
        type("R", (), {"dynamic_effort": None})()
    ).active


def test_soft_limit_can_be_disabled_per_server():
    cfg = DynamicEffortConfig(soft_limit={"enabled": False})
    assert not soft_limit_from_config(cfg.soft_limit).active
    assert soft_limit_from_config(cfg.soft_limit).ramp == 0


def test_soft_limit_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        SoftLimitConfig(max_bias=float("nan"))
    with pytest.raises(ValueError):
        SoftLimitConfig(ramp_tokens=-1)
    with pytest.raises(ValueError, match="must be finite"):
        SoftLimitConfig(curve=float("inf"))


# --------------------------------------------- the two actuators, in lockstep


def _bias_at(sim, budget, drafts, k):
    """Run one step of a request with `budget` and return `{pos: bias}`."""
    sim.forced([(0, drafts)])
    return dict(sim.last_bias[0])


def test_v1_and_v2_write_the_same_ramp_over_a_k7_draft_window():
    # 100 committed think tokens, cap 96: the window straddles the ramp.
    paired = _Paired(7, soft_limit=SOFT)
    paired.add(0, [], 96)
    paired.commit(0, START)
    paired.commit(0, list(range(200, 300)))  # 100 think tokens
    drafts = list(range(900, 907))
    forced = paired.forced([(0, drafts)])[0]
    bias = paired.last_bias[0]
    # Rows 0..3 sit at think 100..103, i.e. 4..7 tokens past the cap.
    assert set(bias) == {0, 1, 2, 3}
    for pos, over in zip(range(4), range(4, 8)):
        assert bias[pos] == pytest.approx(SOFT.bias(96 + over, 96))
    # Row 4 is at think 104 = cap + ramp: the hard force starts there.
    assert min(forced) == 4
    assert forced[4] == END[0]


def test_the_ramp_is_the_only_actuation_below_the_cap_plus_ramp():
    paired = _Paired(0, soft_limit=SOFT)
    paired.add(0, [], 96)
    paired.commit(0, START)
    paired.commit(0, list(range(200, 297)))  # 97 think tokens: 1 past the cap
    forced = paired.forced([(0, [])])[0]
    assert not forced
    assert paired.last_bias[0] == {0: pytest.approx(SOFT.bias(97, 96))}


def test_a_graceful_multi_token_close_is_still_forced_token_by_token():
    # Detection stays on the bare marker; the forced close is the transition.
    natural = [2]
    graceful = [7, 8, 2]
    paired = _Paired(0, end=graceful, natural_end=natural, soft_limit=SOFT)
    paired.add(0, [], 96)
    paired.commit(0, START)
    paired.commit(0, list(range(200, 300)))  # 100 think tokens = cap + 4
    assert not paired.forced([(0, [])])[0]
    assert paired.last_bias[0][0] == pytest.approx(SOFT.bias(100, 96))
    # And the bias lands on the *natural* marker, not on the transition's
    # first token, so a soft close reads like the model's own.
    paired.commit(0, list(range(300, 304)))  # cap + ramp
    forced = paired.forced([(0, [])])[0]
    assert forced == {0: graceful[0]}


# ------------------------------------------ a model that closes inside the ramp


def _decode(sim, budget, threshold, steps=64, base_logit=1.0):
    """Greedy decode where the model closes once the ramp beats `threshold`.

    The scripted target keeps one ordinary token at `base_logit`; the soft
    limit's bias is the only thing that can outvote it, which is exactly the
    mechanism under test.
    """
    sim.add(0, [], budget)
    sim.commit(0, START)
    forced_any = False
    for step in range(steps):
        forced = sim.forced([(0, [])])[0]
        bias = sim.last_bias[0].get(0, 0.0)
        if forced:
            forced_any = True
            token = forced[0]
        elif bias + base_logit > threshold + base_logit:
            token = END[0] if bias > threshold else 100 + step
        else:
            token = 100 + step
        sim.commit(0, [token])
        output = sim.output(0)
        if output[-len(END) :] == END:
            break
    return sim.output(0), forced_any


@pytest.mark.parametrize("sim_factory", [lambda: V1Sim(0, soft_limit=SOFT), V2Sim])
def test_a_natural_close_inside_the_ramp_forces_nothing(sim_factory):
    sim = sim_factory() if sim_factory is not V2Sim else V2Sim(soft_limit=SOFT)
    # threshold 5.0 -> the model gives in once the bias passes half the ramp.
    output, forced_any = _decode(sim, budget=20, threshold=5.0)
    assert not forced_any
    think = _think_len(output)
    assert think == 25  # cap 20 + the first row whose bias exceeds 5.0
    assert classify_close(think, 20, SOFT.ramp) == "soft"


@pytest.mark.parametrize("sim_factory", [lambda: V1Sim(0, soft_limit=SOFT), V2Sim])
def test_a_stubborn_model_is_still_forced_at_the_end_of_the_ramp(sim_factory):
    sim = sim_factory() if sim_factory is not V2Sim else V2Sim(soft_limit=SOFT)
    # A threshold above max_bias: the ramp never wins, so the force must.
    output, forced_any = _decode(sim, budget=20, threshold=99.0)
    assert forced_any
    think = _think_len(output)
    assert think == 20 + SOFT.ramp
    assert classify_close(think, 20, SOFT.ramp) == "forced"


def test_v1_v2_parity_on_a_full_k7_run_with_the_ramp():
    paired = _Paired(7, soft_limit=SOFT)
    paired.add(0, [], 40)
    paired.commit(0, START)
    drafts_fn = _good_drafts(7)
    for _ in range(24):
        n_out = len(paired.output(0))
        drafts = drafts_fn(n_out)
        forced = paired.forced([(0, drafts)])  # asserts V1 == V2 each step
        paired.commit(0, _verify(n_out - 1, drafts, forced[0]))
        if END[0] in paired.output(0):
            break
    output = paired.output(0)
    assert _think_len(output) == 40 + SOFT.ramp  # scripted model never closes


# ------------------------------------------- static budgets stay hard-capped


@pytest.mark.parametrize("holder", ["v1", "v2"])
def test_static_budget_is_untouched_when_the_soft_limit_is_off(holder):
    sim = V1Sim(0) if holder == "v1" else V2Sim()
    sim.add(0, [], 20)
    sim.commit(0, START)
    sim.commit(0, list(range(200, 220)))  # exactly the budget
    forced = sim.forced([(0, [])])[0]
    assert forced == {0: END[0]}  # forced at the cap, as before
    assert not sim.last_bias[0]  # and never biased


def test_static_budget_gets_the_ramp_when_the_server_enables_it():
    # Same request, same budget: only the server's soft_limit changed.
    sim = V1Sim(0, soft_limit=SOFT)
    sim.add(0, [], 20)
    sim.commit(0, START)
    sim.commit(0, list(range(200, 220)))
    assert not sim.forced([(0, [])])[0]
    sim.commit(0, list(range(300, 300 + SOFT.ramp)))
    assert sim.forced([(0, [])])[0] == {0: END[0]}


def test_disabled_soft_limit_leaves_the_v2_reference_bit_identical():
    off = SoftLimit(enabled=False, ramp_tokens=8, max_bias=10.0)
    plain, ramped = V2Sim(), V2Sim(soft_limit=off)
    for sim in (plain, ramped):
        sim.add(0, [], 20)
        sim.commit(0, START)
        sim.commit(0, list(range(200, 220)))
    assert plain.forced([(0, [])]) == ramped.forced([(0, [])])
    assert not ramped.last_bias[0]


# ----------------------------------------------- the controller's close_kind


def _cfg(**kw) -> DynamicEffortConfig:
    base = dict(
        ladder=[100, 400],
        p_uncertain=[0.85],
        min_samples=8,
        dwell_tokens=0,
        baseline_tokens=16,
        novelty_window=32,
        novelty_ngram=4,
    )
    base.update(kw)
    return DynamicEffortConfig(**base)


def _close_after(cfg, think_tokens):
    """Drive one request that emits `think_tokens` and then the end marker."""
    state = new_effort_state("r", cfg, {}, [START[0]], [END[0]], [], None, 100_000)
    policy = EffortPolicy(warm=False)
    stream = [START[0]] + list(range(1000, 1000 + think_tokens)) + [END[0]]
    for tok in stream:
        step_effort(
            state,
            cfg,
            EffortEvent(new_token_ids=[tok], n_rows=1, max_tokens=100_000),
            policy,
        )
    return state


def test_controller_reports_the_close_kind():
    cfg = _cfg(soft_limit={"ramp_tokens": 16, "max_bias": 10.0})
    assert _close_after(cfg, 40).report["close_kind"] == "natural"
    assert _close_after(cfg, 100).report["close_kind"] == "natural"
    assert _close_after(cfg, 108).report["close_kind"] == "soft"
    assert _close_after(cfg, 116).report["close_kind"] == "forced"


def test_controller_close_kind_without_a_soft_limit():
    cfg = _cfg(soft_limit={"enabled": False})
    assert _close_after(cfg, 40).report["close_kind"] == "natural"
    assert _close_after(cfg, 100).report["close_kind"] == "forced"


def test_soft_limit_ramp_reaches_the_actuator_as_the_state_ramp():
    cfg = _cfg(soft_limit={"ramp_tokens": 16})
    state = new_effort_state("r", cfg, {}, [START[0]], [END[0]], [], None, 100_000)
    assert state.soft_ramp == 16
    off = _cfg(soft_limit={"enabled": False})
    state = new_effort_state("r", off, {}, [START[0]], [END[0]], [], None, 100_000)
    assert state.soft_ramp == 0


def test_the_forced_close_is_never_reported_as_natural_under_a_stall_clamp():
    cfg = _cfg(soft_limit={"ramp_tokens": 16}, hard_stop_margin=4, loop_repeats=2)
    state = new_effort_state("r", cfg, {}, [START[0]], [END[0]], [], None, 100_000)
    policy = EffortPolicy(warm=False)
    for tok in [START[0]] + [7, 8, 9] * 40:
        step_effort(
            state,
            cfg,
            EffortEvent(new_token_ids=[tok], n_rows=1, max_tokens=100_000),
            policy,
        )
    assert state.stalled
    step_effort(
        state,
        cfg,
        EffortEvent(new_token_ids=[END[0]], n_rows=1, max_tokens=100_000),
        policy,
    )
    # The clamp lowered the cap to think + margin, so the close lands past it.
    assert state.report["close_kind"] in ("soft", "forced")


def test_force_constant_is_out_of_reach_of_the_ramp():
    # The ramp is a bias, not a mask: it must never look like a force.
    assert SOFT.bias(10_000, 0) < FORCE
    assert torch.tensor(SOFT.max_bias) < FORCE
