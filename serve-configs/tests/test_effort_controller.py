# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for dynamic reasoning effort (P2 controller + frontend).

The controller tests here pin the pre-P6 `rule="score"` path, which stays
selectable; the P6 rank rule, novelty churn, p(end) grace, quantile sketches
and worker-side evaluation are covered by `test_effort_p6.py`.
"""

import itertools
import math
from types import SimpleNamespace

import pytest

from vllm.config.reasoning import (
    QWEN_LOW_EFFORT_SENTENCE,
    DynamicEffortConfig,
    ReasoningConfig,
)
from vllm.entrypoints.openai.chat_completion.dynamic_effort import (
    DynamicEffortError,
    apply_dynamic_effort,
    build_dynamic_effort_overrides,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    EffortInfo,
)
from vllm.v1.core.sched.effort_controller import (
    EffortEvent,
    finish_effort,
    new_effort_state,
    resolve_marker_sequences,
    step_effort,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

START, END = 1, 2
LADDER = [100, 400, 1600]


def _cfg(**kw) -> DynamicEffortConfig:
    base = dict(
        rule="score",
        ladder=LADDER,
        theta=[0.0, 0.5],
        p_uncertain=[0.75, 0.85],
        min_samples=8,
        dwell_tokens=0,
        calibration={"entropy": (0.5, 0.2), "margin": (0.5, 0.2)},
    )
    base.update(kw)
    return DynamicEffortConfig(**base)


# --------------------------------------------------------------------- config


def test_config_defaults_and_theta():
    cfg = DynamicEffortConfig()
    assert cfg.ladder == [1024, 4096, 16384]
    assert cfg.theta == [0.0, 0.5]
    assert cfg.p_uncertain == [0.85, 0.92]
    assert cfg.rule == "rank" and cfg.evaluation == "worker"
    assert cfg.check_at == 0.75 and cfg.final_check_at == 0.9
    assert cfg.loop_ngram == 16 and cfg.loop_repeats == 3 and cfg.loop_window == 512
    assert cfg.floor_enabled is False
    assert cfg.low_effort_sentence == QWEN_LOW_EFFORT_SENTENCE
    assert cfg.top_rung == 2


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"ladder": [4096, 1024]}, "strictly increasing"),
        ({"ladder": [1024]}, "at least two"),
        ({"check_at": 0.9, "final_check_at": 0.8}, "final_check_at"),
        ({"theta": [1.0]}, "one entry per ladder transition"),
        ({"calibration": {"entropy": (0.0, 0.0), "margin": (0.0, 1.0)}}, "std > 0"),
        ({"calibration": {"entropy": (0.0, 1.0)}}, "missing 'margin'"),
        ({"floor_enabled": True}, "not implemented"),
        ({"max_rung_by_batch_size": [(1, 8, 9)]}, "outside"),
        ({"max_rung_by_batch_size": [(1, 8, 2), (4, 16, 1)]}, "non-overlapping"),
        ({"hash_window": 8}, "hash_window"),
    ],
)
def test_config_rejects_bad_values(kw, match):
    with pytest.raises(ValueError, match=match):
        DynamicEffortConfig(**kw)


def test_config_via_reasoning_config_dict_and_batch_cap():
    rc = ReasoningConfig(
        dynamic_effort={
            "ladder": [1024, 4096, 16384, 65536],
            "max_rung_by_batch_size": [[1, 8, 3], [9, 32, 2], [33, 128, 1]],
        }
    )
    cfg = rc.dynamic_effort
    assert cfg is not None
    assert cfg.max_rung_for_batch_size(1) == 3
    assert cfg.max_rung_for_batch_size(20) == 2
    assert cfg.max_rung_for_batch_size(128) == 1
    assert cfg.max_rung_for_batch_size(129) == 3  # outside every range
    assert ReasoningConfig().dynamic_effort is None


# ------------------------------------------------------------------- frontend


def _req(**kw) -> ChatCompletionRequest:
    body = dict(model="m", messages=[{"role": "user", "content": "hi"}])
    body.update(kw)
    return ChatCompletionRequest(**body)


def test_frontend_dynamic_string_content():
    cfg = DynamicEffortConfig()
    req = _req(
        reasoning_effort="dynamic",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "last question"},
        ],
    )
    apply_dynamic_effort(req, cfg)
    assert req.reasoning_effort == "medium"
    assert req.messages[1]["content"] == "first"
    assert req.messages[3]["content"] == "last question\n\n" + QWEN_LOW_EFFORT_SENTENCE
    assert req.thinking_token_budget == 1024
    params = req.to_sampling_params(4096, {})
    assert params.thinking_token_budget == 1024
    assert params.extra_args["effort_telemetry"] is True
    assert params.extra_args["dynamic_effort"] == {
        "ladder": [1024, 4096, 16384],
        "theta": [0.0, 0.5],
        "bias": 0.0,
        "deadline_ms": None,
    }
    chat = req.build_chat_params(None, "string")
    assert chat.chat_template_kwargs["reasoning_effort"] == "medium"
    assert chat.chat_template_kwargs["enable_thinking"] is True


def test_frontend_dynamic_list_content_and_tool_tail():
    cfg = DynamicEffortConfig()
    req = _req(
        reasoning_effort="dynamic",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "look"}],
            },
            {"role": "assistant", "content": "calling"},
            {"role": "tool", "content": "result", "tool_call_id": "x"},
        ],
    )
    apply_dynamic_effort(req, cfg)
    parts = req.messages[0]["content"]
    assert parts[-1] == {"type": "text", "text": QWEN_LOW_EFFORT_SENTENCE}
    assert len(parts) == 2
    assert req.messages[2]["content"] == "result"


def test_frontend_overrides_and_telemetry_opt_out():
    cfg = DynamicEffortConfig()
    req = _req(
        reasoning_effort="dynamic",
        vllm_xargs={
            "dynamic_effort_ladder": [512, 2048],
            "dynamic_effort_theta": [0.25],
            "effort_bias": 1.5,
            "deadline_ms": 30000,
            "effort_telemetry": 0,
        },
    )
    apply_dynamic_effort(req, cfg)
    params = req.to_sampling_params(4096, {})
    assert params.thinking_token_budget == 512
    assert params.extra_args["dynamic_effort"] == {
        "ladder": [512, 2048],
        "theta": [0.25],
        "bias": 1.5,
        "deadline_ms": 30000.0,
    }
    assert params.extra_args["effort_telemetry"] == 0


def test_frontend_ladder_override_without_theta_gets_default_theta():
    o = build_dynamic_effort_overrides(
        DynamicEffortConfig(), {"dynamic_effort_ladder": [100, 200, 300]}
    )
    assert o["theta"] == [0.0, 0.5]


@pytest.mark.parametrize(
    "kw,cfg,match",
    [
        ({"thinking_token_budget": 500}, DynamicEffortConfig(), "static"),
        (
            {"chat_template_kwargs": {"enable_thinking": False}},
            DynamicEffortConfig(),
            "enable_thinking",
        ),
        ({}, None, "not enabled"),
        (
            {"vllm_xargs": {"dynamic_effort_ladder": [200, 100]}},
            DynamicEffortConfig(),
            "strictly increasing",
        ),
        (
            {"vllm_xargs": {"dynamic_effort_theta": [1, 2, 3, 4, 5]}},
            DynamicEffortConfig(),
            "one entry per",
        ),
        ({"vllm_xargs": {"deadline_ms": -1}}, DynamicEffortConfig(), "positive"),
        ({"vllm_xargs": {"dynamic_effort_floor": 1}}, DynamicEffortConfig(), "floor"),
        (
            {"messages": [{"role": "system", "content": "only"}]},
            DynamicEffortConfig(),
            "user message",
        ),
    ],
)
def test_frontend_conflicts_are_rejected(kw, cfg, match):
    req = _req(reasoning_effort="dynamic", **kw)
    with pytest.raises(DynamicEffortError, match=match):
        apply_dynamic_effort(req, cfg)


@pytest.mark.parametrize("effort", [None, "low", "xhigh"])
def test_frontend_non_dynamic_requests_unchanged(effort):
    req = _req(reasoning_effort=effort, vllm_xargs={"k": 1})
    before = req.model_dump()
    apply_dynamic_effort(req, DynamicEffortConfig())
    assert req.model_dump() == before
    params = req.to_sampling_params(4096, {})
    assert params.thinking_token_budget is None
    assert params.extra_args == {"k": 1}
    plain = _req().to_sampling_params(4096, {})
    assert plain.extra_args is None


def test_effort_info_from_report():
    assert EffortInfo.from_report(None) is None
    info = EffortInfo.from_report(
        {"rung": 2, "escalations": 2, "reasoning_tokens": 3000, "late": 1}
    )
    assert info is not None
    assert (info.rung, info.escalations, info.reasoning_tokens, info.late) == (
        2,
        2,
        3000,
        True,
    )


# ------------------------------------------------------------- pure policy


def _state(cfg, prompt=None, max_tokens=100_000, **overrides):
    return new_effort_state(
        "r0",
        cfg,
        overrides,
        [START],
        [END],
        [],
        prompt,
        max_tokens,
        now_ms=0.0,
    )


def _run(state, cfg, tokens, entropy, margin, *, batch=1, max_tokens=None, **kw):
    """Feed tokens one per step with a constant signal; return decisions."""
    decisions = []
    for tok in tokens:
        ev = EffortEvent(
            new_token_ids=[tok],
            entropy=entropy,
            margin=margin,
            n_rows=1 if entropy is not None else 0,
            batch_size=batch,
            max_tokens=max_tokens or state.max_tokens,
            **kw,
        )
        decisions.append(step_effort(state, cfg, ev))
    return decisions


_UNIQUE = itertools.count(10_000)


def _think_tokens(n):
    """`n` never-repeating think tokens (no n-gram can loop)."""
    return [next(_UNIQUE) for _ in range(n)]


def test_prompt_ending_mid_think_starts_in_think():
    cfg = _cfg()
    st = _state(cfg, prompt=[5, 6, START, 7, 8, 9])
    assert st.in_think and st.think_count == 3
    st2 = _state(cfg, prompt=[START, 7, END, 8])
    assert not st2.in_think


def test_escalates_at_check_point_when_uncertain():
    cfg = _cfg()
    st = _state(cfg)
    decisions = _run(st, cfg, [START] + _think_tokens(80), entropy=0.9, margin=0.1)
    updates = [(i, d) for i, d in enumerate(decisions) if d.budget_update]
    assert len(updates) == 1
    i, d = updates[0]
    assert i == 75  # START + 75 think tokens = 0.75 * 100
    assert d.escalation == (0, 1)
    assert d.budget_update == (1, 400)
    assert d.score is not None and d.score >= 0.0
    assert d.vector["think"] == 75
    assert st.rung == 1 and st.cap == 400 and st.escalations == 1
    # The final check of rung 0 is disarmed; the next check is at 300.
    more = _run(st, cfg, _think_tokens(220), entropy=0.9, margin=0.1)  # 80 -> 300
    assert all(d.budget_update is None for d in more[:-1])
    assert more[-1].budget_update == (2, 1600) and more[-1].escalation == (1, 2)
    assert st.rung == 2
    # Top rung: no further checks/escalations.
    tail = _run(st, cfg, _think_tokens(1500), entropy=0.9, margin=0.1)
    assert all(d.budget_update is None for d in tail)


def test_does_not_escalate_when_confident():
    cfg = _cfg()
    st = _state(cfg)
    decisions = _run(st, cfg, [START] + _think_tokens(100), entropy=0.1, margin=0.9)
    checked = [d for d in decisions if d.checked]
    assert len(checked) == 2  # 75 % and 90 % checks fired
    assert all(d.budget_update is None for d in decisions)
    assert all(d.score is not None and d.score < 0.0 for d in checked)
    assert st.rung == 0 and st.escalations == 0


def test_replay_is_deterministic():
    cfg = _cfg()
    tokens = [START] + _think_tokens(120)
    a = _run(_state(cfg), cfg, tokens, entropy=0.9, margin=0.1)
    b = _run(_state(cfg), cfg, tokens, entropy=0.9, margin=0.1)
    assert [(d.budget_update, d.escalation, d.score) for d in a] == [
        (d.budget_update, d.escalation, d.score) for d in b
    ]


@pytest.mark.parametrize("entropy,margin", [(None, None), (float("nan"), 0.1)])
def test_missing_or_nan_samples_never_escalate(entropy, margin):
    cfg = _cfg()
    st = _state(cfg)
    decisions = _run(st, cfg, [START] + _think_tokens(100), entropy, margin)
    assert st.samples == 0
    assert all(d.budget_update is None for d in decisions)
    assert any(d.checked for d in decisions)


def test_min_samples_gate():
    cfg = _cfg(min_samples=1000)
    st = _state(cfg)
    decisions = _run(st, cfg, [START] + _think_tokens(100), entropy=0.9, margin=0.1)
    assert all(d.budget_update is None for d in decisions)


@pytest.mark.parametrize("ramp", [0, 8, 256])
def test_hard_stop_on_repeated_ngram(ramp):
    cfg = _cfg(soft_limit={"enabled": bool(ramp), "ramp_tokens": ramp or 1})
    st = _state(cfg)
    gram = list(range(200, 216))  # 16 tokens
    decisions = _run(st, cfg, [START] + gram * 3, entropy=0.5, margin=0.5)
    stalls = [d for d in decisions if d.stall_clamp]
    assert len(stalls) == 1
    assert st.stalled and st.loop_flag
    # The clamp aims the actuator's force point (cap + ramp), so it stays
    # hard_stop_margin tokens from the third repeat at 48 - except when the
    # ramp itself is longer than that, which the cap floor at 0 bounds.
    revision, cap = stalls[0].budget_update
    assert revision == 1
    assert cap + st.soft_ramp == max(48 + 32, st.soft_ramp)
    # No escalation after a stall, even with an uncertain signal.
    more = _run(st, cfg, _think_tokens(100), entropy=0.9, margin=0.1)
    assert all(d.budget_update is None for d in more)
    assert st.report["stall_clamps"] == 1


def test_hard_stop_on_repeated_hash_window_and_repetition_evidence():
    cfg = _cfg()
    st = _state(cfg)
    # 32-token window seen loop_repeats times, separated by unique filler so
    # the 16-gram counter over the sliding window does not fire first.
    block = list(range(300, 332))
    seq = [START]
    for k in range(3):
        seq += block + [1000 + 40 * k + j for j in range(40)]
    _run(st, cfg, seq, entropy=0.5, margin=0.5)
    assert st.loop_flag and st.stalled
    st2 = _state(cfg)
    _run(st2, cfg, [START] + _think_tokens(10), entropy=0.5, margin=0.5)
    d = step_effort(
        st2,
        cfg,
        EffortEvent(new_token_ids=[111], repetition_evidence=True, max_tokens=1000),
    )
    assert d.stall_clamp
    assert d.budget_update[0] == 1
    assert d.budget_update[1] + st2.soft_ramp == max(
        st2.think_count + 32, st2.soft_ramp
    )


def test_respects_max_rung_by_batch_size():
    cfg = _cfg(max_rung_by_batch_size=[(1, 8, 2), (9, 128, 0)])
    st = _state(cfg)
    decisions = _run(
        st, cfg, [START] + _think_tokens(100), entropy=0.9, margin=0.1, batch=40
    )
    assert all(d.budget_update is None for d in decisions)
    st = _state(cfg)
    decisions = _run(
        st, cfg, [START] + _think_tokens(100), entropy=0.9, margin=0.1, batch=4
    )
    assert any(d.budget_update for d in decisions)


@pytest.mark.parametrize("ramp", [0, 8])
def test_respects_max_tokens_headroom(ramp):
    # The ramp is thinking too, so it comes out of the same headroom: the
    # forced close (cap + ramp) must still leave the answer reserve free.
    cfg = _cfg(
        answer_reserve_tokens=256,
        soft_limit={"enabled": bool(ramp), "ramp_tokens": ramp or 1},
    )
    st = _state(cfg, max_tokens=300)  # 300 - 256 = 44 < cap 100: no room
    decisions = _run(st, cfg, [START] + _think_tokens(100), entropy=0.9, margin=0.1)
    assert all(d.budget_update is None for d in decisions)
    st = _state(cfg, max_tokens=500)  # next cap = min(400, 244 - ramp)
    decisions = _run(st, cfg, [START] + _think_tokens(80), entropy=0.9, margin=0.1)
    updates = [d.budget_update for d in decisions if d.budget_update]
    assert updates == [(1, 244 - st.soft_ramp)]
    assert updates[0][1] + st.soft_ramp == 500 - 256


def test_deadline_blocks_escalation():
    cfg = _cfg()
    st = _state(cfg, deadline_ms=1000.0)
    tokens = [START] + _think_tokens(100)
    decisions = []
    for i, tok in enumerate(tokens):
        ev = EffortEvent(
            new_token_ids=[tok],
            entropy=0.9,
            margin=0.1,
            n_rows=1,
            max_tokens=100_000,
            now_ms=10.0 * (i + 1),  # 0.1 tok/ms: 400 tokens need 4000 ms
        )
        decisions.append(step_effort(st, cfg, ev))
    assert all(d.budget_update is None for d in decisions)


def test_mtp_acceptance_is_corroboration_only():
    # High MTP acceptance (boilerplate) pulls the score below theta; the
    # entropy/margin signal alone would sit right at the threshold.
    cfg = _cfg(w_a=4.0, theta=[0.0, 0.5], acc_baseline_tokens=8)
    st = _state(cfg)
    tokens = [START] + _think_tokens(100)
    for tok in tokens:
        ev = EffortEvent(
            new_token_ids=[tok],
            entropy=0.5,
            margin=0.5,
            n_rows=1,
            max_tokens=100_000,
            num_draft_tokens=4,
            num_accepted_tokens=4 if st.acc_base is not None else 1,
        )
        d = step_effort(st, cfg, ev)
        assert d.budget_update is None
    assert st.acc_base == pytest.approx(0.25) and st.acc_ema is not None
    # Without spec decode the term is 0 and trend alone (w_t = 0.5) escalates.
    st = _state(cfg)
    decisions = _run(st, cfg, tokens, entropy=0.5, margin=0.5)
    assert any(d.budget_update for d in decisions)


def test_revision_monotone_across_escalation_and_stall():
    cfg = _cfg()
    st = _state(cfg)
    decisions = _run(st, cfg, [START] + _think_tokens(80), entropy=0.9, margin=0.1)
    gram = list(range(200, 216))
    decisions += _run(st, cfg, gram * 3, entropy=0.9, margin=0.1)
    revs = [d.budget_update[0] for d in decisions if d.budget_update]
    assert revs == [1, 2]
    assert st.cap < 400  # the stall clamp lowered the rung-1 cap


def test_late_detection_and_ack():
    cfg = _cfg()
    st = _state(cfg)
    _run(st, cfg, [START] + _think_tokens(76), entropy=0.9, margin=0.1)
    assert st.revision == 1 and st.acked_revision == 0
    d = step_effort(st, cfg, EffortEvent(new_token_ids=[END], max_tokens=1000))
    assert d.late and st.late and not st.in_think
    assert st.reasoning_tokens == 76
    assert finish_effort(st) == {
        "rung": 1,
        "escalations": 1,
        "reasoning_tokens": 76,
        "late": 1,
        "stall_clamps": 0,
        "grace_tokens": 0,
        "close_kind": "natural",
    }
    # Acked before the close: not late.
    st = _state(cfg)
    _run(st, cfg, [START] + _think_tokens(76), entropy=0.9, margin=0.1)
    step_effort(
        st, cfg, EffortEvent(new_token_ids=[300], acked_revision=1, max_tokens=1000)
    )
    d = step_effort(st, cfg, EffortEvent(new_token_ids=[END], max_tokens=1000))
    assert not d.late and not st.late and st.acked_revision == 1
    # Finished mid-think with a pending update is late too.
    st = _state(cfg)
    _run(st, cfg, [START] + _think_tokens(76), entropy=0.9, margin=0.1)
    rep = finish_effort(st)
    assert rep["late"] == 1 and rep["reasoning_tokens"] == 76


def test_multi_step_signal_uses_row_count():
    cfg = _cfg()
    st = _state(cfg)
    step_effort(st, cfg, EffortEvent(new_token_ids=[START], max_tokens=1000))
    step_effort(
        st,
        cfg,
        EffortEvent(new_token_ids=[7, 8, 9, 10], entropy=1.0, margin=0.0, n_rows=4),
    )
    assert st.samples == 4 and st.h_fast == 1.0
    step_effort(
        st,
        cfg,
        EffortEvent(new_token_ids=[7, 8, 9, 10], entropy=0.0, margin=0.0, n_rows=4),
    )
    # w = 1 - (1 - 0.3)^4
    assert st.h_fast == pytest.approx((1 - 0.3) ** 4)


def test_marker_sequences_and_churn_veto():
    seqs = resolve_marker_sequences(["Wait", "Hmm"], lambda t: [ord(c) for c in t])
    assert (ord("W"), ord("a"), ord("i"), ord("t")) in seqs
    assert (ord(" "), ord("H"), ord("m"), ord("m")) in seqs
    cfg = _cfg(marker_window=64, marker_max_rate=0.05, backtrack_marker_weight=1.0)
    st = new_effort_state("r", cfg, {}, [START], [END], [(50,)], None, 100_000)
    # A marker every 8 tokens (rate 0.125) with a flat-high entropy = churn.
    tokens = [START] + [50 if i % 8 == 0 else 100 + i for i in range(100)]
    decisions = _run(st, cfg, tokens, entropy=0.9, margin=0.1)
    assert st.churn
    assert all(d.budget_update is None for d in decisions)


# ---------------------------------------------------- data-plane contracts


def test_frozen_interface_fields_default_empty():
    so = SchedulerOutput.make_empty()
    assert so.thinking_budget_updates == {}
    mro = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    assert mro.effort_signals is None and mro.thinking_budget_acks is None


def test_holder_update_budget_revisions():
    rc = SimpleNamespace(
        reasoning_start_token_ids=[START], reasoning_end_token_ids=[END]
    )
    holder = ThinkingBudgetStateHolder(rc, 4, 0, "cpu", False)
    assert holder.update_budget(0, 1, 400) is False  # untracked
    holder._state[0] = holder._init_state_entry(None, 100)
    holder._state[0]["check_count_down"] = 20
    assert holder.update_budget(0, 1, 400) is True
    assert holder._state[0]["thinking_token_budget"] == 400
    assert holder._state[0]["check_count_down"] == 320
    assert holder.update_budget(0, 1, 800) is False  # replay: no change
    assert holder._state[0]["thinking_token_budget"] == 400
    holder._state[0]["in_end"] = True
    # A late raise is still applied; the state keeps forcing (in_end stays).
    assert holder.update_budget(0, 2, 800) is True
    assert holder._state[0]["in_end"] is True
    assert holder.update_budget(0, 3, 50) is True


def test_effort_report_shape():
    cfg = _cfg()
    st = _state(cfg)
    rep = finish_effort(st)
    assert set(rep) == {
        "rung",
        "escalations",
        "reasoning_tokens",
        "late",
        "stall_clamps",
        "grace_tokens",
        "close_kind",
    }
    counters = {k: v for k, v in rep.items() if k != "close_kind"}
    assert all(isinstance(v, int) for v in counters.values())
    assert rep["close_kind"] in ("natural", "soft", "forced")
    assert not math.isnan(rep["rung"])


# ------------------------------------------------------- scheduler glue


def _bare_scheduler(cfg):
    from vllm.v1.core.sched.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched._effort = {}
    sched._effort_pending = {}
    sched._effort_rep_params = {}
    sched._effort_cfg = cfg
    sched._effort_start_ids = [START]
    sched._effort_end_ids = [END]
    sched._effort_marker_seqs = []
    sched._effort_sketches = None
    sched._effort_policy = None
    sched._effort_policy_age = 0
    sched._effort_worker_eval = False
    sched._effort_worker_reqs = set()
    sched.running = []
    return sched


def _request(req_id, extra_args, rep=None):
    from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
    from vllm.v1.request import Request

    params = SamplingParams(
        max_tokens=100_000,
        thinking_token_budget=LADDER[0],
        extra_args=extra_args,
        repetition_detection=RepetitionDetectionParams(**rep) if rep else None,
    )
    return Request(req_id, [5, START], params, None)


def test_scheduler_glue_add_step_ack_free():
    cfg = _cfg()
    sched = _bare_scheduler(cfg)
    dyn = _request("dyn", {"dynamic_effort": {"ladder": LADDER, "theta": [0.0, 0.5]}})
    plain = _request("plain", None)
    sched._maybe_add_effort_state(dyn)
    sched._maybe_add_effort_state(plain)
    assert set(sched._effort) == {"dyn"}
    st = sched._effort["dyn"]
    assert st.in_think  # prompt ends with <think>
    sched.running = [dyn, plain]
    for tok in _think_tokens(75):
        dyn.append_output_token_ids(tok)
        sched._step_effort(dyn, st, [tok], (0.9, 0.1, 0.0, 1), 0, 0, None)
    assert sched._effort_pending == {"dyn": (1, 400)}
    # Unacked updates are re-sent until the worker acks the revision.
    sched._ingest_effort_acks({"dyn": 0, "other": 3})
    assert sched._effort_pending == {"dyn": (1, 400)}
    sched._ingest_effort_acks({"dyn": 1})
    assert sched._effort_pending == {} and st.acked_revision == 1
    dyn.append_output_token_ids(END)
    sched._step_effort(dyn, st, [END], None, 0, 0, None)
    assert not st.late and finish_effort(st)["reasoning_tokens"] == 75
    sched._effort_pending["dyn"] = (2, 9)
    from vllm.v1.request import RequestStatus

    dyn.status = RequestStatus.FINISHED_STOPPED
    # _free_request's controller cleanup, in isolation.
    sched._effort.pop(dyn.request_id, None)
    sched._effort_pending.pop(dyn.request_id, None)
    sched._effort_rep_params.pop(dyn.request_id, None)
    assert not sched._effort and not sched._effort_pending


def test_scheduler_glue_reuses_repetition_params_relaxed():
    cfg = _cfg()
    sched = _bare_scheduler(cfg)
    req = _request(
        "rep",
        {"dynamic_effort": {}},
        rep={"max_pattern_size": 4, "min_pattern_size": 1, "min_count": 3},
    )
    sched._maybe_add_effort_state(req)
    relaxed = sched._effort_rep_params["rep"]
    assert (relaxed.max_pattern_size, relaxed.min_count) == (4, 2)
    st = sched._effort["rep"]
    sched.running = [req]
    for tok in [7, 8, 7, 8]:  # 2 repeats of a 2-gram: evidence, not a stop
        req.append_output_token_ids(tok)
        sched._step_effort(req, st, [tok], (0.5, 0.5, 0.0, 1), 0, 0, None)
    assert st.stalled
    revision, cap = sched._effort_pending["rep"]
    assert revision == 1
    assert cap + st.soft_ramp == max(st.think_count + 32, st.soft_ramp)
