# SPDX-License-Identifier: Apache-2.0
"""The v3 dynamic-effort contract: one level, chosen before thinking, no cap.

`reasoning_effort: "dynamic"` renders one prompt variant per effort level, with
that level's sentence as a trailing user message - the true tail of the prompt,
which is where the model honours it (measured 2026-08-19, see
docs/dynamic-reasoning-v3-results.md §2). Nothing caps the thinking and nothing
watches it: the model closes its own think block, bounded only by the
client's own max_tokens and timeouts.
"""

import asyncio
import random

import pytest

from vllm.config.reasoning import (
    LOW_EFFORT_SENTENCE,
    OFF_VOTE_PROMPT,
    DynamicEffortConfig,
    HiddenEffortConfig,
    ReasoningConfig,
)
from vllm.entrypoints.openai.chat_completion.dynamic_effort import (
    DynamicEffortError,
    apply_default_effort,
    apply_dynamic_effort,
    render_effort_variants,
    split_body_and_tails,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    EffortInfo,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.inputs import tokens_input
from vllm.v1.core.sched.effort_controller import (
    CLOSE_CLIENT_LIMIT,
    CLOSE_NATURAL,
    EffortEvent,
    finish_effort,
    new_effort_state,
    step_effort,
)

START, END = 1, 2


def _cfg(**kw) -> DynamicEffortConfig:
    return DynamicEffortConfig(**kw)


def _request(**kw) -> ChatCompletionRequest:
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c0",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c0", "content": "def f(): pass"},
        ],
        "reasoning_effort": "dynamic",
    }
    body.update(kw)
    return ChatCompletionRequest(**body)


def test_dynamic_effort_rejects_independent_data_parallel_state():
    from types import SimpleNamespace

    reasoning_config = ReasoningConfig(dynamic_effort=DynamicEffortConfig())
    reasoning_config.verify_with_parallel_config(
        parallel_config=SimpleNamespace(data_parallel_size=1)
    )

    with pytest.raises(ValueError, match="not supported with data parallelism"):
        reasoning_config.verify_with_parallel_config(
            parallel_config=SimpleNamespace(data_parallel_size=2)
        )


# ------------------------------------------------------------------ frontend


def test_levels_default_to_low_and_medium():
    cfg = _cfg()
    assert cfg.num_levels == 2
    assert cfg.level_sentences == [LOW_EFFORT_SENTENCE, ""]
    assert LOW_EFFORT_SENTENCE == "Reasoning effort is set to low."


def test_sentence_goes_at_the_true_tail_not_the_last_user_turn():
    """The last message here is a tool result, which is exactly the case the
    last-user-turn placement gets wrong: the sentence would sit behind it."""
    cfg = _cfg()
    request = _request()
    apply_dynamic_effort(request, cfg)
    variants = request._dynamic_effort_variant_messages
    assert len(variants) == 2
    # Level 1 renders no sentence at all - the chat template's own `medium`.
    assert [m["role"] for m in variants[1]] == ["user", "assistant", "tool"]
    assert variants[0][-1] == {"role": "user", "content": LOW_EFFORT_SENTENCE}
    # Everything before the sentence is untouched, so the body is shared.
    assert variants[0][:-1] == variants[1]
    # The submitted prompt is the default level, and the template sees `medium`.
    assert request.messages is variants[0]
    assert request.reasoning_effort == "medium"


def test_dynamic_sets_no_thinking_budget():
    request = _request()
    apply_dynamic_effort(request, _cfg())
    assert request.thinking_token_budget is None
    params = request.to_sampling_params(1000, {})
    assert params.thinking_token_budget is None
    assert params.extra_args["dynamic_effort"]["default_level"] == 0


def test_an_explicit_thinking_budget_is_rejected():
    request = _request(thinking_token_budget=512)
    with pytest.raises(DynamicEffortError):
        apply_dynamic_effort(request, _cfg())


def test_a_forced_level_skips_the_decision():
    request = _request(vllm_xargs={"dynamic_effort_level": 0})
    apply_dynamic_effort(request, _cfg())
    assert request.messages[-1]["content"] == LOW_EFFORT_SENTENCE
    assert request._dynamic_effort["forced_level"] == 0
    with pytest.raises(DynamicEffortError):
        apply_dynamic_effort(_request(vllm_xargs={"dynamic_effort_level": 9}), _cfg())


def test_a_forced_think_off_still_passes_the_off_gate():
    """Forcing the think-off level via vllm_xargs may not bypass the vote:
    the engine gets `force_off` instead of a forced level."""
    cfg = _cfg(hidden_effort=HiddenEffortConfig(think_off_level=True, default_level=1))
    request = _request(vllm_xargs={"dynamic_effort_level": 0})
    apply_dynamic_effort(request, cfg)
    overrides = request._dynamic_effort
    assert overrides["force_off"] is True
    assert "forced_level" not in overrides
    assert overrides["default_level"] == 1
    assert overrides["think_off_levels"] == [0]
    assert overrides["off_votes"] == 3
    assert overrides["off_vote_max_tokens"] == 8
    # With the gate disabled the forced level is honoured directly and no
    # vote parameters are shipped.
    cfg = _cfg(
        hidden_effort=HiddenEffortConfig(
            think_off_level=True, default_level=1, off_vote=False
        )
    )
    request = _request(vllm_xargs={"dynamic_effort_level": 0})
    apply_dynamic_effort(request, cfg)
    overrides = request._dynamic_effort
    assert overrides["forced_level"] == 0
    assert "force_off" not in overrides
    assert "off_votes" not in overrides


def test_no_server_config_is_a_client_error():
    with pytest.raises(DynamicEffortError):
        apply_dynamic_effort(_request(), None)


def test_variants_share_a_body_and_differ_only_in_the_tail():
    variants = render_effort_variants(
        [{"role": "user", "content": "hi"}], ["short", "", "long"]
    )
    ids = [[1, 2, 3, 4, 90, 91], [1, 2, 3, 4], [1, 2, 3, 4, 70, 71, 72]]
    body_len, tails = split_body_and_tails(ids)
    assert body_len == 3
    assert [v[-1]["content"] for v in variants if len(v) > 1] == ["short", "long"]
    for variant, tail in zip(ids, tails):
        assert variant[:body_len] + tail == variant


def test_attachment_preserves_absolute_levels_with_a_medium_default():
    cfg = _cfg(hidden_effort=HiddenEffortConfig(default_level=1))
    request = _request()
    apply_dynamic_effort(request=request, cfg=cfg)

    low = [1, 2, 3, 10]
    medium = [1, 2, 3, 20]
    rendered_by_sentence = {LOW_EFFORT_SENTENCE: low}

    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.model_config = type("ModelConfig", (), {"is_encoder_decoder": False})()

    async def render_chat_request(variant_request):
        last_content = variant_request.messages[-1].get("content")
        token_ids = rendered_by_sentence.get(last_content, medium)
        return [], [tokens_input(token_ids)]

    serving.render_chat_request = render_chat_request
    asyncio.run(
        serving._attach_effort_tails(
            request=request,
            engine_inputs=[tokens_input(medium)],
        )
    )

    body_len = request._dynamic_effort["body_len"]
    tails = request._dynamic_effort["tails"]
    assert [medium[:body_len] + tail for tail in tails] == [low, medium]


# ---------------------------------------------------------------- controller


def _state(cfg, prompt=None):
    return new_effort_state("r", cfg, [START], [END], prompt)


def _run(state, cfg, tokens):
    return [step_effort(state, cfg, EffortEvent(new_token_ids=[t])) for t in tokens]


def test_nothing_caps_or_forces_a_close():
    cfg = _cfg()
    state = _state(cfg)
    # 50k think tokens and the model still closes on its own: nothing on this
    # path can cap, bias or force the close.
    rng = random.Random(0)
    stream = [rng.randrange(1000, 150_000) for _ in range(50_000)]
    _run(state, cfg, [START] + stream + [END])
    assert not state.in_think
    assert state.reasoning_tokens == 50_000
    assert finish_effort(state)["close_kind"] == CLOSE_NATURAL


def test_a_degenerate_loop_is_left_alone():
    """No stall detector: `dynamic` does not touch the think block at all.

    A model that repeats itself is the client's `max_tokens` and timeout to
    stop, exactly as it is at a fixed effort level. Nothing here may cut it.
    """
    cfg = _cfg()
    state = _state(cfg)
    _run(state, cfg, [START] + [7, 8, 9, 10] * 500)
    assert state.in_think
    assert state.think_count == 2000
    # Ending here means the client stopped it, so the entry is right-censored
    # and the memory must not take its length as a value.
    assert finish_effort(state)["close_kind"] == CLOSE_CLIENT_LIMIT


def test_a_think_block_the_client_cut_short_is_censored():
    cfg = _cfg()
    state = _state(cfg)
    _run(state, cfg, [START, 5, 6, 7])
    report = finish_effort(state)
    assert report["close_kind"] == CLOSE_CLIENT_LIMIT
    assert report["reasoning_tokens"] == 3


def test_a_prompt_that_ends_mid_think_starts_in_the_think_block():
    cfg = _cfg()
    state = _state(cfg, prompt=[9, 9, START, 4, 5])
    assert state.in_think and state.think_count == 2


def test_the_report_is_the_level_and_how_it_closed():
    cfg = _cfg()
    state = _state(cfg)
    state.level = 1
    state.decided = True
    state.memory_entries = 4096
    state.neighbours = 16
    _run(state, cfg, [START, 5, 6, 7, END])
    report = finish_effort(state)
    assert report == {
        "level": 1,
        "decided": 1,
        "reasoning_tokens": 3,
        "close_kind": CLOSE_NATURAL,
        "memory_entries": 4096,
        "neighbours": 16,
        "estimate": None,
        "calibrated": None,
        "novelty_rank": None,
        "off_votes": 0,
        "off_vetoed": 0,
        "level_votes": None,
        "vote_probs": None,
    }
    info = EffortInfo.from_report(report)
    assert info.level == 1 and info.decided and info.close_kind == CLOSE_NATURAL
    assert info.reasoning_tokens == 3 and info.memory_entries == 4096


def test_hidden_effort_validates_its_levels():
    with pytest.raises(ValueError):
        HiddenEffortConfig(effort_sentences=["only one"])
    with pytest.raises(ValueError):
        HiddenEffortConfig(default_level=5)
    with pytest.raises(ValueError):
        HiddenEffortConfig(off_votes=0)
    with pytest.raises(ValueError):
        HiddenEffortConfig(off_vote_max_tokens=0)
    assert HiddenEffortConfig(effort_sentences=["a", "b", "c", "d"]).sentences() == [
        "a",
        "b",
        "c",
        "d",
    ]


# ------------------------------------------------------- the omitted default


def _bare(**kw) -> ChatCompletionRequest:
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return ChatCompletionRequest(**body)


def test_an_omitted_effort_takes_the_server_default():
    cfg = _cfg(default_effort="dynamic")
    request = _bare()
    assert request.reasoning_effort is None
    apply_default_effort(request, cfg)
    assert request.reasoning_effort == "dynamic"
    # ...and then routes like any other dynamic request.
    apply_dynamic_effort(request, cfg)
    assert request._dynamic_effort_variant_messages is not None
    assert request.reasoning_effort == cfg.render_effort


def test_an_omitted_effort_is_untouched_without_the_knob():
    """The throughput profile opts out by leaving `default_effort` unset."""
    for cfg in (_cfg(), None):
        request = _bare()
        apply_default_effort(request, cfg)
        assert request.reasoning_effort is None
        assert request._dynamic_effort_variant_messages is None


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh"])
def test_an_explicit_effort_is_never_overridden(effort):
    cfg = _cfg(default_effort="dynamic")
    request = _bare(reasoning_effort=effort)
    apply_default_effort(request, cfg)
    assert request.reasoning_effort == effort
    # Only "dynamic" goes through the router, so nothing else is rewritten.
    apply_dynamic_effort(request, cfg)
    assert request.reasoning_effort == effort
    assert request._dynamic_effort_variant_messages is None


def test_an_explicit_effort_alias_is_rewritten_before_the_template():
    """A proxy that normalises `xhigh` to `high` still reaches the template."""
    cfg = _cfg(default_effort="dynamic", effort_aliases={"high": "xhigh"})
    request = _bare(reasoning_effort="high")
    apply_default_effort(request, cfg)
    assert request.reasoning_effort == "xhigh"
    request = _bare(reasoning_effort="low")
    apply_default_effort(request, cfg)
    assert request.reasoning_effort == "low"
    with pytest.raises(ValueError, match="effort_aliases"):
        _cfg(effort_aliases={"high": "bogus"})


def test_an_explicit_dynamic_routes_even_with_the_knob_off():
    cfg = _cfg()
    request = _bare(reasoning_effort="dynamic")
    apply_default_effort(request, cfg)
    apply_dynamic_effort(request, cfg)
    assert request._dynamic_effort_variant_messages is not None


def test_the_default_effort_knob_is_validated():
    for bad in ("Dynamic", "verylow", ""):
        with pytest.raises(ValueError):
            DynamicEffortConfig(default_effort=bad)
    assert DynamicEffortConfig(default_effort="low").default_effort == "low"
    assert DynamicEffortConfig().default_effort is None


def test_think_off_variant_renders_without_a_sentence():
    from vllm.entrypoints.openai.chat_completion.dynamic_effort import (
        render_effort_variants,
    )

    cfg = HiddenEffortConfig(enabled=True, think_off_level=True, default_level=2)
    sentences = cfg.sentences()
    assert sentences[0] is None and sentences[2] == ""
    msgs = [{"role": "user", "content": "hi"}]
    variants = render_effort_variants(msgs, sentences)
    assert variants[0] == msgs and variants[2] == msgs
    assert variants[1][-1]["content"].startswith("Reasoning effort is set to low")
    assert cfg.low_level == 1
    with pytest.raises(ValueError):
        HiddenEffortConfig(enabled=True, think_off_level=True, default_level=0)


def test_off_vote_variant_asks_the_hidden_question_at_the_tail():
    from vllm.entrypoints.openai.chat_completion.dynamic_effort import (
        off_vote_variant,
    )

    cfg = HiddenEffortConfig(enabled=True, think_off_level=True, default_level=2)
    assert cfg.sentences() == [None, LOW_EFFORT_SENTENCE, ""]
    msgs = [{"role": "user", "content": "hi"}]
    meta = off_vote_variant(msgs)
    assert meta[:-1] == msgs
    assert meta[-1] == {"role": "user", "content": OFF_VOTE_PROMPT}
    # The vote parameters ship with any think-off configuration.
    request = _request()
    apply_dynamic_effort(
        request, _cfg(hidden_effort=HiddenEffortConfig(think_off_level=True,
                                                       default_level=1))
    )
    overrides = request._dynamic_effort
    assert overrides["think_off_levels"] == [0]
    assert overrides["off_votes"] == 3 and overrides["off_vote_max_tokens"] == 8
