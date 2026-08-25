# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The tail-only tokenization of dynamic-effort variants must produce the
same ids as rendering every variant through the full chat path."""

import asyncio
import glob
import os
import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.config import MultiModalConfig
from vllm.config.reasoning import (
    DynamicEffortConfig,
    HiddenEffortConfig,
)
from vllm.entrypoints.openai.chat_completion.dynamic_effort import (
    apply_dynamic_effort,
    render_effort_variants,
)
from vllm.entrypoints.openai.chat_completion.effort_tails import (
    special_token_cut,
    tokenize_variant_tails,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config

_QWEN3_SNAPSHOTS = [
    os.path.expanduser("~/.cache/huggingface/hub/models--RedHatAI--Qwen3.8-27B-INT4"),
    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"),
    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"),
]


def _local_qwen3_path() -> str | None:
    for repo in _QWEN3_SNAPSHOTS:
        for snap in sorted(glob.glob(os.path.join(repo, "snapshots", "*"))):
            if os.path.exists(os.path.join(snap, "tokenizer_config.json")):
                return snap
    return None


QWEN3_PATH = _local_qwen3_path()
MODEL_NAME = "qwen3-effort-test"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = QWEN3_PATH
    tokenizer = QWEN3_PATH
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 262144
    tokenizer_revision = None
    revision = None
    code_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


@dataclass
class MockEngine:
    model_config: MockModelConfig = field(default_factory=MockModelConfig)
    input_processor: MagicMock = field(default_factory=MagicMock)
    renderer: Any = None
    vllm_config: Any = None
    errored: bool = False


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def agent_conversation(turns: int, words_per_tool_result: int, seed: int = 0):
    """A multi-turn agent transcript: tool calls, tool results, user follow-ups."""
    rng = random.Random(seed)
    words = ["def", "return", "self", "kv", "cache", "{", "}", "=", "0x1f", "\n"]

    def blob(n: int) -> str:
        return " ".join(rng.choice(words) for _ in range(n))

    msgs: list[dict] = [
        {"role": "system", "content": "You are a coding agent. " + blob(40)},
        {"role": "user", "content": "Fix the scheduler bug. " + blob(30)},
    ]
    for i in range(1, turns + 1):
        msgs.append(
            {
                "role": "assistant",
                "content": "Let me look. " + blob(10),
                "reasoning_content": "thinking " + blob(15),
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": f'{{"path": "/x/{i}.py"}}',
                        },
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "content": blob(words_per_tool_result),
                "tool_call_id": f"call_{i}",
            }
        )
        if i % 3 == 0:
            msgs.append({"role": "assistant", "content": "Progress: " + blob(10)})
            msgs.append({"role": "user", "content": "Continue. " + blob(8)})
    return msgs


def _effort_config(**hidden) -> DynamicEffortConfig:
    return DynamicEffortConfig(
        default_effort="dynamic",
        hidden_effort=HiddenEffortConfig(enabled=True, **hidden),
    )


def _build_qwen3_serving_chat(cfg: DynamicEffortConfig) -> OpenAIServingChat:
    model_config = MockModelConfig()
    engine = MockEngine(
        model_config=model_config,
        renderer=HfRenderer(
            MockVllmConfig(model_config, MockParallelConfig()),
            cached_tokenizer_from_config(model_config),
        ),
        vllm_config=SimpleNamespace(
            reasoning_config=SimpleNamespace(dynamic_effort=cfg)
        ),
    )
    models = OpenAIServingModels(
        engine_client=engine, base_model_paths=BASE_MODEL_PATHS
    )
    online_renderer = OnlineRenderer(
        model_config=model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
        tool_parser="qwen3_coder",
    )
    return OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        online_renderer=online_renderer,
        chat_template=None,
        chat_template_content_format="auto",
        request_logger=None,
        enable_auto_tools=True,
        tool_parser="qwen3_coder",
    )


async def _effort_overrides(
    serving: OpenAIServingChat,
    cfg: DynamicEffortConfig,
    messages: list[dict],
    *,
    full_render: bool,
    chat_template_kwargs: dict | None = None,
    vllm_xargs: dict | None = None,
) -> tuple[dict, list[int]]:
    request = ChatCompletionRequest(
        model=BASE_MODEL_PATHS[0].name,
        messages=messages,
        tools=TOOLS,
        reasoning_effort="dynamic",
        chat_template_kwargs=chat_template_kwargs,
        vllm_xargs=vllm_xargs,
    )
    apply_dynamic_effort(request, cfg)
    result = await serving.render_chat_request(request)
    assert not isinstance(result, ErrorResponse), result
    _, engine_inputs = result
    await serving._apply_default_effort_layout(
        request, request._dynamic_effort_variants, engine_inputs
    )
    original = serving._tokenize_effort_variants
    fast_results: list = []

    async def observed(*args, **kwargs):
        if full_render:
            return None
        out = await original(*args, **kwargs)
        fast_results.append(out)
        return out

    serving._tokenize_effort_variants = observed  # type: ignore[method-assign]
    try:
        error = await serving._attach_effort_tails(request, engine_inputs)
    finally:
        serving._tokenize_effort_variants = original  # type: ignore[method-assign]
    assert error is None
    if not full_render:
        assert fast_results and fast_results[0] is not None, "tail path fell back"
    assert request._dynamic_effort is not None
    return request._dynamic_effort, list(engine_inputs[0]["prompt_token_ids"])


_CASES = {
    "prod-off-vote": dict(
        hidden=dict(think_off_level=True, default_level=2), kwargs=None
    ),
    "no-think-off": dict(hidden=dict(default_level=1), kwargs=None),
    "forced-think-off-default": dict(
        hidden=dict(think_off_level=True, default_level=1),
        kwargs=None,
        xargs={"dynamic_effort_level": 1},
    ),
    "preserve-thinking-false": dict(
        hidden=dict(think_off_level=True, default_level=2),
        kwargs={"preserve_thinking": False},
    ),
    "think-placement": dict(
        hidden=dict(think_off_level=True, default_level=2, sentence_placement="think"),
        kwargs=None,
    ),
    "think-default-with-suffix": dict(
        hidden=dict(think_off_level=True, default_level=1, sentence_placement="think"),
        kwargs=None,
    ),
    "user-placement": dict(
        hidden=dict(think_off_level=True, default_level=2, sentence_placement="user"),
        kwargs=None,
    ),
    "system-placement": dict(
        hidden=dict(think_off_level=True, default_level=2, sentence_placement="system"),
        kwargs=None,
    ),
    "system-default-with-insert": dict(
        hidden=dict(think_off_level=True, default_level=1, sentence_placement="system"),
        kwargs=None,
    ),
    "system-forced-low": dict(
        hidden=dict(think_off_level=True, default_level=2, sentence_placement="system"),
        kwargs=None,
        xargs={"dynamic_effort_level": 1},
    ),
    "user-forced-low": dict(
        hidden=dict(think_off_level=True, default_level=2),
        kwargs=None,
        xargs={"dynamic_effort_level": 1},
    ),
}


@pytest.mark.skipif(QWEN3_PATH is None, reason="no local Qwen3 tokenizer")
@pytest.mark.parametrize("case", sorted(_CASES))
def test_tail_tokenization_matches_full_render(case: str):
    """Tail-only ids equal full-render ids for every variant, seam included."""
    asyncio.run(_check_tail_tokenization(_CASES[case]))


async def _check_tail_tokenization(spec: dict):
    cfg = _effort_config(**spec["hidden"])
    serving = _build_qwen3_serving_chat(cfg)
    messages = agent_conversation(turns=7, words_per_tool_result=120)
    fast, default_ids = await _effort_overrides(
        serving,
        cfg,
        messages,
        full_render=False,
        chat_template_kwargs=spec["kwargs"],
        vllm_xargs=spec.get("xargs"),
    )
    slow, _ = await _effort_overrides(
        serving,
        cfg,
        messages,
        full_render=True,
        chat_template_kwargs=spec["kwargs"],
        vllm_xargs=spec.get("xargs"),
    )
    assert fast["body_len"] == slow["body_len"]
    assert fast["tails"] == slow["tails"]
    assert fast.get("meta_tail") == slow.get("meta_tail")
    assert fast.get("off_append") == slow.get("off_append")
    assert fast.get("meta_stop_ids") == slow.get("meta_stop_ids")
    assert fast["body_len"] < len(default_ids)
    assert (
        default_ids[: fast["body_len"]] + fast["tails"][fast["default_level"]]
        == default_ids
    )


@pytest.mark.skipif(QWEN3_PATH is None, reason="no local Qwen3 tokenizer")
def test_variants_use_the_tail_path():
    """The fast path is what actually runs: it produces ids, not None."""
    asyncio.run(_check_tail_path_taken())


async def _check_tail_path_taken():
    cfg = _effort_config(
        think_off_level=True, default_level=2, sentence_placement="system"
    )
    serving = _build_qwen3_serving_chat(cfg)
    request = ChatCompletionRequest(
        model=BASE_MODEL_PATHS[0].name,
        messages=agent_conversation(turns=2, words_per_tool_result=20),
        tools=TOOLS,
        reasoning_effort="dynamic",
    )
    apply_dynamic_effort(request, cfg)
    result = await serving.render_chat_request(request)
    assert not isinstance(result, ErrorResponse)
    _, engine_inputs = result
    variants = request._dynamic_effort_variants
    assert variants is not None
    renders = await serving._effort_renders(
        request, variants, engine_inputs[0]["prompt"]
    )
    assert [r.request is None for r in renders] == [False, True, True, False]
    assert renders[1].insert.startswith("<|im_start|>system\n")
    rendered = await serving._tokenize_effort_variants(
        request, engine_inputs[0], renders, renders[2]
    )
    assert rendered is not None
    assert len(rendered) == len(variants.levels) + 1


def test_render_effort_variants_placement():
    """`think` leaves the messages alone and carries the sentence as a suffix;
    `user` appends it as a trailing user message with no suffix."""
    messages = [{"role": "user", "content": "hi"}]
    sentences = [None, "Reasoning effort is set to low.", ""]
    system = render_effort_variants(messages, sentences, "system")
    assert [v.messages is messages for v in system] == [True, True, True]
    assert [v.system for v in system] == ["", "Reasoning effort is set to low.", ""]
    assert [v.suffix for v in system] == ["", "", ""]
    think = render_effort_variants(messages, sentences, "think")
    assert [v.messages is messages for v in think] == [True, True, True]
    assert [v.suffix for v in think] == ["", "Reasoning effort is set to low.\n", ""]
    assert messages == [{"role": "user", "content": "hi"}]
    user = render_effort_variants(messages, sentences, "user")
    assert [v.suffix for v in user] == ["", "", ""]
    assert user[0].messages == messages
    assert user[1].messages == messages + [
        {"role": "user", "content": "Reasoning effort is set to low."}
    ]
    assert user[2].messages == messages


@pytest.mark.skipif(QWEN3_PATH is None, reason="no local Qwen3 tokenizer")
def test_think_placement_tail_is_generation_prompt_plus_sentence():
    """With `think` placement the level-1 tail is the generation prompt plus
    the sentence as the first think line, the body is shared and the request's
    messages are the client's."""
    asyncio.run(_check_think_placement())


async def _check_think_placement():
    cfg = _effort_config(
        think_off_level=True, default_level=2, sentence_placement="think"
    )
    serving = _build_qwen3_serving_chat(cfg)
    tokenizer = serving.renderer.tokenizer
    messages = agent_conversation(turns=2, words_per_tool_result=20)
    request = ChatCompletionRequest(
        model=BASE_MODEL_PATHS[0].name,
        messages=messages,
        tools=TOOLS,
        reasoning_effort="dynamic",
    )
    apply_dynamic_effort(request, cfg)
    assert request.messages == messages
    levels = request._dynamic_effort_variants.levels
    assert all(v.messages is levels[0].messages for v in levels)
    assert levels[1].messages == messages
    overrides, default_ids = await _effort_overrides(
        serving, cfg, messages, full_render=False
    )
    body_len, tails = overrides["body_len"], overrides["tails"]
    assert default_ids[:body_len] + tails[2] == default_ids
    low_tail = tokenizer.decode(tails[1])
    assert low_tail.endswith(
        "<|im_start|>assistant\n<think>\nReasoning effort is set to low.\n"
    )
    assert tokenizer.decode(tails[2]).endswith("<|im_start|>assistant\n<think>\n")
    assert tails[1][: len(tails[2])] == tails[2]
    assert tails[1][len(tails[2]) :] == tokenizer.encode(
        "Reasoning effort is set to low.\n", add_special_tokens=False
    )
    assert tokenizer.decode(tails[0]).endswith("<think>\n\n</think>\n\n")
    assert tokenizer.decode(overrides["meta_tail"]).startswith("<|im_start|>user\n")
    # The full renderer agrees, suffix included.
    slow, _ = await _effort_overrides(serving, cfg, messages, full_render=True)
    assert slow["tails"] == tails and slow["body_len"] == body_len


@pytest.mark.skipif(QWEN3_PATH is None, reason="no local Qwen3 tokenizer")
def test_system_placement_tail_is_system_turn_plus_generation_prompt():
    """With `system` placement the level-1 tail is the
    template's rendering of a system message carrying the sentence followed
    by the level-2 tail (the generation prompt); the body is shared, the
    messages are the client's and the off gate still closes in place."""
    asyncio.run(_check_system_placement())


async def _check_system_placement():
    cfg = _effort_config(
        think_off_level=True, default_level=2, sentence_placement="system"
    )
    serving = _build_qwen3_serving_chat(cfg)
    tokenizer = serving.renderer.tokenizer
    messages = agent_conversation(turns=2, words_per_tool_result=20)
    request = ChatCompletionRequest(
        model=BASE_MODEL_PATHS[0].name,
        messages=messages,
        tools=TOOLS,
        reasoning_effort="dynamic",
    )
    apply_dynamic_effort(request, cfg)
    assert request.messages == messages
    levels = request._dynamic_effort_variants.levels
    assert all(v.messages is levels[0].messages for v in levels)
    overrides, default_ids = await _effort_overrides(
        serving, cfg, messages, full_render=False
    )
    body_len, tails = overrides["body_len"], overrides["tails"]
    assert default_ids[:body_len] + tails[2] == default_ids
    system_turn = "<|im_start|>system\nReasoning effort is set to low.<|im_end|>\n"
    assert tokenizer.decode(tails[1]) == system_turn + tokenizer.decode(tails[2])
    assert tokenizer.decode(tails[2]).endswith("<|im_start|>assistant\n<think>\n")
    assert tails[1] == (
        tokenizer.encode(system_turn, add_special_tokens=False) + tails[2]
    )
    assert tokenizer.decode(tails[0]).endswith("<think>\n\n</think>\n\n")
    assert tokenizer.decode(overrides["meta_tail"]).startswith("<|im_start|>user\n")
    assert tokenizer.decode(overrides["off_append"]) == "</think>\n\n"
    slow, _ = await _effort_overrides(serving, cfg, messages, full_render=True)
    assert slow["tails"] == tails and slow["body_len"] == body_len
    # The template pieces are cached per template and kwargs.
    assert any(k[0] == "gen" for k in serving._effort_pieces)
    assert any(k[0] == "system" for k in serving._effort_pieces)


@pytest.mark.skipif(QWEN3_PATH is None, reason="no local Qwen3 tokenizer")
def test_system_placement_default_level_carries_the_turn():
    """A default level with a sentence gets the system turn spliced into the
    engine prompt (text and ids) before the seam is computed."""
    asyncio.run(_check_system_default_insert())


async def _check_system_default_insert():
    cfg = _effort_config(
        think_off_level=True, default_level=1, sentence_placement="system"
    )
    serving = _build_qwen3_serving_chat(cfg)
    tokenizer = serving.renderer.tokenizer
    messages = agent_conversation(turns=2, words_per_tool_result=20)
    overrides, default_ids = await _effort_overrides(
        serving, cfg, messages, full_render=False
    )
    prompt = tokenizer.decode(default_ids)
    assert prompt.endswith(
        "<|im_start|>system\nReasoning effort is set to low.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    assert default_ids == tokenizer.encode(prompt, add_special_tokens=False)
    tails = overrides["tails"]
    assert default_ids[: overrides["body_len"]] + tails[1] == default_ids
    assert "Reasoning effort" not in tokenizer.decode(tails[2])
    assert tokenizer.decode(overrides["off_append"]) == "</think>\n\n"


@pytest.mark.skipif(QWEN3_PATH is None, reason="no local Qwen3 tokenizer")
def test_user_placement_reproduces_trailing_user_message():
    asyncio.run(_check_user_placement())


async def _check_user_placement():
    cfg = _effort_config(think_off_level=True, default_level=2)
    assert cfg.hidden_effort.sentence_placement == "user"
    serving = _build_qwen3_serving_chat(cfg)
    tokenizer = serving.renderer.tokenizer
    messages = agent_conversation(turns=2, words_per_tool_result=20)
    overrides, default_ids = await _effort_overrides(
        serving, cfg, messages, full_render=False
    )
    tails = overrides["tails"]
    assert default_ids[: overrides["body_len"]] + tails[2] == default_ids
    assert tokenizer.decode(tails[1]).endswith(
        "<|im_start|>user\nReasoning effort is set to low.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    assert "Reasoning effort" not in tokenizer.decode(tails[2])


def test_tokenize_variant_tails_rejects_unproven_boundary():
    """A default tail that does not re-encode to its own ids means fallback."""
    special = ["<|s|>"]
    encode = lambda text: [ord(c) for c in text]  # noqa: E731
    default_text = "abc<|s|>xyz"
    assert special_token_cut([default_text, "abc<|s|>xy!"], special) == 3
    ids = encode(default_text)
    out = tokenize_variant_tails(
        encode, default_text, ids, [default_text, "abc<|s|>q"], special
    )
    assert out == [ids, encode("abc") + encode("<|s|>q")]
    assert (
        tokenize_variant_tails(
            encode, default_text, ids[:-1], [default_text, "abc<|s|>q"], special
        )
        is None
    )
    assert special_token_cut(["abc", "abd"], special) is None
