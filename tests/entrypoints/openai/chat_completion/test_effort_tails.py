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


SYSTEM_TAIL_TEMPLATE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "serve-configs",
    "qwen3_8_chat_template.jinja",
)


def _chat_template(cfg: DynamicEffortConfig) -> str | None:
    """The stock Qwen template rejects a trailing system turn; the placements
    that render one need the serve-configs copy that accepts it."""
    if cfg.hidden_effort.sentence_placement == "user":
        return None
    with open(SYSTEM_TAIL_TEMPLATE) as f:
        return f.read()


def _build_qwen3_serving_chat(cfg: DynamicEffortConfig) -> OpenAIServingChat:
    model_config = MockModelConfig()
    chat_template = _chat_template(cfg)
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
        chat_template=chat_template,
        chat_template_content_format="auto",
        enable_auto_tools=True,
        tool_parser="qwen3_coder",
    )
    return OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        online_renderer=online_renderer,
        chat_template=chat_template,
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
    "system-tail": dict(
        hidden=dict(think_off_level=True, default_level=2, sentence_placement="system"),
        kwargs=None,
    ),
    "auto-tail-after-tool": dict(
        hidden=dict(think_off_level=True, default_level=2, sentence_placement="auto"),
        kwargs=None,
    ),
}


@pytest.mark.parametrize(
    ("placement", "last_role", "expected"),
    [
        ("user", "tool", "user"),
        ("system", "user", "system"),
        ("auto", "user", "user"),
        ("auto", "tool", "system"),
        ("auto", "assistant", "system"),
    ],
)
def test_sentence_role_follows_placement(placement, last_role, expected):
    """`auto` only speaks as the user when the user spoke last."""
    messages = [{"role": "user", "content": "hi"}, {"role": last_role, "content": "x"}]
    variants = render_effort_variants(messages, ["low", "high"], placement)
    assert all(v[-1]["role"] == expected for v in variants)
    assert [v[-1]["content"] for v in variants] == ["low", "high"]


@pytest.mark.skipif(QWEN3_PATH is None, reason="no local Qwen3 tokenizer")
def test_system_tail_renders_trailing_system_turn():
    """With the serve-configs template a system-placed sentence lands as the
    last turn before the generation prompt, byte-for-byte."""
    asyncio.run(_check_system_tail_text())


async def _check_system_tail_text():
    cfg = _effort_config(default_level=1, sentence_placement="auto")
    serving = _build_qwen3_serving_chat(cfg)
    messages = agent_conversation(turns=2, words_per_tool_result=20)
    assert messages[-1]["role"] == "tool"
    overrides, ids = await _effort_overrides(serving, cfg, messages, full_render=False)
    tokenizer = serving.engine_client.renderer.tokenizer
    body = tokenizer.decode(ids[: overrides["body_len"]])
    low = tokenizer.decode(overrides["tails"][0])
    assert body.endswith("<|im_end|>\n")
    assert low.endswith(
        "<|im_start|>system\n"
        + cfg.level_sentences[0]
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )


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
        default_ids[: fast["body_len"]] + fast["tails"][cfg.hidden_effort.default_level]
        == default_ids
    )


@pytest.mark.skipif(QWEN3_PATH is None, reason="no local Qwen3 tokenizer")
def test_variants_use_the_tail_path():
    """The fast path is what actually runs: it produces ids, not None."""
    asyncio.run(_check_tail_path_taken())


async def _check_tail_path_taken():
    cfg = _effort_config(think_off_level=True, default_level=2)
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
    variants = request._dynamic_effort_variant_messages
    assert variants is not None
    requests = []
    for level, msgs in enumerate(variants):
        if level == cfg.hidden_effort.default_level:
            requests.append(None)
            continue
        variant = request.model_copy()
        variant.messages = msgs
        if level == 0:
            variant.chat_template_kwargs = {"enable_thinking": False}
        requests.append(variant)
    rendered = await serving._tokenize_effort_variants(
        request, engine_inputs[0], requests
    )
    assert rendered is not None
    assert len(rendered) == len(variants)


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
