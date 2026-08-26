# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend half of `reasoning_effort: "dynamic"` (docs/dynamic-reasoning §13).

`dynamic` never reaches the chat template. The request is rewritten in place:
the template sees `render_effort` (medium: no effort sentence, so block 0 of the
prompt is identical for every level), and each effort level is rendered as a
**trailing user message carrying only that level's sentence**, after the last
message of the conversation. That is the true tail of the prompt, which is where
the model actually honours it - measured on this box 2026-08-19: with the
sentence on the last *user* message of an agent turn (the placement patch 0009
shipped) the `xhigh` wording moves reasoning length 1.14x against no sentence,
because a tool result sits between it and the generation point; as a trailing
user message it moves it 1.23x up and 0.78x down.

The engine gets the shared body and one tail per level and picks the level from
the body's own pooled hidden state before the model thinks. No thinking budget
is set: on this path the model ends its own think block.
"""

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.config.reasoning import (
    OFF_VOTE_PROMPT,
    DynamicEffortConfig,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

_LEVEL_KEY = "dynamic_effort_level"


class DynamicEffortError(ValueError):
    """Client error in a dynamic-effort request (rendered as HTTP 400)."""


def build_dynamic_effort_overrides(
    cfg: DynamicEffortConfig, xargs: dict[str, Any] | None
) -> dict[str, Any]:
    """Validate the per-request `vllm_xargs` and merge them over `cfg`."""
    xargs = xargs or {}
    overrides: dict[str, Any] = {}
    if _LEVEL_KEY in xargs and xargs[_LEVEL_KEY] is not None:
        raw = xargs[_LEVEL_KEY]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise DynamicEffortError(f"vllm_xargs.{_LEVEL_KEY} must be an integer")
        if not 0 <= raw < cfg.num_levels:
            raise DynamicEffortError(
                f"vllm_xargs.{_LEVEL_KEY} must be in [0, {cfg.num_levels - 1}]"
            )
        overrides["forced_level"] = raw
    return overrides


def ends_with_tool_result(messages: list[Any]) -> bool:
    """True when the last message is a tool result."""
    if not messages:
        return False
    last = messages[-1]
    role = last.get("role") if isinstance(last, dict) else getattr(last, "role", None)
    return role == "tool"


def append_to_last_message(messages: list[Any], sentence: str) -> bool:
    """Append `sentence` as a trailing user message; True if it was added."""
    if not sentence:
        return False
    messages.append({"role": "user", "content": sentence})
    return True


def apply_default_effort(
    request: "ChatCompletionRequest", cfg: DynamicEffortConfig | None
) -> None:
    """Fill in `reasoning_effort` when the client omitted it entirely.

    Only an *absent* value is filled: an explicit level, including `"none"`,
    is left as the client sent it, except for a configured
    `effort_aliases` rewrite. With `default_effort` unset the request is
    untouched and the chat template picks its own default.
    """
    if cfg is None:
        return
    if request.reasoning_effort is not None:
        alias = cfg.effort_aliases.get(request.reasoning_effort)
        if alias is not None:
            request.reasoning_effort = alias  # type: ignore[assignment]
        return
    if not cfg.default_effort:
        return
    request.reasoning_effort = cfg.default_effort  # type: ignore[assignment]


def apply_dynamic_effort(
    request: "ChatCompletionRequest", cfg: DynamicEffortConfig | None
) -> None:
    """Rewrite a `reasoning_effort: "dynamic"` request in place.

    Raises:
        DynamicEffortError: on conflicts, bad overrides or a server without
            `dynamic_effort` in its reasoning config.
    """
    if request.reasoning_effort != "dynamic":
        return
    if cfg is None:
        raise DynamicEffortError(
            "reasoning_effort='dynamic' is not enabled on this server; start it "
            "with --reasoning-config '{\"dynamic_effort\": {...}}'"
        )
    if request.thinking_token_budget is not None:
        raise DynamicEffortError(
            "reasoning_effort='dynamic' does not cap thinking; drop "
            "thinking_token_budget or ask for a fixed effort level"
        )
    kwargs = request.chat_template_kwargs or {}
    if "enable_thinking" in kwargs and not kwargs["enable_thinking"]:
        raise DynamicEffortError(
            "reasoning_effort='dynamic' conflicts with "
            "chat_template_kwargs.enable_thinking=false"
        )
    hidden = cfg.hidden_effort
    if not hidden.tail_after_tool_result and ends_with_tool_result(request.messages):
        request.reasoning_effort = cfg.render_effort  # type: ignore[assignment]
        return
    overrides = build_dynamic_effort_overrides(cfg, request.vllm_xargs)
    forced = overrides.get("forced_level")
    if (
        forced is not None
        and cfg.level_sentences[forced] is None
        and hidden.off_vote
        and not hidden.level_vote
    ):
        # A forced think-off still passes through the off-vote gate: run the
        # normal two-phase path and force the verdict in the engine.
        overrides["force_off"] = True
        del overrides["forced_level"]
    default_level = overrides.get("forced_level", hidden.default_level)
    variants = render_effort_variants(request.messages, cfg.level_sentences)
    request.messages = variants[default_level]
    request._dynamic_effort_variant_messages = variants
    overrides["default_level"] = default_level
    overrides["think_off_levels"] = [
        i for i, sentence in enumerate(cfg.level_sentences) if sentence is None
    ]
    if hidden.level_vote:
        overrides["meta_prompt"] = hidden.level_vote_prompt
        overrides["level_vote_words"] = hidden.vote_words()
        overrides["level_vote_rule"] = hidden.level_vote_rule
    elif overrides["think_off_levels"] and hidden.off_vote:
        overrides["meta_prompt"] = OFF_VOTE_PROMPT
    if "meta_prompt" in overrides:
        overrides["off_votes"] = hidden.off_votes
        overrides["off_vote_max_tokens"] = hidden.off_vote_max_tokens
    if default_level in overrides["think_off_levels"]:
        request.chat_template_kwargs = {**kwargs, "enable_thinking": False}
    request.reasoning_effort = cfg.render_effort  # type: ignore[assignment]
    request._dynamic_effort = overrides


def render_effort_variants(
    messages: list[Any], sentences: list[str | None]
) -> list[list[Any]]:
    """One message list per level, each with that level's tail sentence.

    A `None` sentence is the think-off level: the messages are untouched and
    the variant is rendered with `enable_thinking=false` instead."""
    variants: list[list[Any]] = []
    for sentence in sentences:
        rendered = copy.deepcopy(messages)
        append_to_last_message(rendered, sentence or "")
        variants.append(rendered)
    return variants


def off_vote_variant(messages: list[Any], prompt: str = OFF_VOTE_PROMPT) -> list[Any]:
    """The extra rendering a hidden vote needs: the question (yes/no for the
    off gate, the level words for the level vote), rendered thinking-off."""
    meta = copy.deepcopy(messages)
    append_to_last_message(meta, prompt)
    return meta


def vote_word_token_ids(
    encode: Callable[[str], list[int]], words: list[str]
) -> list[list[int]]:
    """Single-token spellings of each answer word: as written, Capitalized,
    UPPER, and each with a leading space."""
    out: list[list[int]] = []
    for word in words:
        ids: set[int] = set()
        for text in (word, word.capitalize(), word.upper()):
            for spelled in (text, " " + text):
                encoded = encode(spelled)
                if len(encoded) == 1:
                    ids.add(int(encoded[0]))
        out.append(sorted(ids))
    return out


def split_body_and_tails(
    variant_token_ids: list[list[int]],
) -> tuple[int, list[list[int]]] | None:
    """Split the rendered level variants into the shared body and per-level tails.

    The variants differ only in the trailing sentence message, so their longest
    common token prefix *is* the body of the §13.3 seam. The boundary is pulled
    one token back so that the token at position `body_len` - the one an
    eagle-family drafter reads ahead at a chunked-prefill boundary - is the same
    whichever level is chosen.

    Args:
        variant_token_ids: the fully rendered prompt of each level, in level
            order.

    Returns:
        `(body_len, tails)`, or `None` when the variants share no usable body.
    """
    if len(variant_token_ids) < 2 or any(not ids for ids in variant_token_ids):
        return None
    first = variant_token_ids[0]
    common = len(first)
    for ids in variant_token_ids[1:]:
        limit = min(common, len(ids))
        i = 0
        while i < limit and ids[i] == first[i]:
            i += 1
        common = i
    body_len = common - 1
    if body_len < 1 or any(len(ids) <= body_len for ids in variant_token_ids):
        return None
    return body_len, [list(ids[body_len:]) for ids in variant_token_ids]
