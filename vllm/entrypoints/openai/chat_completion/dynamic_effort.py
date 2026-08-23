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
from typing import TYPE_CHECKING, Any

from vllm.config.reasoning import (
    CUSTOM_EFFORT_SENTENCE,
    CUSTOM_META_PROMPT,
    CUSTOM_TAIL_PREFIX,
    CUSTOM_TAIL_SUFFIX,
    DynamicEffortConfig,
)

CUSTOM_PLACEHOLDERS = ("QZQZQZ", "QZQZQZQZQZQZ")
"""Two placeholder notes of different length; the custom tail's prefix and
suffix token ids are the common prefix and suffix of their renderings."""

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
    is left exactly as the client sent it. With `default_effort` unset the
    request is untouched and the chat template picks its own default.
    """
    if request.reasoning_effort is not None:
        return
    if cfg is None or not cfg.default_effort:
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
    overrides = build_dynamic_effort_overrides(cfg, request.vllm_xargs)
    forced = overrides.get("forced_level")
    if forced is not None and cfg.level_sentences[forced] == CUSTOM_EFFORT_SENTENCE:
        # Custom needs the engine's meta pass: run the normal two-phase path
        # and force the verdict there instead of rendering it as the default.
        overrides["force_custom"] = True
        del overrides["forced_level"]
    default_level = overrides.get("forced_level", cfg.hidden_effort.default_level)
    variants = render_effort_variants(request.messages, cfg.level_sentences)
    request.messages = variants[default_level]
    request._dynamic_effort_variant_messages = variants
    overrides["default_level"] = default_level
    overrides["think_off_levels"] = [
        i for i, sentence in enumerate(cfg.level_sentences) if sentence is None
    ]
    custom = [
        i for i, sentence in enumerate(cfg.level_sentences)
        if sentence == CUSTOM_EFFORT_SENTENCE
    ]
    overrides["custom_level"] = custom[0] if custom else None
    overrides["custom_max_tokens"] = cfg.hidden_effort.custom_max_tokens
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
        if sentence == CUSTOM_EFFORT_SENTENCE:
            # The level's own rendering is the first placeholder; the engine
            # splices the generated note between the tail's prefix and suffix.
            sentence = CUSTOM_TAIL_PREFIX + CUSTOM_PLACEHOLDERS[0] + CUSTOM_TAIL_SUFFIX
        append_to_last_message(rendered, sentence or "")
        variants.append(rendered)
    return variants


def custom_aux_variants(messages: list[Any]) -> list[list[Any]]:
    """The two extra renderings a custom level needs: the second placeholder
    and the hidden meta prompt (rendered thinking-off)."""
    second = copy.deepcopy(messages)
    append_to_last_message(
        second, CUSTOM_TAIL_PREFIX + CUSTOM_PLACEHOLDERS[1] + CUSTOM_TAIL_SUFFIX
    )
    meta = copy.deepcopy(messages)
    append_to_last_message(meta, CUSTOM_META_PROMPT)
    return [second, meta]


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
