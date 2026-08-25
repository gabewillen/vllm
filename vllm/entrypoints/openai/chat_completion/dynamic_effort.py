# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend half of `reasoning_effort: "dynamic"` (docs/dynamic-reasoning §13).

`dynamic` never reaches the chat template. The request is rewritten in place:
the template sees `render_effort` (medium: no effort sentence, so block 0 of the
prompt is identical for every level), and each effort level is rendered as a
**tail after the shared body**. By default (`sentence_placement="user"`) that
tail is a trailing user message carrying only the level's sentence, after the
last message of the conversation - the true tail of the prompt, which is where
the model actually honours it (measured 2026-08-19: 1.23x up / 0.78x down
against 1.14x with the sentence on the last user message of an agent turn).
Two opt-in placements exist for A/B: `"system"` inserts the chat template's
rendering of a system turn with the sentence right before the generation
prompt, so the messages are untouched and the sentence cannot be read as a
user request (an agent benchmark 2026-08-24 saw the user form answered as the
user's turn after a tool result; on the 12-prompt grid system matches user,
21/24 at 481 vs 465 avg tokens); `"think"` appends the sentence after the
generation prompt as the first line of the think block, which forces the first
thought and measured worse (19/24 at 1285, one capped, one unclosed think).

The engine gets the shared body and one tail per level and picks the level from
the body's own pooled hidden state before the model thinks. No thinking budget
is set: on this path the model ends its own think block.
"""

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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


@dataclass
class EffortVariant:
    """One effort level's rendering: the messages the chat template sees, the
    sentence rendered as a trailing system turn before the generation prompt
    (`""` for none) and the text appended after the generation prompt (`""`
    for none)."""

    messages: list[Any]
    system: str = ""
    suffix: str = ""


@dataclass
class EffortRender:
    """A variant as the serving layer renders it: the request to run through
    the chat template (`None`: the default level's template output), the text
    inserted before the generation prompt `gen`, and the text appended after
    it."""

    request: Any
    insert: str = ""
    gen: str = ""
    suffix: str = ""

    def compose(self, text: str) -> str | None:
        """`text` (a chat template output) with this render's insert and
        suffix applied; `None` when `text` does not end with `gen`."""
        if self.insert:
            if not self.gen or not text.endswith(self.gen):
                return None
            text = text[: -len(self.gen)] + self.insert + self.gen
        return text + self.suffix

    def base_text(self, text: str) -> str | None:
        """Inverse of `compose`: the template output `text` was built from."""
        if self.suffix:
            if not text.endswith(self.suffix):
                return None
            text = text[: -len(self.suffix)]
        if self.insert:
            tail = self.insert + self.gen
            if not text.endswith(tail):
                return None
            text = text[: -len(tail)] + self.gen
        return text


@dataclass
class EffortVariants:
    """Every level's variant plus, when the off gate is on, the messages of the
    hidden off-vote question."""

    levels: list[EffortVariant]
    meta_messages: list[Any] | None = None
    default_level: int = 0
    think_off_levels: set[int] = field(default_factory=set)


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
    overrides = build_dynamic_effort_overrides(cfg, request.vllm_xargs)
    forced = overrides.get("forced_level")
    hidden = cfg.hidden_effort
    if forced is not None and cfg.level_sentences[forced] is None and hidden.off_vote:
        # A forced think-off still passes through the off-vote gate: run the
        # normal two-phase path and force the verdict in the engine.
        overrides["force_off"] = True
        del overrides["forced_level"]
    default_level = overrides.get("forced_level", hidden.default_level)
    levels = render_effort_variants(
        request.messages, cfg.level_sentences, hidden.sentence_placement
    )
    think_off_levels = [
        i for i, sentence in enumerate(cfg.level_sentences) if sentence is None
    ]
    meta = None
    if think_off_levels and hidden.off_vote:
        meta = off_vote_variant(request.messages)
    request._dynamic_effort_variants = EffortVariants(
        levels=levels,
        meta_messages=meta,
        default_level=default_level,
        think_off_levels=set(think_off_levels),
    )
    request.messages = levels[default_level].messages
    overrides["default_level"] = default_level
    overrides["think_off_levels"] = think_off_levels
    if meta is not None:
        overrides["off_votes"] = hidden.off_votes
        overrides["off_vote_max_tokens"] = hidden.off_vote_max_tokens
    if default_level in overrides["think_off_levels"]:
        request.chat_template_kwargs = {**kwargs, "enable_thinking": False}
    request.reasoning_effort = cfg.render_effort  # type: ignore[assignment]
    request._dynamic_effort = overrides


def render_effort_variants(
    messages: list[Any],
    sentences: list[str | None],
    placement: Literal["user", "system", "think"] = "user",
) -> list[EffortVariant]:
    """One variant per level, each carrying that level's sentence.

    With `placement="user"` (default) the sentence is a trailing user
    message. With `"system"` the messages are shared untouched and the
    sentence is rendered by the serving layer as a trailing system turn right
    before the generation prompt. With `"think"` the sentence (plus a newline)
    is the suffix appended after the generation prompt, i.e. the first line of
    the think block. A `None` sentence is the think-off level: the
    messages are untouched and the variant is rendered with
    `enable_thinking=false` instead."""
    variants: list[EffortVariant] = []
    for sentence in sentences:
        if placement == "user":
            rendered = copy.deepcopy(messages)
            append_to_last_message(rendered, sentence or "")
            variants.append(EffortVariant(rendered))
        elif placement == "think":
            variants.append(
                EffortVariant(messages, suffix=f"{sentence}\n" if sentence else "")
            )
        else:
            variants.append(EffortVariant(messages, system=sentence or ""))
    return variants


def off_vote_variant(messages: list[Any]) -> list[Any]:
    """The extra rendering the off gate needs: the hidden yes/no question,
    rendered thinking-off."""
    meta = copy.deepcopy(messages)
    append_to_last_message(meta, OFF_VOTE_PROMPT)
    return meta


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
