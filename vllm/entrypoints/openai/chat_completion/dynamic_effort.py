# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend half of `reasoning_effort: "dynamic"` (docs/dynamic-reasoning §2b, §7).

`dynamic` never reaches the chat template. The request is rewritten in place:
the template sees `render_effort` (medium: no effort sentence, so block 0 of
the prompt is identical for every effort), the low-effort sentence is appended
to the *last user turn* (rung-0 prior, prefix-cache safe), the static
`thinking_token_budget` becomes `ladder[0]`, and the validated overrides ride
in `SamplingParams.extra_args["dynamic_effort"]` for the scheduler.
"""

import math
from typing import TYPE_CHECKING, Any

from vllm.config.reasoning import DynamicEffortConfig

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

_LADDER_KEY = "dynamic_effort_ladder"
_THETA_KEY = "dynamic_effort_theta"
_BIAS_KEY = "effort_bias"
_DEADLINE_KEY = "deadline_ms"
_FLOOR_KEY = "dynamic_effort_floor"


class DynamicEffortError(ValueError):
    """Client error in a dynamic-effort request (rendered as HTTP 400)."""


def _int_list(value: Any, key: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise DynamicEffortError(f"vllm_xargs.{key} must be a non-empty list")
    out: list[int] = []
    for x in value:
        if isinstance(x, bool) or not isinstance(x, int | float) or x != int(x):
            raise DynamicEffortError(f"vllm_xargs.{key} must contain integers")
        out.append(int(x))
    return out


def _float_list(value: Any, key: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise DynamicEffortError(f"vllm_xargs.{key} must be a non-empty list")
    out: list[float] = []
    for x in value:
        if isinstance(x, bool) or not isinstance(x, int | float):
            raise DynamicEffortError(f"vllm_xargs.{key} must contain numbers")
        if not math.isfinite(float(x)):
            raise DynamicEffortError(f"vllm_xargs.{key} must be finite")
        out.append(float(x))
    return out


def build_dynamic_effort_overrides(
    cfg: DynamicEffortConfig, xargs: dict[str, Any] | None
) -> dict[str, Any]:
    """Validate the per-request `vllm_xargs` and merge them over `cfg`."""
    xargs = xargs or {}
    ladder = list(cfg.ladder)
    theta = list(cfg.theta or [])
    if _LADDER_KEY in xargs:
        ladder = _int_list(xargs[_LADDER_KEY], _LADDER_KEY)
        if len(ladder) < 2 or ladder[0] <= 0:
            raise DynamicEffortError(
                f"vllm_xargs.{_LADDER_KEY} needs at least two positive rungs"
            )
        if any(b <= a for a, b in zip(ladder, ladder[1:])):
            raise DynamicEffortError(
                f"vllm_xargs.{_LADDER_KEY} must be strictly increasing"
            )
        if _THETA_KEY not in xargs and len(theta) != len(ladder) - 1:
            theta = [0.5 * i for i in range(len(ladder) - 1)]
    if _THETA_KEY in xargs:
        theta = _float_list(xargs[_THETA_KEY], _THETA_KEY)
    if len(theta) != len(ladder) - 1:
        raise DynamicEffortError(
            f"vllm_xargs.{_THETA_KEY} needs one entry per ladder transition "
            f"({len(ladder) - 1}), got {len(theta)}"
        )
    bias = 0.0
    if _BIAS_KEY in xargs:
        raw = xargs[_BIAS_KEY]
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise DynamicEffortError(f"vllm_xargs.{_BIAS_KEY} must be a number")
        bias = float(raw)
        if not math.isfinite(bias):
            raise DynamicEffortError(f"vllm_xargs.{_BIAS_KEY} must be finite")
    deadline_ms = None
    if _DEADLINE_KEY in xargs and xargs[_DEADLINE_KEY] is not None:
        raw = xargs[_DEADLINE_KEY]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int | float)
            or not math.isfinite(float(raw))
            or raw <= 0
        ):
            raise DynamicEffortError(
                f"vllm_xargs.{_DEADLINE_KEY} must be a positive number"
            )
        deadline_ms = float(raw)
    if xargs.get(_FLOOR_KEY):
        raise DynamicEffortError(f"vllm_xargs.{_FLOOR_KEY} is not implemented")
    return {
        "ladder": ladder,
        "theta": theta,
        "bias": bias,
        "deadline_ms": deadline_ms,
    }


def append_to_last_user_message(messages: list[Any], sentence: str) -> bool:
    """Append `sentence` (after a blank line) to the last user turn in place.

    String content is extended; list-of-parts content gets a text part.
    Returns False when there is no user message.
    """
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if content is None or isinstance(content, str):
            msg["content"] = f"{content}\n\n{sentence}" if content else sentence
        elif isinstance(content, list):
            content.append({"type": "text", "text": sentence})
        else:
            raise DynamicEffortError(
                "dynamic reasoning_effort needs string or list user content"
            )
        return True
    return False


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
            "reasoning_effort='dynamic' conflicts with a static thinking_token_budget"
        )
    kwargs = request.chat_template_kwargs or {}
    if "enable_thinking" in kwargs and not kwargs["enable_thinking"]:
        raise DynamicEffortError(
            "reasoning_effort='dynamic' conflicts with "
            "chat_template_kwargs.enable_thinking=false"
        )
    overrides = build_dynamic_effort_overrides(cfg, request.vllm_xargs)
    if cfg.low_effort_sentence and not append_to_last_user_message(
        request.messages, cfg.low_effort_sentence
    ):
        raise DynamicEffortError(
            "reasoning_effort='dynamic' needs at least one user message"
        )
    request.reasoning_effort = cfg.render_effort  # type: ignore[assignment]
    request.thinking_token_budget = overrides["ladder"][0]
    request._dynamic_effort = overrides
