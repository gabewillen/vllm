# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Soft-limit close for the thinking budget (docs/dynamic-reasoning.claude.md §5).

A hard force at the cap cuts the model off mid-sentence. Instead, from the cap
onward the *first token of the natural reasoning end sequence* gets a rising
logit bias

    bias(t) = max_bias * clamp((t - cap) / ramp_tokens, 0, 1) ** curve

over the ``ramp_tokens`` tokens after the cap, and only at ``cap + ramp_tokens``
does the existing hard force fire. A model that was already close to done
closes on its own inside the ramp (``close_kind = "soft"``) and nothing is
forced; a model that keeps going still terminates, deterministically, one ramp
later.

The rule is pure arithmetic on the request's think position, so the V1
``ThinkingBudgetStateHolder``, the V2 Triton kernel and its torch reference all
implement the same three lines and the CPU tests are the specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_RAMP_TOKENS = 256
DEFAULT_MAX_BIAS = 10.0
DEFAULT_CURVE = 1.0

CLOSE_NATURAL = "natural"
CLOSE_SOFT = "soft"
CLOSE_FORCED = "forced"


@dataclass(frozen=True)
class SoftLimit:
    """Resolved soft-limit parameters, as the actuators see them."""

    enabled: bool = False
    ramp_tokens: int = DEFAULT_RAMP_TOKENS
    max_bias: float = DEFAULT_MAX_BIAS
    curve: float = DEFAULT_CURVE

    @property
    def active(self) -> bool:
        """The ramp only exists when it is on, non-empty and biasing."""
        return self.enabled and self.ramp_tokens > 0 and self.max_bias > 0.0

    @property
    def ramp(self) -> int:
        """Tokens between the cap and the hard force; 0 when inactive."""
        return self.ramp_tokens if self.active else 0

    def bias(self, think_count: int, cap: int) -> float:
        """The logit bias for a row whose think prefix holds ``think_count``."""
        return soft_limit_bias(
            think_count, cap, self.ramp_tokens, self.max_bias, self.curve
        )


def soft_limit_bias(
    think_count: int,
    cap: int,
    ramp_tokens: int,
    max_bias: float,
    curve: float = DEFAULT_CURVE,
) -> float:
    """``max_bias * clamp((think_count - cap) / ramp_tokens, 0, 1) ** curve``."""
    if ramp_tokens <= 0:
        return 0.0
    x = (think_count - cap) / float(ramp_tokens)
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    if curve != 1.0:
        x = x**curve
    return max_bias * x


def classify_close(think_count: int, cap: int, ramp_tokens: int) -> str:
    """How a think block ended, from its reasoning-token count.

    Args:
        think_count: reasoning tokens before the end sequence.
        cap: the request's thinking cap at the close.
        ramp_tokens: the soft-limit ramp; ``0`` when the soft limit is off.

    Returns:
        ``"natural"`` (closed before the bias could do anything - the bias is
        exactly 0 at the cap), ``"soft"`` (closed inside the ramp, so the bias
        carried it) or ``"forced"`` (the hard force fired).
    """
    if ramp_tokens <= 0:
        return CLOSE_FORCED if think_count >= cap else CLOSE_NATURAL
    if think_count >= cap + ramp_tokens:
        return CLOSE_FORCED
    if think_count > cap:
        return CLOSE_SOFT
    return CLOSE_NATURAL


def soft_limit_from_config(soft_limit_config: Any | None) -> SoftLimit:
    """Build a :class:`SoftLimit` from a `DynamicEffortConfig.soft_limit`."""
    if soft_limit_config is None:
        return SoftLimit(enabled=False)
    return SoftLimit(
        enabled=bool(getattr(soft_limit_config, "enabled", False)),
        ramp_tokens=int(getattr(soft_limit_config, "ramp_tokens", DEFAULT_RAMP_TOKENS)),
        max_bias=float(getattr(soft_limit_config, "max_bias", DEFAULT_MAX_BIAS)),
        curve=float(getattr(soft_limit_config, "curve", DEFAULT_CURVE)),
    )


def soft_limit_from_reasoning_config(reasoning_config: Any | None) -> SoftLimit:
    """Resolve the soft limit a runner should apply.

    The parameters live under ``dynamic_effort``; a server without a
    ``dynamic_effort`` block has no soft limit, so plain static
    ``thinking_token_budget`` deployments keep the exact hard-cap semantics.
    """
    if reasoning_config is None:
        return SoftLimit(enabled=False)
    dynamic_effort = getattr(reasoning_config, "dynamic_effort", None)
    if dynamic_effort is None:
        return SoftLimit(enabled=False)
    return soft_limit_from_config(getattr(dynamic_effort, "soft_limit", None))
