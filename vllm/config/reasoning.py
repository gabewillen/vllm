# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import field

from pydantic import Field

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.sample.soft_limit import (
    DEFAULT_CURVE,
    DEFAULT_MAX_BIAS,
    DEFAULT_RAMP_TOKENS,
)

logger = init_logger(__name__)


QWEN_LOW_EFFORT_SENTENCE = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration."
)

QWEN_GRACEFUL_FORCE_END_STR = (
    "\n\nConsidering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n</think>\n\n"
)
"""Qwen's own budget-forcing transition (docs/dynamic-reasoning.claude.md §5).
Forcing this instead of a bare `</think>` keeps the close in-distribution."""


@config
class SoftLimitConfig:
    """Soft-limit close: a ramped bias on the reasoning end token at the cap.

    Instead of forcing the end sequence the moment a request reaches its
    thinking cap, the first token of the *natural* end marker gets a bias that
    rises from 0 at the cap to `max_bias` `ramp_tokens` later, where the hard
    force takes over. Applies to dynamic and to static `thinking_token_budget`
    requests alike (docs/dynamic-reasoning.claude.md §5).
    """

    enabled: bool = True
    """Ramp the close instead of cutting at the cap. Off restores the pre-soft
    behaviour: the end sequence is forced at the cap itself."""
    ramp_tokens: int = Field(default=DEFAULT_RAMP_TOKENS, ge=0)
    """Think tokens between the cap and the hard force. The model has this many
    tokens to close on its own under a rising bias; 0 disables the ramp."""
    max_bias: float = Field(default=DEFAULT_MAX_BIAS, ge=0.0)
    """Logits added to the end token at the far end of the ramp. 10 makes the
    close overwhelmingly likely without being a hard mask, so a model that is
    mid-word can still finish it."""
    curve: float = Field(default=DEFAULT_CURVE, gt=0.0)
    """Exponent of the ramp: 1.0 linear, >1 keeps the bias small until late,
    <1 pushes early."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_bias) or not math.isfinite(self.curve):
            raise ValueError("soft_limit.max_bias and curve must be finite")


QWEN_HIGH_EFFORT_SENTENCE = (
    "Reasoning effort is set to high. Think this through carefully and "
    "completely before answering, checking your work as you go."
)


@config
class HiddenEffortConfig:
    """Prefill hidden-state routing of the *starting* rung (§13).

    The last prompt token's final hidden state - the vector `lm_head` consumes,
    produced by the prefill that was going to happen anyway - is matched against
    an online memory of the server's own finished requests, keyed by cosine and
    valued by the reasoning tokens each of them actually spent. Nothing is
    fitted; the cuts are percentile ranks of running digests.

    Off by default: a deployment that does not set this keeps the pre-v3
    behaviour, where every dynamic request starts at rung 0.
    """

    enabled: bool = False
    """Split the prompt at the effort sentence and choose the starting rung
    from the body's pooled hidden state."""
    shadow: bool = False
    """Compute and log the decision but always start at rung 0 (§13.8 step 1).
    The memory still fills, so a shadow day warms it for free."""
    memory_size: int = Field(default=4096, ge=16)
    """Entries in the ring. 512 already reach the measured AUC of 2048, so
    4096 is headroom; it costs `memory_size * hidden_size * 4` bytes of host
    RAM (84 MB at 4096 x 5120) and half that on disk."""
    min_entries: int = Field(default=128, ge=1)
    """Entries the memory needs before it may decide. Below this the request
    is rendered at rung 0 exactly as before, and the query result - which is
    still computed once there are `k` entries - only warms the digests."""
    k: int = Field(default=16, ge=1)
    """Neighbours the value estimate averages over."""
    temperature: float = Field(default=0.05, gt=0.0)
    """Softmax temperature on cosine similarity in the neighbour weights."""
    q_mid: float = Field(default=0.35, ge=0.0, le=1.0)
    """Estimate rank at or above which the request starts at rung 1."""
    q_high: float = Field(default=0.60, ge=0.0, le=1.0)
    """Estimate rank at or above which it starts at rung 2."""
    novelty_gate_q: float = Field(default=0.60, ge=0.0, le=1.0)
    """Novelty rank the *downward* band requires: above it the memory has
    nothing similar, so it cannot be trusted to say "easy"."""
    spread_gate_q: float = Field(default=0.60, ge=0.0, le=1.0)
    """Neighbour-disagreement rank the downward band requires."""
    digest_compression: float = Field(default=100.0, ge=10.0)
    """t-digest compression of the estimate / novelty / spread digests."""
    memory_path: str | None = None
    """`.npz` the memory is persisted to and warmed from. `None` keeps it in
    memory, so a restart starts cold."""
    flush_every: int = Field(default=256, ge=0)
    """Inserts between two writes of `memory_path`; 0 disables."""
    split_min_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    """Safety net: the fraction of the prompt the body must cover before the
    split is made at all. The effort sentence sits at the very tail of the
    prompt, so the body is normally everything the model reads except the
    sentence - but on a hybrid model the KV block can be wide (1648 tokens on
    the Qwen3.8 profile) and a non-final prefill chunk may only end on a block
    boundary, so the body is quantised down to one. A prompt with no boundary
    at or above this fraction takes no decision at all and runs at
    `default_level`, with a byte-identical prompt."""
    effort_sentences: list[str] | None = None
    """One prompt sentence per effort level, lowest first. `None` uses
    `[low, "", high]`: the model's own `low` and `xhigh` wording, with the
    middle level rendering no sentence at all - the chat template's `medium`.
    The sentence is the *whole* actuator: it is placed at the true tail of the
    prompt, after the last message, so the body before it is byte-identical
    across levels and one body per conversation is cached."""
    default_level: int = Field(default=0, ge=0)
    """Level a request gets when no decision can be made - a cold memory, a
    missing vector, a prompt with no usable seam. 0 is the `low` sentence,
    which is what `dynamic` rendered before the hidden-state decision existed,
    so a cold server behaves exactly as it did."""

    def __post_init__(self) -> None:
        if self.q_high < self.q_mid:
            raise ValueError("hidden_effort.q_high must be >= q_mid")
        if self.k > self.memory_size:
            raise ValueError("hidden_effort.k must not exceed memory_size")
        if self.effort_sentences is not None and len(self.effort_sentences) < 2:
            raise ValueError("hidden_effort.effort_sentences needs at least two levels")
        if self.default_level >= len(self.sentences()):
            raise ValueError("hidden_effort.default_level is outside the levels")

    def sentences(self) -> list[str]:
        """The tail sentence of each effort level, lowest first."""
        if self.effort_sentences is not None:
            return list(self.effort_sentences)
        return [QWEN_LOW_EFFORT_SENTENCE, "", QWEN_HIGH_EFFORT_SENTENCE]


@config
class DynamicEffortConfig:
    """Server defaults for `reasoning_effort: "dynamic"`.

    A dynamic request is given one effort **level** before it thinks, chosen
    from its own pooled prefill hidden state (`hidden_effort`), and rendered as
    that level's sentence at the tail of the prompt. Nothing else touches the
    think block: no thinking cap, no forced close, no mid-generation
    escalation and no stall detector. The model ends its own reasoning, bounded
    only by the client's `max_tokens` and timeouts, exactly as a fixed effort
    level is.
    """

    hidden_effort: HiddenEffortConfig = field(default_factory=HiddenEffortConfig)
    """Which level a request gets, and the memory that decides it."""
    render_effort: str = "medium"
    """`reasoning_effort` value handed to the chat template for dynamic
    requests, so block 0 of the prompt is identical for every level and the
    level lives only in the tail sentence."""

    @property
    def level_sentences(self) -> list[str]:
        return self.hidden_effort.sentences()

    @property
    def num_levels(self) -> int:
        return len(self.level_sentences)


@config
class ReasoningConfig:
    """Configuration for reasoning models.

    Set `reasoning_start_str` and `reasoning_end_str` to the strings used to
    enter and forcibly terminate reasoning. The end string may include a
    transition phrase before the parser's natural reasoning end marker. Token
    IDs are derived automatically by `initialize_token_ids`.
    """

    reasoning_parser: str = ""
    """The name of the ReasoningParser to use for this model."""
    reasoning_start_str: str = ""
    """String that indicates the start of reasoning."""
    reasoning_end_str: str = ""
    """String that ends reasoning; used for *detection* and, unless
    `force_end_str` is set, also as the forced close."""
    force_end_str: str = ""
    """String forced when an explicit `thinking_token_budget` is exhausted.
    Empty falls back to `reasoning_end_str`. Splitting the two lets the forced
    close be an in-distribution transition phrase while detection stays on the
    bare end marker (docs/dynamic-reasoning.claude.md §5). `dynamic` never
    forces a close, so this does not apply to it."""
    dynamic_effort: DynamicEffortConfig | None = None
    """Server defaults for `reasoning_effort: "dynamic"`; `None` rejects it."""

    _reasoning_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `reasoning_start_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""
    _reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for forced reasoning end token IDs."""
    _natural_reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Token IDs that naturally terminate reasoning, as defined by the parser."""

    _enabled: bool = field(default=False, init=False, repr=False)
    """Private field indicating whether reasoning token IDs have been initialized.
    Set to True by `initialize_token_ids` once token IDs are initialized."""

    @property
    def enabled(self) -> bool:
        """Returns True if reasoning is enabled (i.e. if token IDs have been
        initialized), False otherwise."""
        return self._enabled

    @property
    def reasoning_start_token_ids(self) -> list[int] | None:
        """Token IDs derived from `reasoning_start_str`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._reasoning_start_token_ids

    @property
    def reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs forced when the thinking budget is exhausted."""
        return self._reasoning_end_token_ids

    @property
    def natural_reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs that indicate the model naturally ended reasoning."""
        return self._natural_reasoning_end_token_ids

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Initialize reasoning token IDs from strings using the tokenizer."""
        if (
            self._reasoning_start_token_ids is not None
            and self._reasoning_end_token_ids is not None
            and self._natural_reasoning_end_token_ids is not None
        ):
            self._enabled = True
            return  # Already initialized

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        reasoning_start_str = self.reasoning_start_str
        reasoning_end_str = self.reasoning_end_str
        natural_reasoning_end_str = ""
        if self.reasoning_parser:
            parser_cls = ReasoningParserManager.get_reasoning_parser(
                self.reasoning_parser
            )
            reasoning_parser = parser_cls(tokenizer)
            start_token = reasoning_parser.reasoning_start_str
            if start_token and not reasoning_start_str:
                reasoning_start_str = start_token

            end_token = reasoning_parser.reasoning_end_str
            if end_token and not reasoning_end_str:
                reasoning_end_str = end_token
            natural_reasoning_end_str = end_token or ""

        if not natural_reasoning_end_str:
            natural_reasoning_end_str = reasoning_end_str

        force_end_str = self.force_end_str
        if not force_end_str:
            force_end_str = reasoning_end_str
        if natural_reasoning_end_str and not force_end_str.endswith(
            natural_reasoning_end_str.rstrip()
        ):
            logger.warning(
                "ReasoningConfig: the forced close %r does not end with the "
                "detected reasoning end marker %r; reasoning may never close.",
                force_end_str,
                natural_reasoning_end_str,
            )

        if not reasoning_start_str or not force_end_str:
            # If we don't have valid strings to tokenize,
            # we can't initialize the token IDs.
            return
        self._reasoning_start_token_ids = tokenizer.encode(
            reasoning_start_str, add_special_tokens=False
        )
        self._reasoning_end_token_ids = tokenizer.encode(
            force_end_str, add_special_tokens=False
        )
        self._natural_reasoning_end_token_ids = tokenizer.encode(
            natural_reasoning_end_str, add_special_tokens=False
        )

        if (
            not self._reasoning_start_token_ids
            or not self._reasoning_end_token_ids
            or not self._natural_reasoning_end_token_ids
        ):
            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: "
                f"reasoning_start_str='{self.reasoning_start_str}', "
                f"reasoning_end_str='{self.reasoning_end_str}', "
                f"force_end_str='{self.force_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
        self._enabled = True
