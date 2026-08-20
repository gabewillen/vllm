# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import field

from pydantic import Field

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config


QWEN_LOW_EFFORT_SENTENCE = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration."
)


@config
class DynamicEffortConfig:
    """Server defaults for `reasoning_effort: "dynamic"`.

    Every dynamic request starts at `ladder[0]` and the scheduler-side
    controller escalates one rung at a time from live signals; see
    `vllm/v1/core/sched/effort_controller.py` for the policy.
    """

    ladder: list[int] = field(default_factory=lambda: [1024, 4096, 16384, 65536])
    """Thinking-token caps per rung, strictly increasing."""
    check_at: float = Field(default=0.75, gt=0.0, lt=1.0)
    """Fraction of the current cap at which the escalation check fires."""
    final_check_at: float = Field(default=0.9, gt=0.0, lt=1.0)
    """Fraction of the current cap for the second (last) escalation check."""
    theta: list[float] | None = None
    """Escalation score threshold per transition (`len(ladder) - 1` entries).
    Defaults to `[0.0, 0.5, 1.0, ...]` (harder to climb the higher rungs)."""
    w_h: float = 1.0
    """Weight of z(H_fast) in the escalation score."""
    w_m: float = 1.0
    """Weight of -z(margin_ema) in the escalation score."""
    w_t: float = 0.5
    """Weight of the entropy-trend indicator `[H_fast >= H_slow]`."""
    w_a: float = 0.5
    """Weight of the MTP acceptance drop `(acc_base - acc_ema)`."""
    ema_fast_alpha: float = Field(default=0.3, gt=0.0, le=1.0)
    """EMA weight of a new sample for `H_fast`, `margin_ema` and `acc_ema`."""
    ema_slow_alpha: float = Field(default=0.05, gt=0.0, le=1.0)
    """EMA weight of a new sample for `H_slow`."""
    min_samples: int = Field(default=64, ge=1)
    """Signal samples (committed think tokens with signals) required before
    an escalation may fire."""
    acc_baseline_tokens: int = Field(default=256, ge=1)
    """Draft-token observations that form the per-request MTP acceptance
    baseline (`acc_base`)."""
    dwell_tokens: int = Field(default=128, ge=0)
    """Think tokens that must pass at a rung before its checks may fire."""
    cooldown_tokens: int = Field(default=0, ge=0)
    """Think tokens after an escalation before the next check may fire."""
    loop_ngram: int = Field(default=16, ge=2)
    """N-gram length for the degenerate-loop detector."""
    loop_repeats: int = Field(default=3, ge=2)
    """Repeats of an n-gram (or 32-token window) that flag a loop."""
    loop_window: int = Field(default=512, ge=32)
    """Think tokens the n-gram loop detector looks back over."""
    hash_window: int = Field(default=32, ge=2)
    """Length of the rolling token windows hashed for the loop detector."""
    hard_stop_margin: int = Field(default=32, ge=1)
    """Tokens of thinking left after a stall clamp (`cap = think + margin`)."""
    backtrack_markers: list[str] = field(
        default_factory=lambda: ["Wait", "Hmm", "Actually", "Let me re-check"]
    )
    """Self-correction phrases whose density is tracked as churn evidence."""
    marker_window: int = Field(default=256, ge=16)
    """Think tokens over which the backtrack-marker density is measured."""
    marker_max_rate: float = Field(default=0.05, gt=0.0)
    """Markers per token above which non-converging thinking counts as churn
    (vetoes escalation; never a hard stop)."""
    answer_reserve_tokens: int = Field(default=256, ge=0)
    """Tokens kept free below `max_tokens` for the answer after thinking;
    a rung whose cap cannot leave this reserve is never entered."""
    max_rung_by_batch_size: list[tuple[int, int, int]] | None = None
    """`(range_start, range_end, max_rung)` with inclusive batch-size ranges;
    the top rung is withheld under load. Batch sizes outside every range
    keep the full ladder."""
    floor_enabled: bool = False
    """Thinking-floor actuator (P5). Rejected while unimplemented."""
    calibration: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"entropy": (0.0, 1.0), "margin": (0.0, 1.0)}
    )
    """Per-signal `(mean, std)` used by the z-scores; keys `entropy`, `margin`."""
    render_effort: str = "medium"
    """`reasoning_effort` value handed to the chat template for dynamic
    requests (block-0-stable rendering)."""
    low_effort_sentence: str = QWEN_LOW_EFFORT_SENTENCE
    """Sentence appended to the last user turn of a dynamic request (the
    rung-0 prior). Empty disables the append."""
    default_effort: str | None = None
    """Effort applied when a request omits `reasoning_effort` (e.g.
    "dynamic"). `None` keeps the stock behaviour (template default thinking).
    Explicit values, including "none", are never overridden."""

    def __post_init__(self) -> None:
        if len(self.ladder) < 2:
            raise ValueError("dynamic_effort.ladder needs at least two rungs")
        if any(cap <= 0 for cap in self.ladder) or any(
            b <= a for a, b in zip(self.ladder, self.ladder[1:])
        ):
            raise ValueError(
                "dynamic_effort.ladder must be positive and strictly increasing"
            )
        if self.final_check_at <= self.check_at:
            raise ValueError("dynamic_effort.final_check_at must exceed check_at")
        if self.theta is None:
            self.theta = [0.5 * i for i in range(len(self.ladder) - 1)]
        if len(self.theta) != len(self.ladder) - 1:
            raise ValueError(
                "dynamic_effort.theta needs one entry per ladder transition "
                f"({len(self.ladder) - 1}), got {len(self.theta)}"
            )
        if any(not math.isfinite(t) for t in self.theta):
            raise ValueError("dynamic_effort.theta must be finite")
        if self.hash_window <= self.loop_ngram:
            raise ValueError("dynamic_effort.hash_window must exceed loop_ngram")
        if self.floor_enabled:
            raise ValueError("dynamic_effort.floor_enabled is not implemented")
        for key in ("entropy", "margin"):
            if key not in self.calibration:
                raise ValueError(f"dynamic_effort.calibration is missing '{key}'")
        for key, (mean, std) in self.calibration.items():
            if not (math.isfinite(mean) and math.isfinite(std)) or std <= 0.0:
                raise ValueError(
                    f"dynamic_effort.calibration['{key}'] needs finite mean and std > 0"
                )
        if self.max_rung_by_batch_size is not None:
            top = len(self.ladder) - 1
            prev_end = 0
            for start, end, rung in self.max_rung_by_batch_size:
                if start < 1 or end < start or start <= prev_end:
                    raise ValueError(
                        "dynamic_effort.max_rung_by_batch_size ranges must be "
                        "1-based, ordered and non-overlapping"
                    )
                if not 0 <= rung <= top:
                    raise ValueError(
                        f"dynamic_effort.max_rung_by_batch_size rung {rung} is "
                        f"outside [0, {top}]"
                    )
                prev_end = end

    @property
    def top_rung(self) -> int:
        return len(self.ladder) - 1

    def max_rung_for_batch_size(self, batch_size: int) -> int:
        """Highest rung allowed at `batch_size` (S11)."""
        if self.max_rung_by_batch_size:
            for start, end, rung in self.max_rung_by_batch_size:
                if start <= batch_size <= end:
                    return rung
        return self.top_rung


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
    """String forced when the thinking budget is exhausted."""
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

        if not reasoning_start_str or not reasoning_end_str:
            # If we don't have valid strings to tokenize,
            # we can't initialize the token IDs.
            return
        self._reasoning_start_token_ids = tokenizer.encode(
            reasoning_start_str, add_special_tokens=False
        )
        self._reasoning_end_token_ids = tokenizer.encode(
            reasoning_end_str, add_special_tokens=False
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
                f"reasoning_end_str='{self.reasoning_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
        self._enabled = True
