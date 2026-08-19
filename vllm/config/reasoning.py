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
from vllm.v1.sample.effort_policy import DEFAULT_P_UNCERTAIN
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
    split_min_fraction: float = Field(default=0.75, ge=0.0, le=1.0)
    """Fraction of the prompt the body must cover before the *two-phase* form
    (which also chooses the rung's prompt sentence) is used instead of the
    cap-only form. An agent turn that ends in a tool result puts the effort
    sentence - and so the seam - thousands of tokens from the end of the
    prompt, and a wide KV block quantises the boundary further; below this
    fraction the vector would describe a small prefix of what the model reads,
    and reading the whole prompt and moving only the cap is better informed."""
    effort_sentences: list[str] | None = None
    """One prompt sentence per rung, appended to the *tail* of the last user
    turn. `None` uses `[low, "", high]` padded to the ladder with `""`
    (the template's own `medium` rendering). The body before the sentence is
    byte-identical across rungs, so one body per conversation is cached."""

    def __post_init__(self) -> None:
        if self.q_high < self.q_mid:
            raise ValueError("hidden_effort.q_high must be >= q_mid")
        if self.k > self.memory_size:
            raise ValueError("hidden_effort.k must not exceed memory_size")

    def sentences_for(self, ladder_len: int) -> list[str]:
        """The per-rung tail sentences, padded/truncated to the ladder."""
        if self.effort_sentences is not None:
            base = list(self.effort_sentences)
        else:
            base = [QWEN_LOW_EFFORT_SENTENCE, "", QWEN_HIGH_EFFORT_SENTENCE]
        if len(base) < ladder_len:
            base += [base[-1] if base else ""] * (ladder_len - len(base))
        return base[:ladder_len]


@config
class DynamicEffortConfig:
    """Server defaults for `reasoning_effort: "dynamic"`.

    Every dynamic request starts at `ladder[0]` and the scheduler-side
    controller escalates one rung at a time from live signals; see
    `vllm/v1/core/sched/effort_controller.py` for the policy.
    """

    ladder: list[int] = field(default_factory=lambda: [1024, 4096, 16384])
    """Thinking-token caps per rung, strictly increasing. The pre-P6 default
    had a fourth 65536 rung; across 1199 measured requests nothing ever passed
    16384 think tokens and only 5 passed 4096 (docs/dynamic-reasoning-v3-
    analysis.md §3), so the top rung was dead weight. It stays configurable."""
    rule: str = "length"
    """Which escalation rule decides a check point.

    `length` (default, P7): termination/length-based - escalate when the
    request is *still* in the think block at the check point, is not looping or
    churning, is not converging (p(reasoning end) rising) and passes the MTP
    corroboration veto. The entropy/margin percentile-rank features are added
    on top **only** when the model's calibration file reports a discriminative
    AUC of at least `uncertainty_min_auc`; the v3 measurement found them at
    chance on Qwen3.8 (0.41-0.54, length controlled), so by default they are
    inert rather than silently load-bearing.
    `rank` (P6): always applies the rank features, whatever the calibration
    says. `score`: the pre-P6 weighted z-score against the fixed `calibration`
    table (deprecated)."""
    uncertainty_min_auc: float = Field(default=0.60, ge=0.0, le=1.0)
    """Discriminative AUC (from `quantile_path`, written by
    `serve-configs/effort_calibrate.py`) the entropy/margin features must reach
    on *this* model before `rule="length"` consults them. No AUC in the file
    means no evidence, which means the features stay off."""
    soft_limit: SoftLimitConfig = field(default_factory=SoftLimitConfig)
    """Ramped close at the cap instead of a hard cut; see `SoftLimitConfig`.
    Honoured by both actuators and by static `thinking_token_budget`
    requests."""
    hidden_effort: HiddenEffortConfig = field(default_factory=HiddenEffortConfig)
    """Prefill hidden-state routing of the starting rung; see
    `HiddenEffortConfig`. Off by default."""
    evaluation: str = "worker"
    """Where the escalation rule runs. `worker` (default) evaluates it in the
    V2 sampler next to the cap actuator, so a decision can never arrive late;
    `scheduler` keeps the pre-P6 scheduler-side path. The V1 runner always
    falls back to `scheduler`."""
    check_at: float = Field(default=0.75, gt=0.0, lt=1.0)
    """Fraction of the current cap at which the escalation check fires."""
    final_check_at: float = Field(default=0.9, gt=0.0, lt=1.0)
    """Fraction of the current cap for the second (last) escalation check."""
    p_uncertain: list[float] | None = None
    """Uncertainty **percentile rank** required to climb each rung, the only
    tunable of the rank rule. Defaults to `[0.85, 0.92, 0.96, ...]`: the v3
    measurement found entropy/margin near chance at separating requests that
    needed more thinking (AUC 0.41-0.54 with length controlled), so the rule is
    deliberately conservative and the p(end) grace window carries the load."""
    quantile_path: str | None = None
    """JSON file the per-model quantile sketches are persisted to and warmed
    from at startup. `None` keeps them in memory (cold after every restart)."""
    quantile_min_samples: int = Field(default=2048, ge=1)
    """Observations a signal sketch needs before any request may escalate."""
    quantile_compression: float = Field(default=100.0, ge=10.0)
    """t-digest compression of the sketches."""
    quantile_flush_every: int = Field(default=5000, ge=0)
    """Observations between two writes of `quantile_path`; 0 disables."""
    quantile_edges: int = Field(default=33, ge=2)
    """Points of the monotone quantile grid shipped to the worker; the worker
    turns a signal into a rank by searching this grid."""
    baseline_tokens: int = Field(default=128, ge=1)
    """Think tokens that form the within-request signal baseline; escalation
    keys on the rise over it, not on the absolute level."""
    baseline_rise: float = Field(default=0.10, ge=0.0, le=1.0)
    """Uncertainty-rank rise over the request's own baseline required to
    escalate."""
    grace_tokens: int = Field(default=256, ge=0)
    """Pre-soft-limit P6 mechanism: think tokens granted once, near the cap,
    when p(reasoning end) is rising. The soft-limit ramp grants the same room
    *unconditionally* and biases the close on top, so it subsumes this window;
    the scheduler zeroes `grace_tokens` while `soft_limit` is active and this
    setting only takes effect with `soft_limit.enabled = false`."""
    p_end_rise_eps: float = Field(default=0.0, ge=0.0)
    """How much the fast p(end) EMA must exceed the slow one to count as
    rising."""
    acc_veto_rank: float = Field(default=0.85, gt=0.0, le=1.0)
    """MTP acceptance rank above which escalation is vetoed (the drafter
    predicts the target, so the text is predictable). Corroboration only."""
    novelty_ngram: int = Field(default=8, ge=2)
    """N-gram length of the language-agnostic novelty (churn) detector."""
    novelty_window: int = Field(default=256, ge=16)
    """Think tokens the novelty rate is measured over."""
    novelty_min_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    """Distinct-new-n-grams / total below which the window counts as churn
    and vetoes escalation."""
    backtrack_marker_weight: float = Field(default=0.0, ge=0.0)
    """**Legacy, off by default.** Weight of the backtrack-marker density in
    the churn evidence. A lexical marker list is exactly what
    docs/dynamic-reasoning.claude.md §11.0 rules out - it is English-only and
    model-specific - so the churn detector runs on the language-agnostic n-gram
    novelty rate instead. 0 disables the markers entirely; the setting survives
    only so an existing deployment can reproduce a pre-P6 decision."""
    graceful_force_end: bool = True
    """Force `force_end_str` (an in-distribution transition phrase ending in
    the model's own end marker) instead of a bare end marker."""
    force_end_str: str = QWEN_GRACEFUL_FORCE_END_STR
    """The forced close. Detection keeps `ReasoningConfig.reasoning_end_str`."""
    theta: list[float] | None = None
    """Deprecated (pre-P6 `rule="score"`): escalation score threshold per
    transition. Defaults to `[0.0, 0.5, 1.0, ...]`."""
    w_h: float = 1.0
    """Deprecated (`rule="score"`): weight of z(H_fast)."""
    w_m: float = 1.0
    """Deprecated (`rule="score"`): weight of -z(margin_ema)."""
    w_t: float = 0.5
    """Deprecated (`rule="score"`): weight of the entropy trend."""
    w_a: float = 0.5
    """Deprecated (`rule="score"`): weight of the MTP acceptance drop."""
    ema_fast_alpha: float = Field(default=0.3, gt=0.0, le=1.0)
    """EMA weight of a new sample for `H_fast`, `margin_ema` and `acc_ema`."""
    ema_slow_alpha: float = Field(default=0.05, gt=0.0, le=1.0)
    """EMA weight of a new sample for `H_slow` and the slow p(end) EMA."""
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
    """Tokens of thinking left after a stall clamp. The clamp aims the
    actuator's *force point*, so with `soft_limit` on the cap is set to
    `think + margin - ramp_tokens` and the close still lands `margin` tokens
    out (floored: a loop found inside the first ramp closes at
    `ramp_tokens`)."""
    backtrack_markers: list[str] = field(
        default_factory=lambda: ["Wait", "Hmm", "Actually", "Let me re-check"]
    )
    """Legacy (see `backtrack_marker_weight`): self-correction phrases whose
    density is tracked as churn evidence when the weight is non-zero. Never
    consulted at the default weight of 0."""
    marker_window: int = Field(default=256, ge=16)
    """Think tokens over which the backtrack-marker density is measured."""
    marker_max_rate: float = Field(default=0.05, gt=0.0)
    """Legacy: markers per token above which non-converging thinking counts as
    churn. Only consulted when `backtrack_marker_weight > 0`."""
    answer_reserve_tokens: int = Field(default=256, ge=0)
    """Tokens kept free below `max_tokens` for the answer after thinking;
    a rung whose cap cannot leave this reserve is never entered. The
    `soft_limit` ramp is thinking too, so it is subtracted from the same
    headroom."""
    max_rung_by_batch_size: list[tuple[int, int, int]] | None = None
    """`(range_start, range_end, max_rung)` with inclusive batch-size ranges;
    the top rung is withheld under load. Batch sizes outside every range
    keep the full ladder."""
    floor_enabled: bool = False
    """Thinking-floor actuator (P5). Rejected while unimplemented."""
    calibration: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"entropy": (0.0, 1.0), "margin": (0.0, 1.0)}
    )
    """Deprecated (`rule="score"`): per-signal `(mean, std)` for the z-scores;
    keys `entropy`, `margin`. The rank rule replaces it with running sketches."""
    render_effort: str = "medium"
    """`reasoning_effort` value handed to the chat template for dynamic
    requests (block-0-stable rendering)."""
    low_effort_sentence: str = QWEN_LOW_EFFORT_SENTENCE
    """Sentence appended to the last user turn of a dynamic request (the
    rung-0 prior). Empty disables the append."""

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
        if self.rule not in ("length", "rank", "score"):
            raise ValueError("dynamic_effort.rule must be 'length', 'rank' or 'score'")
        if self.evaluation not in ("worker", "scheduler"):
            raise ValueError(
                "dynamic_effort.evaluation must be 'worker' or 'scheduler'"
            )
        num_transitions = len(self.ladder) - 1
        if self.p_uncertain is None:
            defaults = list(DEFAULT_P_UNCERTAIN)
            self.p_uncertain = [
                defaults[min(i, len(defaults) - 1)] for i in range(num_transitions)
            ]
        if len(self.p_uncertain) != num_transitions:
            raise ValueError(
                "dynamic_effort.p_uncertain needs one entry per ladder transition "
                f"({num_transitions}), got {len(self.p_uncertain)}"
            )
        if any(not 0.0 < p <= 1.0 for p in self.p_uncertain):
            raise ValueError("dynamic_effort.p_uncertain entries must be in (0, 1]")
        if self.graceful_force_end and not self.force_end_str:
            raise ValueError(
                "dynamic_effort.graceful_force_end needs a non-empty force_end_str"
            )
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
        if self.novelty_window <= self.novelty_ngram:
            raise ValueError("dynamic_effort.novelty_window must exceed novelty_ngram")
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

    def uncertainty_features(self, auc: float | None) -> tuple[bool, str]:
        """Whether the entropy/margin rank features are consulted, and why.

        Args:
            auc: the discriminative AUC recorded for this model in the
                calibration file, or `None` when it holds none.

        Returns:
            `(active, reason)`; `reason` is a short phrase for the startup log.
        """
        if self.rule == "score":
            return True, "rule='score' scores the z-features directly"
        if self.rule == "rank":
            return True, "rule='rank' always applies them"
        if auc is None:
            return False, (
                "no discriminative AUC in the calibration file (run "
                "serve-configs/effort_calibrate.py build)"
            )
        if auc >= self.uncertainty_min_auc:
            return True, f"calibration AUC {auc:.3f} >= {self.uncertainty_min_auc:.2f}"
        return False, f"calibration AUC {auc:.3f} < {self.uncertainty_min_auc:.2f}"


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
    """String forced when the thinking budget is exhausted. Empty falls back
    to `dynamic_effort.force_end_str` (when `graceful_force_end` is on) and
    then to `reasoning_end_str`. Splitting the two lets the forced close be an
    in-distribution transition phrase while detection stays on the bare end
    marker (docs/dynamic-reasoning.claude.md §5)."""
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
        if (
            not force_end_str
            and self.dynamic_effort is not None
            and self.dynamic_effort.graceful_force_end
        ):
            force_end_str = self.dynamic_effort.force_end_str
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
