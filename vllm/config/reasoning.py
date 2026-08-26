# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig

VALID_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "dynamic"}
)
"""The `reasoning_effort` values the chat completion API accepts; kept in step
with `ChatCompletionRequest.reasoning_effort`."""

LOW_EFFORT_SENTENCE = "Reasoning effort is set to low."
"""The resting level's whole tail. Measured 2026-08-23 on the 12-prompt grid
(2 reps): 22/24 correct at 611 avg tokens vs 21/24 at 774 with no sentence;
every upward word (high/xhigh/max) lost accuracy at 1.5-2x tokens with cap
runaways, so there is no upward level."""

OFF_VOTE_PROMPT = (
    "Could you answer the request above correctly with no step-by-step "
    "working? Reply with one word: yes or no."
)
"""The hidden question that gates a think-off verdict. Sampled `off_votes`
times at temperature 0.7; only a unanimous yes skips thinking. Measured
2026-08-23: unanimity-of-3 had zero wrong-offs on the grid, a single vote
wrongly said yes on arithmetic."""

LEVEL_VOTE_PROMPT = (
    "How much step-by-step working does the request above need to answer "
    "correctly? Reply with one word: none, brief, or extended."
)
"""The hidden question behind `level_vote`: the model names its own level.
One word per level, lowest first, aligned with the level sentences."""

DEFAULT_LEVEL_VOTE_WORDS = ["none", "brief", "extended"]
"""Answer words of `LEVEL_VOTE_PROMPT` for the think-off / low / medium
ladder; a ladder without a think-off level drops `none`."""


@config
class HiddenEffortConfig:
    """Prefill hidden-state routing of the effort level (§13).

    The last prompt token's final hidden state - the vector `lm_head` consumes,
    produced by the prefill that was going to happen anyway - is matched against
    an online memory of the server's own finished requests, keyed by cosine and
    valued by the reasoning tokens each of them actually spent. Nothing is
    fitted; the cuts are percentile ranks of running digests.

    Off by default: a deployment that does not set this renders every dynamic
    request at `default_level`.
    """

    enabled: bool = False
    """Split the prompt at the effort sentence and choose the level from the
    body's pooled hidden state."""
    tail_after_tool_result: bool = True
    """Also decide and render the level sentence when the conversation ends on
    a tool result (an agent loop continuation). False renders those requests
    at `render_effort` with no sentence and no decision: the level is only set
    on the user's own turns."""
    shadow: bool = False
    """Compute and log the decision but always render `default_level` (§13.8
    step 1). The memory still fills, so a shadow day warms it for free."""
    memory_size: int = Field(default=4096, ge=16)
    """Entries in the ring. 512 already reach the measured AUC of 2048, so
    4096 is headroom; it costs `memory_size * hidden_size * 4` bytes of host
    RAM (84 MB at 4096 x 5120) and half that on disk."""
    k: int = Field(default=16, ge=1)
    """Neighbours the value estimate averages over."""
    min_entries: int | None = Field(default=None, ge=1)
    """Entries the memory needs before it may decide. `None` means 1: the
    second request is already decided from the first, bluntly - small-lane
    ranks are smoothed toward the middle and sharpen as the lanes fill - and
    the calibration pulls an unproven estimate toward the mean. Only an empty
    memory renders `default_level`."""
    temperature: float = Field(default=0.05, gt=0.0)
    """Softmax temperature on cosine similarity in the neighbour weights."""
    q_mid: float = Field(default=0.35, ge=0.0, le=1.0)
    """Neighbour difficulty (within-level spend percentile, 0..1) at or
    above which the request leaves the resting low level for the middle."""
    q_high: float = Field(default=0.60, ge=0.0, le=1.0)
    """Neighbour difficulty at or above which it gets the top level."""
    spread_gate_q: float = Field(default=0.60, ge=0.0, le=1.0)
    """Kept for telemetry; the spread rank is reported, not cut on."""
    probe_every: int = Field(default=8, ge=0)
    """Every N-th decision above low renders one level lower, so lanes keep
    receiving cheaper samples and neighbourhoods can be pulled down. 0
    disables."""
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
    `[low, "", xhigh]`: the model's own `low` and `xhigh` wording, with the
    middle level rendering no sentence at all - the chat template's `medium`.
    The sentence is the *whole* actuator: it is placed at the true tail of the
    prompt, after the last message, so the body before it is byte-identical
    across levels and one body per conversation is cached."""
    default_level: int = Field(default=0, ge=0)
    """Level a request gets when no decision can be made - a cold memory, a
    missing vector, a prompt with no usable seam. 0 is the `low` sentence,
    or the think-off level when `think_off_level` is set (then 1 is `low`).
    The default may not be the think-off level."""
    think_off_level: bool = False
    """Add a bottom level that renders the chat template's
    `enable_thinking=false` - an instant answer with no think block. The
    memory sends a request there when its neighbours' difficulty is below
    `q_none`. A think-off request has no thinking length to rank, so its
    memory entry carries the difficulty it was decided with: it keeps the
    neighbourhood where it is and adds no confidence; the probe clock renders
    every `probe_every`-th think-off verdict at `low` instead, and that
    evidence is what can lift the neighbourhood back."""
    q_none: float = Field(default=0.15, ge=0.0, le=1.0)
    """Neighbour difficulty below which a request skips thinking entirely
    (only with `think_off_level`)."""
    off_vote: bool = True
    """Gate every think-off verdict on the model itself: before thinking is
    skipped, the engine asks `OFF_VOTE_PROMPT` hidden from the client,
    `off_votes` times. Only a unanimous yes renders think-off; any dissent
    demotes the request to the resting low level. Free-text guidance was
    removed after a fluent wrong note anchored an implementation on a wrong
    contract - the model's voice is effort-only, one word."""
    off_votes: int = Field(default=3, ge=1)
    """Votes the off gate samples. Each is a short hidden generation over the
    fully cached prompt at a different seed."""
    off_vote_max_tokens: int = Field(default=8, ge=1)
    """Cap on one vote's generation; an unclassifiable vote counts as no."""
    level_vote: bool = False
    """Let the model choose its own level: the hidden question is
    `level_vote_prompt` instead of `OFF_VOTE_PROMPT`, its first-token
    distribution over `level_vote_words` is sampled `off_votes` times at the
    vote temperature, and `level_vote_rule` picks the level from the draws.
    The memory still records and reports but no longer decides; a forced
    level (`vllm_xargs.dynamic_effort_level`) still overrides. Mass on any
    other token counts as `default_level`, so an unparseable answer never
    lowers the effort."""
    level_vote_prompt: str = LEVEL_VOTE_PROMPT
    """The hidden question `level_vote` asks."""
    level_vote_words: list[str] | None = None
    """One answer word per level, lowest first, aligned with `sentences()`.
    `None` derives them from `DEFAULT_LEVEL_VOTE_WORDS`, dropping `none`
    when there is no think-off level."""
    level_vote_rule: Literal["max", "median"] = "max"
    """`max`: the highest level any draw chose - the level rises on a single
    draw and only falls by consensus, the same conservatism as the unanimous
    off gate. `median`: the middle draw."""

    def __post_init__(self) -> None:
        if self.q_high < self.q_mid:
            raise ValueError("hidden_effort.q_high must be >= q_mid")
        if self.k > self.memory_size:
            raise ValueError("hidden_effort.k must not exceed memory_size")
        if self.effort_sentences is not None and len(self.effort_sentences) < 2:
            raise ValueError("hidden_effort.effort_sentences needs at least two levels")
        if self.default_level >= len(self.sentences()):
            raise ValueError("hidden_effort.default_level is outside the levels")
        if self.think_off_level and self.default_level == 0:
            raise ValueError(
                "hidden_effort.default_level may not be the think-off level"
            )
        if self.think_off_level and self.q_none > self.q_mid:
            raise ValueError("hidden_effort.q_none must be <= q_mid")
        if self.level_vote:
            self.vote_words()

    def vote_words(self) -> list[str]:
        """The `level_vote` answer word of each level, lowest first.

        Raises:
            ValueError: when the words do not align 1:1 with the levels.
        """
        num_levels = len(self.sentences())
        if self.level_vote_words is not None:
            words = list(self.level_vote_words)
        else:
            words = list(DEFAULT_LEVEL_VOTE_WORDS)
            if not self.think_off_level:
                words = words[1:]
        if len(words) != num_levels:
            raise ValueError(
                f"hidden_effort.level_vote_words needs one word per level "
                f"({num_levels}), got {len(words)}"
            )
        if any(not w.strip() for w in words) or len(set(words)) != len(words):
            raise ValueError("hidden_effort.level_vote_words must be distinct words")
        return words

    def sentences(self) -> list[str | None]:
        """The tail sentence of each effort level, lowest first. `None` is the
        think-off level: no sentence, `enable_thinking=false`."""
        if self.effort_sentences is not None:
            base: list[str | None] = list(self.effort_sentences)
        else:
            base = [LOW_EFFORT_SENTENCE, ""]
        return ([None] if self.think_off_level else []) + base

    @property
    def low_level(self) -> int:
        """Index of the resting `low` level."""
        return 1 if self.think_off_level else 0


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
    default_effort: str | None = None
    """`reasoning_effort` to assume when a chat completion **omits** it.

    `None` (the default) keeps today's behaviour: an omitted value reaches the
    chat template as `None` and the template picks its own default, which for
    Qwen3.8 is `xhigh` - the most expensive level there is. Setting this to
    `"dynamic"` makes the omitted case route itself instead. An explicit
    `reasoning_effort` on the request is never overridden, including `"none"`.
    A deployment that wants the template default back sets it to `None`."""
    effort_aliases: dict[str, str] = field(default_factory=dict)
    """Client `reasoning_effort` values rewritten before anything reads them,
    e.g. `{"high": "xhigh"}` for a proxy that normalises levels to the OpenAI
    set on the way through. Applied to explicit values only."""
    render_effort: str = "medium"
    """`reasoning_effort` value handed to the chat template for dynamic
    requests, so block 0 of the prompt is identical for every level and the
    level lives only in the tail sentence."""

    def __post_init__(self) -> None:
        if self.default_effort is not None and self.default_effort not in (
            VALID_REASONING_EFFORTS
        ):
            raise ValueError(
                f"dynamic_effort.default_effort must be null or one of "
                f"{sorted(VALID_REASONING_EFFORTS)}, got {self.default_effort!r}"
            )
        for src, dst in self.effort_aliases.items():
            if dst not in VALID_REASONING_EFFORTS or dst in self.effort_aliases:
                raise ValueError(
                    f"dynamic_effort.effort_aliases[{src!r}] must map to one of "
                    f"{sorted(VALID_REASONING_EFFORTS)} that is not itself "
                    f"aliased, got {dst!r}"
                )

    @property
    def level_sentences(self) -> list[str | None]:
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
    """String forced when the thinking budget is exhausted."""
    dynamic_effort: DynamicEffortConfig | None = None
    """Server defaults for `reasoning_effort: "dynamic"`; `None` rejects it.

    Dynamic effort requires data-parallel size one because its routing memory
    and shutdown persistence have one process owner.
    """

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

    def verify_with_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject dynamic routing where each DP rank owns independent state."""
        if self.dynamic_effort is not None and parallel_config.data_parallel_size > 1:
            raise ValueError(
                "dynamic effort is not supported with data parallelism; "
                "use data_parallel_size=1"
            )
