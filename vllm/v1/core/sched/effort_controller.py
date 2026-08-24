# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic reasoning-effort state: one level per request, chosen before thinking.

`reasoning_effort: "dynamic"` picks an effort **level** from the prompt's own
pooled prefill hidden state, before the model has produced a single reasoning
token, and renders that level's sentence at the tail of the prompt
(docs/dynamic-reasoning.claude.md §13).

That sentence is the whole actuator. Nothing here touches the think block: no
thinking cap, no forced close, no mid-generation escalation, no stall detector.
The model ends its own reasoning, bounded only by the client's `max_tokens` and
timeouts - exactly as it is at a fixed effort level. What is left is
bookkeeping: how many reasoning tokens the request spent and whether it closed
its think block itself, which is what the memory needs in order to value the
entry (a request cut off by the client is right-censored).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.config.reasoning import DynamicEffortConfig

CLOSE_NATURAL = "natural"
"""The model ended its own think block."""
CLOSE_CLIENT_LIMIT = "client-limit"
"""The request ran out of `max_tokens`, was aborted, or otherwise ended while
still inside the think block. The length it would have spent is unknown."""


@dataclass
class EffortEvent:
    """Per-step observations for one request (all CPU scalars)."""

    new_token_ids: Sequence[int]
    """Committed output tokens of this step, in commit order."""


@dataclass
class EffortState:
    """Controller state for one request."""

    request_id: str
    num_levels: int
    start_ids: list[int]
    end_ids: list[int]

    level: int = 0
    """Effort level the prefill decision chose; index into the level sentences."""
    decided: bool = False
    """A hidden-state decision was made (as opposed to the server default)."""

    in_think: bool = False
    think_count: int = 0
    reasoning_tokens: int = 0
    close_kind: str = CLOSE_NATURAL
    finished: bool = False

    # Decision provenance, for telemetry only.
    novelty: float | None = None
    novelty_rank: float | None = None
    estimate: float | None = None
    """Raw kNN estimate the decision saw; calibrates the memory at finish."""
    decided_difficulty: float | None = None
    """Calibrated estimate the level was chosen from; a think-off request is
    remembered with it."""
    off_votes: int = 0
    """Yes votes collected by the off gate before its verdict."""
    off_vetoed: bool = False
    """True when the off gate demoted a think-off verdict to low."""
    spread: float | None = None
    neighbours: int = 0
    memory_entries: int = 0

    _tail: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from collections import deque

        self._tail = deque(maxlen=max(len(self.start_ids), len(self.end_ids), 1))

    @property
    def top_level(self) -> int:
        return max(self.num_levels - 1, 0)

    @property
    def report(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "decided": int(self.decided),
            "reasoning_tokens": self.reasoning_tokens,
            "close_kind": self.close_kind,
            "memory_entries": self.memory_entries,
            "neighbours": self.neighbours,
            "estimate": self.estimate,
            "calibrated": self.decided_difficulty,
            "novelty_rank": self.novelty_rank,
            "off_votes": self.off_votes,
            "off_vetoed": int(self.off_vetoed),
        }


def _rfind(seq: Sequence[int], sub: Sequence[int]) -> int:
    """Index of the last occurrence of `sub` in `seq`, or -1."""
    if not sub or len(sub) > len(seq):
        return -1
    if len(sub) == 1:
        rev = list(seq)[::-1]
        try:
            return len(seq) - 1 - rev.index(sub[0])
        except ValueError:
            return -1
    for i in range(len(seq) - len(sub), -1, -1):
        if list(seq[i : i + len(sub)]) == list(sub):
            return i
    return -1


def new_effort_state(
    request_id: str,
    cfg: DynamicEffortConfig,
    start_ids: list[int],
    end_ids: list[int],
    prompt_token_ids: Sequence[int] | None,
) -> EffortState:
    """Build the state for a request; a prompt ending mid-think starts in it."""
    state = EffortState(
        request_id=request_id,
        num_levels=cfg.num_levels,
        start_ids=start_ids,
        end_ids=end_ids,
    )
    if prompt_token_ids:
        last_start = _rfind(prompt_token_ids, start_ids)
        last_end = _rfind(prompt_token_ids, end_ids)
        if last_start >= 0 and last_start > last_end:
            state.in_think = True
            state.think_count = len(prompt_token_ids) - last_start - len(start_ids)
    return state


def _tail_endswith(tail, seq: Sequence[int]) -> bool:
    n = len(seq)
    if n == 0 or len(tail) < n:
        return False
    if n == 1:
        return tail[-1] == seq[0]
    return list(tail)[-n:] == list(seq)


def step_effort(state: EffortState, cfg: DynamicEffortConfig, ev: EffortEvent) -> None:
    """Advance one request by one step.

    Pure bookkeeping: it counts reasoning tokens and notices the model's own
    end marker. There is no action to take - the level was chosen before the
    request started thinking and nothing may change it afterwards.
    """
    for tok in ev.new_token_ids:
        state._tail.append(tok)
        if not state.in_think:
            if _tail_endswith(state._tail, state.start_ids):
                state.in_think = True
                state.think_count = 0
            continue
        state.think_count += 1
        if _tail_endswith(state._tail, state.end_ids):
            state.in_think = False
            # The end sequence itself is not reasoning content.
            state.think_count = max(state.think_count - len(state.end_ids), 0)
            state.reasoning_tokens += state.think_count


def finish_effort(state: EffortState) -> dict[str, Any]:
    """Close the state at request finish and return the report."""
    if state.in_think:
        # Still thinking when the request ended: the client's max_tokens, a
        # timeout or an abort stopped it, so the length it would have spent is
        # unknown and the memory must not value this entry.
        state.reasoning_tokens += state.think_count
        state.in_think = False
        state.close_kind = CLOSE_CLIENT_LIMIT
    state.finished = True
    return state.report
