# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request thinking token budgets for Model Runner V2.

State per batch slot: the (versioned) budget plus a cache of the last
reasoning start/end marker positions in the committed token history. Each
step, ``apply`` forces the reasoning end sequence on every target logits row
whose effective prefix (committed tokens plus the draft tokens before that
row) is inside a think block that has reached the budget. The rules match the
V1 ``ThinkingBudgetStateHolder``: the first over-budget row forces the first
end token, later rows in the same draft window force the next end tokens
while the drafts follow the sequence, and a rejected draft simply re-forces
next step because the decision is recomputed from committed tokens.

``apply_thinking_budget`` dispatches to Triton kernels on CUDA and to a pure
torch reference (``apply_thinking_budget_torch``) on other devices; the
reference is the CPU-testable specification of the semantics.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

_INT32_MAX = np.iinfo(np.int32).max
_COLD_SCAN_BLOCK = 1024
NO_REVISION = -1


class ThinkingBudgetState:
    """Model Runner V2 state for per-request thinking token budgets."""

    def __init__(
        self,
        req_states: RequestState,
        reasoning_config: "ReasoningConfig | None",
    ):
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device

        start_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_start_token_ids or []
        )
        end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_end_token_ids or []
        )
        natural_end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.natural_reasoning_end_token_ids or []
        )
        self.enabled = bool(start_ids and end_ids and natural_end_ids)
        self.use_thinking_budget = np.zeros(self.max_num_reqs, dtype=bool)
        if not self.enabled:
            return

        self.thinking_token_budget = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.thinking_token_budget.np.fill(-1)
        self.thinking_token_budget.copy_to_uva()
        # Revision of the last applied budget update per slot (CPU only).
        self.budget_revision = np.full(self.max_num_reqs, NO_REVISION, dtype=np.int64)

        self.cached_last_start = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
        )
        self.cached_last_end = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
        )
        self.cached_scan_pos = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self._reset_reqs: list[int] = []
        self._budget_dirty = False

        self.reasoning_start_token_ids = torch.tensor(
            start_ids, dtype=torch.int32, device=self.device
        )
        self.natural_reasoning_end_token_ids = torch.tensor(
            natural_end_ids, dtype=torch.int32, device=self.device
        )
        self.reasoning_end_token_ids = torch.tensor(
            end_ids, dtype=torch.int32, device=self.device
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        if not self.enabled:
            return
        budget = sampling_params.thinking_token_budget
        self.use_thinking_budget[req_idx] = budget is not None
        self.budget_revision[req_idx] = NO_REVISION
        if budget is None:
            budget = -1
        else:
            budget = min(budget, _INT32_MAX)
            self._reset_reqs.append(req_idx)
        if self.thinking_token_budget.np[req_idx] != budget:
            self.thinking_token_budget.np[req_idx] = budget
            self._budget_dirty = True

    def update_budgets(
        self,
        updates: dict[str, tuple[int, int]],
        req_id_to_index: dict[str, int],
    ) -> dict[str, int]:
        """Apply versioned budget updates; return req_id -> revision applied.

        Only requests that opted into a budget at ``add_request`` and only
        strictly increasing revisions are applied. The staged budget tensor
        is flushed by ``apply_staged_writes``.
        """
        if not self.enabled or not updates:
            return {}
        acks: dict[str, int] = {}
        for req_id, (revision, budget) in updates.items():
            req_idx = req_id_to_index.get(req_id)
            if req_idx is None or not self.use_thinking_budget[req_idx]:
                continue
            if revision <= self.budget_revision[req_idx]:
                continue
            self.budget_revision[req_idx] = revision
            budget = min(max(budget, 0), _INT32_MAX)
            if self.thinking_token_budget.np[req_idx] != budget:
                self.thinking_token_budget.np[req_idx] = budget
                self._budget_dirty = True
            acks[req_id] = revision
        return acks

    def apply_staged_writes(self) -> None:
        if not self.enabled:
            return
        if self._reset_reqs:
            idx = async_tensor_h2d(
                self._reset_reqs, dtype=torch.int64, device=self.device
            )
            self.cached_last_start.index_fill_(0, idx, -1)
            self.cached_last_end.index_fill_(0, idx, -1)
            self.cached_scan_pos.index_fill_(0, idx, 0)
            self._reset_reqs.clear()
        if self._budget_dirty:
            self.thinking_token_budget.copy_to_uva()
            self._budget_dirty = False

    def requires_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:
        return self.enabled and bool(np.any(self.use_thinking_budget[idx_mapping_np]))

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        if not self.requires_logits_processing(idx_mapping_np):
            return

        apply_thinking_budget(
            logits,
            idx_mapping,
            expanded_idx_mapping,
            self.thinking_token_budget.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
            input_ids,
            expanded_local_pos,
            self.cached_last_start,
            self.cached_last_end,
            self.cached_scan_pos,
            self.reasoning_start_token_ids,
            self.natural_reasoning_end_token_ids,
            self.reasoning_end_token_ids,
        )


@triton.jit
def _load_effective_token(
    all_token_ids_ptr,
    all_token_ids_stride,
    input_ids_ptr,
    cur_req_first_pos,
    req_state_idx,
    total_len,
    pos,
):
    if pos < total_len:
        return tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos)
    # In decode/spec-decode, input_ids at local position 0 is the already
    # committed last sampled token. Effective draft-prefix positions start at
    # local position 1.
    input_pos = cur_req_first_pos + pos - total_len + 1
    return tl.load(input_ids_ptr + input_pos)


@triton.jit
def _update_committed_marker_cache_kernel(
    req_ids_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    cached_scan_pos_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_state_idx = tl.load(req_ids_ptr + tl.program_id(0))
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    scan_pos = tl.load(cached_scan_pos_ptr + req_state_idx)
    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    if scan_pos > total_len:
        scan_pos = 0
        last_start = -1
        last_end = -1

    if scan_pos == 0 and last_start < 0 and last_end < 0:
        # Cold scan: walk backward in vectorized blocks, stopping at the first
        # block with a marker; only the relative order of the two positions
        # found matters below.
        block_hi = total_len
        while block_hi > 0 and last_start < 0 and last_end < 0:
            block_lo = block_hi - BLOCK
            if block_lo < 0:
                block_lo = 0
            offs = block_lo + tl.arange(0, BLOCK)

            start_match = (offs < block_hi) & (offs + START_LEN <= total_len)
            for j in tl.static_range(0, START_LEN):
                expected = tl.load(reasoning_start_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                start_match = start_match & (actual == expected)

            end_match = (offs < block_hi) & (offs + NATURAL_END_LEN <= total_len)
            for j in tl.static_range(0, NATURAL_END_LEN):
                expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                end_match = end_match & (actual == expected)

            last_start = tl.max(tl.where(start_match, offs, -1), axis=0)
            last_end = tl.max(tl.where(end_match, offs, -1), axis=0)
            block_hi = block_lo
    else:
        for i in tl.range(scan_pos, total_len):
            if i + START_LEN <= total_len:
                start_match = True
                for j in tl.static_range(0, START_LEN):
                    expected = tl.load(reasoning_start_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j
                    )
                    start_match = start_match & (actual == expected)
                if start_match:
                    last_start = i

            if i + NATURAL_END_LEN <= total_len:
                end_match = True
                for j in tl.static_range(0, NATURAL_END_LEN):
                    expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j
                    )
                    end_match = end_match & (actual == expected)
                if end_match:
                    last_end = i

    tl.store(cached_last_start_ptr + req_state_idx, last_start)
    tl.store(cached_last_end_ptr + req_state_idx, last_end)
    new_scan_pos = total_len - (MAX_LEN - 1)
    if new_scan_pos < 0:
        new_scan_pos = 0
    tl.store(cached_scan_pos_ptr + req_state_idx, new_scan_pos)


@triton.jit
def _thinking_budget_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - local_pos
    total_len = tl.load(total_len_ptr + req_state_idx)
    effective_len = total_len + local_pos

    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    start_lo = total_len - START_LEN + 1
    if start_lo < 0:
        start_lo = 0
    for i in tl.range(start_lo, effective_len - START_LEN + 1):
        start_match = True
        for j in tl.static_range(0, START_LEN):
            expected = tl.load(reasoning_start_token_ids_ptr + j)
            actual = _load_effective_token(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                i + j,
            )
            start_match = start_match & (actual == expected)
        if start_match:
            last_start = i

    end_lo = total_len - NATURAL_END_LEN + 1
    if end_lo < 0:
        end_lo = 0
    for i in tl.range(end_lo, effective_len - NATURAL_END_LEN + 1):
        end_match = True
        for j in tl.static_range(0, NATURAL_END_LEN):
            expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
            actual = _load_effective_token(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                i + j,
            )
            end_match = end_match & (actual == expected)
        if end_match:
            last_end = i

    if last_start < 0 or last_start <= last_end:
        return

    reasoning_start = last_start + START_LEN
    # If the request resumes from a prompt that already contains generated
    # reasoning content, count it against the remaining budget.
    num_reasoning_tokens = effective_len - reasoning_start
    if num_reasoning_tokens < budget:
        return

    # If the tail already ends with a prefix of the forced end sequence
    # (even from a resumed prompt), continue from the next marker token.
    end_prefix_len = 0
    max_prefix_len = END_LEN - 1
    if effective_len < max_prefix_len:
        max_prefix_len = effective_len

    for prefix_len in tl.static_range(1, END_LEN):
        if prefix_len <= max_prefix_len:
            prefix_match = True
            suffix_start = effective_len - prefix_len
            for j in tl.static_range(0, END_LEN):
                if j < prefix_len:
                    expected = tl.load(reasoning_end_token_ids_ptr + j)
                    actual = _load_effective_token(
                        all_token_ids_ptr,
                        all_token_ids_stride,
                        input_ids_ptr,
                        cur_req_first_pos,
                        req_state_idx,
                        total_len,
                        suffix_start + j,
                    )
                    prefix_match = prefix_match & (actual == expected)
            if prefix_match:
                end_prefix_len = prefix_len

    force_token_id = tl.load(reasoning_end_token_ids_ptr + end_prefix_len)
    tl.store(logits_ptr + token_idx * logits_stride + force_token_id, 1.0e9)


def find_last_marker(tokens: torch.Tensor, marker: torch.Tensor, lo: int = 0) -> int:
    """Largest ``i >= lo`` with ``tokens[i:i+len(marker)] == marker``, else -1."""
    n = marker.numel()
    if n == 0 or tokens.numel() - lo < n:
        return -1
    hits = (tokens[lo:].unfold(0, n, 1) == marker).all(dim=1)
    if not bool(hits.any()):
        return -1
    return lo + int(hits.nonzero()[-1].item())


def update_marker_cache_torch(
    req_ids: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    cached_scan_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
) -> None:
    """Torch reference of ``_update_committed_marker_cache_kernel``.

    Refreshes the cached last start / natural-end marker positions of each
    budgeted request from its committed tokens, scanning only what the
    previous scan could not have covered.
    """
    max_len = max(
        reasoning_start_token_ids.numel(), natural_reasoning_end_token_ids.numel()
    )
    for req_idx in req_ids.tolist():
        if int(thinking_token_budget[req_idx]) < 0:
            continue
        seq_len = int(total_len[req_idx])
        scan_pos = int(cached_scan_pos[req_idx])
        last_start = int(cached_last_start[req_idx])
        last_end = int(cached_last_end[req_idx])
        if scan_pos > seq_len:
            scan_pos, last_start, last_end = 0, -1, -1
        lo = 0 if (scan_pos == 0 and last_start < 0 and last_end < 0) else scan_pos
        tokens = all_token_ids[req_idx, :seq_len]
        last_start = max(
            last_start, find_last_marker(tokens, reasoning_start_token_ids, lo)
        )
        last_end = max(
            last_end, find_last_marker(tokens, natural_reasoning_end_token_ids, lo)
        )
        cached_last_start[req_idx] = last_start
        cached_last_end[req_idx] = last_end
        cached_scan_pos[req_idx] = max(0, seq_len - (max_len - 1))


def forced_end_tokens_torch(
    expanded_idx_mapping: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    reasoning_end_token_ids: torch.Tensor,
) -> tuple[list[int], list[int]]:
    """Torch reference of ``_thinking_budget_kernel``'s decision.

    Returns the ``(row, token_id)`` pairs to force. Row ``t`` of request ``r``
    at local position ``p`` sees the effective prefix = committed tokens plus
    ``input_ids[t-p+1 : t+1]`` (the drafts before it); it is forced when that
    prefix is inside a think block holding at least ``budget`` tokens, to the
    end-sequence token following the longest end-sequence prefix the effective
    prefix already ends with.
    """
    start_len = reasoning_start_token_ids.numel()
    natural_end_len = natural_reasoning_end_token_ids.numel()
    end_len = reasoning_end_token_ids.numel()
    rows: list[int] = []
    force_tokens: list[int] = []
    for row, (req_idx, local_pos) in enumerate(
        zip(expanded_idx_mapping.tolist(), expanded_local_pos.tolist())
    ):
        budget = int(thinking_token_budget[req_idx])
        if budget < 0:
            continue
        seq_len = int(total_len[req_idx])
        first_row = row - local_pos
        effective = torch.cat(
            [
                all_token_ids[req_idx, :seq_len].to(torch.int64),
                input_ids[first_row + 1 : first_row + local_pos + 1].to(torch.int64),
            ]
        )
        effective_len = seq_len + local_pos
        last_start = max(
            int(cached_last_start[req_idx]),
            find_last_marker(
                effective, reasoning_start_token_ids, max(0, seq_len - start_len + 1)
            ),
        )
        last_end = max(
            int(cached_last_end[req_idx]),
            find_last_marker(
                effective,
                natural_reasoning_end_token_ids,
                max(0, seq_len - natural_end_len + 1),
            ),
        )
        if last_start < 0 or last_start <= last_end:
            continue
        if effective_len - (last_start + start_len) < budget:
            continue
        end_prefix_len = 0
        for prefix_len in range(1, min(end_len - 1, effective_len) + 1):
            if torch.equal(
                effective[effective_len - prefix_len :],
                reasoning_end_token_ids[:prefix_len].to(torch.int64),
            ):
                end_prefix_len = prefix_len
        rows.append(row)
        force_tokens.append(int(reasoning_end_token_ids[end_prefix_len]))
    return rows, force_tokens


def apply_thinking_budget_torch(
    logits: torch.Tensor,
    req_ids: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    cached_scan_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    reasoning_end_token_ids: torch.Tensor,
) -> None:
    """Torch reference of ``apply_thinking_budget`` (runs on CPU tensors)."""
    update_marker_cache_torch(
        req_ids,
        thinking_token_budget,
        all_token_ids,
        total_len,
        cached_last_start,
        cached_last_end,
        cached_scan_pos,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
    )
    rows, force_tokens = forced_end_tokens_torch(
        expanded_idx_mapping,
        thinking_token_budget,
        all_token_ids,
        total_len,
        input_ids,
        expanded_local_pos,
        cached_last_start,
        cached_last_end,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        reasoning_end_token_ids,
    )
    if rows:
        logits[
            torch.tensor(rows, dtype=torch.int64, device=logits.device),
            torch.tensor(force_tokens, dtype=torch.int64, device=logits.device),
        ] = 1.0e9


def apply_thinking_budget(
    logits: torch.Tensor,
    req_ids: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    cached_scan_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    reasoning_end_token_ids: torch.Tensor,
) -> None:
    if logits.device.type != "cuda":
        apply_thinking_budget_torch(
            logits,
            req_ids,
            expanded_idx_mapping,
            thinking_token_budget,
            all_token_ids,
            total_len,
            input_ids,
            expanded_local_pos,
            cached_last_start,
            cached_last_end,
            cached_scan_pos,
            reasoning_start_token_ids,
            natural_reasoning_end_token_ids,
            reasoning_end_token_ids,
        )
        return

    num_tokens = logits.shape[0]
    start_len = reasoning_start_token_ids.shape[0]
    natural_end_len = natural_reasoning_end_token_ids.shape[0]
    end_len = reasoning_end_token_ids.shape[0]

    _update_committed_marker_cache_kernel[(req_ids.shape[0],)](
        req_ids,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        cached_last_start,
        cached_last_end,
        cached_scan_pos,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        MAX_LEN=max(start_len, natural_end_len),
        BLOCK=_COLD_SCAN_BLOCK,
    )

    _thinking_budget_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        input_ids,
        expanded_local_pos,
        cached_last_start,
        cached_last_end,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        END_LEN=end_len,
    )
