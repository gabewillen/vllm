# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model Runner V2 state for per-request effort telemetry (entropy / margin)."""

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.effort_signals import (
    NUM_ROW_SIGNALS,
    effort_row_signals,
    effort_row_signals_scattered,
    flagged_row_indices,
    reduce_committed,
    wants_effort_signals,
)
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig


class EffortState:
    """Opt-in flags plus the per-step row-signal accumulator.

    ``begin`` arms the state for a batch, ``compute`` is called once per logits
    chunk at the canonical stage (after penalties / bad words, before the
    thinking-budget force and temperature) and ``finish`` reduces the rows to
    per-request means over the committed prefix once ``num_sampled`` is known.
    """

    def __init__(
        self,
        max_num_reqs: int,
        device: torch.device,
        reasoning_config: "ReasoningConfig | None" = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.device = device
        natural_end = (
            None
            if reasoning_config is None
            else reasoning_config.natural_reasoning_end_token_ids
        )
        self.end_token_id: int | None = natural_end[0] if natural_end else None
        self.use_effort = np.zeros(max_num_reqs, dtype=bool)
        # req_state_idx -> 1 when flagged; read on GPU for expanded rows.
        self.flags = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self._flags_dirty = False
        self._active = False
        self._pending: list[torch.Tensor] = []
        self._pending_mask: list[torch.Tensor] = []

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        flagged = wants_effort_signals(sampling_params)
        self.use_effort[req_idx] = flagged
        if bool(self.flags.np[req_idx]) != flagged:
            self.flags.np[req_idx] = int(flagged)
            self._flags_dirty = True

    def apply_staged_writes(self) -> None:
        if self._flags_dirty:
            self.flags.copy_to_uva()
            self._flags_dirty = False

    def batch_flags(self, idx_mapping_np: np.ndarray) -> np.ndarray:
        return self.use_effort[idx_mapping_np]

    def begin(self, idx_mapping_np: np.ndarray) -> bool:
        """Arm for a batch; returns whether any request in it opted in."""
        self._pending.clear()
        self._pending_mask.clear()
        self._active = bool(np.any(self.use_effort[idx_mapping_np]))
        return self._active

    def compute(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        """Append canonical-stage row signals for one logits chunk."""
        if not self._active:
            return
        num_rows = logits.shape[0]
        device = logits.device
        chunk_flags = self.use_effort[idx_mapping_np]
        if not np.any(chunk_flags):
            sig = torch.zeros(
                (num_rows, NUM_ROW_SIGNALS), dtype=torch.float32, device=device
            )
            row_flags = torch.zeros(num_rows, dtype=torch.bool, device=device)
        elif num_rows == idx_mapping_np.shape[0]:
            # One row per request: exact host-side row selection.
            rows = flagged_row_indices(chunk_flags)
            sig = effort_row_signals_scattered(
                logits, async_tensor_h2d(rows, device=device), self.end_token_id
            )
            row_flags = async_tensor_h2d(chunk_flags, device=device)
        else:
            # Spec-decode rows: the exact per-request layout only exists on
            # the GPU (adaptive verification compacts drafts), so compute the
            # chunk and zero the rows of requests that did not opt in.
            row_flags = self.flags.gpu[expanded_idx_mapping] != 0
            sig = (
                effort_row_signals(logits, self.end_token_id)
                * row_flags.to(torch.float32)[:, None]
            )
        self._pending.append(sig)
        self._pending_mask.append(row_flags)

    def finish(
        self,
        cu_num_logits: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> torch.Tensor | None:
        """Reduce the accumulated rows to ``[num_reqs, 4]`` per-request means."""
        if not self._active:
            return None
        self._active = False
        if not self._pending:
            return None
        rows = self._pending[0] if len(self._pending) == 1 else torch.cat(self._pending)
        mask = (
            self._pending_mask[0]
            if len(self._pending_mask) == 1
            else torch.cat(self._pending_mask)
        )
        self._pending.clear()
        self._pending_mask.clear()
        return reduce_committed(rows, cu_num_logits, num_sampled, row_mask=mask)
