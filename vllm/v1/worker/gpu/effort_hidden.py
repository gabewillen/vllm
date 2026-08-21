# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The free half of the v3 effort signal: the body's last prefill row.

`hidden_states` in `GPUModelRunner.sample_tokens` is `[num_tokens, hidden]` -
the tensor `compute_logits` feeds to `lm_head`, already full width on every TP
rank because the final row-parallel all-reduce has happened. The row the
scheduler wants is the last row of the request's query window on the step its
*body* prefill completes (docs/dynamic-reasoning.claude.md §13.3).

Taking the last scheduled row rather than counting prompt tokens is what makes
this correct for a fully prefix-cached body: such a request schedules a single
token, which is exactly the last body token, and the naive prefill accounting
produces no record at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch


def gather_prefill_states(
    hidden_states: torch.Tensor,
    input_batch: InputBatch,
    capture_req_ids: list[str] | None,
) -> tuple[list[str], torch.Tensor] | None:
    """Last-row hidden state of each request the scheduler asked to capture.

    Args:
        hidden_states: `[num_tokens, hidden]` final-layer states of this step.
        input_batch: the step's batch; `query_start_loc_np` locates each
            request's rows.
        capture_req_ids: request ids whose body prefill ends this step.

    Returns:
        `(req_ids, [n, hidden] fp16 device tensor)`, or `None` when nothing was
        asked for or none of the ids are in this batch.
    """
    if not capture_req_ids or hidden_states is None:
        return None
    wanted = set(capture_req_ids)
    qsl = input_batch.query_start_loc_np
    req_ids: list[str] = []
    rows: list[int] = []
    for i, req_id in enumerate(input_batch.req_ids):
        if req_id not in wanted:
            continue
        row = int(qsl[i + 1]) - 1
        if 0 <= row < hidden_states.shape[0]:
            req_ids.append(req_id)
            rows.append(row)
    if not req_ids:
        return None
    index = torch.tensor(rows, dtype=torch.long, device=hidden_states.device)
    return req_ids, hidden_states.index_select(0, index).to(torch.float16)
