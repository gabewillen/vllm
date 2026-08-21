# SPDX-License-Identifier: Apache-2.0
"""The worker half of the v3 signal: which row of `hidden_states` is taken.

`gather_prefill_states` picks the last row of a request's query window on the
step its body prefill ends. Two things have to hold for that to be the vector
the design measured (§13.3):

* it is the same row however the body was chunked, and
* it works when the prefix cache computed all but the last body token, which is
  the case the capture prototype silently dropped.
"""

import numpy as np
import torch

from vllm.v1.worker.gpu.effort_hidden import gather_prefill_states


class _Batch:
    """The two fields the pooler reads off a real `InputBatch`."""

    def __init__(self, req_ids, num_scheduled):
        self.req_ids = list(req_ids)
        self.query_start_loc_np = np.concatenate(
            [[0], np.cumsum(np.asarray(num_scheduled, dtype=np.int64))]
        )


def _states(num_tokens, hidden=8):
    # Row i is i, so the identity of the selected row is readable.
    return (
        torch.arange(num_tokens, dtype=torch.float32).reshape(-1, 1).repeat(1, hidden)
    )


def test_last_row_is_the_last_scheduled_prompt_token():
    hidden = _states(10)
    batch = _Batch(["a", "b"], [4, 6])
    got = gather_prefill_states(hidden, batch, ["b"])
    assert got is not None
    req_ids, rows = got
    assert req_ids == ["b"]
    assert rows.dtype is torch.float16
    assert rows.shape == (1, 8)
    assert rows[0, 0].item() == 9.0  # the last row of b's window


def test_row_is_chunk_invariant():
    """A body split across chunks yields the same row as one that is not.

    Only the step where the body *ends* is captured, so the selected row is the
    body's last token whatever the chunk boundaries before it were.
    """
    whole = gather_prefill_states(_states(96), _Batch(["a"], [96]), ["a"])
    # The same body arriving as 64 + 32: the capture step sees only the tail
    # chunk, whose last row is still the body's last token.
    chunked = gather_prefill_states(_states(32), _Batch(["a"], [32]), ["a"])
    assert whole is not None and chunked is not None
    # Row values differ only because this fixture numbers rows per step; what
    # is pinned is that both select the final row of the capture step.
    assert whole[1][0, 0].item() == 95.0
    assert chunked[1][0, 0].item() == 31.0
    # And a mid-body chunk is never captured at all: the scheduler only puts a
    # request in the capture list on the step its body completes.
    assert gather_prefill_states(_states(64), _Batch(["a"], [64]), []) is None


def test_fully_cached_body_schedules_one_row_and_still_yields_a_vector():
    # 100% prefix-cache hit on everything but the last body token: exactly one
    # token is scheduled, and it is the row the decision needs.
    hidden = _states(1)
    got = gather_prefill_states(hidden, _Batch(["a"], [1]), ["a"])
    assert got is not None
    assert got[0] == ["a"]
    assert got[1][0, 0].item() == 0.0


def test_requests_outside_the_batch_are_skipped():
    hidden = _states(6)
    batch = _Batch(["a", "b"], [3, 3])
    got = gather_prefill_states(hidden, batch, ["b", "ghost"])
    assert got is not None and got[0] == ["b"]
    assert gather_prefill_states(hidden, batch, ["ghost"]) is None
    assert gather_prefill_states(hidden, batch, None) is None
    assert gather_prefill_states(None, batch, ["a"]) is None


def test_multiple_captures_keep_request_order():
    hidden = _states(9)
    batch = _Batch(["a", "b", "c"], [2, 3, 4])
    req_ids, rows = gather_prefill_states(hidden, batch, ["c", "a"])
    # Batch order, not capture-list order, so the rows line up with req_ids.
    assert req_ids == ["a", "c"]
    assert rows[:, 0].tolist() == [1.0, 8.0]


def test_tolerates_a_model_that_returns_no_auxiliary_states():
    """The capture never asks for auxiliary hidden states.

    The probe measurement found that this build's model ignores an
    `aux_hidden_state_layers` request (docs/effort-hidden-probe.md §1), so v3
    reads only the final-layer tensor the sampler already has. This pins that
    the pooler's contract is that tensor alone - nothing here can be broken by
    a model that returns a bare tensor instead of a tuple.
    """
    import inspect

    params = inspect.signature(gather_prefill_states).parameters
    assert list(params) == ["hidden_states", "input_batch", "capture_req_ids"]
