# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba speculative state slots follow the drafts a step actually verifies.

A Mamba/GDN layer keeps one recurrent-state block per verified draft token so
a rejected draft can be rolled back. Reserving `num_speculative_tokens` of
them for a request's whole life costs K+1 full state copies per request even
when the per-batch-size schedule has switched drafting off, which caps the
number of concurrent requests far below the KV pool. The manager sizes the
reservation by the draft count of each step instead, holding a slot until no
step that may still read it (the previous one, plus the one in flight under
async scheduling) needed it.
"""

import pytest
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

NUM_SPEC = 3
BLOCK_SIZE = 64
# Draft NUM_SPEC tokens up to batch 2, none above.
SCHEDULE = [(1, 2, NUM_SPEC), (3, 16, 0)]


def _mamba_spec(mode: str) -> MambaSpec:
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode=mode,
        num_speculative_blocks=NUM_SPEC,
    )


def _scheduler(mode: str, schedule=SCHEDULE, **kwargs) -> Scheduler:
    return create_scheduler(
        block_size=BLOCK_SIZE,
        max_model_len=4 * BLOCK_SIZE,
        max_num_batched_tokens=4 * BLOCK_SIZE,
        num_speculative_tokens=NUM_SPEC,
        num_speculative_tokens_per_batch_size=schedule,
        kv_cache_spec=_mamba_spec(mode),
        **kwargs,
    )


def _manager(scheduler: Scheduler):
    return scheduler.kv_cache_manager.coordinator.single_type_managers[0]


def _num_blocks(scheduler: Scheduler, req_id: str) -> int:
    return len(_manager(scheduler).req_to_blocks[req_id])


def _free_blocks(scheduler: Scheduler) -> int:
    return scheduler.kv_cache_manager.block_pool.get_num_free_blocks()


def _run_step(
    scheduler: Scheduler, accept_all: bool = True
) -> tuple[SchedulerOutput, dict[str, int]]:
    """Schedule one step, accept the verified drafts, draft for the next step.

    Returns the scheduler output and the number of drafts verified per
    request in this step.
    """
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    verified = {
        req_id: len(output.scheduled_spec_decode_tokens.get(req_id, ()))
        for req_id in req_ids
    }
    sampled = [
        [0] * (1 + verified[req_id]) if accept_all else [0] for req_id in req_ids
    ]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    num_drafts = output.num_spec_tokens_to_schedule
    running = [req.request_id for req in scheduler.running]
    scheduler.update_draft_token_ids(
        DraftTokenIds(running, [list(range(1, 1 + num_drafts)) for _ in running])
    )
    return output, verified


@pytest.mark.parametrize("mode", ["none", "align"])
def test_slots_follow_the_batch_size_schedule(mode: str):
    scheduler = _scheduler(mode)
    first = create_requests(num_requests=2, num_tokens=8, max_tokens=64)
    for req in first:
        scheduler.add_request(req)
    ids = [req.request_id for req in first]

    # Prefill: no drafts yet, one state block each.
    _run_step(scheduler)
    assert all(_num_blocks(scheduler, req_id) == 1 for req_id in ids)

    # Batch 2 verifies NUM_SPEC drafts: the state block plus one slot per draft.
    _, verified = _run_step(scheduler)
    assert all(verified[req_id] == NUM_SPEC for req_id in ids)
    assert all(_num_blocks(scheduler, req_id) == 1 + NUM_SPEC for req_id in ids)
    free_with_drafts = _free_blocks(scheduler)

    # Two more requests push the batch past the schedule boundary: drafting
    # stops, and the slots are released once the steps that could still read
    # them (previous + in flight) are past.
    more = create_requests(
        num_requests=2, num_tokens=8, max_tokens=64, req_ids=["r2", "r3"]
    )
    for req in more:
        scheduler.add_request(req)
    # The drafts proposed at batch 2 are still verified in the first step.
    verified_per_step = []
    trims_per_step = []
    for _ in range(4):
        output, verified = _run_step(scheduler)
        verified_per_step.append({verified[req_id] for req_id in ids})
        trims_per_step.append(output.scheduled_cached_reqs.trimmed_block_counts)
        assert not output.preempted_req_ids
    assert verified_per_step == [{NUM_SPEC}, {0}, {0}, {0}]
    assert trims_per_step[:3] == [{}, {}, {}]
    assert trims_per_step[3] == {req_id: (NUM_SPEC,) for req_id in ids}
    assert all(_num_blocks(scheduler, req_id) == 1 for req_id in ids)
    assert all(_num_blocks(scheduler, req.request_id) == 1 for req in more)
    assert _free_blocks(scheduler) == free_with_drafts + 2 * NUM_SPEC - 2

    # The extra requests finish: batch 2 drafts again and the slots come back
    # through the normal append path.
    scheduler.finish_requests(
        [req.request_id for req in more], RequestStatus.FINISHED_ABORTED
    )
    _, verified = _run_step(scheduler)  # proposes at batch 2 again
    assert all(verified[req_id] == 0 for req_id in ids)
    output, verified = _run_step(scheduler)
    assert all(verified[req_id] == NUM_SPEC for req_id in ids)
    assert all(_num_blocks(scheduler, req_id) == 1 + NUM_SPEC for req_id in ids)
    new_blocks = dict(
        zip(
            output.scheduled_cached_reqs.req_ids,
            output.scheduled_cached_reqs.new_block_ids,
        )
    )
    assert all(len(new_blocks[req_id][0]) == NUM_SPEC for req_id in ids)
    assert not output.scheduled_cached_reqs.trimmed_block_counts


@pytest.mark.parametrize("mode", ["none", "align"])
def test_steady_state_matches_static_reservation_without_schedule(mode: str):
    scheduler = _scheduler(mode, schedule=None)
    (req,) = create_requests(num_requests=1, num_tokens=8, max_tokens=64)
    scheduler.add_request(req)
    _run_step(scheduler)
    for _ in range(4):
        _, verified = _run_step(scheduler)
        assert verified[req.request_id] == NUM_SPEC
        assert _num_blocks(scheduler, req.request_id) == 1 + NUM_SPEC
        assert _manager(scheduler)._spec_slots[req.request_id] == NUM_SPEC


@pytest.mark.parametrize("mode", ["none", "align"])
def test_drafts_are_dropped_before_anyone_is_preempted(mode: str):
    # Pool: null block + (1 + NUM_SPEC) for the first request + 1 for the
    # second; the second request's drafts do not fit.
    scheduler = _scheduler(mode, schedule=None, num_blocks=1 + (1 + NUM_SPEC) + 1)
    first, second = create_requests(num_requests=2, num_tokens=8, max_tokens=64)
    scheduler.add_request(first)
    _run_step(scheduler)
    _run_step(scheduler)
    assert _num_blocks(scheduler, first.request_id) == 1 + NUM_SPEC

    scheduler.add_request(second)
    _run_step(scheduler)
    assert _num_blocks(scheduler, second.request_id) == 1
    assert _free_blocks(scheduler) == 0

    output, verified = _run_step(scheduler)
    assert verified[first.request_id] == NUM_SPEC
    assert verified[second.request_id] == 0
    assert second.request_id in output.num_scheduled_tokens
    assert output.num_scheduled_tokens[second.request_id] == 1
    assert not output.preempted_req_ids
    assert len(scheduler.running) == 2


def test_slot_release_waits_for_the_step_that_may_read_it():
    """A slot written in step t can be the initial state of step t+1 and,
    under async scheduling, of the step still in flight: it goes two steps
    after the last draft that needed it, never earlier."""
    scheduler = _scheduler("align")
    reqs = create_requests(num_requests=2, num_tokens=8, max_tokens=64)
    for req in reqs:
        scheduler.add_request(req)
    ids = [req.request_id for req in reqs]
    _run_step(scheduler)
    _run_step(scheduler)
    manager = _manager(scheduler)
    assert all(manager._spec_slots[req_id] == NUM_SPEC for req_id in ids)

    more = create_requests(
        num_requests=2, num_tokens=8, max_tokens=64, req_ids=["r2", "r3"]
    )
    for req in more:
        scheduler.add_request(req)
    # Step 1 still verifies the drafts proposed at batch 2; steps 2 and 3 may
    # read the slot they wrote; step 4 releases it.
    held_per_step = []
    for _ in range(4):
        _run_step(scheduler)
        held_per_step.append(manager._spec_slots[ids[0]])
    assert held_per_step == [NUM_SPEC, NUM_SPEC, NUM_SPEC, 0]
