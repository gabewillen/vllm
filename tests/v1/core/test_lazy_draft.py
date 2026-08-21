# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy drafting: the drafter is skipped on zero-draft steps.

A stateful drafter (EAGLE/MTP KV) that skips a step can never be brought back
in sync, because its inputs are the target hidden states of the skipped
tokens and those are gone. The scheduler therefore tracks completeness: a
request scheduled into a zero-draft step is `draft_stale` and is never
verified against drafts again; the blocks it caches carry the same bit, a
request that is going to draft does not reuse them, and a request that is
not may. Exactness of the target distribution never depends on any of this,
only how much speculation happens.
"""

import pytest

from vllm.config import SpeculativeConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

NUM_SPEC = 2
BLOCK = 16
# Draft up to batch 2, nothing above.
SCHEDULE = [(1, 2, NUM_SPEC), (3, 16, 0)]


def _scheduler(**kwargs) -> Scheduler:
    scheduler = create_scheduler(
        block_size=BLOCK,
        max_model_len=8 * BLOCK,
        max_num_batched_tokens=8 * BLOCK,
        enable_prefix_caching=True,
        num_speculative_tokens=NUM_SPEC,
        num_speculative_tokens_per_batch_size=SCHEDULE,
        use_v2_model_runner=True,
        **kwargs,
    )
    # The test harness drafts with ngram (stateless); the completeness
    # bookkeeping is independent of the drafter, so arm it directly.
    scheduler.lazy_draft = True
    scheduler.lazy_draft_state = True
    return scheduler


def _step(scheduler: Scheduler) -> tuple[SchedulerOutput, dict[str, int]]:
    """Schedule one step, accept every draft, propose for the next step."""
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    verified = {
        req_id: len(output.scheduled_spec_decode_tokens.get(req_id, ()))
        for req_id in req_ids
    }
    sampled = [[0] * (1 + verified[req_id]) for req_id in req_ids]
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
    # The runner proposes only when the step asked for drafts.
    num_drafts = output.num_spec_tokens_to_schedule
    if num_drafts > 0:
        running = [req.request_id for req in scheduler.running]
        scheduler.update_draft_token_ids(
            DraftTokenIds(running, [list(range(1, 1 + num_drafts))] * len(running))
        )
    return output, verified


def _add(scheduler: Scheduler, n: int, prefix: str, **kw) -> list[str]:
    reqs = create_requests(
        num_requests=n,
        num_tokens=2 * BLOCK,
        max_tokens=64,
        req_ids=[f"{prefix}{i}" for i in range(n)],
        **kw,
    )
    for req in reqs:
        scheduler.add_request(req)
    return [req.request_id for req in reqs]


def test_draft_keeps_state_by_method():
    cfg = SpeculativeConfig(model="ngram", num_speculative_tokens=2)
    assert not cfg.draft_keeps_state()
    for method in ("eagle", "eagle3", "mtp", "draft_model"):
        cfg.method = method
        assert cfg.draft_keeps_state()


def test_zero_draft_step_marks_requests_stale_for_good():
    scheduler = _scheduler()
    ids = _add(scheduler, 2, "a")
    _step(scheduler)  # prefill at batch 2: drafts proposed
    _, verified = _step(scheduler)
    assert all(verified[r] == NUM_SPEC for r in ids)
    assert not any(scheduler.requests[r].draft_stale for r in ids)

    # Two more requests: batch 4 schedules zero drafts, the drafter is skipped
    # and every request of that step is stale.
    more = _add(scheduler, 2, "b")
    output, verified = _step(scheduler)
    assert output.num_spec_tokens_to_schedule == 0
    assert all(scheduler.requests[r].draft_stale for r in ids + more)
    _, verified = _step(scheduler)
    assert all(verified[r] == 0 for r in ids + more)

    # Back to batch 2: the schedule drafts again, but the survivors never do.
    scheduler.finish_requests(more, RequestStatus.FINISHED_ABORTED)
    for _ in range(3):
        output, verified = _step(scheduler)
        assert output.num_spec_tokens_to_schedule == NUM_SPEC
        assert all(verified[r] == 0 for r in ids)
        assert all(not scheduler.requests[r].spec_token_ids for r in ids)

    # A fresh request next to a stale one drafts; the stale one still does not.
    scheduler.finish_requests(ids[1:], RequestStatus.FINISHED_ABORTED)
    (fresh,) = _add(scheduler, 1, "c")
    _step(scheduler)  # prefill of the fresh request
    _, verified = _step(scheduler)
    assert verified[fresh] == NUM_SPEC
    assert verified[ids[0]] == 0
    assert not scheduler.requests[fresh].draft_stale


def test_async_scheduler_gives_stale_requests_no_placeholders():
    scheduler = _scheduler(async_scheduling=True, speculative_method="ngram_gpu")
    ids = _add(scheduler, 2, "a")
    scheduler.schedule()
    assert all(scheduler.requests[r].spec_token_ids == [-1] * NUM_SPEC for r in ids)
    _add(scheduler, 2, "b")
    output = scheduler.schedule()
    assert output.num_spec_tokens_to_schedule == 0
    for req in scheduler.running:
        assert req.draft_stale
        assert req.spec_token_ids == []


def test_skipped_tokens_are_counted():
    scheduler = _scheduler()
    _add(scheduler, 3, "a")
    output = scheduler.schedule()
    assert output.num_spec_tokens_to_schedule == 0
    req_ids = list(output.num_scheduled_tokens)
    outputs = scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[0] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    stats = next(iter(outputs.values())).scheduler_stats.spec_decoding_stats
    assert stats.num_draft_skipped_tokens == output.total_num_scheduled_tokens
    assert stats.num_drafts == 0


def _cached_blocks(scheduler: Scheduler):
    pool = scheduler.kv_cache_manager.block_pool
    return [b for b in pool.blocks if b.block_hash is not None]


def test_stale_blocks_are_recomputed_by_a_drafting_request():
    scheduler = _scheduler()
    # Four requests share a prompt; batch 4 schedules no drafts, so the
    # blocks they cache have no drafter KV.
    first = _add(scheduler, 4, "a", same_prompt=True)
    _step(scheduler)
    _step(scheduler)
    assert all(scheduler.requests[r].draft_stale for r in first)
    scheduler.finish_requests(first, RequestStatus.FINISHED_ABORTED)
    cached = _cached_blocks(scheduler)
    assert cached and all(b.draft_stale for b in cached)

    # Alone, the next request with that prompt would draft: it must not
    # inherit the stale blocks, so it recomputes the prompt and stays fresh.
    (solo,) = _add(scheduler, 1, "b", same_prompt=True)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[solo] == 2 * BLOCK
    assert not scheduler.requests[solo].draft_stale
    _finish_step(scheduler, output)
    _step(scheduler)
    scheduler.finish_requests([solo], RequestStatus.FINISHED_ABORTED)
    # Its blocks are complete duplicates of the stale ones and win lookups.
    assert any(not b.draft_stale for b in _cached_blocks(scheduler))
    (again,) = _add(scheduler, 1, "c", same_prompt=True)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[again] < 2 * BLOCK
    assert not scheduler.requests[again].draft_stale
    _finish_step(scheduler, output)
    scheduler.finish_requests([again], RequestStatus.FINISHED_ABORTED)


def test_non_drafting_admission_reuses_stale_blocks_and_inherits():
    scheduler = _scheduler()
    first = _add(scheduler, 4, "a", same_prompt=True)
    _step(scheduler)
    _step(scheduler)
    scheduler.finish_requests(first[1:], RequestStatus.FINISHED_ABORTED)
    # Three running requests: a newcomer lands in the zero-draft band, so it
    # may take the stale blocks - and is stale itself from then on.
    others = _add(scheduler, 2, "b")
    _step(scheduler)
    (late,) = _add(scheduler, 1, "c", same_prompt=True)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[late] < 2 * BLOCK
    assert scheduler.requests[late].draft_stale
    _finish_step(scheduler, output)
    scheduler.finish_requests(first[:1] + others, RequestStatus.FINISHED_ABORTED)
    for _ in range(2):
        output, verified = _step(scheduler)
        assert output.num_spec_tokens_to_schedule == NUM_SPEC
        assert verified[late] == 0


def test_preemption_forgets_staleness():
    scheduler = _scheduler()
    ids = _add(scheduler, 3, "a")
    _step(scheduler)
    req = scheduler.requests[ids[0]]
    assert req.draft_stale
    scheduler.running.remove(req)
    scheduler._preempt_request(req, timestamp=0.0)
    assert not req.draft_stale


def _finish_step(scheduler: Scheduler, output: SchedulerOutput) -> None:
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[
                [0] * (1 + len(output.scheduled_spec_decode_tokens.get(r, ())))
                for r in req_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    if output.num_spec_tokens_to_schedule > 0:
        running = [r.request_id for r in scheduler.running]
        scheduler.update_draft_token_ids(
            DraftTokenIds(
                running,
                [list(range(1, 1 + output.num_spec_tokens_to_schedule))] * len(running),
            )
        )


def test_inherited_complete_blocks_keep_their_mark():
    scheduler = _scheduler()
    # One request alone computes a prompt with drafting on.
    (owner,) = _add(scheduler, 1, "a", same_prompt=True)
    _step(scheduler)
    _step(scheduler)
    complete = [b for b in _cached_blocks(scheduler) if not b.draft_stale]
    assert complete
    # Two more push the batch into the zero-draft band; a third newcomer with
    # the same prompt reuses the owner's blocks and goes stale, but those
    # blocks were computed with drafter KV and stay complete.
    _add(scheduler, 2, "b")
    _step(scheduler)
    (late,) = _add(scheduler, 1, "c", same_prompt=True)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[late] < 2 * BLOCK
    assert scheduler.requests[late].draft_stale
    assert all(not b.draft_stale for b in complete)
    _finish_step(scheduler, output)
    assert all(not b.draft_stale for b in complete)
