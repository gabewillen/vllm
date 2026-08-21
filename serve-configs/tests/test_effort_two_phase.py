# SPDX-License-Identifier: Apache-2.0
"""End-to-end scheduler tests for the v3 two-phase prefill decision (§13.3).

These run a real `Scheduler` on the served model's own config (no GPU, no
weights): the prompt is split at the effort-sentence seam, the body prefills
alone, the last body row comes back as a pooled hidden state, and the chosen
level's tail replaces the default-level one before generation starts.

Every failure mode has to land on today's behaviour with a byte-identical
prompt, because that is what makes the split safe to ship.
"""

import numpy as np
import pytest

from vllm.config.reasoning import DynamicEffortConfig, HiddenEffortConfig
from vllm.entrypoints.openai.chat_completion.dynamic_effort import (
    split_body_and_tails,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.effort_memory import EffortMemory
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

MODEL = "Qwen/Qwen3.8-27B-FP8"
HIDDEN = 5120
BLOCK = 16
NUM_LEVELS = 3
START, END = 151667, 151668
BODY = 96
TAILS = [[10, 11, 12, 13], [20, 21], [30, 31, 32, 33, 34, 35]]


def _scheduler(max_num_batched_tokens=2048, async_scheduling=None, **hidden_kw):
    from tests.v1.core.utils import create_scheduler

    kw = {} if async_scheduling is None else {"async_scheduling": async_scheduling}
    scheduler = create_scheduler(
        model=MODEL,
        enable_prefix_caching=True,
        block_size=BLOCK,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=2048,
        use_v2_model_runner=True,
        **kw,
    )
    kwargs = dict(enabled=True, memory_size=128, min_entries=4, k=4, flush_every=0)
    kwargs.update(hidden_kw)
    hidden = HiddenEffortConfig(**kwargs)
    cfg = DynamicEffortConfig(hidden_effort=hidden)
    scheduler._effort_cfg = cfg
    scheduler._effort_start_ids = [START]
    scheduler._effort_end_ids = [END]
    scheduler._effort_marker_seqs = []
    scheduler._effort_memory = EffortMemory(
        HIDDEN, hidden, model=MODEL, levels=NUM_LEVELS
    )
    return scheduler


def _fill_memory(scheduler, n=8, tokens=100):
    """Fill the memory and warm its digests, as a cold-phase server would.

    The digests only see the decisions the server actually faced, so the very
    first query of a fresh memory has no rank and falls back to the safe level.
    """
    memory = scheduler._effort_memory
    rng = np.random.default_rng(0)
    for i in range(n):
        memory.insert(
            rng.normal(size=HIDDEN).astype(np.float32),
            tokens,
            "natural",
            session_id=f"warm{i}",
        )
    for _ in range(4):
        result = memory.query(rng.normal(size=HIDDEN).astype(np.float32))
        memory.ranks(result)


def _prompt(seed: int, tail: list[int]) -> list[int]:
    rng = np.random.default_rng(seed)
    body = [int(x) for x in rng.integers(1000, 20000, size=BODY)]
    return body + tail


_NONE_HASH_READY = False


def _block_hasher():
    """One `init_none_hash` per process: it re-randomises the sentinel, which
    would make two identical prompts hash differently."""
    global _NONE_HASH_READY
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

    if not _NONE_HASH_READY:
        init_none_hash(sha256)
        _NONE_HASH_READY = True
    return get_request_block_hasher(BLOCK, sha256)


def _add(scheduler, req_id: str, seed: int = 7, body_len: int = BODY) -> Request:
    params = SamplingParams(
        max_tokens=60000,
        extra_args={
            "dynamic_effort": {
                "default_level": 0,
                "body_len": body_len,
                "tails": TAILS,
            }
        },
    )
    request = Request(
        request_id=req_id,
        prompt_token_ids=_prompt(seed, TAILS[0]),
        sampling_params=params,
        pooling_params=None,
        block_hasher=_block_hasher(),
    )
    scheduler.add_request(request)
    return request


def _runner_output(scheduler_output, vectors=None, sampled=None):
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[sampled.get(r, []) if sampled else [] for r in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        effort_prefill_states=vectors,
    )


# --------------------------------------------------------------- the frontend


def test_split_body_and_tails_keeps_the_lookahead_token_shared():
    shared = [1, 2, 3, 4, 5]
    variants = [shared + [90, 91], shared + [80], shared + [70, 71, 72]]
    body_len, tails = split_body_and_tails(variants)
    # One token back from the divergence, so the token an eagle drafter reads
    # ahead at the body boundary is the same whichever level is chosen.
    assert body_len == len(shared) - 1
    assert {t[0] for t in tails} == {shared[-1]}
    for variant, tail in zip(variants, tails):
        assert variant[:body_len] + tail == variant
    # Identical variants still split: the tails match, but the level's sentence
    # is an actuator of its own.
    assert split_body_and_tails([shared, shared]) == (4, [[5], [5]])
    # A body of fewer than two tokens has no usable seam.
    assert split_body_and_tails([[1, 2], [1, 3]]) is None
    assert split_body_and_tails([[1, 2, 3]]) is None


# ------------------------------------------------------------- the body phase


def test_body_prefill_emits_no_token():
    scheduler = _scheduler()
    request = _add(scheduler, "a")
    assert request.effort_decision_pending and request.effort_hold_prefill
    assert request.effort_body_len == BODY

    output = scheduler.schedule()
    # Only the body is scheduled, so the step is a non-final prefill chunk and
    # the model samples nothing for this request.
    assert output.num_scheduled_tokens["a"] == BODY
    assert request.num_computed_tokens == BODY
    assert request.is_prefill_chunk
    assert output.effort_prefill_capture == ["a"]

    engine_outputs = scheduler.update_from_output(output, _runner_output(output))
    assert all(not batch.outputs for batch in engine_outputs.values())


def test_a_held_request_is_not_scheduled_before_its_level_is_chosen():
    scheduler = _scheduler()
    _add(scheduler, "a")
    scheduler.schedule()  # body
    # No output processed yet: the tail must not prefill, or it would commit
    # the default level before the decision.
    held = scheduler.schedule()
    assert "a" not in held.num_scheduled_tokens
    assert scheduler.requests["a"].effort_decision_skips == 1


def test_held_request_falls_back_rather_than_stalling():
    """Liveness: a vector that never arrives must not park the request forever.

    The bound is generous on purpose - the async batch queue can put several
    schedule() calls between the body prefill and its output, and giving up
    early silently drops the decision.
    """
    from vllm.v1.core.sched.scheduler import MAX_EFFORT_DECISION_SKIPS

    scheduler = _scheduler()
    request = _add(scheduler, "a")
    scheduler.schedule()
    for _ in range(MAX_EFFORT_DECISION_SKIPS):
        assert "a" not in scheduler.schedule().num_scheduled_tokens
        assert request.effort_decision_pending
    scheduler.schedule()
    # The vector never arrived; the request resolves to the default level
    # instead of sitting in the running queue forever.
    assert not request.effort_decision_pending
    assert scheduler._effort_held_timeouts == 1
    assert scheduler.schedule().num_scheduled_tokens["a"] == len(TAILS[0])


def test_tail_appended_and_no_budget_shipped():
    scheduler = _scheduler(q_mid=0.0, q_high=0.0)  # everything routes to level 2
    _fill_memory(scheduler)
    request = _add(scheduler, "a")
    body = list(request.prompt_token_ids[:BODY])

    output = scheduler.schedule()
    vector = np.ones(HIDDEN, dtype=np.float16)
    scheduler.update_from_output(output, _runner_output(output, {"a": vector}))

    assert not request.effort_decision_pending
    assert list(request.prompt_token_ids) == body + TAILS[2]
    assert list(request._all_token_ids) == body + TAILS[2]
    assert request.num_prompt_tokens == BODY + len(TAILS[2])
    assert scheduler._effort["a"].level == 2

    # The tail prefills next, and only the tail: the body is already computed.
    # No thinking budget exists on this path - the level is the whole actuator.
    tail_step = scheduler.schedule()
    assert tail_step.num_scheduled_tokens["a"] == len(TAILS[2])
    assert not hasattr(tail_step, "thinking_budget_updates")
    assert request.sampling_params.thinking_token_budget is None


def test_decision_unavailable_falls_back_to_the_default_level():
    for kwargs, vector in (
        ({}, None),  # no vector at all
        ({"min_entries": 10_000}, np.ones(HIDDEN, dtype=np.float16)),  # cold
        ({"shadow": True, "q_mid": 0.0, "q_high": 0.0}, np.ones(HIDDEN, np.float16)),
    ):
        scheduler = _scheduler(**kwargs)
        _fill_memory(scheduler)
        request = _add(scheduler, "a")
        before = list(request.prompt_token_ids)
        output = scheduler.schedule()
        scheduler.update_from_output(
            output,
            _runner_output(output, {"a": vector} if vector is not None else None),
        )
        assert not request.effort_decision_pending
        assert list(request.prompt_token_ids) == before  # byte-identical prompt
        assert scheduler._effort["a"].level == 0
        assert request.status == RequestStatus.RUNNING


def test_fully_cached_body_still_yields_a_vector():
    scheduler = _scheduler()
    first = _add(scheduler, "a")
    output = scheduler.schedule()
    scheduler.update_from_output(
        output, _runner_output(output, {"a": np.ones(HIDDEN, dtype=np.float16)})
    )
    # Finish the first request so its blocks are cached and freed.
    tail = scheduler.schedule()
    scheduler.update_from_output(tail, _runner_output(tail, sampled={"a": [END]}))
    scheduler.finish_requests("a", RequestStatus.FINISHED_STOPPED)

    # The same prompt again: the body (and the default-level tail) are in the prefix
    # cache, but the decision has not been made, so the last body row must
    # still be computed and captured.
    second = _add(scheduler, "b", seed=7)
    assert list(second.prompt_token_ids) == list(first.prompt_token_ids)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens["b"] >= 1
    assert second.num_computed_tokens == BODY
    assert output.effort_prefill_capture == ["b"]


def test_prefix_cache_body_shared_across_levels():
    scheduler = _scheduler(q_mid=0.0, q_high=0.0)
    _fill_memory(scheduler)
    first = _add(scheduler, "a")
    out = scheduler.schedule()
    scheduler.update_from_output(
        out, _runner_output(out, {"a": np.ones(HIDDEN, dtype=np.float16)})
    )
    assert list(first.prompt_token_ids)[BODY:] == TAILS[2]
    tail = scheduler.schedule()
    scheduler.update_from_output(tail, _runner_output(tail, sampled={"a": [END]}))
    scheduler.finish_requests("a", RequestStatus.FINISHED_STOPPED)

    # A second request with the same body: the body blocks are shared, whatever
    # level the first one ended up on.
    second = _add(scheduler, "b", seed=7)
    out = scheduler.schedule()
    cached = second.num_computed_tokens - out.num_scheduled_tokens["b"]
    assert cached >= (BODY // BLOCK - 1) * BLOCK
    assert second.num_computed_tokens == BODY


def test_memory_records_the_finished_request():
    scheduler = _scheduler()
    request = _add(scheduler, "a")
    out = scheduler.schedule()
    vector = np.ones(HIDDEN, dtype=np.float16)
    scheduler.update_from_output(out, _runner_output(out, {"a": vector}))
    assert scheduler._effort_memory.n_entries == 0

    tail = scheduler.schedule()
    scheduler.update_from_output(tail, _runner_output(tail, sampled={"a": [START]}))
    state = scheduler._effort["a"]
    state.think_count = 321
    step = scheduler.schedule()
    scheduler.update_from_output(step, _runner_output(step, sampled={"a": [END]}))
    scheduler.finish_requests("a", RequestStatus.FINISHED_STOPPED)

    assert scheduler._effort_memory.n_entries == 1
    assert scheduler._effort_memory.n_valued == 1
    assert "a" not in scheduler._effort_vectors
    assert request.request_id not in scheduler._effort


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_split_survives_async_scheduling(async_scheduling):
    from tests.v1.core.utils import create_scheduler

    scheduler = create_scheduler(
        model=MODEL,
        enable_prefix_caching=True,
        block_size=BLOCK,
        max_num_batched_tokens=2048,
        max_model_len=2048,
        use_v2_model_runner=True,
        async_scheduling=async_scheduling,
    )
    hidden = HiddenEffortConfig(
        enabled=True,
        memory_size=128,
        min_entries=4,
        k=4,
        flush_every=0,
        q_mid=0.0,
        q_high=0.0,
    )
    scheduler._effort_cfg = DynamicEffortConfig(hidden_effort=hidden)
    scheduler._effort_start_ids = [START]
    scheduler._effort_end_ids = [END]
    scheduler._effort_marker_seqs = []
    scheduler._effort_memory = EffortMemory(
        HIDDEN, hidden, model=MODEL, levels=NUM_LEVELS
    )
    _fill_memory(scheduler)
    request = _add(scheduler, "a")
    body_step = scheduler.schedule()
    assert body_step.num_scheduled_tokens["a"] == BODY
    # The scheduler runs ahead of the output under async scheduling; the held
    # request must not be given a decode step in the meantime.
    ahead = scheduler.schedule()
    assert "a" not in ahead.num_scheduled_tokens
    scheduler.update_from_output(
        body_step, _runner_output(body_step, {"a": np.ones(HIDDEN, dtype=np.float16)})
    )
    assert list(request.prompt_token_ids)[BODY:] == TAILS[2]
    assert request.num_output_placeholders == 0


def test_a_multi_chunk_body_is_decided_by_the_step_that_computed_it():
    """The vector belongs to the step whose output carries it, not to a counter.

    A body wider than one prefill chunk finishes on a later step, and under
    async scheduling that step is already scheduled - so the request's token
    counter has already reached the body boundary - by the time the *earlier*
    chunk's output is processed. Resolving off the counter consumes the
    decision against an output that never captured anything, and the request
    silently runs at the default level with reason `no-vector`.
    """
    scheduler = _scheduler(
        max_num_batched_tokens=64, async_scheduling=True, q_mid=0.0, q_high=0.0
    )
    _fill_memory(scheduler)
    request = _add(scheduler, "a")
    body = list(request.prompt_token_ids[:BODY])

    first = scheduler.schedule()
    assert first.num_scheduled_tokens["a"] == 64
    assert first.effort_prefill_capture == []

    # Async scheduling runs a step ahead of the outputs: the chunk that
    # finishes the body is scheduled before the first chunk's output arrives.
    second = scheduler.schedule()
    assert second.num_scheduled_tokens["a"] == BODY - 64
    assert second.effort_prefill_capture == ["a"]
    assert request.num_computed_tokens == BODY

    scheduler.update_from_output(first, _runner_output(first))
    assert request.effort_decision_pending

    scheduler.update_from_output(
        second, _runner_output(second, {"a": np.ones(HIDDEN, dtype=np.float16)})
    )
    assert not request.effort_decision_pending
    assert list(request.prompt_token_ids) == body + TAILS[2]
    assert scheduler._effort["a"].level == 2
    assert "no-vector" not in scheduler._effort_default_reason


def test_body_stops_at_a_cacheable_block_boundary_under_mamba_align():
    """Qwen3.5 is hybrid, and "align" mode only caches SSM state on a block
    boundary, so a body whose seam is mid-block would deadlock the scheduler:
    the aligned chunk shrinks to zero and the request is never admitted.

    The scheduler therefore stops the body at the last cacheable boundary at or
    before the frontend's seam, and the tokens in between ride at the head of
    every tail - they are the same in every variant.
    """
    scheduler = _scheduler(q_mid=0.0, q_high=0.0)
    _fill_memory(scheduler)
    # The test scheduler builds a full-attention KV spec; the served profile is
    # hybrid GDN with an MTP drafter, which is what sets these two.
    scheduler.need_mamba_block_aligned_split = True
    scheduler.use_eagle = True

    seam = 90  # not a multiple of the 16-token block
    request = _add(scheduler, "a", body_len=seam)
    prompt = list(request.prompt_token_ids)
    boundary = request.effort_body_len
    assert boundary % BLOCK == 0 and boundary <= seam
    assert boundary == min(seam, len(prompt) - len(prompt) % BLOCK) // BLOCK * BLOCK
    for tail, variant in zip(request.effort_tail_variants, TAILS):
        assert tail == prompt[boundary:seam] + variant

    output = scheduler.schedule()
    assert output.num_scheduled_tokens["a"] == boundary
    assert output.effort_prefill_capture == ["a"]
    scheduler.update_from_output(
        output, _runner_output(output, {"a": np.ones(HIDDEN, dtype=np.float16)})
    )
    assert list(request.prompt_token_ids) == prompt[:seam] + TAILS[2]
    # And the rest of the prompt still prefills.
    assert scheduler.schedule().num_scheduled_tokens["a"] == (seam - boundary) + len(
        TAILS[2]
    )


def test_no_usable_seam_keeps_the_default_level():
    """No cacheable boundary, or a seam far from the prompt's tail, means no
    decision at all: the request runs at the server default level.

    The prompt is then byte-identical to the pre-v3 rendering, the whole prompt
    prefills in one go with nothing held back, and - since the level is the only
    actuator - nothing else about the request changes either.
    """
    scheduler = _scheduler(q_mid=0.0, q_high=0.0)
    _fill_memory(scheduler)
    request = _add(scheduler, "a", body_len=8)
    before = list(request.prompt_token_ids)
    assert not request.effort_decision_pending
    assert not request.effort_hold_prefill
    assert scheduler._effort["a"].level == 0

    output = scheduler.schedule()
    assert output.num_scheduled_tokens["a"] == len(before)
    assert output.effort_prefill_capture == []

    scheduler.update_from_output(output, _runner_output(output, sampled={"a": [START]}))
    assert list(request.prompt_token_ids) == before
    assert request.status == RequestStatus.RUNNING


def test_a_seam_far_from_the_prompt_tail_is_not_worth_a_split():
    # An agent turn whose effort sentence is not at the tail: the body would
    # describe a small prefix of what the model reads, so no split is made.
    scheduler = _scheduler(split_min_fraction=0.75)
    _fill_memory(scheduler)
    request = _add(scheduler, "a", body_len=32)  # 32 of 100 prompt tokens
    assert not request.effort_decision_pending and not request.effort_hold_prefill
    # The same seam is worth a split when it does cover the prompt.
    lenient = _scheduler(split_min_fraction=0.1)
    _fill_memory(lenient)
    other = _add(lenient, "a", body_len=32)
    assert other.effort_hold_prefill and other.effort_body_len == 32
