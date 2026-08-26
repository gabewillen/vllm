# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic effort, full-default form: what a resubmission recomputes.

A non-default level (and the think-off vote) splices another tail at the seam
and re-admits the request. Its prefix hit can only resume at a mamba-align
chunk end, so the decision prefill must end a chunk at the last aligned
boundary before the seam, and the vote must not pay that resubmission once
per vote.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.config.reasoning import DynamicEffortConfig, HiddenEffortConfig
from vllm.sampling_params import RequestOutputKind
from vllm.v1.core.sched.effort_controller import new_effort_state
from vllm.v1.core.sched.scheduler import (
    Scheduler,
    draw_effort_level,
    effort_level_vote_verdict,
)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import RoutedPromptUpdate
from vllm.v1.engine.output_processor import RequestState
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


def _run_step(scheduler: Scheduler, request: Request, sampled: list[int]):
    output = scheduler.schedule()
    runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[sampled],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, runner_output)
    return output


def _requeue(scheduler: Scheduler, request: Request) -> None:
    """What `update_from_output` does with an effort-requeued request."""
    scheduler.running.remove(request)
    scheduler.waiting.prepend_request(request)


def _create_hybrid_align_scheduler() -> Scheduler:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = 100
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["fa"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        log_stats=True,
    )


def _hold_full_default(request: Request, seam: int, tails: list[list[int]]) -> None:
    """Mark `request` as the full-default form held for its decision."""
    request.effort_body_len = request.num_prompt_tokens
    request.effort_seam = seam
    request.effort_tail_variants = tails
    request.effort_hold_prefill = True
    request.effort_default_level = len(tails) - 1


def test_decision_prefill_ends_chunk_at_seam_boundary():
    """The held prefill stops at the last aligned boundary before the seam,
    on top of the eagle back-off stop, so that state gets cached."""
    prompt_len = 6 * BLOCK_SIZE + 5
    seam = 6 * BLOCK_SIZE + 3
    [request] = create_requests(
        num_requests=1, num_tokens=prompt_len, block_size=BLOCK_SIZE
    )
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        use_eagle=True,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        hash_block_size=BLOCK_SIZE,
    )

    def chunks() -> list[int]:
        request.num_computed_tokens = 0
        sizes = []
        while request.num_computed_tokens < prompt_len:
            num_new = Scheduler._mamba_block_aligned_split(
                stub, request, prompt_len - request.num_computed_tokens
            )
            sizes.append(num_new)
            request.num_computed_tokens += num_new
        return sizes

    assert chunks() == [5 * BLOCK_SIZE, BLOCK_SIZE + 5]
    _hold_full_default(request, seam, [[7], [8]])
    assert chunks() == [5 * BLOCK_SIZE, BLOCK_SIZE, 5]
    request.effort_hold_prefill = False
    assert chunks() == [5 * BLOCK_SIZE, BLOCK_SIZE + 5]


@pytest.mark.parametrize("stop_at_seam", [True, False])
def test_resubmission_resumes_at_seam_boundary_on_hybrid_align(stop_at_seam: bool):
    """Re-admission after `_resubmit_effort_tail` reuses every cached block up
    to the aligned boundary before the seam - the FA blocks and the mamba
    state the decision prefill left there - so only the seam's partial block
    and the new tail are scheduled. Without a chunk end at that boundary
    (`stop_at_seam=False` prefills as the plain split would) the mamba group
    has no state to resume from and the whole prompt is recomputed."""
    scheduler = _create_hybrid_align_scheduler()
    prompt_len = 100
    seam = 70
    keep = seam // BLOCK_SIZE * BLOCK_SIZE
    [request] = create_requests(
        num_requests=1, num_tokens=prompt_len, max_tokens=8, block_size=BLOCK_SIZE
    )
    default_tail = list(request.prompt_token_ids[seam:])
    low_tail = [200, 201, 202, 203, 204]
    _hold_full_default(request, seam if stop_at_seam else 0, [low_tail, default_tail])
    scheduler.add_request(request)

    chunks = []
    while request.num_computed_tokens < prompt_len:
        output = _run_step(scheduler, request, [])
        chunks.append(output.num_scheduled_tokens[request.request_id])
    assert chunks == ([keep, 96 - keep, 4] if stop_at_seam else [96, 4])
    assert _mamba_state_cached(scheduler, request, keep) == stop_at_seam
    _run_step(scheduler, request, [42])
    assert request.num_computed_tokens == prompt_len

    request.effort_hold_prefill = False
    request.effort_seam = seam
    scheduler._resubmit_effort_tail(request, low_tail)
    _requeue(scheduler, request)
    assert request.status == RequestStatus.WAITING
    assert request.num_computed_tokens == 0
    assert list(request.prompt_token_ids) == (
        list(request.prompt_token_ids[:seam]) + low_tail
    )

    new_len = seam + len(low_tail)
    recomputed = 0
    while request.num_computed_tokens < new_len:
        output = _run_step(scheduler, request, [])
        recomputed += output.num_scheduled_tokens[request.request_id]
    assert recomputed == new_len - (keep if stop_at_seam else 0)


def _mamba_state_cached(scheduler: Scheduler, request: Request, num_tokens: int):
    """Whether the prefix cache holds the mamba state after `num_tokens`."""
    block_pool = scheduler.kv_cache_manager.block_pool
    block_hash = request.block_hashes[num_tokens // BLOCK_SIZE - 1]
    return block_pool.get_cached_block(block_hash, [1]) is not None


def _vote_request(scheduler: Scheduler, off_votes: int = 3) -> Request:
    [request] = create_requests(
        num_requests=1, num_tokens=100, max_tokens=8, block_size=BLOCK_SIZE
    )
    seam = 70
    default_tail = list(request.prompt_token_ids[seam:])
    _hold_full_default(request, seam, [[], [300, 301], default_tail])
    request.effort_off_append = [400]
    request.effort_meta_tail = [500, 501, 502]
    request.effort_meta_stop_ids = {3}
    request.effort_yes_ids = {1}
    request.effort_no_ids = {2}
    request.effort_off_votes = off_votes
    request.effort_meta_max_tokens = 8
    scheduler.add_request(request)
    _run_step(scheduler, request, [42])
    assert request.num_computed_tokens == 100
    request.effort_hold_prefill = False
    return request


def _assert_frontend_sees_prompt(update: RoutedPromptUpdate, request: Request):
    """The rewritten prompt reaches the frontend's request state, so the
    reasoning parser adjusts to the tail the engine actually ran."""
    assert update.prompt_token_ids == list(request.prompt_token_ids)
    state = RequestState(
        request_id=request.request_id,
        external_req_id=request.request_id,
        parent_req=None,
        request_index=0,
        lora_request=None,
        output_kind=RequestOutputKind.FINAL_ONLY,
        prompt=None,
        prompt_token_ids=list(request.prompt_token_ids[:70]),
        prompt_embeds=None,
        logprobs_processor=None,
        detokenizer=None,
        max_tokens_param=None,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
    )
    state.routed_prompt_revision = update.revision - 1
    assert state.apply_routed_prompt(update, tokenizer=None)
    assert state.prompt_token_ids == list(request.prompt_token_ids)
    assert state.prompt_len == request.num_prompt_tokens


def test_extend_effort_prompt_surfaces_off_tail():
    """Think-off in place: the off tail is appended to the prompt and the
    routed prompt update carries the full extended prompt."""
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    request = _vote_request(scheduler)
    body = list(request.prompt_token_ids)
    update = scheduler._extend_effort_prompt(request, request.effort_off_append)
    assert update.revision == 1
    assert list(request.prompt_token_ids) == body + [400]
    assert request.num_prompt_tokens == len(body) + 1
    assert list(request.all_token_ids) == body + [400]
    _assert_frontend_sees_prompt(update, request)


def _count_resubmissions(scheduler: Scheduler) -> list[list[int]]:
    tails: list[list[int]] = []
    original = scheduler._resubmit_effort_tail

    def counting(request: Request, tail: list[int]):
        tails.append(list(tail))
        return original(request, tail)

    scheduler._resubmit_effort_tail = counting  # type: ignore[method-assign]
    return tails


def _vote_step(scheduler: Scheduler, request: Request, entries: dict[int, float]):
    """Run the gate's step: the sampled token plus its top logprobs."""
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] > 0
    token_ids = [next(iter(entries))] + list(entries)
    scores = [math.log(entries[t]) for t in token_ids]
    runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[token_ids[0]]],
        logprobs=LogprobsLists(
            logprob_token_ids=np.array([token_ids]),
            logprobs=np.array([scores], dtype=np.float32),
            sampled_token_ranks=np.array([0]),
        ),
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    return scheduler.update_from_output(output, runner_output)


@pytest.mark.parametrize("off", [True, False])
def test_off_vote_is_one_resubmission(off: bool):
    """All votes come from one step's logprobs: the gate resubmits once for
    the question and once for the verdict, not once per vote."""
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    scheduler._effort_cfg = DynamicEffortConfig(
        hidden_effort=HiddenEffortConfig(
            enabled=True, think_off_level=True, default_level=2
        )
    )
    request = _vote_request(scheduler)
    params = request.sampling_params
    assert params is not None
    client_sampling = (params.temperature, params.seed, params.logprobs)
    tails = _count_resubmissions(scheduler)

    update = scheduler._start_effort_off_vote(request)
    assert update is not None
    _requeue(scheduler, request)
    assert tails == [request.effort_meta_tail]
    assert params.logprobs == 20 and params.temperature == 0.7

    # Every entry is a yes (or a no): the draws are certain either way.
    entries = {1: 1.0} if off else {2: 1.0}
    outputs = _vote_step(scheduler, request, entries)
    [engine_output] = outputs[0].outputs
    assert engine_output.new_token_ids == []
    assert engine_output.routed_prompt_update is not None

    assert request.effort_meta_votes == [off] * 3
    assert len(tails) == 2
    seam_prefix = list(request.prompt_token_ids[:70])
    expected_tail = list(request.effort_tail_variants[2]) + [400] if off else [300, 301]
    assert tails[1] == expected_tail
    assert list(request.prompt_token_ids) == seam_prefix + expected_tail
    _assert_frontend_sees_prompt(engine_output.routed_prompt_update, request)
    assert request.status == RequestStatus.WAITING
    assert (params.temperature, params.seed, params.logprobs) == client_sampling
    assert not request.effort_meta_phase


def test_off_vote_without_logprobs_samples_each_vote():
    """A runner that returns no logprobs keeps the sampled-walk gate: one
    resubmission per vote."""
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    scheduler._effort_cfg = DynamicEffortConfig(
        hidden_effort=HiddenEffortConfig(
            enabled=True, think_off_level=True, default_level=2
        )
    )
    request = _vote_request(scheduler)
    tails = _count_resubmissions(scheduler)
    scheduler._start_effort_off_vote(request)
    _requeue(scheduler, request)
    for _ in range(3):
        _run_step(scheduler, request, [1])
    assert request.effort_meta_votes == [True, True, True]
    assert [len(t) for t in tails] == [
        3,
        3,
        3,
        len(request.effort_tail_variants[2]) + 1,
    ]


def test_level_vote_draws_and_rules():
    """Draws index the categorical by cumulative mass; `max` raises on one
    draw and lowers only by consensus, `median` takes the middle draw."""
    probs = [0.2, 0.5, 0.3]
    assert [draw_effort_level(u, probs) for u in (0.0, 0.19, 0.2, 0.69, 0.7, 1.0)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert effort_level_vote_verdict([0, 0, 2], "max", 1) == 2
    assert effort_level_vote_verdict([0, 0, 0], "max", 1) == 0
    assert effort_level_vote_verdict([0, 0, 2], "median", 1) == 0
    assert effort_level_vote_verdict([0, 2, 2], "median", 1) == 2
    assert effort_level_vote_verdict([], "max", 1) == 1


def _level_vote_request(scheduler: Scheduler, rule: str = "max") -> Request:
    request = _vote_request(scheduler)
    request.effort_yes_ids = set()
    request.effort_no_ids = set()
    request.effort_level_word_ids = [{1}, {2}, {5}]
    request.effort_level_vote_rule = rule
    cfg = scheduler._effort_cfg
    assert cfg is not None
    scheduler._effort[request.request_id] = new_effort_state(
        request.request_id, cfg, [], [], request.prompt_token_ids
    )
    scheduler._effort[request.request_id].level = 2
    return request


_LEVEL_VOTE_CASES = {
    # entries -> (level, rendered tail)
    "none": ({1: 1.0}, 0, "off"),
    "brief": ({2: 1.0}, 1, [300, 301]),
    "extended": ({5: 1.0}, 2, "default"),
    "unparseable": ({9: 1.0}, 2, "default"),
    "no-mass-in-top": ({9: 0.1}, 2, "default"),
}


@pytest.mark.parametrize("case", sorted(_LEVEL_VOTE_CASES))
def test_level_vote_is_one_resubmission(case: str):
    """All draws come from the first token's logprobs; the memory decision is
    bypassed and the voted level's tail is rendered in one resubmission."""
    entries, expected_level, expected_tail = _LEVEL_VOTE_CASES[case]
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    scheduler._effort_cfg = DynamicEffortConfig(
        hidden_effort=HiddenEffortConfig(
            enabled=True, think_off_level=True, default_level=2, level_vote=True
        )
    )
    request = _level_vote_request(scheduler)
    params = request.sampling_params
    assert params is not None
    client_sampling = (params.temperature, params.seed, params.logprobs)
    tails = _count_resubmissions(scheduler)

    # No memory, no vector: the vote still decides.
    request.effort_decision_pending = True
    update = scheduler._resolve_effort_decision(request, None)
    assert update is not None and request.effort_meta_phase
    _requeue(scheduler, request)
    assert tails == [request.effort_meta_tail]

    outputs = _vote_step(scheduler, request, entries)
    [engine_output] = outputs[0].outputs
    assert engine_output.new_token_ids == []
    assert engine_output.routed_prompt_update is not None
    assert len(tails) == 2
    if expected_tail == "off":
        expected = list(request.effort_tail_variants[2]) + [400]
    elif expected_tail == "default":
        expected = list(request.effort_tail_variants[2])
    else:
        expected = expected_tail
    assert tails[1] == expected
    assert list(request.prompt_token_ids) == list(request.prompt_token_ids[:70]) + (
        expected
    )
    _assert_frontend_sees_prompt(engine_output.routed_prompt_update, request)
    assert max(request.effort_level_votes) == expected_level
    assert len(request.effort_level_votes) == 3
    state = scheduler._effort[request.request_id]
    report = state.report
    assert report["level"] == expected_level and report["decided"] == 1
    assert report["level_votes"] == request.effort_level_votes
    assert len(report["vote_probs"]) == 3
    assert sum(report["vote_probs"]) == pytest.approx(1.0, abs=1e-3)
    if case == "unparseable":
        assert report["vote_probs"] == [0.0, 0.0, 1.0]
    assert (params.temperature, params.seed, params.logprobs) == client_sampling
    assert not request.effort_meta_phase
    assert request.status == RequestStatus.WAITING


def test_level_vote_median_rule():
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    scheduler._effort_cfg = DynamicEffortConfig(
        hidden_effort=HiddenEffortConfig(
            enabled=True,
            think_off_level=True,
            default_level=2,
            level_vote=True,
            level_vote_rule="median",
        )
    )
    request = _level_vote_request(scheduler, rule="median")
    tails = _count_resubmissions(scheduler)
    request.effort_decision_pending = True
    scheduler._resolve_effort_decision(request, None)
    _requeue(scheduler, request)
    _vote_step(scheduler, request, {1: 0.5, 2: 0.5, 5: 1e-9})
    votes = request.effort_level_votes
    assert sorted(votes)[1] == scheduler._effort[request.request_id].level
    assert tails[1] != [] or votes == [0, 0, 0]


def test_level_vote_without_logprobs_samples_each_vote():
    """Without logprobs each draw is one sampled walk: the first answer token
    names the level, a stop id is the default level."""
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    scheduler._effort_cfg = DynamicEffortConfig(
        hidden_effort=HiddenEffortConfig(
            enabled=True, think_off_level=True, default_level=2, level_vote=True
        )
    )
    request = _level_vote_request(scheduler)
    tails = _count_resubmissions(scheduler)
    request.effort_decision_pending = True
    scheduler._resolve_effort_decision(request, None)
    _requeue(scheduler, request)
    for sampled in ([1], [3], [2]):
        _run_step(scheduler, request, sampled)
    assert request.effort_level_votes == [0, 2, 1]
    assert scheduler._effort[request.request_id].level == 2
    assert [len(t) for t in tails] == [3, 3, 3, len(request.effort_tail_variants[2])]


def test_forced_off_and_shadow_skip_the_level_vote():
    """`force_off` keeps the off gate's verdict; shadow renders the default."""
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    scheduler._effort_cfg = DynamicEffortConfig(
        hidden_effort=HiddenEffortConfig(
            enabled=True, think_off_level=True, default_level=2, level_vote=True
        )
    )
    request = _level_vote_request(scheduler)
    request.effort_force_off = True
    request.effort_decision_pending = True
    update = scheduler._resolve_effort_decision(request, None)
    assert update is not None
    assert request.effort_meta_phase and request.effort_level_votes == []
    _requeue(scheduler, request)
    _vote_step(scheduler, request, {1: 1.0})
    assert request.effort_level_votes == [0, 0, 0]
    assert scheduler._effort[request.request_id].level == 0


def test_vote_probability_is_tempered_top_logprob_mass():
    scheduler = create_scheduler(block_size=BLOCK_SIZE)
    [request] = create_requests(num_requests=1, num_tokens=8, block_size=BLOCK_SIZE)
    request.effort_yes_ids = {1, 5}
    logprobs = LogprobsLists(
        logprob_token_ids=np.array([[1, 1, 2, 5, 7]]),
        logprobs=np.array(
            [[math.log(0.5), math.log(0.5), math.log(0.3), math.log(0.1), -np.inf]],
            dtype=np.float32,
        ),
        sampled_token_ranks=np.array([0]),
    )
    p_yes = scheduler._effort_vote_yes_probability(request, logprobs)
    assert p_yes is not None
    weights = {t: p ** (1 / 0.7) for t, p in {1: 0.5, 2: 0.3, 5: 0.1}.items()}
    expected = (weights[1] + weights[5]) / sum(weights.values())
    assert p_yes == pytest.approx(expected, rel=1e-5)
