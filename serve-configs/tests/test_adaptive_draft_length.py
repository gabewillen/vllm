# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the adaptive draft-length patch (0005), CPU only."""

import pytest
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.core.sched.scheduler import adaptive_num_spec_tokens, update_accepted_ema
from vllm.v1.worker.gpu.cudagraph_utils import (
    decode_query_len_allowed,
    decode_query_lens_for_spec,
)

SCHEDULE = [[1, 8, 7], [9, 32, 2], [33, 96, 0]]


def _cfg(**kw):
    return SpeculativeConfig(method="ngram", prompt_lookup_max=3, **kw)


def test_adaptive_requires_at_least_two_tokens():
    with pytest.raises(ValueError, match="num_speculative_tokens >= 2"):
        _cfg(num_speculative_tokens=1, adaptive_draft_length=True)


def test_adaptive_min_tokens_bounded_by_k():
    with pytest.raises(ValueError, match="must not exceed"):
        _cfg(num_speculative_tokens=4, adaptive_draft_length=True,
             adaptive_draft_min_tokens=5)


def test_adaptive_defaults_accept():
    cfg = _cfg(num_speculative_tokens=7, adaptive_draft_length=True)
    assert cfg.adaptive_draft_margin == 2.0
    assert cfg.adaptive_draft_min_tokens == 1
    assert cfg.draft_lm_head_dtype == "auto"


def test_draft_lm_head_dtype_is_validated():
    with pytest.raises(ValueError):
        _cfg(num_speculative_tokens=2, draft_lm_head_dtype="int3")


@pytest.mark.parametrize(
    "emas,expected",
    [
        ([], 7),  # no requests -> full length
        ([None], 7),  # fresh request keeps the full length
        ([1.6], 4),  # ceil(1.6 + 2.0)
        ([1.6, 4.4], 7),  # most-accepting request wins, capped at K
        ([0.0], 2),  # ceil(0 + 2) = 2, above the floor
        ([None, 3.0], 7),  # any fresh request -> full length
    ],
)
def test_adaptive_num_spec_tokens(emas, expected):
    assert (
        adaptive_num_spec_tokens(emas=emas, max_tokens=7, min_tokens=1, margin=2.0)
        == expected
    )


def test_adaptive_num_spec_tokens_respects_min():
    assert adaptive_num_spec_tokens(emas=[0.0], max_tokens=7, min_tokens=3,
                                    margin=0.0) == 3


def test_update_accepted_ema_seeds_then_smooths():
    assert update_accepted_ema(prev=None, num_accepted=3, alpha=0.7) == 3.0
    ema = update_accepted_ema(prev=2.0, num_accepted=0, alpha=0.7)
    assert ema == pytest.approx(1.4)


def test_capture_lengths_follow_schedule_and_adaptive_ranges():
    cfg = _cfg(num_speculative_tokens=7, adaptive_draft_length=True,
               num_speculative_tokens_per_batch_size=SCHEDULE)
    lens, ranged = decode_query_lens_for_spec(
        speculative_config=cfg, decode_query_len=8, max_num_reqs=96
    )
    assert sorted(lens) == list(range(1, 9))
    assert ranged == [(1, 8, {2, 3, 4, 5, 6, 7, 8}), (9, 32, {2, 3})]
    schedule_lens = set(lens) - {q for _, _, qs in ranged for q in qs}
    assert schedule_lens == {1}
    # adaptive length only for batch sizes inside its range
    assert decode_query_len_allowed(query_len=5, num_reqs=4,
                                    schedule_query_lens=schedule_lens,
                                    ranged_query_lens=ranged)
    assert not decode_query_len_allowed(query_len=5, num_reqs=12,
                                        schedule_query_lens=schedule_lens,
                                        ranged_query_lens=ranged)
    # schedule-derived length stays unrestricted
    assert decode_query_len_allowed(query_len=1, num_reqs=90,
                                    schedule_query_lens=schedule_lens,
                                    ranged_query_lens=ranged)


def test_capture_lengths_draft_decode_manager_is_untouched():
    cfg = _cfg(num_speculative_tokens=7, adaptive_draft_length=True,
               num_speculative_tokens_per_batch_size=SCHEDULE)
    assert decode_query_lens_for_spec(
        speculative_config=cfg, decode_query_len=1, max_num_reqs=96
    ) == ([1], [])


def test_capture_lengths_static_k():
    cfg = _cfg(num_speculative_tokens=3)
    assert decode_query_lens_for_spec(
        speculative_config=cfg, decode_query_len=4, max_num_reqs=8
    ) == ([4], [])
