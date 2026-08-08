# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the predictive expert prefetcher (CPU-only)."""

import threading

import numpy as np
import pytest
import torch

from vllm.config.offload import ExpertTierOffloadConfig
from vllm.model_executor.offloader.expert_prefetch import ExpertPrefetcher

LAYERS = (0, 1, 2, 3)
NUM_EXPERTS = 16


class RecordingCache:
    """Minimal cache stub recording prefetch() calls thread-safely."""

    def __init__(self):
        self.calls: list[tuple[int, tuple[int, ...], bool]] = []
        self._lock = threading.Lock()

    def prefetch(self, layer_id, expert_ids, to_gpu=True):
        with self._lock:
            self.calls.append((layer_id, tuple(expert_ids), to_gpu))

    def by_layer(self, layer_id):
        with self._lock:
            return [ids for lid, ids, _ in self.calls if lid == layer_id]

    def union_by_layer(self, layer_id):
        return {e for ids in self.by_layer(layer_id) for e in ids}


@pytest.fixture
def cache():
    return RecordingCache()


def make_prefetcher(cache, predictors, **kwargs):
    kwargs.setdefault("layer_ids", LAYERS)
    kwargs.setdefault("num_experts", NUM_EXPERTS)
    return ExpertPrefetcher(cache=cache, predictors=predictors, **kwargs)


def observe(pf, layer_id, ids):
    pf.observe(layer_id, torch.tensor(ids, dtype=torch.int32))


def test_temporal_lookahead_targets(cache):
    pf = make_prefetcher(cache, {"temporal"}, lookahead=2)
    try:
        step1 = {0: [1, 2], 1: [3, 4], 2: [5], 3: [6, 7]}
        for layer_id, ids in step1.items():
            observe(pf, layer_id, ids)
        pf.flush(timeout=10)
        cache.calls.clear()

        observe(pf, 0, [8, 9])
        pf.flush(timeout=10)
        # Layer 0 observation targets only layer 0+lookahead with that
        # layer's previous-step ids (closer layers were already targeted
        # by their own k-th predecessors).
        assert cache.by_layer(1) == []
        assert cache.by_layer(2) == [(5,)]
        assert cache.by_layer(3) == []
    finally:
        pf.close()


def test_temporal_lookahead_wraps(cache):
    pf = make_prefetcher(cache, {"temporal"}, lookahead=2)
    try:
        for layer_id, ids in {0: [1], 1: [2], 2: [3], 3: [4]}.items():
            observe(pf, layer_id, ids)
        pf.flush(timeout=10)
        cache.calls.clear()

        observe(pf, 3, [10, 11])
        pf.flush(timeout=10)
        # Wraps to the next step's early layers.
        assert cache.by_layer(0) == []
        assert cache.by_layer(1) == [(2,)]
    finally:
        pf.close()


def test_popular_topk_and_ema(cache):
    pf = make_prefetcher(cache, {"popular"}, lookahead=1, popular_k=2)
    try:
        # Layer 1 popularity: 5 (x3), 7 (x2), 1 (x1).
        observe(pf, 1, [5, 7, 1])
        observe(pf, 1, [5, 7])
        observe(pf, 1, [5])
        pf.flush(timeout=10)
        counts = pf._counts[1]
        assert counts[5] > counts[7] > counts[1] > 0
        assert np.isclose(counts[1], 0.98**2)

        cache.calls.clear()
        observe(pf, 0, [0])
        pf.flush(timeout=10)
        assert cache.by_layer(1) == [(5, 7)]
    finally:
        pf.close()


def test_popular_skips_zero_counts(cache):
    pf = make_prefetcher(cache, {"popular"}, lookahead=1, popular_k=8)
    try:
        observe(pf, 1, [3])
        pf.flush(timeout=10)
        cache.calls.clear()
        observe(pf, 0, [0])
        pf.flush(timeout=10)
        # Only the single seen expert qualifies despite popular_k=8.
        assert cache.by_layer(1) == [(3,)]
    finally:
        pf.close()


def test_temporal_and_popular_union(cache):
    pf = make_prefetcher(cache, {"temporal", "popular"}, lookahead=1, popular_k=1)
    try:
        observe(pf, 1, [3, 4])
        observe(pf, 1, [3])
        pf.flush(timeout=10)
        cache.calls.clear()
        observe(pf, 0, [0])
        pf.flush(timeout=10)
        # Temporal = last step's {3}; popular top-1 = {3}; union with
        # the earlier step's temporal ids not included.
        assert cache.by_layer(1) == [(3,)]
    finally:
        pf.close()


def test_hash_predictor_exact_ids(cache):
    pf = make_prefetcher(cache, {"hash"})
    try:
        vocab = 10
        table0 = torch.arange(vocab * 2, dtype=torch.int32).reshape(vocab, 2) % 16
        table1 = (table0 + 1) % 16
        pf.note_hash_table(0, table0)
        pf.note_hash_table(1, table1)

        tokens = torch.tensor([[3], [7], [-1], [vocab]])
        pf.observe_sampled_tokens(tokens)
        pf.flush(timeout=10)

        expect0 = set(table0[[3, 7]].flatten().tolist())
        expect1 = set(table1[[3, 7]].flatten().tolist())
        assert cache.union_by_layer(0) == expect0
        assert cache.union_by_layer(1) == expect1
        assert cache.by_layer(2) == []
        assert pf.stats()["token_batches"] == 1
    finally:
        pf.close()


def test_hash_predictor_requires_tables(cache):
    pf = make_prefetcher(cache, {"hash"})
    try:
        pf.observe_sampled_tokens(torch.tensor([[1]]))
        pf.flush(timeout=10)
        assert cache.calls == []
    finally:
        pf.close()


class FakeGate(torch.nn.Module):
    def __init__(self, num_experts, hidden, bias=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(num_experts, hidden), requires_grad=False
        )
        self.e_score_correction_bias = bias


def test_gate_lookahead_predicts_next_layer(cache):
    hidden = 8
    pf = make_prefetcher(cache, {"gate"})
    try:
        gate = FakeGate(NUM_EXPERTS, hidden)
        # Make experts 2 and 5 win for an all-ones input.
        gate.weight.data[2] = 3.0
        gate.weight.data[5] = 2.0
        pf.note_gate(1, gate)

        pf.observe_hidden(0, torch.ones(1, hidden))
        pf.flush(timeout=10)
        ((layer, ids, to_gpu),) = cache.calls
        assert layer == 1
        assert {2, 5} <= set(ids)
        # Gate predictions promote to the pinned tier only.
        assert to_gpu is False
        assert pf.stats()["gate_batches"] == 1

        # Prefill-sized inputs are ignored (decode-only predictor).
        cache.calls.clear()
        pf.observe_hidden(0, torch.ones(8, hidden))
        pf.flush(timeout=10)
        assert cache.calls == []
    finally:
        pf.close()


def test_gate_lookahead_skips_hash_layers(cache):
    pf = make_prefetcher(cache, {"gate"})
    try:
        pf.note_hash_table(1, torch.zeros(4, 2, dtype=torch.int32))
        pf.note_gate(1, FakeGate(NUM_EXPERTS, 8))
        pf.observe_hidden(0, torch.ones(1, 8))
        pf.flush(timeout=10)
        assert cache.calls == []
    finally:
        pf.close()


def test_disabled_predictors_never_prefetch(cache):
    pf = make_prefetcher(cache, frozenset())
    try:
        observe(pf, 0, [1, 2])
        pf.note_hash_table(0, torch.zeros(4, 2, dtype=torch.int32))
        pf.observe_sampled_tokens(torch.tensor([[1]]))
        pf.observe_hidden(0, torch.ones(1, 8))
        pf.flush(timeout=10)
        assert cache.calls == []
        assert pf.stats()["observes"] == 0
    finally:
        pf.close()


def test_out_of_range_ids_filtered(cache):
    pf = make_prefetcher(cache, {"temporal"}, lookahead=1)
    try:
        observe(pf, 1, [-3, 2, NUM_EXPERTS, NUM_EXPERTS + 5])
        pf.flush(timeout=10)
        cache.calls.clear()
        observe(pf, 0, [0])
        pf.flush(timeout=10)
        assert cache.by_layer(1) == [(2,)]
    finally:
        pf.close()


def test_unknown_predictor_rejected(cache):
    with pytest.raises(ValueError, match="unknown predictors"):
        make_prefetcher(cache, {"psychic"})


def test_config_predictor_set_parsing():
    config = ExpertTierOffloadConfig()
    assert config.predictor_set == {"hash"}
    assert ExpertTierOffloadConfig(prefetch_predictors="off").predictor_set == set()
    assert ExpertTierOffloadConfig(prefetch_predictors="").predictor_set == set()
    assert ExpertTierOffloadConfig(prefetch_predictors="hash, gate").predictor_set == {
        "hash",
        "gate",
    }
    with pytest.raises(ValueError, match="unknown expert-tier prefetch"):
        ExpertTierOffloadConfig(prefetch_predictors="temporal,bogus")
