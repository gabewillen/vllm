# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the v3 online effort memory (§13.4, §13.5).

The memory is the whole v3 signal: a ring of pooled prefill states the server
filled itself, queried by cosine and valued by the reasoning tokens each entry
actually spent. Nothing here is trained, so what has to be pinned is the
bookkeeping - who evicts whom, which entries may contribute a value, and that
the asymmetric map can only lower a request's level when both confidence gates
agree.
"""

import math

import numpy as np
import pytest

from vllm.config.reasoning import HiddenEffortConfig
from vllm.v1.core.sched.effort_memory import (
    MEMORY_VERSION,
    EffortMemory,
    MemoryQuery,
    decide_effort_level,
)

DIM = 8


def _cfg(**kw) -> HiddenEffortConfig:
    base = dict(enabled=True, memory_size=128, min_entries=4, k=4, flush_every=0)
    base.update(kw)
    return HiddenEffortConfig(**base)


def _mem(**kw) -> EffortMemory:
    return EffortMemory(DIM, _cfg(**kw), model="test-model", levels=3)


def _vec(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=DIM).astype(np.float32)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


# ------------------------------------------------------------------ eviction


def test_ring_evicts_fifo_and_caps_one_session():
    # memory_size 128 -> at most ceil(128/64) = 2 entries per session id.
    memory = _mem(memory_size=128)
    assert memory.per_session_cap == 2

    for i in range(5):
        memory.insert(_vec(i), 100 + i, "natural", session_id="chatty")
    # One conversation can never hold more than its cap, so it cannot evict
    # the rest of the memory.
    assert memory.n_entries == 2
    assert len(memory._by_session["chatty"]) == 2

    memory = _mem(memory_size=16)
    for i in range(20):
        memory.insert(_vec(i), i, "natural", session_id=f"s{i}")
    assert memory.n_entries == 16
    # FIFO: the four oldest sessions were overwritten by the four newest.
    assert set(memory._by_session) == {f"s{i}" for i in range(4, 20)}


def test_censored_closes_are_keys_not_values():
    memory = _mem(k=5)
    query_vec = _vec(1)
    # One natural close far away, three forced/soft closes right on top of the
    # query: the estimate must come from the natural close alone.
    memory.insert(-query_vec, 10_000, "natural", session_id="a")
    for i, kind in enumerate(("forced", "soft", "forced")):
        memory.insert(query_vec + 0.01 * _vec(51 + i), 5, kind, session_id=f"b{i}")
    memory.insert(_vec(9), 1_000, "natural", session_id="c")

    result = memory.query(query_vec)
    assert result is not None
    assert result.n_entries == 5
    # The three censored neighbours are the nearest, so they dominate novelty...
    assert result.max_cos > 0.9
    # ...and contribute nothing to the value.
    assert result.n_valued == 2
    values = {math.log1p(10_000), math.log1p(1_000)}
    assert min(values) <= result.estimate <= max(values)


# --------------------------------------------------------------------- query


def test_query_matches_numpy_reference():
    memory = _mem(memory_size=64, k=4, temperature=0.05)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(20, DIM)).astype(np.float32)
    tokens = rng.integers(16, 8192, size=20)
    for i, (vec, tok) in enumerate(zip(vectors, tokens)):
        memory.insert(vec, int(tok), "natural", session_id=f"s{i}")

    probe = rng.normal(size=DIM).astype(np.float32)
    got = memory.query(probe)
    assert got is not None

    keys = np.stack([_unit(v) for v in vectors])
    sims = keys @ _unit(probe)
    top = np.argsort(-sims)[:4]
    best = sims[top[0]]
    weights = np.exp((sims[top] - best) / 0.05)
    weights /= weights.sum()
    values = np.log1p(tokens[top].astype(np.float64))
    expected = float((weights * values).sum())
    expected_spread = float(math.sqrt((weights * (values - expected) ** 2).sum()))

    assert got.estimate == pytest.approx(expected, rel=1e-4)
    assert got.spread == pytest.approx(expected_spread, rel=1e-4)
    assert got.novelty == pytest.approx((1.0 - best) / 2.0, rel=1e-5)


def test_cold_memory_returns_none():
    memory = _mem(k=4, min_entries=8)
    assert memory.query(_vec(0)) is None  # nothing to compare against
    for i in range(4):
        memory.insert(_vec(i), 100, "natural", session_id=f"s{i}")
    # k entries answer the query - the digests warm during the cold phase - but
    # the memory is not `ready`, so the caller keeps the default-level path.
    assert memory.query(_vec(0)) is not None
    assert not memory.ready
    for i in range(4, 8):
        memory.insert(_vec(i), 100, "natural", session_id=f"s{i}")
    assert memory.ready


# --------------------------------------------------------------- persistence


def test_persistence_roundtrip_and_version_mismatch(tmp_path):
    path = str(tmp_path / "memory.npz")
    memory = _mem(memory_path=path, flush_every=0)
    for i in range(10):
        memory.insert(_vec(i), 100 + i, "natural", session_id=f"s{i}")
    memory.query(_vec(0))  # feeds the digests, which must survive too
    memory.save()

    warm = EffortMemory(DIM, memory.cfg, model="test-model", levels=3)
    assert warm.load()
    assert warm.n_entries == memory.n_entries
    np.testing.assert_allclose(
        warm._keys[: warm.n_entries],
        memory._keys[: memory.n_entries].astype(np.float16).astype(np.float32),
        atol=1e-3,
    )
    np.testing.assert_allclose(warm._values[: warm.n_entries], memory._values[:10])
    assert set(warm._by_session) == set(memory._by_session)

    # A different model, hidden size or level count invalidates the stored values.
    other_model = EffortMemory(DIM, memory.cfg, model="other", levels=2)
    assert not other_model.load()
    other_dim = EffortMemory(DIM + 1, memory.cfg, model="test-model", levels=0)
    assert not other_dim.load()
    other_levels = EffortMemory(DIM, memory.cfg, model="test-model", levels=4)
    assert not other_levels.load()
    assert memory.state_dict()["version"] == MEMORY_VERSION


def test_flush_every_writes_atomically(tmp_path):
    path = str(tmp_path / "memory.npz")
    memory = _mem(memory_path=path, flush_every=3)
    for i in range(2):
        memory.insert(_vec(i), 10, "natural", session_id=f"s{i}")
    assert not (tmp_path / "memory.npz").exists()
    memory.insert(_vec(2), 10, "natural", session_id="s2")
    assert (tmp_path / "memory.npz").exists()
    # The temp file is gone: the write was a rename, not a truncate-in-place.
    assert [p.name for p in tmp_path.iterdir()] == ["memory.npz"]


# ------------------------------------------------------------ asymmetric map


def _query(novelty=0.1, spread=0.1, estimate=1.0) -> MemoryQuery:
    return MemoryQuery(
        estimate=estimate,
        novelty=novelty,
        spread=spread,
        max_cos=1.0 - 2 * novelty,
        n_entries=1000,
        n_valued=16,
    )


def test_asymmetric_map_never_lowers_without_both_gates():
    cfg = _cfg(q_mid=0.35, q_high=0.60, novelty_gate_q=0.6, spread_gate_q=0.6)
    top = 2
    q = _query()

    # Upward band: no gate, ever.
    assert decide_effort_level(q, (0.99, 0.99, 0.99), cfg, top).level == 2
    assert decide_effort_level(q, (0.60, 0.99, 0.99), cfg, top).level == 2
    assert decide_effort_level(q, (0.35, 0.99, 0.99), cfg, top).level == 1

    # Downward band needs BOTH gates. Either one above its rank keeps the
    # request at the safe level.
    assert decide_effort_level(q, (0.10, 0.10, 0.10), cfg, top).level == 0
    assert decide_effort_level(q, (0.10, 0.90, 0.10), cfg, top).level == 1
    assert decide_effort_level(q, (0.10, 0.10, 0.90), cfg, top).level == 1
    assert decide_effort_level(q, (0.10, 0.90, 0.90), cfg, top).level == 1
    # A missing gate rank is not a passing gate.
    assert decide_effort_level(q, (0.10, None, 0.10), cfg, top).level == 1
    assert decide_effort_level(q, (0.10, 0.10, None), cfg, top).level == 1

    # Exactly at a gate is inside it; exactly at a cut is inside the band above.
    assert decide_effort_level(q, (0.10, 0.60, 0.60), cfg, top).level == 0
    assert decide_effort_level(q, (0.35, 0.10, 0.10), cfg, top).level == 1
    assert decide_effort_level(q, (0.60, 0.10, 0.10), cfg, top).level == 2


def test_map_falls_back_to_the_safe_level_without_an_estimate():
    cfg = _cfg()
    assert decide_effort_level(None, (None, None, None), cfg, 2).level == 1
    assert decide_effort_level(_query(), (None, 0.1, 0.1), cfg, 2).level == 1
    # A two-level server cannot reach level 2.
    assert decide_effort_level(_query(), (0.99, 0.1, 0.1), cfg, 1).level == 1


def test_ranks_are_streaming_and_absorb_the_observation():
    memory = _mem()
    first = _query(estimate=1.0, novelty=0.2, spread=0.3)
    # An empty digest cannot rank anything.
    assert memory.ranks(first) == (None, None, None)
    second = _query(estimate=2.0, novelty=0.5, spread=0.9)
    est, nov, spread = memory.ranks(second)
    assert est == 1.0 and nov == 1.0 and spread == 1.0
    below = _query(estimate=0.0, novelty=0.0, spread=0.0)
    assert memory.ranks(below) == (0.0, 0.0, 0.0)
