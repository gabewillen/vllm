# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the v3 online effort memory (§13.4, §13.5).

The memory is the whole v3 signal: a ring of pooled prefill states the server
filled itself, queried by cosine and valued by each entry's difficulty - its
reasoning spend ranked within the effort level it was rendered at. Nothing
here is trained, so what has to be pinned is the bookkeeping - who evicts
whom, which entries may contribute a value, that the level label cancels out
of the value, and that the low-resting map only leaves low on evidence and
keeps probing downward.
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
    memory = _mem(k=5, min_entries=5)
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
    assert 0.0 <= result.estimate <= 1.0


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
    # Every entry went in at level 0: each value is its spend rank in that
    # lane, smoothed toward the middle by a prior worth one neighbourhood.
    lane = memory._level_digests[0]
    k = memory.cfg.k
    values = np.array(
        [
            (lane.rank(float(np.float32(math.log1p(int(t))))) * lane.count + 0.5 * k)
            / (lane.count + k)
            for t in tokens[top]
        ]
    )
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


def test_low_resting_map_leaves_low_only_on_difficulty():
    cfg = _cfg(q_mid=0.35, q_high=0.60, default_level=1)
    top = 2
    q = _query()

    # The estimate is the neighbours' difficulty; the cuts apply to it.
    assert decide_effort_level(q, (0.99, 0.10, 0.99), cfg, top).level == 2
    assert decide_effort_level(q, (0.60, 0.10, 0.99), cfg, top).level == 2
    assert decide_effort_level(q, (0.35, 0.10, 0.99), cfg, top).level == 1
    assert decide_effort_level(q, (0.34, 0.10, 0.99), cfg, top).level == 0
    assert decide_effort_level(q, (0.00, 0.10, 0.00), cfg, top).level == 0
    # Spread is reported, never cut on.
    assert decide_effort_level(q, (0.10, 0.10, None), cfg, top).level == 0

    # Novelty is not a gate: the calibrated estimate alone decides.
    assert decide_effort_level(q, (0.99, 0.90, 0.10), cfg, top).level == 2
    assert decide_effort_level(q, (0.00, 0.99, 0.10), cfg, top).level == 0


def _calibration_memory(k=4, **kw):
    cfg = _cfg(k=k, min_entries=k, memory_size=256, **kw)
    memory = EffortMemory(8, cfg, levels=3)
    # Warm the lane so realised difficulty is a meaningful rank.
    rng = np.random.default_rng(99)
    for tokens in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        memory.insert(rng.standard_normal(8), tokens, "natural", level=1)
    return memory, cfg


def test_uninformative_neighbours_calibrate_to_the_mean():
    """When the estimate has not predicted realised difficulty at a given
    novelty, calibration collapses it to the mean rank instead of letting a
    meaningless neighbourhood pick the top or bottom level."""
    memory, cfg = _calibration_memory()
    rng = np.random.default_rng(0)
    # Estimates and outcomes unrelated: high estimates close short and vice
    # versa, at the same novelty bin.
    for est, tokens in ((0.9, 5), (0.1, 500), (0.95, 8), (0.05, 400),
                        (0.85, 6), (0.15, 450), (0.9, 4), (0.1, 480)):
        memory.insert(rng.standard_normal(8), tokens, "natural", level=1,
                      estimate=est, novelty_rank=0.3)
    low = memory.calibrate(0.05, 0.3)
    high = memory.calibrate(0.95, 0.3)
    assert abs(high - low) < 0.15, (low, high)
    assert 0.3 < low < 0.7 and 0.3 < high < 0.7


def test_informative_neighbours_pass_through_calibration():
    memory, cfg = _calibration_memory()
    rng = np.random.default_rng(1)
    for est, tokens in ((0.9, 500), (0.1, 5), (0.95, 480), (0.05, 8),
                        (0.85, 450), (0.15, 6), (0.9, 400), (0.1, 4)):
        memory.insert(rng.standard_normal(8), tokens, "natural", level=1,
                      estimate=est, novelty_rank=0.3)
    assert memory.calibrate(0.95, 0.3) > cfg.q_high
    assert memory.calibrate(0.05, 0.3) < cfg.q_mid


def test_calibration_is_per_novelty_bin_and_persists(tmp_path):
    memory, cfg = _calibration_memory()
    rng = np.random.default_rng(2)
    for est, tokens in ((0.9, 5), (0.1, 500), (0.95, 8), (0.05, 400)):
        memory.insert(rng.standard_normal(8), tokens, "natural", level=1,
                      estimate=est, novelty_rank=0.9)
    # The far bin is uninformative; the near bin has no fit yet and, with the
    # pooled fit also at k, the pooled (uninformative) fit stands in.
    assert abs(memory.calibrate(0.95, 0.9) - memory.calibrate(0.05, 0.9)) < 0.15
    memory.cfg.memory_path = str(tmp_path / "m.npz")
    memory.save()
    again = EffortMemory(8, memory.cfg, levels=3)
    assert again.load()
    assert again.calibrate(0.95, 0.9) == memory.calibrate(0.95, 0.9)


def test_probe_renders_one_level_below_the_verdict():
    cfg = _cfg(default_level=1)
    q = _query()
    probed = decide_effort_level(q, (0.99, 0.1, 0.1), cfg, 2, probe=True)
    assert (probed.level, probed.reason) == (1, "probe/q>=q_high")
    assert decide_effort_level(q, (0.40, 0.1, 0.1), cfg, 2, probe=True).level == 0
    # Low cannot go lower; a novel or estimate-less request is not probed.
    assert decide_effort_level(q, (0.10, 0.1, 0.1), cfg, 2, probe=True).level == 0
    assert decide_effort_level(q, (0.99, 0.9, 0.1), cfg, 2, probe=True).level == 1
    assert decide_effort_level(None, (None, None, None), cfg, 2, probe=True).level == 1


def test_probe_clock_fires_every_nth_decision():
    memory = _mem(probe_every=4)
    fired = [memory.take_probe() for _ in range(8)]
    assert fired == [False, False, False, True] * 2
    assert not any(_mem(probe_every=0).take_probe() for _ in range(8))


def test_map_falls_back_to_the_default_level_without_an_estimate():
    cfg = _cfg(default_level=1)
    assert decide_effort_level(None, (None, None, None), cfg, 2).level == 1
    assert decide_effort_level(_query(), (None, 0.1, 0.1), cfg, 2).level == 1
    # A two-level server cannot reach level 2.
    assert decide_effort_level(_query(), (0.99, 0.1, 0.1), cfg, 1).level == 1


def test_level_label_cancels_out_of_the_value():
    """The same task rendered at the top level thinks ~10x longer than at low.

    Ranked within its lane that is the *same* difficulty, so a neighbourhood
    the server over-routed does not read back as hard."""
    memory = _mem(memory_size=256, k=4, min_entries=4)
    rng = np.random.default_rng(3)
    # Two lanes, both seeing the same spread of tasks: low spends 10..1000,
    # top spends 100..10000 (x10 from the sentence alone).
    for i in range(40):
        spend = int(10 * 10 ** (2 * i / 39))
        memory.insert(rng.normal(size=DIM), spend, "natural", level=0)
        memory.insert(rng.normal(size=DIM), spend * 10, "natural", level=2)
    # A cluster of trivial tasks that the server happened to route to the top
    # level: they spent 150 tokens there - huge for low, tiny for top.
    centre = _vec(7)
    for i in range(6):
        memory.insert(centre + 0.02 * _vec(100 + i), 150, "natural", level=2)
    got = memory.query(centre)
    assert got is not None and got.n_valued == 4
    # Within the top lane 150 sits near the bottom: this neighbourhood is easy.
    assert got.estimate < 0.15
    # The same spend recorded at low would read as mid-difficulty.
    assert memory._difficulty(0, math.log1p(150)) > 0.4


def test_legacy_file_migrates_to_within_level_difficulty(tmp_path):
    path = str(tmp_path / "legacy.npz")
    legacy = _mem(memory_path=path, k=2)
    for i in range(8):
        legacy.insert(_vec(i), 100 * (i + 1), "natural", level=i % 2, session_id=str(i))
    legacy.save()
    # Strip the lane digests as a file from before they existed.
    import json

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(bytes(data["meta"]).decode())
        arrays = {k: data[k] for k in data.files if k != "meta"}
    for key in ("level_digests", "spend_digest", "probe_clock"):
        meta.pop(key, None)
    np.savez(path, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)

    fresh = _mem(memory_path=path, k=2)
    assert fresh.load()
    assert fresh.n_entries == 8
    assert {k: int(d.count) for k, d in fresh._level_digests.items()} == {0: 4, 1: 4}
    assert fresh._est_digest.count == 0


def test_ranks_are_streaming_and_absorb_the_observation():
    memory = _mem()
    first = _query(estimate=0.7, novelty=0.2, spread=0.3)
    # The estimate passes through; empty digests cannot rank the rest.
    assert memory.ranks(first) == (0.7, None, None)
    second = _query(estimate=0.2, novelty=0.5, spread=0.9)
    est, nov, spread = memory.ranks(second)
    assert est == 0.2 and nov == 1.0 and spread == 1.0
    below = _query(estimate=0.0, novelty=0.0, spread=0.0)
    assert memory.ranks(below) == (0.0, 0.0, 0.0)


def test_calibration_learns_from_the_first_pair_and_from_the_memory_itself():
    """No warm-up gate: one observed pair already moves the posterior, and each
    insert also yields leave-one-out pairs for the entry and its neighbours,
    so the calibration fills in ~k times faster than one pair per request."""
    memory, cfg = _calibration_memory()
    before = memory._calibration_all.n
    rng = np.random.default_rng(3)
    memory.insert(rng.standard_normal(8), 300, "natural", level=1,
                  estimate=0.9, novelty_rank=0.3)
    assert memory._calibration_all.n - before > 1  # live pair + LOO pairs
    # Two wildly wrong pairs at one novelty already pull that bin's line
    # toward the mean, while leaving it above the identity-anchored pooled fit.
    identity = memory.calibrate(0.95, None)
    for est, tokens in ((0.95, 2), (0.9, 3)):
        memory.insert(rng.standard_normal(8), tokens, "natural", level=1,
                      estimate=est, novelty_rank=0.95)
    assert memory.calibrate(0.95, 0.95) < identity


def test_think_off_level_sits_under_low_and_probes_upward():
    """With `think_off_level` the map has four levels: below `q_none` the
    request skips thinking; the resting level is still `low` (index 1); the
    probe clock renders a think-off verdict at low so the neighbourhood keeps
    receiving thinking-length evidence."""
    cfg = _cfg(think_off_level=True, q_none=0.15, q_mid=0.35, q_high=0.60, default_level=2)
    q = _query()
    top = 3
    assert decide_effort_level(q, (0.10, 0.1, 0.1), cfg, top).level == 0
    assert decide_effort_level(q, (0.10, 0.1, 0.1), cfg, top).reason == "q<q_none"
    assert decide_effort_level(q, (0.20, 0.1, 0.1), cfg, top).level == 1
    assert decide_effort_level(q, (0.40, 0.1, 0.1), cfg, top).level == 2
    assert decide_effort_level(q, (0.70, 0.1, 0.1), cfg, top).level == 3
    # Probes: down from above low, up from think-off.
    assert decide_effort_level(q, (0.70, 0.1, 0.1), cfg, top, probe=True).level == 2
    up = decide_effort_level(q, (0.10, 0.1, 0.1), cfg, top, probe=True)
    assert (up.level, up.reason) == (1, "probe/q<q_none")
    # No estimate -> default, never the think-off level.
    assert decide_effort_level(None, (None, None, None), cfg, top).level == 2


def test_think_off_entry_echoes_its_difficulty_and_teaches_nothing(tmp_path):
    """A think-off request has no thinking length. Its entry carries the
    difficulty it was decided with - neighbours see it, the lanes and the
    calibration do not."""
    memory, cfg = _calibration_memory(think_off_level=True, default_level=2)
    lane_before = {k: d.count for k, d in memory._level_digests.items()}
    cal_before = memory._calibration_all.n
    v = np.ones(8)
    memory.insert(v, 0, "natural", level=0, difficulty=0.12)
    assert memory._level_digests.get(0) is None
    assert {k: d.count for k, d in memory._level_digests.items()} == lane_before
    assert memory._calibration_all.n == cal_before
    got = memory.query(v)
    assert got is not None and got.estimate is not None
    # The nearest neighbour is the think-off entry at 0.12; with temperature
    # 0.05 it dominates the weighted mean.
    assert got.estimate < 0.3
    # It survives a save/load round trip as a think-off entry.
    memory.cfg.memory_path = str(tmp_path / "m.npz")
    memory.save()
    again = EffortMemory(8, memory.cfg, levels=3)
    assert again.load()
    assert again.query(v).estimate == pytest.approx(got.estimate, rel=1e-5)


def test_enabling_the_think_off_level_shifts_a_saved_memory(tmp_path):
    cfg3 = _cfg(memory_size=64, memory_path=str(tmp_path / "m.npz"))
    three = EffortMemory(8, cfg3, levels=3)
    rng = np.random.default_rng(5)
    for lv, tokens in ((0, 10), (1, 100), (2, 1000)):
        three.insert(rng.standard_normal(8), tokens, "natural", level=lv)
    three.save()
    cfg4 = _cfg(memory_size=64, memory_path=str(tmp_path / "m.npz"),
                think_off_level=True, default_level=2)
    four = EffortMemory(8, cfg4, levels=4)
    assert four.load()
    assert sorted(int(x) for x in four._levels_used[:3]) == [1, 2, 3]
    assert sorted(four._level_digests) == [1, 2, 3]


def test_three_level_map_off_default_custom():
    """off / default / custom: one level above the resting default, so only
    `q_high` matters above it; `q_none` below; probes go down from custom
    and up from off."""
    cfg = _cfg(think_off_level=True, custom_level=True, q_none=0.3, q_high=0.6, default_level=1)
    q = _query()
    top = 2
    assert decide_effort_level(q, (0.20, 0.1, 0.1), cfg, top).level == 0
    assert decide_effort_level(q, (0.45, 0.1, 0.1), cfg, top).level == 1
    assert decide_effort_level(q, (0.59, 0.1, 0.1), cfg, top).level == 1
    assert decide_effort_level(q, (0.60, 0.1, 0.1), cfg, top).level == 2
    assert decide_effort_level(q, (0.90, 0.1, 0.1), cfg, top, probe=True).level == 1
    assert decide_effort_level(q, (0.20, 0.1, 0.1), cfg, top, probe=True).level == 1
