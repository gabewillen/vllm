# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the tiered expert-weight cache (GPU / pinned-CPU / mmap cold)."""

import random
import threading

import numpy as np
import pytest
import torch

from vllm.model_executor.offloader.expert_tier import (
    ExpertTensorSpec,
    ExpertTierCache,
    MmapColdBacking,
)

LAYERS = (0, 1, 2)
NUM_EXPERTS = 16
SPECS = [
    ExpertTensorSpec("w_a", (8, 4), torch.int8),
    ExpertTensorSpec("w_b", (2,), torch.float32),
]
BYTES_PER_EXPERT = 8 * 4 * 1 + 2 * 4
GPU_SLOTS = {0: 4, 1: 4, 2: 6}
PINNED_SLOTS = {0: 8, 1: 6, 2: 8}


def expected(layer_id: int, expert_id: int, name: str) -> torch.Tensor:
    """Deterministic contents for (layer, expert, tensor)."""
    if name == "w_a":
        base = layer_id * 977 + expert_id * 131
        vals = (base + np.arange(32) * 7) % 251 - 125
        return torch.from_numpy(vals.astype(np.int8)).reshape(8, 4)
    base = float(layer_id * 100 + expert_id)
    return torch.tensor([base + 0.25, -base - 0.5], dtype=torch.float32)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def device(request):
    return request.param


@pytest.fixture
def manifest(tmp_path):
    result = {}
    for layer_id in LAYERS:
        for spec in SPECS:
            rows = np.stack(
                [expected(layer_id, e, spec.name).numpy() for e in range(NUM_EXPERTS)]
            )
            path = tmp_path / f"l{layer_id}_{spec.name}.bin"
            rows.tofile(path)
            result[(layer_id, spec.name)] = str(path)
    return result


@pytest.fixture
def make_cache(manifest):
    caches = []

    def _make(device, gpu_slots=None, pinned_slots=None, cold=None, io_threads=4):
        cache = ExpertTierCache(
            device=device,
            gpu_slots_per_layer=dict(gpu_slots or GPU_SLOTS),
            pinned_slots_per_layer=dict(pinned_slots or PINNED_SLOTS),
            layer_specs={layer_id: list(SPECS) for layer_id in LAYERS},
            num_experts=NUM_EXPERTS,
            cold=cold if cold is not None else MmapColdBacking(manifest),
            io_threads=io_threads,
        )
        caches.append(cache)
        return cache

    yield _make
    for cache in caches:
        cache.close()


def verify_resident(result, layer_id, expert_ids):
    """Check GPU slot contents for the given experts against expected()."""
    if result.slot_of.device.type != "cpu":
        torch.cuda.synchronize()
    slot_of = result.slot_of.cpu()
    for expert_id in expert_ids:
        slot = int(slot_of[expert_id])
        assert slot >= 0, f"expert {expert_id} not resident in layer {layer_id}"
        for spec in SPECS:
            got = result.gpu_tensors[spec.name][slot].cpu()
            exp = expected(layer_id, expert_id, spec.name)
            assert torch.equal(got, exp), (layer_id, expert_id, spec.name)


def test_mmap_cold_backing_reads(manifest):
    cold = MmapColdBacking(manifest)
    for layer_id, expert_id, name in [(0, 0, "w_a"), (2, 15, "w_b"), (1, 7, "w_a")]:
        spec = next(s for s in SPECS if s.name == name)
        dst = torch.empty(spec.shape, dtype=spec.dtype)
        cold.read_into(layer_id, expert_id, name, dst)
        assert torch.equal(dst, expected(layer_id, expert_id, name))
    with pytest.raises(ValueError, match="too small"):
        dst = torch.empty(SPECS[0].shape, dtype=SPECS[0].dtype)
        cold.read_into(0, NUM_EXPERTS, "w_a", dst)
    with pytest.raises(KeyError):
        cold.read_into(99, 0, "w_a", torch.empty(SPECS[0].shape, dtype=torch.int8))
    with pytest.raises(FileNotFoundError):
        MmapColdBacking({(0, "w_a"): "/nonexistent/file.bin"})


def test_ensure_cold_promotion(device, make_cache):
    cache = make_cache(device)
    for layer_id in LAYERS:
        ids = [0, 3, 5]
        result = cache.ensure(layer_id, torch.tensor(ids))
        verify_resident(result, layer_id, ids)
        slot_of = result.slot_of.cpu()
        for expert_id in set(range(NUM_EXPERTS)) - set(ids):
            assert int(slot_of[expert_id]) == -1
        cache.release(layer_id)
    total = cache.stats()["total"]
    assert total["cold_misses"] == 9
    assert total["bytes_cold_to_pinned"] == 9 * BYTES_PER_EXPERT
    assert total["bytes_pinned_to_gpu"] == 9 * BYTES_PER_EXPERT


def test_repeat_ensure_pure_hits(device, make_cache):
    cache = make_cache(device)
    ids = [1, 2, 8]
    cache.ensure(0, ids)
    cache.release(0)
    before = cache.stats()["total"]
    result = cache.ensure(0, torch.tensor(ids + ids))
    cache.release(0)
    verify_resident(result, 0, ids)
    after = cache.stats()["total"]
    assert after["gpu_hits"] - before["gpu_hits"] == 3
    assert after["cold_misses"] == before["cold_misses"]
    assert after["bytes_cold_to_pinned"] == before["bytes_cold_to_pinned"]
    assert after["bytes_pinned_to_gpu"] == before["bytes_pinned_to_gpu"]


def test_gpu_lru_eviction(device, make_cache):
    cache = make_cache(device)
    cache.ensure(0, [0, 1, 2, 3])
    cache.release(0)
    result = cache.ensure(0, [4])
    cache.release(0)
    slot_of = result.slot_of.cpu()
    assert int(slot_of[0]) == -1  # LRU expert 0 evicted
    verify_resident(result, 0, [4])
    stats = cache.stats()["per_layer"][0]
    assert stats["gpu_evictions"] == 1
    before = cache.stats()["total"]
    result = cache.ensure(0, [0])  # pinned hit, no new cold read
    cache.release(0)
    verify_resident(result, 0, [0])
    after = cache.stats()["total"]
    assert after["cold_misses"] == before["cold_misses"]
    assert after["pinned_hits"] - before["pinned_hits"] == 1
    assert after["bytes_pinned_to_gpu"] - before["bytes_pinned_to_gpu"] == (
        BYTES_PER_EXPERT
    )


def test_pinned_lru_eviction(device, make_cache):
    cache = make_cache(device)
    for expert_id in range(10):  # pinned pool of layer 0 holds 8
        cache.ensure(0, [expert_id])
        cache.release(0)
    stats = cache.stats()["per_layer"][0]
    assert stats["pinned_evictions"] >= 2
    before = cache.stats()["total"]
    result = cache.ensure(0, [0])  # evicted from pinned tier: cold again
    cache.release(0)
    verify_resident(result, 0, [0])
    after = cache.stats()["total"]
    assert after["cold_misses"] - before["cold_misses"] == 1


def test_pins_unevictable_and_overflow_raises(device, make_cache):
    cache = make_cache(device)
    result = cache.ensure(0, [0, 1, 2])
    cache.ensure(0, [3])  # union of pins == pool size, still fine
    with pytest.raises(RuntimeError, match="working set"):
        cache.ensure(0, [4])
    verify_resident(result, 0, [0, 1, 2, 3])  # pins survived the failure
    cache.release(0)
    result = cache.ensure(0, [4])  # evictable again after release
    verify_resident(result, 0, [4])
    cache.release(0)
    with pytest.raises(RuntimeError, match="working set"):
        cache.ensure(1, list(range(5)))  # single call larger than the pool


def test_prefetch_then_ensure_hits(device, make_cache):
    cache = make_cache(device)
    cache.prefetch(0, [7, 8], to_gpu=True)
    cache.wait_io()
    before = cache.stats()["total"]
    assert before["bytes_cold_to_pinned"] == 0  # published lazily
    result = cache.ensure(0, [7, 8])
    cache.release(0)
    verify_resident(result, 0, [7, 8])
    after = cache.stats()["total"]
    assert after["cold_misses"] == 0
    assert after["gpu_hits"] == 2
    assert after["bytes_cold_to_pinned"] == 2 * BYTES_PER_EXPERT

    cache.prefetch(1, [4, 5], to_gpu=False)
    cache.wait_io()
    result = cache.ensure(1, [4, 5])
    cache.release(1)
    verify_resident(result, 1, [4, 5])
    after = cache.stats()["total"]
    assert after["cold_misses"] == 0
    assert after["pinned_hits"] == 2

    # No wait: ensure must block out the in-flight read and still be right.
    cache.prefetch(2, [9], to_gpu=True)
    result = cache.ensure(2, [9])
    cache.release(2)
    verify_resident(result, 2, [9])
    assert cache.stats()["total"]["cold_misses"] == 0


def test_prefetch_junk_does_not_corrupt(device, make_cache):
    cache = make_cache(device)
    result = cache.ensure(0, [0, 1, 2])  # unreleased pins
    rng = random.Random(7)
    for _ in range(20):
        cache.prefetch(0, rng.sample(range(NUM_EXPERTS), rng.randint(1, 8)))
    cache.wait_io()
    cache.prefetch(0, [])  # sweep finished loads
    verify_resident(result, 0, [0, 1, 2])
    cache.release(0)
    with pytest.raises(ValueError, match="out of range"):
        cache.prefetch(0, [NUM_EXPERTS])
    with pytest.raises(ValueError, match="out of range"):
        cache.ensure(0, [-1])


def test_concurrent_prefetch_stress(device, make_cache):
    cache = make_cache(device)
    errors = []
    stop = threading.Event()

    def prefetcher():
        rng = random.Random(1337)
        try:
            for _ in range(200):
                if stop.is_set():
                    break
                layer_id = rng.choice(LAYERS)
                ids = rng.sample(range(NUM_EXPERTS), rng.randint(1, 6))
                cache.prefetch(layer_id, ids, to_gpu=rng.random() < 0.5)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    thread = threading.Thread(target=prefetcher)
    thread.start()
    rng = random.Random(42)
    ensured = 0
    try:
        for _ in range(200):
            layer_id = rng.choice(LAYERS)
            ids = rng.sample(range(NUM_EXPERTS), rng.randint(1, 3))
            result = cache.ensure(layer_id, torch.tensor(ids))
            verify_resident(result, layer_id, ids)
            cache.release(layer_id)
            ensured += len(ids)
    finally:
        stop.set()
        thread.join()
    assert not errors
    total = cache.stats()["total"]
    served = (
        total["gpu_hits"]
        + total["pinned_hits"]
        + total["cold_misses"]
        + total["inflight_waits"]
    )
    assert served == ensured
    assert total["bytes_cold_to_pinned"] >= total["cold_misses"] * BYTES_PER_EXPERT


def test_stats_accounting(device, make_cache):
    cache = make_cache(device)
    stats = cache.stats()
    assert stats["total"] == stats["per_layer"][0]  # all zero
    cache.ensure(0, [1, 2])
    cache.release(0)
    stats = cache.stats()
    layer = stats["per_layer"][0]
    assert layer["ensure_calls"] == 1
    assert layer["cold_misses"] == 2
    assert layer["gpu_hits"] == 0
    assert layer["bytes_cold_to_pinned"] == 2 * BYTES_PER_EXPERT
    assert layer["bytes_pinned_to_gpu"] == 2 * BYTES_PER_EXPERT
    cache.ensure(1, [1])
    cache.release(1)
    stats = cache.stats()
    for key, value in stats["total"].items():
        assert value == sum(s[key] for s in stats["per_layer"].values())
    assert stats["total"]["cold_misses"] == 3


def test_preload(device, make_cache):
    cache = make_cache(device)
    cache.preload(0, range(8))
    stats = cache.stats()["per_layer"][0]
    assert stats["cold_misses"] == 0
    assert stats["bytes_cold_to_pinned"] == 8 * BYTES_PER_EXPERT
    result = cache.ensure(0, [0, 1, 2, 3])
    cache.release(0)
    verify_resident(result, 0, [0, 1, 2, 3])
    assert cache.stats()["per_layer"][0]["pinned_hits"] == 4

    cache.preload(1, [4, 5], to_gpu=True)
    result = cache.ensure(1, [4, 5])
    cache.release(1)
    verify_resident(result, 1, [4, 5])
    layer = cache.stats()["per_layer"][1]
    assert layer["gpu_hits"] == 2
    assert layer["cold_misses"] == 0

    with pytest.raises(RuntimeError, match="pinned pool"):
        cache.preload(2, range(9))  # pinned pool of layer 2 holds 8


def test_cold_read_error_propagates(device, make_cache, manifest):
    class FlakyBacking:
        def __init__(self):
            self.inner = MmapColdBacking(manifest)
            self.fail = True

        def read_into(self, layer_id, expert_id, name, dst):
            if self.fail and expert_id == 13:
                raise OSError("injected read failure")
            self.inner.read_into(layer_id, expert_id, name, dst)

    cold = FlakyBacking()
    cache = make_cache(device, cold=cold)
    with pytest.raises(OSError, match="injected"):
        cache.ensure(0, [13])
    assert cache.stats()["per_layer"][0]["io_errors"] == 1
    result = cache.ensure(0, [12])  # cache still healthy, slot was reclaimed
    verify_resident(result, 0, [12])
    cache.release(0)
    cold.fail = False
    result = cache.ensure(0, [13])  # retry after the fault clears
    verify_resident(result, 0, [13])
    cache.release(0)


def test_large_row_tensors(device, tmp_path):
    """Rows above the staging threshold take the per-row DMA path."""
    num_experts = 6
    specs = [
        ExpertTensorSpec("w_big", (130, 1024), torch.int8),  # >64 KiB row
        ExpertTensorSpec("w_small", (2,), torch.float32),
    ]

    def big(expert_id):
        vals = (np.arange(130 * 1024, dtype=np.int64) * 3 + expert_id * 17) % (
            251
        ) - 125
        return vals.astype(np.int8).reshape(130, 1024)

    def small(expert_id):
        return np.array([expert_id + 0.5, -float(expert_id)], dtype=np.float32)

    manifest = {}
    for name, gen in (("w_big", big), ("w_small", small)):
        rows = np.stack([gen(e) for e in range(num_experts)])
        path = tmp_path / f"{name}.bin"
        rows.tofile(path)
        manifest[(0, name)] = str(path)
    cache = ExpertTierCache(
        device=device,
        gpu_slots_per_layer={0: 3},
        pinned_slots_per_layer={0: 4},
        layer_specs={0: specs},
        num_experts=num_experts,
        cold=MmapColdBacking(manifest),
    )

    def check(result, expert_ids):
        if result.slot_of.device.type != "cpu":
            torch.cuda.synchronize()
        slot_of = result.slot_of.cpu()
        for expert_id in expert_ids:
            slot = int(slot_of[expert_id])
            assert slot >= 0
            got_big = result.gpu_tensors["w_big"][slot].cpu().numpy()
            np.testing.assert_array_equal(got_big, big(expert_id))
            got_small = result.gpu_tensors["w_small"][slot].cpu().numpy()
            np.testing.assert_array_equal(got_small, small(expert_id))

    try:
        result = cache.ensure(0, [0, 1, 2])
        check(result, [0, 1, 2])
        cache.release(0)
        result = cache.ensure(0, [3])  # evicts LRU expert 0
        check(result, [1, 2, 3])
        cache.release(0)
        result = cache.ensure(0, [0])  # back from the pinned tier
        check(result, [0])
        cache.release(0)
    finally:
        cache.close()


def test_invalid_construction_and_lookups(device, make_cache, manifest):
    with pytest.raises(ValueError, match="same layer ids"):
        ExpertTierCache(
            device=device,
            gpu_slots_per_layer={0: 2},
            pinned_slots_per_layer={0: 2, 1: 2},
            layer_specs={0: SPECS, 1: SPECS},
            num_experts=NUM_EXPERTS,
            cold=MmapColdBacking(manifest),
        )
    cache = make_cache(device)
    with pytest.raises(KeyError, match="unknown layer"):
        cache.ensure(99, [0])
