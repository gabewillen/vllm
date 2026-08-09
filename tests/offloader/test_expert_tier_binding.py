# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the expert-tier cache binding into the fused-MoE runtime.

The end-to-end test builds a tiny Marlin MXFP4 MoE layer twice: once with
resident weights (normal path) and once served from the tier cache via
synthetic cold files written from the resident layer's post-processed
tensors. Both must produce identical outputs.
"""

import json

import pytest
import torch

from vllm.config import (
    ExpertTierOffloadConfig,
    OffloadConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.fused_moe import expert_tier_binding as binding
from vllm.model_executor.layers.fused_moe.expert_tier_binding import (
    ExpertTierLayerBinding,
    cluster_tokens_by_experts,
    compute_slot_counts,
    load_manifest,
    plan_expert_sub_batches,
)
from vllm.model_executor.offloader.expert_tier import ExpertTensorSpec
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

NUM_EXPERTS = 4
TOP_K = 2
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 128


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _write_cache(cache_dir, rank: int, num_experts: int, layer_tensors, meta=None):
    """Write raw cold files + manifest for {layer_id: {name: tensor[E,...]}}."""
    rank_dir = cache_dir / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    layers = {}
    for layer_id, tensors in layer_tensors.items():
        entries = {}
        for name, tensor in tensors.items():
            tensor = tensor.detach().cpu().contiguous()
            assert tensor.shape[0] == num_experts
            file_name = f"l{layer_id}_{name}.bin"
            tensor.view(torch.uint8).numpy().tofile(rank_dir / file_name)
            entries[name] = {
                "file": file_name,
                "dtype": _dtype_name(tensor.dtype),
                "per_expert_shape": list(tensor.shape[1:]),
                "row_bytes": tensor[0].numel() * tensor.element_size(),
            }
        layers[str(layer_id)] = entries
    manifest = {"num_experts": num_experts, "layers": layers, "meta": meta or {}}
    with open(rank_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    return rank_dir / "manifest.json"


@pytest.fixture
def tier_state_reset():
    binding.reset_expert_tier_state()
    yield
    binding.reset_expert_tier_state()


#
# Manifest parsing
#


def test_load_manifest(tmp_path):
    tensors = {
        3: {
            "w13_weight": torch.arange(4 * 6, dtype=torch.int32).reshape(4, 2, 3),
            "w13_weight_scale": torch.randn(4, 5, dtype=torch.bfloat16),
        }
    }
    _write_cache(tmp_path, 0, 4, tensors)
    manifest = load_manifest(str(tmp_path), 0)
    assert manifest.num_experts == 4
    assert set(manifest.layer_specs) == {3}
    specs = {s.name: s for s in manifest.layer_specs[3]}
    assert specs["w13_weight"].shape == (2, 3)
    assert specs["w13_weight"].dtype == torch.int32
    assert specs["w13_weight"].row_bytes == 2 * 3 * 4
    assert specs["w13_weight_scale"].dtype == torch.bfloat16
    # Relative paths resolve against the rank dir.
    for key, path in manifest.files.items():
        assert path.startswith(str(tmp_path / "rank0"))

    # The converter writes dtypes as "torch.<name>"; both forms parse.
    assert binding._parse_dtype("torch.float8_e8m0fnu") == torch.float8_e8m0fnu
    assert binding._parse_dtype("int32") == torch.int32
    with pytest.raises(ValueError, match="not a torch dtype"):
        binding._parse_dtype("torch.nonsense")

    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_manifest(str(tmp_path), 1)


def test_load_manifest_row_bytes_mismatch(tmp_path):
    path = _write_cache(
        tmp_path, 0, 2, {0: {"w": torch.zeros(2, 4, dtype=torch.uint8)}}
    )
    data = json.loads(path.read_text())
    data["layers"]["0"]["w"]["row_bytes"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="row_bytes"):
        load_manifest(str(tmp_path), 0)


#
# Pool sizing math
#


def _manifest_for_sizing(num_experts, layer_specs):
    return binding.ExpertTierManifest(
        path="<test>",
        num_experts=num_experts,
        layer_specs=layer_specs,
        files={},
        meta={},
    )


def test_compute_slot_counts():
    # Layer 0: one large tensor (128 KiB rows, direct DMA) and one small
    # (staged) tensor of 1 KiB rows. Layer 1: one small 2 KiB tensor.
    specs = {
        0: [
            ExpertTensorSpec("big", (128 * 1024,), torch.uint8),
            ExpertTensorSpec("small", (1024,), torch.uint8),
        ],
        1: [ExpertTensorSpec("tiny", (2048,), torch.uint8)],
    }
    manifest = _manifest_for_sizing(64, specs)
    # Per GPU slot: layer0 pool 129K + 1K staging, layer1 pool 2K + 2K
    # staging = 134 KiB. Per pinned slot: 129K + 2K = 131 KiB, plus a
    # fixed gpu_slots * 3 KiB pinned staging buffer.
    kib = 1024
    gpu_cost = 134 * kib
    gpu_slots, pinned_slots = compute_slot_counts(
        manifest,
        gpu_bytes=10 * gpu_cost + 5,
        pinned_bytes=131 * kib * 20 + 10 * 3 * kib,
    )
    assert gpu_slots == {0: 10, 1: 10}
    assert pinned_slots == {0: 20, 1: 20}

    # Slots are clamped at num_experts.
    gpu_slots, pinned_slots = compute_slot_counts(
        manifest, gpu_bytes=1000 * gpu_cost, pinned_bytes=131 * kib * 1000
    )
    assert gpu_slots == {0: 64, 1: 64}
    assert pinned_slots == {0: 64, 1: 64}

    # Pinned budget below the GPU working set is raised to gpu_slots.
    gpu_slots, pinned_slots = compute_slot_counts(
        manifest, gpu_bytes=8 * gpu_cost, pinned_bytes=131 * kib
    )
    assert gpu_slots == {0: 8, 1: 8}
    assert pinned_slots == {0: 8, 1: 8}

    # GPU budget below one slot per layer fails.
    with pytest.raises(ValueError, match="cannot hold one GPU slot"):
        compute_slot_counts(manifest, gpu_bytes=gpu_cost - 1, pinned_bytes=2**30)


#
# Sub-batch planning
#


def test_plan_expert_sub_batches():
    # Whole batch within the bound: no split.
    topk = torch.tensor([[0, 1], [0, 1], [2, 3]])
    assert plan_expert_sub_batches(topk, 4) is None
    # Single token never splits.
    assert plan_expert_sub_batches(torch.tensor([[0, 1, 2]]), 2) is None

    # Greedy contiguous split at the distinct-expert bound.
    assert plan_expert_sub_batches(topk, 2) == [(0, 2), (2, 3)]
    topk = torch.tensor([[0, 1], [0, 1], [2, 3], [2, 3], [0, 3]])
    assert plan_expert_sub_batches(topk, 2) == [(0, 2), (2, 4), (4, 5)]

    # A single token over the bound still gets its own sub-batch.
    topk = torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert plan_expert_sub_batches(topk, 2) == [(0, 1), (1, 2)]


def test_cluster_tokens_by_experts_groups_interleaved_sets():
    # Tokens alternate between two disjoint expert sets: request order
    # saturates a set-sized bound at every token, clustered order packs
    # each set's tokens into a single range.
    a, b = [0, 1, 2], [3, 4, 5]
    topk = torch.tensor([a if i % 2 == 0 else b for i in range(20)])
    assert len(plan_expert_sub_batches(topk, 3)) == 20

    perm = cluster_tokens_by_experts(topk)
    assert sorted(perm.tolist()) == list(range(20))
    assert plan_expert_sub_batches(topk[perm], 3) == [(0, 10), (10, 20)]

    # Stable within equal sets: original order preserved per cluster.
    assert perm.tolist() == [i for i in range(20) if i % 2 == 0] + [
        i for i in range(20) if i % 2 == 1
    ]

    # Random routing: ranges partition the batch and each range's
    # distinct-expert count stays within the bound.
    torch.manual_seed(3)
    topk = torch.randint(0, 64, (200, 8))
    bound = 24
    bounds = plan_expert_sub_batches(topk, bound)
    assert bounds is not None
    assert bounds[0][0] == 0 and bounds[-1][1] == 200
    for (_, prev_end), (start, _) in zip(bounds, bounds[1:]):
        assert prev_end == start
    for start, end in bounds:
        assert int(torch.unique(topk[start:end]).numel()) <= bound


#
# Config validation and CLI plumbing
#


def test_expert_tier_config_requires_cache_dir():
    with pytest.raises(Exception, match="cache_dir"):
        ExpertTierOffloadConfig(enabled=True)


def test_expert_tier_forces_marlin_backend():
    oc = OffloadConfig(
        expert_tier=ExpertTierOffloadConfig(enabled=True, cache_dir="/nonexistent")
    )
    config = VllmConfig(offload_config=oc)
    assert config.kernel_config.moe_backend == "marlin"

    from vllm.config import KernelConfig

    with pytest.raises(ValueError, match="marlin"):
        VllmConfig(offload_config=oc, kernel_config=KernelConfig(moe_backend="triton"))


def test_expert_tier_cli_args(tmp_path):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--expert-tier-cache-dir",
            str(tmp_path),
            "--expert-tier-gpu-gb",
            "0.5",
            "--expert-tier-pinned-gb",
            "1.5",
            "--expert-tier-io-threads",
            "2",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.expert_tier_cache_dir == str(tmp_path)
    assert engine_args.expert_tier_gpu_gb == 0.5
    assert engine_args.expert_tier_pinned_gb == 1.5
    assert engine_args.expert_tier_io_threads == 2
    # Mode off by default: no cache dir, not enabled.
    default_args = EngineArgs.from_cli_args(parser.parse_args([]))
    assert default_args.expert_tier_cache_dir is None


def test_disabled_mode_binding_not_reached():
    """With expert tier off, maybe_init_expert_tier is a no-op returning
    None before touching the layer at all."""
    with set_current_vllm_config(VllmConfig()):
        result = binding.maybe_init_expert_tier(object())  # type: ignore[arg-type]
    assert result is None


#
# End-to-end: tiny Marlin MXFP4 MoE layer, tier cache vs resident weights
#


class _TierTestMxfp4Config:
    """Deferred import wrapper (mxfp4 import needs torch/cuda bits)."""

    def __new__(cls):
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )
        from vllm.model_executor.layers.quantization.mxfp4 import (
            Mxfp4Config,
            Mxfp4MoEMethod,
        )

        class _Config(Mxfp4Config):
            def get_quant_method(self, layer, prefix):
                if isinstance(layer, RoutedExperts):
                    return Mxfp4MoEMethod(layer.moe_config)
                return None

        return _Config()


def _make_vllm_config(expert_tier: ExpertTierOffloadConfig | None) -> VllmConfig:
    offload_config = OffloadConfig()
    if expert_tier is not None:
        offload_config = OffloadConfig(expert_tier=expert_tier)
    config = VllmConfig(offload_config=offload_config)
    config.kernel_config.moe_backend = "marlin"
    config.compilation_config.static_forward_context = dict()
    return config


def _make_moe_layer(vllm_config: VllmConfig, prefix: str):
    from vllm.model_executor.layers.fused_moe import FusedMoE

    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            params_dtype=torch.bfloat16,
            renormalize=True,
            quant_config=_TierTestMxfp4Config(),
            tp_size=1,
            dp_size=1,
            pcp_size=1,
            prefix=prefix,
        )
    return layer


def _randomize_mxfp4_weights(experts) -> None:
    """Fill checkpoint-format MXFP4 tensors with valid random content."""
    with torch.no_grad():
        for name in ("w13_weight", "w2_weight"):
            param = getattr(experts, name)
            param.copy_(
                torch.randint(0, 256, param.shape, dtype=torch.uint8, device="cuda")
            )
        for name in ("w13_weight_scale", "w2_weight_scale"):
            # e8m0 exponents around 1.0 (biased 127) keep outputs sane.
            param = getattr(experts, name)
            param.copy_(
                torch.randint(121, 128, param.shape, dtype=torch.uint8, device="cuda")
            )


def _forward(layer, vllm_config, hidden_states, router_logits):
    from vllm.forward_context import set_forward_context

    with set_forward_context(None, vllm_config, num_tokens=hidden_states.shape[0]):
        return layer(hidden_states, router_logits)


def _build_reference_and_cache(tmp_path):
    """Build the resident-weight reference layer and write its
    post-processed Marlin tensors as the synthetic cold cache."""
    cfg_ref = _make_vllm_config(expert_tier=None)
    ref = _make_moe_layer(cfg_ref, prefix="model.layers.0.mlp.ref")
    assert ref.routed_experts.expert_tier is None
    _randomize_mxfp4_weights(ref.routed_experts)
    with set_current_vllm_config(cfg_ref):
        ref.routed_experts.quant_method.process_weights_after_loading(
            ref.routed_experts
        )
    tensors = {
        name: getattr(ref.routed_experts, name)
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    }
    _write_cache(tmp_path, 0, NUM_EXPERTS, {0: tensors})
    return cfg_ref, ref, tensors


def _tier_budgets(tensors, gpu_slots: int, pinned_slots: int) -> tuple[float, float]:
    """Exact GiB budgets yielding the requested uniform slot counts."""
    row_bytes = [t[0].numel() * t.element_size() for t in tensors.values()]
    bytes_all = sum(row_bytes)
    small_all = sum(b for b in row_bytes if b <= 65536)
    gpu_bytes = gpu_slots * (bytes_all + small_all)
    pinned_bytes = pinned_slots * bytes_all + gpu_slots * small_all
    return gpu_bytes / 2**30, pinned_bytes / 2**30


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("gpu_slots", [NUM_EXPERTS, 2])
def test_tier_layer_matches_resident_layer(
    tmp_path, dist_init, workspace_init, tier_state_reset, monkeypatch, gpu_slots
):
    """Forward through the tier-cached layer equals the resident layer.

    gpu_slots == NUM_EXPERTS: whole expert set fits on the GPU tier.
    gpu_slots == 2: routing is restricted to expert pairs, and rotating
    the pairs across calls forces GPU-slot eviction and cold reloads.
    """
    torch.set_default_device("cuda")
    try:
        set_random_seed(7)
        cfg_ref, ref, tensors = _build_reference_and_cache(tmp_path)

        gpu_gb, pinned_gb = _tier_budgets(
            tensors, gpu_slots=gpu_slots, pinned_slots=NUM_EXPERTS
        )
        cfg_tier = _make_vllm_config(
            ExpertTierOffloadConfig(
                enabled=True,
                cache_dir=str(tmp_path),
                gpu_gb=gpu_gb,
                pinned_gb=pinned_gb,
                io_threads=2,
            )
        )
        tier = _make_moe_layer(cfg_tier, prefix="model.layers.0.mlp.tier")
        tier_binding = tier.routed_experts.expert_tier
        assert tier_binding is not None
        state = binding._state
        assert state is not None
        assert state.gpu_slots_per_layer == {0: gpu_slots}
        assert state.pinned_slots_per_layer == {0: NUM_EXPERTS}
        # The pool tensors are the layer's registered weight parameters.
        assert (
            tier.routed_experts.w13_weight.data_ptr()
            == tier_binding.gpu_pools["w13_weight"].data_ptr()
        )
        assert tier.routed_experts.w13_weight.shape[0] == gpu_slots
        with set_current_vllm_config(cfg_tier):
            tier.routed_experts.quant_method.process_weights_after_loading(
                tier.routed_experts
            )

        # Count ensure() calls: never on the reference layer, once per
        # forward on the tier layer.
        ensure_calls = []
        real_ensure = ExpertTierLayerBinding.ensure

        def counting_ensure(self, topk_ids):
            ensure_calls.append(self.layer_id)
            return real_ensure(self, topk_ids)

        monkeypatch.setattr(ExpertTierLayerBinding, "ensure", counting_ensure)

        num_tokens = 16
        if gpu_slots == NUM_EXPERTS:
            routed_pairs = [None, None, None]  # unrestricted routing
        else:
            routed_pairs = [(0, 1), (2, 3), (0, 1), (1, 2)]

        for step, pair in enumerate(routed_pairs):
            hidden = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16) / 10
            if pair is None:
                logits = torch.randn(num_tokens, NUM_EXPERTS, dtype=torch.float32)
            else:
                logits = torch.full(
                    (num_tokens, NUM_EXPERTS), -20.0, dtype=torch.float32
                )
                logits[:, list(pair)] = 20.0

            out_ref = _forward(ref, cfg_ref, hidden, logits)
            calls_before = len(ensure_calls)
            out_tier = _forward(tier, cfg_tier, hidden, logits)
            assert len(ensure_calls) == calls_before + 1

            # Same kernel, same per-expert bytes (slot pools), same
            # routing: outputs must match bitwise.
            torch.testing.assert_close(
                out_tier, out_ref, atol=0.0, rtol=0.0, msg=f"step {step}"
            )

        stats = state.cache.stats()["per_layer"][0]
        assert stats["ensure_calls"] == len(routed_pairs) + 1  # +1 bind-time
        assert stats["cold_misses"] >= min(gpu_slots, NUM_EXPERTS)
        if gpu_slots < NUM_EXPERTS:
            assert stats["gpu_evictions"] > 0
        assert stats["io_errors"] == 0
    finally:
        torch.set_default_device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tier_sub_batched_forward_matches_resident_layer(
    tmp_path, dist_init, workspace_init, tier_state_reset, monkeypatch
):
    """A batch routing more distinct experts than the GPU slot pool is
    applied in contiguous token sub-batches and must match the resident
    layer's unsplit forward."""
    torch.set_default_device("cuda")
    try:
        set_random_seed(11)
        cfg_ref, ref, tensors = _build_reference_and_cache(tmp_path)

        gpu_slots = 2  # batch below routes 4 distinct experts: must split
        gpu_gb, pinned_gb = _tier_budgets(
            tensors, gpu_slots=gpu_slots, pinned_slots=NUM_EXPERTS
        )
        cfg_tier = _make_vllm_config(
            ExpertTierOffloadConfig(
                enabled=True,
                cache_dir=str(tmp_path),
                gpu_gb=gpu_gb,
                pinned_gb=pinned_gb,
                io_threads=2,
            )
        )
        tier = _make_moe_layer(cfg_tier, prefix="model.layers.0.mlp.tier")
        tier_binding = tier.routed_experts.expert_tier
        assert tier_binding is not None
        assert tier_binding.gpu_slots == gpu_slots
        with set_current_vllm_config(cfg_tier):
            tier.routed_experts.quant_method.process_weights_after_loading(
                tier.routed_experts
            )

        ensure_calls = []
        real_ensure = ExpertTierLayerBinding.ensure

        def counting_ensure(self, topk_ids):
            ensure_calls.append(topk_ids.shape[0])
            return real_ensure(self, topk_ids)

        monkeypatch.setattr(ExpertTierLayerBinding, "ensure", counting_ensure)

        # Tokens 0..7 route to experts (0, 1), tokens 8..15 to (2, 3):
        # 4 distinct experts vs a 2-slot pool forces the split [0:8][8:16].
        num_tokens = 16
        hidden = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16) / 10
        logits = torch.full((num_tokens, NUM_EXPERTS), -20.0, dtype=torch.float32)
        logits[:8, [0, 1]] = 20.0
        logits[8:, [2, 3]] = 20.0

        out_ref = _forward(ref, cfg_ref, hidden, logits)
        out_tier = _forward(tier, cfg_tier, hidden, logits)
        assert ensure_calls == [8, 8]  # one ensure per sub-batch

        torch.testing.assert_close(out_tier, out_ref, atol=0.0, rtol=0.0)

        state = binding._state
        assert state is not None
        stats = state.cache.stats()["per_layer"][0]
        assert stats["ensure_calls"] == 3  # bind-time + 2 sub-batches
        assert stats["io_errors"] == 0
    finally:
        torch.set_default_device("cpu")
