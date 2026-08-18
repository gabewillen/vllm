# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the DP>1 adaptive guard (0005) and dense-DBO gate (0006)."""

import torch
from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows,
    quantize_weights,
)
from vllm.scalar_type import scalar_types
from vllm.v1.spec_decode.draft_lm_head import _pack_rows_gptq
from vllm.v1.worker import ubatching


class _Parallel:
    def __init__(self, dp: int):
        self.data_parallel_size = dp


class _Stub:
    """Only the attributes _maybe_disable_dynamic_sd_for_data_parallel reads."""

    def __init__(self, spec: SpeculativeConfig, dp: int):
        self.speculative_config = spec
        self.parallel_config = _Parallel(dp)


def _spec(**kw):
    return SpeculativeConfig(method="ngram", prompt_lookup_max=3, **kw)


def test_adaptive_disabled_under_data_parallel():
    spec = _spec(num_speculative_tokens=4, adaptive_draft_length=True)
    VllmConfig._maybe_disable_dynamic_sd_for_data_parallel(_Stub(spec, dp=2))
    assert spec.adaptive_draft_length is False


def test_adaptive_kept_with_single_dp_rank():
    spec = _spec(num_speculative_tokens=4, adaptive_draft_length=True)
    VllmConfig._maybe_disable_dynamic_sd_for_data_parallel(_Stub(spec, dp=1))
    assert spec.adaptive_draft_length is True


def test_schedule_still_disabled_under_data_parallel():
    spec = _spec(
        num_speculative_tokens=4,
        num_speculative_tokens_per_batch_size=[[1, 8, 4], [9, 32, 0]],
    )
    VllmConfig._maybe_disable_dynamic_sd_for_data_parallel(_Stub(spec, dp=2))
    assert spec.num_speculative_tokens_per_batch_size is None


def test_gptq_row_packing_matches_reference():
    torch.manual_seed(0)
    w = torch.randn(512, 64)  # [K, N]
    _, q_w, _, _ = quantize_weights(
        w=w, quant_type=scalar_types.uint4b8, group_size=128
    )
    ours = _pack_rows_gptq(q_w=q_w, num_bits=4)
    ref = pack_rows(q_w, 4, 512, 64)
    assert torch.equal(ours, ref)


def test_dense_dbo_gate_requires_flag_and_ubatch_thread():
    ubatching.set_overlap_tp_all_reduce(False)
    assert ubatching.dbo_overlap_tp_all_reduce() is False
    ubatching.set_overlap_tp_all_reduce(True)
    # no micro-batch context on this thread -> still False
    assert ubatching.dbo_enabled() is False
    assert ubatching.dbo_overlap_tp_all_reduce() is False
    ubatching.set_overlap_tp_all_reduce(False)
