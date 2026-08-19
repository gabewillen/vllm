# SPDX-License-Identifier: Apache-2.0
"""Micro-batch slicing keeps one state slot for a split request (patch 0006)."""

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.ubatch_utils import UBatchSlice, _make_metadata_with_slice


def _metadata():
    # two requests: req0 has 100 computed + 300 new tokens, req1 20 + 100
    query_start_loc = torch.tensor([0, 300, 400], dtype=torch.int32)
    seq_lens = torch.tensor([400, 120], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=seq_lens,
        num_reqs=2,
        num_actual_tokens=400,
        max_query_len=300,
        max_seq_len=400,
        block_table_tensor=torch.zeros((2, 4), dtype=torch.int32),
        slot_mapping=torch.arange(400, dtype=torch.int64),
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        _seq_lens_cpu=seq_lens.clone(),
        _num_computed_tokens_cpu=torch.tensor([100, 20], dtype=torch.int32),
    )


def test_split_request_keeps_full_state_seq_len():
    md = _metadata()
    first = _make_metadata_with_slice(UBatchSlice(slice(0, 1), slice(0, 200)), md)
    second = _make_metadata_with_slice(UBatchSlice(slice(0, 2), slice(200, 400)), md)
    # attention lengths describe each slice ...
    assert first.seq_lens.tolist() == [300]  # 400 - 100 skipped tokens
    assert second.seq_lens.tolist() == [400, 120]
    # ... but the state slot follows the request's full step length
    assert first.state_seq_lens.tolist() == [400]
    assert second.state_seq_lens.tolist() == [400, 120]
    # the continuation has computed the first slice's tokens
    assert second.compute_num_computed_tokens().tolist() == [300, 20]
    assert second._num_computed_tokens_cpu.tolist() == [300, 20]


def test_unpadded_keeps_state_seq_lens():
    md = _metadata()
    md.mamba_state_seq_lens = torch.tensor([400, 120], dtype=torch.int32)
    assert md.unpadded(300, 1).mamba_state_seq_lens.tolist() == [400]


def test_state_seq_lens_defaults_to_seq_lens():
    md = _metadata()
    assert md.mamba_state_seq_lens is None
    assert torch.equal(md.state_seq_lens, md.seq_lens)
