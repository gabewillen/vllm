# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights,
)
from vllm.v1.spec_decode import draft_lm_head


def _monolithic_int4_pack(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_ref, qweight, scales, _ = quantize_weights(
        w=weight.t().contiguous().float(),
        quant_type=draft_lm_head.MARLIN_WEIGHT_TYPE,
        group_size=draft_lm_head.MARLIN_GROUP_SIZE,
    )
    packed = draft_lm_head._pack_rows_gptq(
        q_w=qweight,
        num_bits=draft_lm_head.MARLIN_WEIGHT_TYPE.size_bits,
    )
    return packed, scales, weight_ref


@pytest.mark.parametrize(
    ("output_size", "chunk_size"),
    [(1, 1), (7, 4), (13, 8), (16, 7)],
)
def test_chunked_int4_pack_matches_monolithic(
    output_size: int, chunk_size: int
) -> None:
    input_size = 256
    weight = torch.linspace(
        start=-1,
        end=1,
        steps=output_size * input_size,
        dtype=torch.float32,
    ).reshape(output_size, input_size)
    weight = weight.to(torch.bfloat16)
    expected_packed, expected_scales, _ = _monolithic_int4_pack(weight)

    actual_packed, actual_scales = draft_lm_head._quantize_int4_gptq_chunked(
        w=weight,
        chunk_size=chunk_size,
    )

    assert torch.equal(actual_packed, expected_packed)
    assert torch.equal(actual_scales, expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunked_int4_head_matches_monolithic_marlin(monkeypatch) -> None:
    if not check_marlin_supported(
        quant_type=draft_lm_head.MARLIN_WEIGHT_TYPE,
        group_size=draft_lm_head.MARLIN_GROUP_SIZE,
    ):
        pytest.skip("requires Marlin uint4b8 group-128 support")

    torch.manual_seed(seed=17)
    device = torch.device("cuda")
    input_size = 256
    output_size = 192
    target = nn.Linear(
        input_size,
        output_size,
        bias=False,
        dtype=torch.bfloat16,
        device=device,
    )
    expected_packed, expected_scales, weight_ref = _monolithic_int4_pack(
        target.weight.data
    )
    empty = torch.empty(size=(0,), dtype=torch.int32, device=device)
    expected_qweight = ops.gptq_marlin_repack(
        b_q_weight=expected_packed,
        perm=empty,
        size_k=input_size,
        size_n=output_size,
        num_bits=draft_lm_head.MARLIN_WEIGHT_TYPE.size_bits,
    )
    expected_marlin_scales = marlin_permute_scales(
        s=expected_scales.to(target.weight.dtype),
        size_k=input_size,
        size_n=output_size,
        group_size=draft_lm_head.MARLIN_GROUP_SIZE,
    )
    monkeypatch.setattr(
        target=draft_lm_head,
        name="INT4_DRAFT_HEAD_CHUNK_SIZE",
        value=128,
    )

    actual = draft_lm_head.QuantizedDraftLMHead(
        target_head=target,
        dtype="int4",
    )

    assert torch.equal(actual.marlin_qweight, expected_qweight)
    assert torch.equal(actual.marlin_scales, expected_marlin_scales)
    activation = torch.randn(
        size=(3, input_size),
        dtype=torch.bfloat16,
        device=device,
    )
    expected_output = activation @ weight_ref.to(torch.bfloat16)
    actual_output = actual.quant_method.apply(layer=actual, x=activation)
    torch.testing.assert_close(
        actual=actual_output,
        expected=expected_output,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize(
    ("weight", "chunk_size", "error"),
    [
        (torch.empty(size=(128,)), 1, "must be 2D"),
        (torch.empty(size=(2, 129)), 1, "multiple of group size"),
        (torch.empty(size=(2, 128)), 0, "must be positive"),
        (torch.empty(size=(0, 128)), 1, "N must be positive"),
        (torch.empty(size=(2, 0)), 1, "K must be positive"),
    ],
)
def test_chunked_int4_pack_rejects_invalid_inputs(
    weight: torch.Tensor,
    chunk_size: int,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        draft_lm_head._quantize_int4_gptq_chunked(
            w=weight,
            chunk_size=chunk_size,
        )


def test_chunked_int4_pack_clamps_caller_chunk_size(monkeypatch) -> None:
    weight = torch.zeros(size=(5, 128), dtype=torch.bfloat16)
    observed_chunk_sizes = []
    original_quantize_weights = draft_lm_head.quantize_weights

    def observe_chunk(*, w, quant_type, group_size):
        observed_chunk_sizes.append(w.shape[1])
        return original_quantize_weights(
            w=w,
            quant_type=quant_type,
            group_size=group_size,
        )

    monkeypatch.setattr(
        target=draft_lm_head,
        name="INT4_DRAFT_HEAD_CHUNK_SIZE",
        value=2,
    )
    monkeypatch.setattr(
        target=draft_lm_head,
        name="quantize_weights",
        value=observe_chunk,
    )

    draft_lm_head._quantize_int4_gptq_chunked(
        w=weight,
        chunk_size=1024,
    )

    assert observed_chunk_sizes == [2, 2, 1]
