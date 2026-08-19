# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the P0 effort telemetry (entropy / margin) lane."""

import io
import json
import math

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample import effort_signals as es
from vllm.v1.sample.effort_signals import (
    ENTROPY,
    MARGIN,
    NUM_ROWS,
    EffortTelemetrySink,
    ThinkTracker,
    commit_order_permutation,
    effort_row_signals,
    effort_row_signals_scattered,
    flagged_row_indices,
    format_sink_record,
    reduce_committed,
    signals_to_dict,
    wants_effort_signals,
)
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID, RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu.sample import effort as v2_effort


def _np_reference(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = logits.astype(np.float64)
    m = z.max(-1, keepdims=True)
    p = np.exp(z - m)
    p /= p.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log(p)).sum(-1, where=p > 0)
    h /= math.log(z.shape[-1])
    top2 = -np.sort(-z, axis=-1)[:, :2]
    return h, top2[:, 0] - top2[:, 1]


# --------------------------------------------------------------------------
# Row scalars
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_row_signals_match_numpy_reference(dtype):
    torch.manual_seed(0)
    logits = (torch.randn(16, 257) * 3).to(dtype)
    got = effort_row_signals(logits)
    h_ref, m_ref = _np_reference(logits.float().numpy())
    assert got.dtype == torch.float32 and got.shape == (16, 3)
    np.testing.assert_allclose(got[:, ENTROPY].numpy(), h_ref, atol=1e-5)
    np.testing.assert_allclose(got[:, MARGIN].numpy(), m_ref, atol=1e-5)


def test_row_signals_masked_vocab_and_bounds():
    vocab = 128
    uniform = torch.zeros(1, vocab)
    one_hot = torch.full((1, vocab), -float("inf"))
    one_hot[0, 3] = 0.0
    partial = torch.randn(1, vocab)
    partial[0, 10:] = -float("inf")
    got = effort_row_signals(torch.cat([uniform, one_hot, partial]))
    assert got[0, ENTROPY] == pytest.approx(1.0, abs=1e-6)
    assert got[0, MARGIN] == 0.0
    assert got[1, ENTROPY] == pytest.approx(0.0, abs=1e-6)
    # Fully peaked rows have an infinite margin; it is clamped to finite.
    assert torch.isfinite(got[1, MARGIN]) and got[1, MARGIN] > 1e30
    h_ref, m_ref = _np_reference(partial.numpy())
    assert got[2, ENTROPY] == pytest.approx(h_ref[0], abs=1e-5)
    assert got[2, MARGIN] == pytest.approx(m_ref[0], abs=1e-5)
    torch.manual_seed(1)
    rnd = effort_row_signals(torch.randn(64, vocab) * 5)
    assert torch.all(rnd[:, ENTROPY] >= 0) and torch.all(rnd[:, ENTROPY] <= 1)
    assert torch.all(rnd[:, MARGIN] >= 0)


def test_row_signals_scattered_only_touches_selected_rows(monkeypatch):
    calls: list[int] = []
    real = es.effort_row_signals

    def spy(logits, end_token_id=None):
        calls.append(logits.shape[0])
        return real(logits, end_token_id)

    monkeypatch.setattr(es, "effort_row_signals", spy)
    logits = torch.randn(6, 32)
    rows = torch.tensor([1, 4])
    out = effort_row_signals_scattered(logits, rows)
    assert calls == [2]
    assert torch.equal(out[[0, 2, 3, 5]], torch.zeros(4, 3))
    torch.testing.assert_close(out[rows], real(logits[rows]))
    empty = effort_row_signals_scattered(logits, torch.zeros(0, dtype=torch.long))
    assert calls == [2] and torch.equal(empty, torch.zeros(6, 3))


# --------------------------------------------------------------------------
# Commit-aware reduction
# --------------------------------------------------------------------------


def test_reduce_committed_mixed_acceptance():
    # 4 requests, K=3 drafts each -> 4 rows per request in commit order.
    torch.manual_seed(2)
    k = 3
    num_reqs = 4
    rows = torch.rand(num_reqs * (k + 1), 3)
    cu = torch.arange(num_reqs + 1) * (k + 1)
    committed = torch.tensor([1, 2, 4, 3])  # 0..K accepted (+1 bonus/recovery)
    out = reduce_committed(rows, cu, committed)
    assert out.shape == (num_reqs, 4)
    for i in range(num_reqs):
        n = int(committed[i])
        seg = rows[i * (k + 1) : i * (k + 1) + n]
        assert out[i, NUM_ROWS] == n
        torch.testing.assert_close(out[i, :3], seg.mean(0))


def test_reduce_committed_zero_rows_and_ragged():
    rows = torch.tensor([[0.2, 1.0, 0.1], [0.4, 3.0, 0.2], [0.9, 9.0, 0.3]])
    cu = torch.tensor([0, 2, 3])
    out = reduce_committed(rows, cu, torch.tensor([0, 1]))
    assert out[0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert out[1].tolist() == pytest.approx([0.9, 9.0, 0.3, 1.0])
    masked = reduce_committed(
        rows, cu, torch.tensor([2, 1]), row_mask=torch.tensor([True, False, True])
    )
    assert masked[0].tolist() == pytest.approx([0.2, 1.0, 0.1, 1.0])
    empty = reduce_committed(torch.zeros(0, 3), torch.tensor([0]), torch.zeros(0))
    assert empty.shape == (0, 4)


def test_commit_order_permutation_and_flagged_rows():
    perm = commit_order_permutation([2, 0, 1])
    # target rows: [r0d0, r0d1, r2d0]; bonus rows: [b0, b1, b2] at 3,4,5.
    assert perm.tolist() == [0, 1, 3, 4, 2, 5]
    flags = np.array([True, False, True])
    assert flagged_row_indices(flags).tolist() == [0, 2]
    assert flagged_row_indices(flags, np.array([2, 3, 1])).tolist() == [0, 1, 5]
    assert flagged_row_indices(np.zeros(3, dtype=bool)).size == 0


# --------------------------------------------------------------------------
# V1 sampler / rejection sampler hooks
# --------------------------------------------------------------------------


def _v1_metadata(num_reqs: int, effort_mask) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(num_reqs),
        presence_penalties=torch.zeros(num_reqs),
        repetition_penalties=torch.ones(num_reqs),
        output_token_ids=[[] for _ in range(num_reqs)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        effort_mask=effort_mask,
    )


def test_v1_sampler_emits_flagged_rows_only_and_keeps_tokens(monkeypatch):
    calls: list[int] = []
    real = es.effort_row_signals

    def spy(logits, end_token_id=None):
        calls.append(logits.shape[0])
        return real(logits, end_token_id)

    monkeypatch.setattr(es, "effort_row_signals", spy)
    torch.manual_seed(3)
    logits = torch.randn(4, 96)
    sampler = Sampler()
    mask = np.array([True, False, True, False])
    out = sampler(logits.clone(), _v1_metadata(4, mask))
    assert calls == [2]
    assert torch.equal(out.sampled_token_ids.view(-1).long(), logits.argmax(-1))
    sig = out.effort_signals
    assert sig is not None and sig.shape == (4, 4)
    ref = real(logits[[0, 2]])
    torch.testing.assert_close(sig[[0, 2], :3], ref)
    assert sig[:, NUM_ROWS].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert out.effort_flags is mask
    d = signals_to_dict(["a", "b", "c", "d"], sig.numpy(), mask)
    assert set(d) == {"a", "c"} and d["a"][2] == 1

    # No flagged request: no work, no output.
    out2 = sampler(logits.clone(), _v1_metadata(4, None))
    assert calls == [2]
    assert out2.effort_signals is None and out2.effort_flags is None
    assert sampler._effort_rows is None
    assert torch.equal(out2.sampled_token_ids, out.sampled_token_ids)


def test_v1_rejection_target_rows_and_committed_reduction(monkeypatch):
    calls: list[int] = []
    real = es.effort_row_signals

    def spy(logits, end_token_id=None):
        calls.append(logits.shape[0])
        return real(logits, end_token_id)

    monkeypatch.setattr(es, "effort_row_signals", spy)
    torch.manual_seed(4)
    num_draft = [2, 3, 0]
    total = sum(num_draft)
    vocab = 40
    rs = RejectionSampler(Sampler())
    md = SpecDecodeMetadata.make_dummy(
        [[1] * d for d in num_draft], device=torch.device("cpu")
    )
    target_logits = torch.randn(total, vocab)
    mask = np.array([True, False, True])
    rs.apply_logits_processors(target_logits.clone(), _v1_metadata(3, mask), md)
    # Only request 0's two draft rows are flagged (request 2 has no drafts).
    assert calls == [2]
    rows = rs._effort_target_rows
    assert rows is not None and rows.shape == (total, 2)
    torch.testing.assert_close(rows[:2], real(target_logits[:2]))
    assert torch.equal(rows[2:], torch.zeros(3, 2))

    bonus_rows = torch.rand(3, 2)
    # Request 0: 1 of 2 accepted -> rows d0, d1 (recovery at pos 1).
    # Request 1: all 3 accepted -> rows d0..d2 + bonus.
    # Request 2: no drafts -> bonus only.
    p = PLACEHOLDER_TOKEN_ID
    output = torch.tensor([[5, 6, p, p], [1, 2, 3, 4], [9, p, p, p]])
    red = RejectionSampler._reduce_effort_signals(rows, bonus_rows, num_draft, output)
    assert red[:, NUM_ROWS].tolist() == [2.0, 4.0, 1.0]
    torch.testing.assert_close(red[0, :2], rows[:2].mean(0))
    torch.testing.assert_close(
        red[1, :2], torch.cat([rows[2:5], bonus_rows[1:2]]).mean(0)
    )
    torch.testing.assert_close(red[2, :2], bonus_rows[2])

    # Unflagged batch: nothing computed.
    rs.apply_logits_processors(target_logits.clone(), _v1_metadata(3, None), md)
    assert calls == [2] and rs._effort_target_rows is None


# --------------------------------------------------------------------------
# V2 EffortState (CPU stand-in for the UVA flag tensor)
# --------------------------------------------------------------------------


class _CpuBacked:
    def __init__(self, size, dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = self.cpu
        self.copies = 0

    def copy_to_uva(self, n=None):
        self.copies += 1
        return self.gpu


def _sp(**extra) -> SamplingParams:
    return SamplingParams(extra_args=extra or None)


def test_wants_effort_signals():
    assert not wants_effort_signals(None)
    assert not wants_effort_signals(_sp())
    assert not wants_effort_signals(_sp(effort_telemetry=False))
    assert wants_effort_signals(_sp(effort_telemetry=True))
    assert wants_effort_signals(_sp(dynamic_effort={"ladder": [1024]}))


def test_v2_effort_state_paths(monkeypatch):
    monkeypatch.setattr(v2_effort, "UvaBackedTensor", _CpuBacked)
    calls: list[int] = []
    real = es.effort_row_signals

    def spy(logits, end_token_id=None):
        calls.append(logits.shape[0])
        return real(logits, end_token_id)

    monkeypatch.setattr(es, "effort_row_signals", spy)
    monkeypatch.setattr(v2_effort, "effort_row_signals", spy)
    st = v2_effort.EffortState(8, torch.device("cpu"))
    st.add_request(2, _sp(effort_telemetry=True))
    st.add_request(5, _sp())
    st.apply_staged_writes()
    assert st.flags.copies == 1 and st.flags.np[2] == 1 and st.flags.np[5] == 0

    # Batch of unflagged requests: begin False, compute/finish are no-ops.
    idx = np.array([5, 0])
    assert st.begin(idx) is False
    st.compute(torch.randn(2, 16), torch.tensor(idx), idx)
    assert calls == [] and st.finish(torch.tensor([0, 1, 2]), torch.ones(2)) is None

    # Decode batch (one row per request): exact host row selection.
    idx = np.array([5, 2, 0])
    assert st.begin(idx) is True
    logits = torch.randn(3, 16)
    st.compute(logits, torch.tensor(idx), idx)
    assert calls == [1]
    out = st.finish(torch.tensor([0, 1, 2, 3]), torch.tensor([1, 1, 0]))
    assert out is not None and out[:, NUM_ROWS].tolist() == [0.0, 1.0, 0.0]
    torch.testing.assert_close(out[1, :2], real(logits[1:2])[0])
    assert st.batch_flags(idx).tolist() == [False, True, False]

    # Spec batch (rows != reqs): whole chunk computed, unflagged rows zeroed,
    # reduction over committed prefix (num_sampled) only.
    idx = np.array([2, 5])
    expanded = torch.tensor([2, 2, 2, 5, 5])
    cu = torch.tensor([0, 3, 5])
    st.begin(idx)
    logits = torch.randn(5, 16)
    st.compute(logits, expanded, idx)
    assert calls == [1, 5]
    out = st.finish(cu, torch.tensor([2, 2]))
    assert out[:, NUM_ROWS].tolist() == [2.0, 0.0]
    torch.testing.assert_close(out[0, :2], real(logits[:2]).mean(0))
    assert out[1, :2].tolist() == [0.0, 0.0]

    # Chunk without flagged requests still keeps row alignment.
    st.begin(np.array([2, 5]))
    st.compute(torch.randn(2, 16), torch.tensor([5, 5]), np.array([5]))
    st.compute(torch.randn(1, 16), torch.tensor([2]), np.array([2]))
    assert calls == [1, 5, 1]
    out = st.finish(torch.tensor([0, 2, 3]), torch.tensor([1, 1]))
    assert out[:, NUM_ROWS].tolist() == [0.0, 1.0]
    assert out[0, :2].tolist() == [0.0, 0.0]


# --------------------------------------------------------------------------
# Scheduler sink
# --------------------------------------------------------------------------


def test_think_tracker_rules():
    t = ThinkTracker([10, 11], [20])
    assert t.update([1, 2]) is False
    assert t.update([1, 2, 10]) is False
    assert t.update([1, 2, 10, 11, 3]) is True
    assert t.update([1, 2, 10, 11, 3, 20]) is False
    assert t.update([1, 2, 10, 11, 3, 20, 10, 11]) is True
    assert ThinkTracker(None, [20]).update([10, 11]) is None
    # Incremental scan sees a start sequence split across two updates.
    t2 = ThinkTracker([10, 11], [20])
    assert t2.update([10]) is False
    assert t2.update([10, 11]) is True


def test_sink_record_schema_and_flush():
    line = format_sink_record("r1", 3, 17, (0.5, 2.25, 0.125, 4), 7, 3, True)
    rec = json.loads(line)
    assert list(rec) == [
        "req_id",
        "step",
        "num_output_tokens",
        "entropy",
        "margin",
        "p_end",
        "n_rows",
        "num_draft_tokens",
        "num_accepted",
        "in_think",
    ]
    assert rec == {
        "req_id": "r1",
        "step": 3,
        "num_output_tokens": 17,
        "entropy": 0.5,
        "margin": 2.25,
        "p_end": 0.125,
        "n_rows": 4,
        "num_draft_tokens": 7,
        "num_accepted": 3,
        "in_think": True,
    }

    buf = io.StringIO()
    sink = EffortTelemetrySink("unused", [10], [20], stream=buf)
    sink.FLUSH_EVERY = 3
    sink.record("a", [10, 1], (0.1, 1.0, 0.0, 1), None, None, False)
    sink.record("b", [2], (0.2, 2.0, 0.0, 1), 3, 1, False)
    assert buf.getvalue() == ""  # buffered
    sink.record("a", [10, 1, 2], (0.3, 3.0, 0.0, 2), None, None, False)
    lines = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert [(r["req_id"], r["step"]) for r in lines] == [("a", 1), ("b", 1), ("a", 2)]
    assert lines[0]["in_think"] is True and lines[1]["in_think"] is False
    assert lines[1]["num_draft_tokens"] == 3 and lines[0]["num_accepted"] is None
    # Finish flushes immediately and forgets the per-request state.
    sink.record("b", [2, 20], (0.4, 4.0, 0.0, 1), 3, 0, True)
    assert json.loads(buf.getvalue().splitlines()[-1])["step"] == 2
    sink.record("b", [5], (0.5, 5.0, 0.0, 1), None, None, True)
    assert json.loads(buf.getvalue().splitlines()[-1])["step"] == 1


def test_sink_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_EFFORT_TELEMETRY", raising=False)
    assert EffortTelemetrySink.from_env([1], [2]) is None
    path = tmp_path / "effort.jsonl"
    monkeypatch.setenv("VLLM_EFFORT_TELEMETRY", str(path))
    sink = EffortTelemetrySink.from_env([1], [2])
    assert sink is not None
    sink.record("x", [1, 3], (0.9, 0.1, 0.0, 1), None, None, True)
    sink.close()
    assert json.loads(path.read_text().strip())["in_think"] is True


def test_think_tracker_seeded_from_prompt_tail():
    from vllm.v1.sample.effort_signals import ThinkTracker

    t = ThinkTracker([7], [8])
    t.seed_from_prompt([1, 2, 7, 3])  # "<think>\n" opened in the prompt
    assert t.update([5, 6]) is True
    assert t.update([5, 6, 8, 9]) is False
    assert t.update([5, 6, 8, 9, 7, 1]) is True
    u = ThinkTracker([7], [8])
    u.seed_from_prompt([1, 7, 2, 8])  # closed in the prompt
    assert u.update([5]) is False
