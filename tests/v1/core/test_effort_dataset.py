# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dynamic-effort training-data collector (docs §13.13)."""

import math

import numpy as np
import pytest

from vllm.v1.core.sched.effort_dataset import (
    ABORTED,
    EffortDatasetWriter,
    load_dataset,
    main,
    shard_paths,
)

pytestmark = pytest.mark.cpu_test

HIDDEN = 8


def _finish_kwargs(**over):
    base = dict(
        level=1,
        decided_by="vote",
        vote_probs=[0.1, 0.7, 0.2],
        level_votes=[1, 1, 2],
        estimate=0.4,
        calibrated=0.5,
        novelty_rank=0.3,
        neighbours=16,
        reasoning_tokens=120,
        num_output_tokens=200,
        close_kind="natural",
        finish_reason="stop",
    )
    base.update(over)
    return base


def _vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(HIDDEN).astype(np.float32)


def test_shards_rotate_and_round_trip(tmp_path):
    """`shard_size` finished examples make one shard; every field survives
    the trip through npz, including NaN-padded vote columns."""
    writer = EffortDatasetWriter(str(tmp_path), HIDDEN, shard_size=2)
    for i in range(5):
        assert writer.begin(f"r{i}", _vector(i), 100 + i, 70, tag="job-a")
        kwargs = _finish_kwargs()
        if i == 4:
            kwargs.update(vote_probs=None, level_votes=None, decided_by="memory")
        assert writer.finish(f"r{i}", **kwargs)
    writer.close()
    assert writer.num_written == 5 and writer.num_dropped == 0
    assert len(shard_paths(str(tmp_path))) == 3

    data = load_dataset(str(tmp_path))
    assert data["req_id"].tolist() == [f"r{i}" for i in range(5)]
    assert data["vector"].dtype == np.float16 and data["vector"].shape == (5, HIDDEN)
    np.testing.assert_allclose(data["vector"][3], _vector(3), rtol=1e-2)
    assert data["level"].tolist() == [1] * 5
    assert data["decided_by"].tolist() == ["vote"] * 4 + ["memory"]
    assert data["vote_probs"].shape == (5, 3)
    assert data["vote_probs"][0].tolist() == pytest.approx([0.1, 0.7, 0.2])
    assert all(math.isnan(p) for p in data["vote_probs"][4])
    assert data["level_votes"][4].tolist() == [-1, -1, -1]
    assert data["tag"].tolist() == ["job-a"] * 5
    assert data["num_prompt_tokens"].tolist() == [100, 101, 102, 103, 104]
    assert data["reasoning_tokens"].tolist() == [120] * 5
    assert data["finish_reason"].tolist() == ["stop"] * 5


def test_abort_and_pending_at_close(tmp_path):
    """An aborted request is written as `aborted`; a request still pending at
    close is flushed the same way; `forget` drops it silently."""
    writer = EffortDatasetWriter(str(tmp_path), HIDDEN, shard_size=100)
    writer.begin("done", _vector(1), 50, 30)
    writer.begin("aborted", _vector(2), 50, 30)
    writer.begin("forgotten", _vector(3), 50, 30)
    writer.begin("pending", _vector(4), 50, 30)
    writer.finish("done", **_finish_kwargs())
    writer.finish(
        "aborted",
        **_finish_kwargs(finish_reason=None, close_kind="client-limit"),
    )
    writer.forget("forgotten")
    assert not writer.finish("never-begun", **_finish_kwargs())
    assert writer.num_pending == 1
    writer.close()
    data = load_dataset(str(tmp_path))
    assert data["req_id"].tolist() == ["done", "aborted", "pending"]
    assert data["finish_reason"].tolist() == ["stop", ABORTED, ABORTED]
    assert data["level"].tolist() == [1, 1, -1]


def test_wrong_width_and_full_buffer_drop(tmp_path):
    writer = EffortDatasetWriter(str(tmp_path), HIDDEN, shard_size=1000, max_buffered=2)
    assert not writer.begin("bad", np.zeros(HIDDEN + 1), 10, 5)
    for i in range(3):
        writer.begin(f"r{i}", _vector(i), 10, 5)
    assert writer.finish("r0", **_finish_kwargs())
    assert writer.finish("r1", **_finish_kwargs())
    assert not writer.finish("r2", **_finish_kwargs())
    assert writer.num_dropped == 1
    writer.close()
    assert len(load_dataset(str(tmp_path))["req_id"]) == 2


def test_empty_directory_and_cli(tmp_path, capsys):
    assert len(load_dataset(str(tmp_path))["req_id"]) == 0
    writer = EffortDatasetWriter(str(tmp_path), HIDDEN)
    writer.begin("r", _vector(0), 10, 5, tag="t1")
    writer.finish("r", **_finish_kwargs())
    writer.close()
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1 examples in 1 shards" in out
    assert "level: 1=1" in out and "t1=1" in out
    assert main([]) == 2
