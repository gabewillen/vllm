# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-logic tests for tools/dsv4_expert_cache_convert.py.

Covers spec parsing, manifest schema, shard slicing math, resume logic,
and shard-size computation with synthetic small tensors. GPU-dependent
conversion paths are exercised by the tool's own --verify mode.
"""

import json
import struct

import numpy as np
import pytest
import torch

from tools.dsv4_expert_cache_convert import (
    CACHE_TENSOR_NAMES,
    build_layer_manifest,
    cache_file_name,
    ckpt_shard_slice,
    expected_marlin_specs,
    file_complete,
    parse_int_list,
    safetensors_expected_size,
    update_manifest,
)


class TestParseIntList:
    def test_range(self):
        assert parse_int_list("0-4") == [0, 1, 2, 3, 4]

    def test_mixed(self):
        assert parse_int_list("5-7,0,2") == [0, 2, 5, 6, 7]

    def test_dedup(self):
        assert parse_int_list("1,1,1-2") == [1, 2]

    def test_descending_range_rejected(self):
        with pytest.raises(ValueError, match="descending"):
            parse_int_list("4-0")

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            parse_int_list(",")


class TestExpectedMarlinSpecs:
    def test_dsv4_tp4_shapes(self):
        specs = expected_marlin_specs(4, hidden_size=4096, intermediate_size=2048)
        assert specs["w13_weight"]["per_expert_shape"] == [256, 2048]
        assert specs["w13_weight"]["dtype"] == "torch.int32"
        assert specs["w13_weight"]["row_bytes"] == 256 * 2048 * 4
        assert specs["w13_weight_scale"]["per_expert_shape"] == [128, 1024]
        assert specs["w13_weight_scale"]["dtype"] == "torch.float8_e8m0fnu"
        assert specs["w13_weight_scale"]["row_bytes"] == 128 * 1024
        assert specs["w2_weight"]["per_expert_shape"] == [32, 8192]
        assert specs["w2_weight"]["row_bytes"] == 32 * 8192 * 4
        assert specs["w2_weight_scale"]["per_expert_shape"] == [16, 4096]
        assert specs["w2_weight_scale"]["row_bytes"] == 16 * 4096

    def test_tp1_shapes(self):
        specs = expected_marlin_specs(1, hidden_size=4096, intermediate_size=2048)
        assert specs["w13_weight"]["per_expert_shape"] == [256, 8192]
        assert specs["w2_weight"]["per_expert_shape"] == [128, 8192]

    def test_indivisible_tp_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            expected_marlin_specs(3, hidden_size=4096, intermediate_size=2048)

    def test_marlin_padding_rejected(self):
        with pytest.raises(ValueError, match="pad"):
            expected_marlin_specs(4, hidden_size=4096, intermediate_size=768)


class TestCkptShardSlice:
    def test_w1_w3_shard_rows(self):
        for shard_id in ("w1", "w3"):
            for rank in range(4):
                dim, start, length = ckpt_shard_slice(
                    shard_id, (2048, 2048), rank, 4
                )
                assert (dim, start, length) == (0, 512 * rank, 512)

    def test_w2_shards_packed_cols(self):
        for rank in range(4):
            dim, start, length = ckpt_shard_slice("w2", (4096, 1024), rank, 4)
            assert (dim, start, length) == (1, 256 * rank, 256)

    def test_scale_shapes(self):
        assert ckpt_shard_slice("w1", (2048, 128), 3, 4) == (0, 1536, 512)
        assert ckpt_shard_slice("w2", (4096, 64), 2, 4) == (1, 32, 16)

    def test_bad_shard_id(self):
        with pytest.raises(ValueError, match="shard_id"):
            ckpt_shard_slice("w4", (8, 8), 0, 4)

    def test_slices_partition_synthetic_tensor(self):
        t = torch.arange(16 * 8, dtype=torch.int32).reshape(16, 8)
        parts = []
        for rank in range(4):
            dim, start, length = ckpt_shard_slice("w1", tuple(t.shape), rank, 4)
            parts.append(t.narrow(dim, start, length))
        assert torch.equal(torch.cat(parts, dim=0), t)
        parts = []
        for rank in range(4):
            dim, start, length = ckpt_shard_slice("w2", tuple(t.shape), rank, 4)
            parts.append(t.narrow(dim, start, length))
        assert torch.equal(torch.cat(parts, dim=1), t)


class TestManifest:
    def test_layer_manifest_schema(self):
        specs = expected_marlin_specs(4)
        entries = build_layer_manifest(7, specs, 256)
        assert set(entries) == set(CACHE_TENSOR_NAMES)
        for name, entry in entries.items():
            assert entry["file"] == f"layer007.{name}.raw"
            assert entry["nbytes"] == entry["row_bytes"] * 256
            assert isinstance(entry["per_expert_shape"], list)
            assert entry["dtype"].startswith("torch.")

    def test_update_manifest_merge_and_sort(self, tmp_path):
        path = str(tmp_path / "manifest.json")
        specs = expected_marlin_specs(4)
        meta = {"tp_size": 4, "tp_rank": 0, "converter_version": 1}
        update_manifest(path, meta, 256, 10, build_layer_manifest(10, specs, 256))
        update_manifest(path, meta, 256, 2, build_layer_manifest(2, specs, 256))
        with open(path) as f:
            manifest = json.load(f)
        assert manifest["num_experts"] == 256
        assert manifest["meta"] == meta
        assert list(manifest["layers"]) == ["2", "10"]
        for entries in manifest["layers"].values():
            assert set(entries) == set(CACHE_TENSOR_NAMES)


class TestResumeLogic:
    def test_file_complete(self, tmp_path):
        path = tmp_path / "layer000.w13_weight.raw"
        assert not file_complete(str(path), 16)
        path.write_bytes(b"\x00" * 16)
        assert file_complete(str(path), 16)
        assert not file_complete(str(path), 17)

    def test_cache_file_name(self):
        assert cache_file_name(0, "w13_weight") == "layer000.w13_weight.raw"
        assert cache_file_name(42, "w2_weight_scale") == (
            "layer042.w2_weight_scale.raw"
        )


class TestSafetensorsExpectedSize:
    def _write_shard(self, path, truncate=0):
        data = np.arange(6, dtype=np.float32).tobytes()
        header = {
            "a": {
                "dtype": "F32",
                "shape": [2, 3],
                "data_offsets": [0, len(data)],
            }
        }
        raw = json.dumps(header).encode()
        blob = struct.pack("<Q", len(raw)) + raw + data
        if truncate:
            blob = blob[:-truncate]
        path.write_bytes(blob)

    def test_complete_shard(self, tmp_path):
        path = tmp_path / "shard.safetensors"
        self._write_shard(path)
        assert safetensors_expected_size(str(path)) == path.stat().st_size

    def test_truncated_shard_detected(self, tmp_path):
        path = tmp_path / "shard.safetensors"
        self._write_shard(path, truncate=4)
        assert safetensors_expected_size(str(path)) == path.stat().st_size + 4
