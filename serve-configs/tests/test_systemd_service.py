# SPDX-License-Identifier: Apache-2.0
"""Deployment service contract tests."""

from configparser import ConfigParser
from pathlib import Path

from vllm.v1.utils import PROCESS_SHUTDOWN_CLEANUP_GRACE_S

SERVE_CONFIGS = Path(__file__).parents[1]


def _setting(path: Path, name: str) -> int:
    prefix = f"{name}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return int(line.removeprefix(prefix).strip())
    raise AssertionError(f"missing {name} in {path}")


def test_systemd_preserves_workers_during_engine_drain():
    """SIGTERM must reach the API first so EngineCore can drain its workers."""
    unit = ConfigParser(interpolation=None, strict=False)
    unit.read(
        filenames=SERVE_CONFIGS / "systemd" / "vllm-qwen38.service",
        encoding="utf-8",
    )

    service = unit["Service"]
    shutdown_timeout = _setting(
        path=SERVE_CONFIGS / "qwen3_8_27b_fp8_mtp_latency.yaml",
        name="shutdown-timeout",
    )

    assert service["KillMode"] == "mixed"
    assert int(service["TimeoutStopSec"]) >= (
        shutdown_timeout + PROCESS_SHUTDOWN_CLEANUP_GRACE_S
    )
