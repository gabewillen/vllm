# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


def test_shutdown_gives_a_sigtermed_process_time_to_clean_up():
    """timeout=0 aborts in-flight requests; it must not SIGKILL the engine
    before its SIGTERM handler has run its own teardown."""
    import multiprocessing
    import time

    from vllm.v1.utils import shutdown

    def slow_exit():
        import signal
        import sys

        signal.signal(signal.SIGTERM, lambda *_: time.sleep(1.0) or sys.exit(0))
        time.sleep(60)

    proc = multiprocessing.get_context("fork").Process(target=slow_exit)
    proc.start()
    time.sleep(0.3)
    shutdown([proc], timeout=0)
    proc.join(1.0)
    assert proc.exitcode == 0, proc.exitcode


def test_shutdown_lifecycle_has_explicit_terminal_and_ignored_edges():
    from vllm.v1.engine.core import (
        EngineShutdownActivity,
        EngineShutdownDisposition,
        EngineShutdownEvent,
        EngineShutdownEventType,
        EngineShutdownLifecycle,
        EngineShutdownState,
    )

    now = [100.0]
    lifecycle = EngineShutdownLifecycle(
        timeout=10,
        monotonic_clock=lambda: now[0],
        wall_clock_ns=lambda: int(now[0] * 1_000_000_000),
    )
    assert lifecycle.snapshot().state == EngineShutdownState.RUNNING
    assert lifecycle.activity() == EngineShutdownActivity.PROCESS_ENGINE_WORK

    requested = EngineShutdownEvent(
        event_type=EngineShutdownEventType.SHUTDOWN_REQUESTED
    )
    assert lifecycle.dispatch(requested) == EngineShutdownDisposition.APPLIED
    assert lifecycle.snapshot().state == EngineShutdownState.DRAINING
    assert lifecycle.activity() == EngineShutdownActivity.DRAIN_ENGINE_WORK
    assert lifecycle.snapshot().deadline == 110.0
    assert lifecycle.dispatch(requested) == EngineShutdownDisposition.IGNORED
    assert lifecycle.snapshot().deadline == 110.0
    assert (
        lifecycle.dispatch(
            EngineShutdownEvent(event_type=EngineShutdownEventType.ABORT_SUCCEEDED)
        )
        == EngineShutdownDisposition.REJECTED
    )

    assert (
        lifecycle.dispatch(
            EngineShutdownEvent(event_type=EngineShutdownEventType.DEADLINE_EXPIRED)
        )
        == EngineShutdownDisposition.APPLIED
    )
    assert (
        lifecycle.dispatch(
            EngineShutdownEvent(
                event_type=EngineShutdownEventType.ABORT_SUCCEEDED,
                aborted_requests=3,
            )
        )
        == EngineShutdownDisposition.APPLIED
    )
    assert lifecycle.snapshot().state == EngineShutdownState.TEARING_DOWN
    assert lifecycle.activity() == EngineShutdownActivity.TEARDOWN_RESOURCES
    assert (
        lifecycle.dispatch(
            EngineShutdownEvent(event_type=EngineShutdownEventType.TEARDOWN_SUCCEEDED)
        )
        == EngineShutdownDisposition.APPLIED
    )
    assert (
        lifecycle.dispatch(
            EngineShutdownEvent(event_type=EngineShutdownEventType.REPORT_SUCCEEDED)
        )
        == EngineShutdownDisposition.APPLIED
    )
    assert lifecycle.snapshot().state == EngineShutdownState.FINAL
    assert lifecycle.dispatch(requested) == EngineShutdownDisposition.TERMINAL

    failed_abort = EngineShutdownLifecycle(timeout=0)
    assert failed_abort.dispatch(requested) == EngineShutdownDisposition.APPLIED
    assert (
        failed_abort.dispatch(
            EngineShutdownEvent(event_type=EngineShutdownEventType.ABORT_FAILED)
        )
        == EngineShutdownDisposition.APPLIED
    )
    assert failed_abort.snapshot().state == EngineShutdownState.TEARING_DOWN
    assert failed_abort.snapshot().outcome == "abort_failed"


def test_shutdown_request_prevents_another_engine_step():
    from types import SimpleNamespace

    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineCoreProc, EngineShutdownLifecycle

    class Scheduler:
        def get_num_unfinished_requests(self):
            return 0

        def finish_requests(self, *, request_ids, finished_status):
            return []

    class ShutdownDriver(EngineCoreProc):
        def __init__(self):
            self.vllm_config = SimpleNamespace(shutdown_timeout=0)
            self.shutdown_lifecycle = EngineShutdownLifecycle(timeout=0)
            self.scheduler = Scheduler()
            self.steps = 0
            self.enable_fault_tolerance = False

        def _process_input_queue(self):
            self._handle_client_request(
                request_type=EngineCoreRequestType.SHUTDOWN,
                request=None,
            )

        def _process_engine_step(self):
            self.steps += 1

        def _maybe_publish_request_counts(self):
            pass

    driver = ShutdownDriver()
    with pytest.raises(SystemExit):
        driver.run_busy_loop()
    assert driver.steps == 0


def test_engine_core_records_shutdown_after_resource_teardown(monkeypatch):
    from types import SimpleNamespace

    import vllm.v1.engine.core as core_module
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import (
        EngineCoreProc,
        EngineShutdownLifecycle,
        EngineShutdownState,
    )

    class Scheduler:
        def get_num_unfinished_requests(self):
            return 0

        def finish_requests(self, *, request_ids, finished_status):
            return []

    class ShutdownDriver(EngineCoreProc):
        def __init__(self):
            self.vllm_config = SimpleNamespace(shutdown_timeout=0)
            self.shutdown_lifecycle = EngineShutdownLifecycle(
                timeout=0,
                wall_clock_ns=lambda: 100_000_000_000,
            )
            self.scheduler = Scheduler()

        def shutdown(self):
            events.append("teardown")

    events = []
    monkeypatch.setattr(
        core_module,
        "instrument_manual",
        lambda **kwargs: events.append(("telemetry", kwargs)),
    )
    driver = ShutdownDriver()
    driver._handle_client_request(
        request_type=EngineCoreRequestType.SHUTDOWN,
        request=None,
    )
    assert not driver._handle_shutdown()
    driver.finish_shutdown()

    assert events[0] == "teardown"
    assert events[1][0] == "telemetry"
    assert driver.shutdown_lifecycle.snapshot().state == EngineShutdownState.FINAL


def test_engine_core_telemetry_does_not_mask_teardown_failure(monkeypatch):
    from types import SimpleNamespace

    import vllm.v1.engine.core as core_module
    from vllm.v1.engine.core import (
        EngineCoreProc,
        EngineShutdownEvent,
        EngineShutdownEventType,
        EngineShutdownLifecycle,
        EngineShutdownState,
    )

    class ShutdownDriver(EngineCoreProc):
        def __init__(self):
            self.vllm_config = SimpleNamespace(shutdown_timeout=0)
            self.shutdown_lifecycle = EngineShutdownLifecycle(timeout=0)

        def shutdown(self):
            raise RuntimeError("teardown failed")

    def telemetry_failure(**kwargs):
        del kwargs
        raise ValueError("telemetry failed")

    monkeypatch.setattr(core_module, "instrument_manual", telemetry_failure)
    driver = ShutdownDriver()
    driver.shutdown_lifecycle.dispatch(
        EngineShutdownEvent(event_type=EngineShutdownEventType.ENGINE_FAILED)
    )

    with pytest.raises(RuntimeError, match="teardown failed"):
        driver.finish_shutdown()
    assert driver.shutdown_lifecycle.snapshot().state == EngineShutdownState.FINAL


def test_normal_loop_exit_does_not_suppress_teardown_failure():
    from vllm.v1.engine.core import _finish_engine_core_process

    class EngineCore:
        def finish_shutdown(self):
            raise RuntimeError("teardown failed")

    with pytest.raises(RuntimeError, match="teardown failed"):
        _finish_engine_core_process(
            engine_core=EngineCore(),
            active_error=SystemExit(),
        )


@pytest.mark.parametrize(
    ("timeout", "expected_join_timeout"),
    [(None, 5.0), (-1.0, 5.0), (0.0, 5.0), (10.0, 15.0)],
)
def test_process_shutdown_adds_cleanup_grace_after_the_drain_budget(
    monkeypatch, timeout, expected_join_timeout
):
    import vllm.v1.utils as utils

    class Process:
        name = "engine"
        pid = 123

        def __init__(self):
            self.join_timeout = None

        def is_alive(self):
            return True

        def terminate(self):
            pass

        def join(self, timeout):
            self.join_timeout = timeout

    process = Process()
    spans = []
    monkeypatch.setattr(utils.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(utils, "kill_process_tree", lambda _pid: None)
    monkeypatch.setattr(
        utils,
        "instrument_manual",
        lambda **kwargs: spans.append(kwargs),
    )

    utils.shutdown(procs=[process], timeout=timeout)

    assert process.join_timeout == expected_join_timeout
    assert spans[0]["attributes"] == {
        "vllm.shutdown.force_killed_processes": 1,
        "vllm.shutdown.outcome": "force_killed",
        "vllm.shutdown.process_count": 1,
    }


def test_shutdown_telemetry_does_not_mask_process_failure(monkeypatch):
    import vllm.v1.utils as utils

    class Process:
        name = "engine"
        pid = 123

        def is_alive(self):
            raise RuntimeError("process inspection failed")

    def telemetry_failure(**kwargs):
        del kwargs
        raise ValueError("telemetry failed")

    monkeypatch.setattr(utils, "instrument_manual", telemetry_failure)

    with pytest.raises(RuntimeError, match="process inspection failed"):
        utils.shutdown(procs=[Process()], timeout=0)
