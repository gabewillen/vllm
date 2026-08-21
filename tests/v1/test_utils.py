# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


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
