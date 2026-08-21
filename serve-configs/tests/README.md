# Tests for the venv-local vLLM patches and the keepalive middleware

Run with the deployment venv (never system python):

    cd serve-configs/tests && /shared/vllm/.venv-qwen38/bin/python -m pytest -q .

(Run from the tests directory: from the repo root the source tree `vllm/`
would shadow the installed wheel. `uv pip install pytest` into the venv once.)

They exercise the pure/CPU parts of `serve-configs/patches/0005-*.patch` and
`0006-*.patch` (config validation, adaptive draft-length policy, cudagraph
query-length capture policy, micro-batch metadata slicing) and the
`serve-configs/middleware/vllm_keepalive.py` ASGI behavior. GPU behavior is
covered by the goal-run evidence linked from `serve-configs/patches/README.md`.

Patch 0009 (dynamic reasoning effort v3 + telemetry) is covered by
`test_effort_telemetry.py`, `test_effort_levels.py`, `test_effort_memory.py`,
`test_effort_two_phase.py`, `test_effort_hidden_pooling.py` and
`test_v2_thinking_budget.py`. To run them against
a lane worktree instead of the installed package use `work/run-tests.sh <worktree> .`.
