# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the Model Runner V2 thinking-budget actuator.

Drives the pure torch reference (``apply_thinking_budget_torch`` and the
``ThinkingBudgetState`` slot logic) on CPU tensors through a scripted target
model plus a greedy rejection sampler, and cross-checks the committed token
streams against the V1 ``ThinkingBudgetStateHolder`` on the same drafts.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample import thinking_budget_state as v1_module
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder
from vllm.v1.worker.gpu.sample import thinking_budget as v2_module
from vllm.v1.worker.gpu.sample.thinking_budget import (
    ThinkingBudgetState,
    apply_thinking_budget_torch,
    find_last_marker,
    forced_end_tokens_torch,
)

VOCAB = 64
START = [1]  # <think>
END = [2]  # </think>
FORCE = 1.0e9
MAX_LEN = 256


def _plain_h2d(data, dtype, device):
    return torch.tensor(data, dtype=dtype, device=device)


@pytest.fixture(autouse=True)
def _no_pinned_memory(monkeypatch):
    # Both holders build small index tensors with async_tensor_h2d, which pins
    # host memory (a CUDA context) on this box; use plain CPU tensors instead.
    monkeypatch.setattr(v1_module, "async_tensor_h2d", _plain_h2d)
    monkeypatch.setattr(v2_module, "async_tensor_h2d", _plain_h2d)


def _script(n: int) -> int:
    """Target model: deterministic thinking token at output position n."""
    return 10 + (n % 40)


class V2Sim:
    """CPU mirror of the V2 per-slot tensors, driven through the torch path."""

    def __init__(
        self,
        max_num_reqs=4,
        start=START,
        end=END,
        natural_end=None,
    ):
        self.max_num_reqs = max_num_reqs
        self.all_token_ids = torch.zeros((max_num_reqs, MAX_LEN), dtype=torch.int32)
        self.total_len = torch.zeros(max_num_reqs, dtype=torch.int32)
        self.budget = torch.full((max_num_reqs,), -1, dtype=torch.int32)
        self.cached_last_start = torch.full((max_num_reqs,), -1, dtype=torch.int32)
        self.cached_last_end = torch.full((max_num_reqs,), -1, dtype=torch.int32)
        self.cached_scan_pos = torch.zeros(max_num_reqs, dtype=torch.int32)
        self.start_ids = torch.tensor(start, dtype=torch.int32)
        self.end_ids = torch.tensor(end, dtype=torch.int32)
        self.natural_end_ids = torch.tensor(
            natural_end if natural_end is not None else end, dtype=torch.int32
        )
        self.prompt_len = [0] * max_num_reqs

    def add(self, req_idx: int, prompt: list[int], budget: int | None) -> None:
        self.all_token_ids[req_idx, : len(prompt)] = torch.tensor(
            prompt, dtype=torch.int32
        )
        self.total_len[req_idx] = len(prompt)
        self.prompt_len[req_idx] = len(prompt)
        self.budget[req_idx] = -1 if budget is None else budget
        self.cached_last_start[req_idx] = -1
        self.cached_last_end[req_idx] = -1
        self.cached_scan_pos[req_idx] = 0

    def output(self, req_idx: int) -> list[int]:
        return self.all_token_ids[
            req_idx, self.prompt_len[req_idx] : int(self.total_len[req_idx])
        ].tolist()

    def commit(self, req_idx: int, tokens: list[int]) -> None:
        n = int(self.total_len[req_idx])
        self.all_token_ids[req_idx, n : n + len(tokens)] = torch.tensor(
            tokens, dtype=torch.int32
        )
        self.total_len[req_idx] = n + len(tokens)

    def forced(self, batch: list[tuple[int, list[int]]]) -> dict[int, dict[int, int]]:
        """Run one step; batch = [(req_idx, drafts)]; returns req -> {row: tok}."""
        req_ids = torch.tensor([r for r, _ in batch], dtype=torch.int64)
        eim, elp, input_ids = [], [], []
        for req_idx, drafts in batch:
            last = int(self.all_token_ids[req_idx, int(self.total_len[req_idx]) - 1])
            for p, tok in enumerate([last] + list(drafts)):
                eim.append(req_idx)
                elp.append(p)
                input_ids.append(tok)
        logits = torch.zeros((len(eim), VOCAB), dtype=torch.float32)
        apply_thinking_budget_torch(
            logits,
            req_ids,
            torch.tensor(eim, dtype=torch.int64),
            self.budget,
            self.all_token_ids,
            self.total_len,
            torch.tensor(input_ids, dtype=torch.int64),
            torch.tensor(elp, dtype=torch.int32),
            self.cached_last_start,
            self.cached_last_end,
            self.cached_scan_pos,
            self.start_ids,
            self.natural_end_ids,
            self.end_ids,
        )
        out: dict[int, dict[int, int]] = {r: {} for r, _ in batch}
        rows, toks = (logits == FORCE).nonzero(as_tuple=True)
        for row, tok in zip(rows.tolist(), toks.tolist()):
            out[eim[row]][elp[row]] = tok
        assert int((logits == FORCE).sum()) == len(rows)
        return out


class V1Sim:
    """The V1 holder driven exactly like the V1 sampler / rejection sampler."""

    def __init__(
        self,
        num_spec_tokens: int,
        max_num_reqs=4,
        start=START,
        end=END,
        natural_end=None,
    ):
        natural = list(natural_end if natural_end is not None else end)
        cfg = SimpleNamespace(
            reasoning_start_token_ids=start,
            reasoning_end_token_ids=end,
            natural_reasoning_end_token_ids=natural,
        )
        self.holder = ThinkingBudgetStateHolder(
            cfg, max_num_reqs, num_spec_tokens, torch.device("cpu"), False
        )
        self.outputs: dict[int, list[int]] = {}
        self.max_num_reqs = max_num_reqs

    def add(self, req_idx: int, prompt: list[int], budget: int | None) -> None:
        self.outputs[req_idx] = []
        params = SamplingParams(thinking_token_budget=budget)
        self.holder.sync_batch(
            BatchUpdate(
                batch_size=self.max_num_reqs,
                removed=[],
                added=[(req_idx, params, list(prompt), self.outputs[req_idx])],
                moved=[],
            )
        )

    def output(self, req_idx: int) -> list[int]:
        return self.outputs[req_idx]

    def commit(self, req_idx: int, tokens: list[int]) -> None:
        self.outputs[req_idx].extend(tokens)

    def forced(self, batch: list[tuple[int, list[int]]]) -> dict[int, dict[int, int]]:
        n = max(r for r, _ in batch) + 1
        outputs = [self.outputs.get(i, []) for i in range(n)]
        specs = [[] for _ in range(n)]
        for r, drafts in batch:
            specs[r] = list(drafts)
        out: dict[int, dict[int, int]] = {r: {} for r, _ in batch}
        if self.holder.in_spec_mode and any(specs):
            # Rejection-sampler step: bonus rows first, then the target rows.
            self.holder.update_state(outputs, specs)
            bonus = torch.zeros((n, VOCAB))
            self.holder.apply_to_logits(bonus, True, specs)
            target = torch.zeros((sum(len(s) for s in specs), VOCAB))
            self.holder.apply_to_logits(target, False, specs)
            for r, drafts in batch:
                hit = (bonus[r] == FORCE).nonzero()
                if len(hit):
                    out[r][len(drafts)] = int(hit[0])
            cu = 0
            for i in range(n):
                for p in range(len(specs[i])):
                    hit = (target[cu + p] == FORCE).nonzero()
                    if len(hit):
                        out[i][p] = int(hit[0])
                cu += len(specs[i])
        else:
            # Plain sampler step (no drafts this step, e.g. right after prefill).
            self.holder.update_state(outputs, specs)
            logits = torch.zeros((n, VOCAB))
            self.holder.apply_to_logits(logits, False, specs)
            for r, _ in batch:
                hit = (logits[r] == FORCE).nonzero()
                if len(hit):
                    out[r][0] = int(hit[0])
        return out


def _verify(n_out: int, drafts: list[int], forced: dict[int, int]) -> list[int]:
    """Greedy rejection sampling against the scripted target (+ forcing)."""
    committed = []
    for j, d in enumerate(drafts):
        target = forced.get(j, _script(n_out + j))
        if d != target:
            committed.append(target)
            return committed
        committed.append(d)
    committed.append(forced.get(len(drafts), _script(n_out + len(drafts))))
    return committed


def _think_len(output: list[int], end=END) -> int:
    """Number of thinking tokens between the leading <think> and the end seq."""
    assert output[: len(START)] == START
    body = output[len(START) :]
    for i in range(len(body) - len(end) + 1):
        if body[i : i + len(end)] == end:
            return i
    raise AssertionError("no end sequence in output")


def _run(sim, req_idx, budget, drafts_fn, steps):
    """Drive one request: <think> sampled at prefill, then scripted decode."""
    sim.add(req_idx, [], budget)
    sim.commit(req_idx, START)
    forced_log = []
    for _ in range(steps):
        n_out = len(sim.output(req_idx))
        drafts = drafts_fn(n_out)
        forced = sim.forced([(req_idx, drafts)])[req_idx]
        forced_log.append((n_out, dict(forced)))
        sim.commit(req_idx, _verify(n_out - 1, drafts, forced))
    return forced_log


class _Paired:
    """V1 + V2 running in lock-step on identical drafts."""

    def __init__(self, k: int, start=START, end=END, natural_end=None):
        self.k = k
        self.v1 = V1Sim(k, start=start, end=end, natural_end=natural_end)
        self.v2 = V2Sim(start=start, end=end, natural_end=natural_end)

    def add(self, req_idx, prompt, budget):
        self.v1.add(req_idx, prompt, budget)
        self.v2.add(req_idx, prompt, budget)

    def output(self, req_idx):
        return self.v2.output(req_idx)

    def commit(self, req_idx, tokens):
        self.v1.commit(req_idx, tokens)
        self.v2.commit(req_idx, tokens)

    def forced(self, batch):
        f1 = self.v1.forced(batch)
        f2 = self.v2.forced(batch)
        for r, _ in batch:
            if f1[r]:
                first = min(f1[r])
                assert first == min(f2[r]), (f1, f2)
                assert f1[r][first] == f2[r][first], (f1, f2)
            else:
                assert not f2[r], (f1, f2)
        return f2


def _good_drafts(k):
    def fn(n_out):
        return [_script(n_out - 1 + j) for j in range(k)]

    return fn


def _bad_draft_at(k, bad_pos):
    """Drafts correct except position ``bad_pos`` (0-based) is wrong."""

    def fn(n_out):
        d = [_script(n_out - 1 + j) for j in range(k)]
        d[bad_pos] = 63
        return d

    return fn


@pytest.mark.parametrize("budget", [1, 5, 8, 16, 21])
def test_step_boundary_no_spec(budget):
    sim = _Paired(0)
    _run(sim, 0, budget, lambda n: [], steps=budget + 3)
    assert _think_len(sim.output(0)) == budget


@pytest.mark.parametrize("budget", [1, 3, 7, 8, 9, 15, 20])
def test_budget_inside_k7_window(budget):
    k = 7
    sim = _Paired(k)
    log = _run(sim, 0, budget, _good_drafts(k), steps=6)
    assert _think_len(sim.output(0)) == budget
    # The step that hits the budget forces at column budget - think_count and
    # every later column in the window (single-token end: the same token).
    hit = next((n, f) for n, f in log if f)
    think_count = hit[0] - 1
    col = budget - think_count
    assert min(hit[1]) == col
    assert set(hit[1]) == set(range(col, k + 1))
    assert set(hit[1].values()) == {END[0]}


def test_rollback_when_rejected_draft_precedes_force():
    k = 7
    budget = 10
    sim = _Paired(k)
    # Step 1: 7 good drafts, no force (think_count 0 + 8 <= 10 -> 8 committed).
    # Step 2: think_count 8, force at column 2, but draft 0 is wrong -> the
    # rejection sampler commits only the recovery token; forcing must recur.
    drafts_by_step = [_good_drafts(k), _bad_draft_at(k, 0), _good_drafts(k)]
    sim.add(0, [], budget)
    sim.commit(0, START)
    forced_cols = []
    for step in range(4):
        n_out = len(sim.output(0))
        drafts = drafts_by_step[min(step, 2)](n_out)
        forced = sim.forced([(0, drafts)])[0]
        forced_cols.append(min(forced) if forced else None)
        sim.commit(0, _verify(n_out - 1, drafts, forced))
    assert forced_cols[0] is None
    assert forced_cols[1] == 2  # forced at column 2, lost to the rejection
    assert forced_cols[2] == 1  # only one draft landed; still 1 short of 10
    assert _think_len(sim.output(0)) == budget


def test_multi_token_end_sequence_forced_in_order():
    k = 3
    end = [40, 41, 2]
    budget = 4
    sim = _Paired(k, end=end)
    sim.add(0, [], budget)
    sim.commit(0, START)
    seen = []
    for _ in range(6):
        n_out = len(sim.output(0))
        drafts = _good_drafts(k)(n_out)
        forced = sim.forced([(0, drafts)])[0]
        seen.append(forced)
        sim.commit(0, _verify(n_out - 1, drafts, forced))
    out = sim.output(0)
    assert _think_len(out, end) == budget
    # Scripted drafts never propose the end sequence, so each end token is
    # forced on the first row and lands one per step (the drafts are rejected).
    forced_first_rows = [f[0] for f in seen if f]
    assert forced_first_rows[:3] == end
    assert out[len(START) + budget : len(START) + budget + 3] == end


def test_multi_token_end_drafts_that_follow_the_sequence_are_kept():
    end = [40, 41, 2]
    sim = V2Sim(end=end)
    sim.add(0, [], 2)
    sim.commit(0, START + [_script(0), _script(1)])
    # Over budget; the drafter proposes the whole end sequence.
    forced = sim.forced([(0, [40, 41, 2])])[0]
    assert forced == {0: 40, 1: 41, 2: 2}
    committed = _verify(2, [40, 41, 2], forced)
    assert committed[:3] == end
    sim.commit(0, committed[:3])
    # Natural end committed: nothing more is forced.
    assert sim.forced([(0, [5, 6, 7])])[0] == {}


def test_prompt_already_mid_think_counts_prompt_tokens():
    k = 4
    budget = 6
    prompt = [7, 8, START[0], 30, 31, 32]  # 3 thinking tokens in the prompt
    sim = _Paired(k)
    sim.add(0, prompt, budget)
    for step in range(4):
        n_out = len(sim.output(0))
        drafts = [] if step == 0 else _good_drafts(k)(n_out + 1)
        forced = sim.forced([(0, drafts)])[0]
        sim.commit(0, _verify(n_out, drafts, forced))
    out = sim.output(0)
    assert out.index(END[0]) == budget - 3


def test_prompt_exhausted_budget_forces_first_token():
    prompt = [START[0], 30, 31, 32]
    sim = _Paired(0)
    sim.add(0, prompt, 2)
    assert sim.forced([(0, [])])[0] == {0: END[0]}
    sim = _Paired(0)
    sim.add(0, prompt, 3)
    assert sim.forced([(0, [])])[0] == {0: END[0]}


def test_no_budget_and_no_think_block_untouched():
    sim = V2Sim()
    sim.add(0, [5, 6], None)
    sim.add(1, [5, 6], 3)  # budget but never enters a think block
    sim.commit(0, START + [_script(0)] * 8)
    sim.commit(1, [_script(0)] * 8)
    assert sim.forced([(0, [1, 2, 3]), (1, [1, 2, 3])]) == {0: {}, 1: {}}


def test_natural_end_stops_forcing_and_new_think_restarts_count():
    sim = V2Sim()
    sim.add(0, [], 3)
    sim.commit(0, START + [_script(0), _script(1), END[0], 50, 51])
    assert sim.forced([(0, [52, 53])])[0] == {}
    sim.commit(0, START + [_script(0)])
    forced = sim.forced([(0, [_script(1), _script(2), _script(3)])])[0]
    assert forced == {2: END[0], 3: END[0]}


class _FakeUva:
    def __init__(self, size, dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = self.cpu.clone()

    def copy_to_uva(self):
        self.gpu = self.cpu.clone()
        return self.gpu


class _FakeStaged:
    def __init__(self, t):
        self.gpu = t


def _cpu_state(max_num_reqs=4, start=START, end=END, natural_end=None):
    req_states = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        device=torch.device("cpu"),
        all_token_ids=_FakeStaged(
            torch.zeros((max_num_reqs, MAX_LEN), dtype=torch.int32)
        ),
        total_len=_FakeStaged(torch.zeros(max_num_reqs, dtype=torch.int32)),
    )
    cfg = SimpleNamespace(
        reasoning_start_token_ids=start,
        reasoning_end_token_ids=end,
        natural_reasoning_end_token_ids=natural_end or end,
    )
    return ThinkingBudgetState(req_states, cfg)


@pytest.fixture(autouse=True)
def _cpu_uva(monkeypatch):
    # UvaBackedTensor needs pinned host memory; mirror it with CPU tensors.
    monkeypatch.setattr(v2_module, "UvaBackedTensor", _FakeUva)


def test_state_disabled_without_reasoning_config():
    req_states = SimpleNamespace(max_num_reqs=2, device=torch.device("cpu"))
    state = ThinkingBudgetState(req_states, None)
    assert not state.enabled
    state.add_request(0, SamplingParams(thinking_token_budget=3))
    assert not state.requires_logits_processing(np.array([0]))


def test_state_slot_reuse_scrubs_budget_cache():
    state = _cpu_state()
    st = state.req_states
    st.all_token_ids.gpu[1, :4] = torch.tensor(START + [_script(0)] * 3)
    st.total_len.gpu[1] = 4
    state.add_request(1, SamplingParams(thinking_token_budget=2))
    state.apply_staged_writes()
    logits = torch.zeros((1, VOCAB))
    state.apply(
        logits,
        torch.tensor([1]),
        torch.tensor([1]),
        np.array([1]),
        torch.tensor([_script(2)]),
        torch.tensor([0], dtype=torch.int32),
    )
    assert logits[0, END[0]] == FORCE
    assert int(state.cached_last_start[1]) == 0
    # Slot 1 is reused by a request without a budget: nothing applies and the
    # cache / budget are scrubbed.
    st.all_token_ids.gpu[1, :2] = torch.tensor([START[0], 9])
    st.total_len.gpu[1] = 2
    state.add_request(1, SamplingParams())
    state.apply_staged_writes()
    assert not state.use_thinking_budget[1]
    assert int(state.thinking_token_budget.gpu[1]) == -1
    assert not state.requires_logits_processing(np.array([1]))
    # And by a budgeted request whose prompt is not in a think block: the
    # cache from the earlier occupant must not leak.
    st.all_token_ids.gpu[1, :2] = torch.tensor([7, 9])
    st.total_len.gpu[1] = 2
    state.add_request(1, SamplingParams(thinking_token_budget=1))
    state.apply_staged_writes()
    assert int(state.cached_last_start[1]) == -1
    logits = torch.zeros((1, VOCAB))
    state.apply(
        logits,
        torch.tensor([1]),
        torch.tensor([1]),
        np.array([1]),
        torch.tensor([9]),
        torch.tensor([0], dtype=torch.int32),
    )
    assert int((logits == FORCE).sum()) == 0


def test_requires_logits_processing_only_for_budgeted_rows():
    state = _cpu_state()
    state.add_request(0, SamplingParams())
    state.add_request(2, SamplingParams(thinking_token_budget=4))
    assert not state.requires_logits_processing(np.array([0]))
    assert state.requires_logits_processing(np.array([0, 2]))


def test_find_last_marker_and_multi_token_prefix_rule():
    toks = torch.tensor([3, 40, 41, 3, 40, 41, 2, 40], dtype=torch.int64)
    assert find_last_marker(toks, torch.tensor([40, 41]), 0) == 4
    assert find_last_marker(toks, torch.tensor([40, 41]), 5) == -1
    assert find_last_marker(toks, torch.tensor([40, 41, 2]), 0) == 4
    assert find_last_marker(toks[:1], torch.tensor([40, 41]), 0) == -1
    # Over budget with the tail already holding end[:2]: force end[2].
    rows, forced = forced_end_tokens_torch(
        torch.tensor([0]),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[1, 5, 40, 41]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([41]),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([-1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([40, 41, 2], dtype=torch.int32),
    )
    assert (rows, forced) == ([0], [2])


def test_frozen_interfaces_have_defaults():
    mro = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    assert mro.effort_signals is None
