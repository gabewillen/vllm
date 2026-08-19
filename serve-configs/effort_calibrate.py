#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the dynamic-effort quantile sketch file for a model.

The P6 controller is self-normalizing: it ranks each request's signals against
running per-model quantile sketches instead of a hand-fitted mean/sd table.
Cold sketches never escalate, so a fresh deployment behaves like a fixed rung-0
cap until it has seen `quantile_min_samples` observations. This script warms
them up front from a prompt set, so the first request after a restart already
decides on a full distribution.

Two modes, usually run back to back:

    # 1. drive a running server with the prompt set (server must have
    #    VLLM_EFFORT_TELEMETRY set to the sink path)
    python serve-configs/effort_calibrate.py run \\
        --base-url http://localhost:8012/v1 --model Qwen3.8-27B \\
        --prompts serve-configs/effort_calibration_prompts.txt

    # 2. turn the sink into the sketch file named by
    #    --reasoning-config '{"dynamic_effort":{"quantile_path": ...}}'
    python serve-configs/effort_calibrate.py build \\
        --telemetry /data/effort-telemetry/latency.jsonl \\
        --out /data/effort-sketches/qwen38.json --model Qwen3.8-27B

`build` reads only in-think rows (`in_think: true`) and weights each row by its
committed row count, exactly as the scheduler does at run time, so a warmed
file and a self-warmed server converge to the same distribution.

`build` also measures whether the entropy/margin features carry any signal on
*this* model and stores the answer in the same file (`auc` block). The default
escalation rule (`dynamic_effort.rule = "length"`) consults those features only
when that AUC clears `uncertainty_min_auc`; a file without an `auc` block means
"no evidence", so they stay off. See docs/dynamic-reasoning-v3-analysis.md §4
for the method - this is the same label and the same rank statistic, run
against the current sink instead of by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vllm.v1.core.sched.effort_quantiles import SignalSketches  # noqa: E402

DEFAULT_PROMPTS = [
    "What is 17 * 23? Answer with the number only.",
    "Write a Python function that returns the median of a list of numbers.",
    (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
        "the ball. How much does the ball cost?"
    ),
    "Explain in two sentences why quicksort is O(n log n) on average.",
    "Find the smallest positive integer divisible by 7 whose digits sum to 20.",
    "Refactor this to be O(1) memory: `def f(xs): return sum(sorted(xs))`.",
    (
        "Five houses in a row, each a different colour, with owners of "
        "different nationalities. Who owns the fish? Reason briefly."
    ),
    "Summarise the trade-offs between a B-tree and an LSM tree in a paragraph.",
    "Given a stream of integers, describe an algorithm for the running median.",
    "Prove that the square root of 2 is irrational.",
    "Debug: a top-k heap returns the k largest but in the wrong order. Why?",
    "Write a regex matching ISO-8601 timestamps with optional nanoseconds.",
]


def iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
    """Yield the parsed records of a JSONL file, skipping unparsable lines."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_sketches(
    telemetry_paths: list[str],
    out_path: str,
    model: str | None,
    min_samples: int,
    compression: float,
    include_all_rows: bool = False,
    ladder: list[int] | None = None,
    with_auc: bool = True,
) -> dict[str, float]:
    """Fold a telemetry sink into a sketch file.

    Args:
        telemetry_paths: JSONL sinks written by `VLLM_EFFORT_TELEMETRY`.
        out_path: sketch file to write (`dynamic_effort.quantile_path`).
        model: model name recorded in the file for provenance.
        min_samples: `quantile_min_samples` of the target server.
        compression: t-digest compression.
        include_all_rows: also fold rows outside the think block (debug only).
        ladder: the rung caps the traffic ran with (for the AUC labels).
        with_auc: also measure and store the uncertainty-feature AUC.

    Returns:
        Per-signal observation counts.
    """
    sketches = SignalSketches(
        min_samples=min_samples, compression=compression, path=out_path
    )
    sketches.model = model
    # The scheduler observes a per-request acceptance EMA, not the raw per-step
    # ratio; mirror that so a warmed file and a self-warmed server agree.
    acc_ema: dict[str, float] = {}
    for path in telemetry_paths:
        for rec in iter_jsonl(path):
            n_rows = rec.get("n_rows") or 0
            if n_rows <= 0:
                continue
            if not include_all_rows and not rec.get("in_think"):
                continue
            weight = float(n_rows)
            sketches.observe("entropy", float(rec.get("entropy", 0.0)), weight)
            sketches.observe("margin", float(rec.get("margin", 0.0)), weight)
            if rec.get("p_end") is not None:
                sketches.observe("p_end", float(rec["p_end"]), weight)
            drafted = rec.get("num_draft_tokens")
            if drafted:
                req_id = str(rec.get("req_id", ""))
                ratio = float(rec.get("num_accepted") or 0) / float(drafted)
                prev = acc_ema.get(req_id)
                ema = ratio if prev is None else 0.7 * prev + 0.3 * ratio
                acc_ema[req_id] = ema
                sketches.observe("acceptance", ema)
    if with_auc:
        sketches.auc = compute_uncertainty_auc(telemetry_paths, ladder)
    sketches.save()
    return {key: sketches.count(key) for key in sketches.digests}


DEFAULT_LADDER = [1024, 4096, 16384]
AUC_WINDOW = 128
AUC_MIN_GROUP = 20
AUC_CAP_SLACK = 8


def _rank_auc(positive: list[float], negative: list[float]) -> float | None:
    """`P(positive > negative)` with ties at 0.5 (Mann-Whitney).

    Args:
        positive: feature values of the "needed more thinking" group.
        negative: feature values of the "closed at or before the cap" group.

    Returns:
        The AUC, or `None` when either group is empty.
    """
    if not positive or not negative:
        return None
    merged = sorted(
        (v, i) for i, values in enumerate((positive, negative)) for v in values
    )
    ranks = [0.0] * len(merged)
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = average
        i = j + 1
    rank_sum = sum(r for r, (_, group) in zip(ranks, merged) if group == 0)
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _window_means(
    steps: list[tuple[float, float, float]], window: int
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Token-weighted (entropy, margin) means of the first and last `window`.

    Args:
        steps: per-step `(entropy, margin, n_rows)` of one request's in-think
            steps, in commit order.
        window: think tokens each end of the request contributes.

    Returns:
        `(first, last)` mean dicts, or `None` when the request is too short for
        the two windows to be disjoint.
    """
    total = sum(int(n) for _, _, n in steps)
    if total < 2 * window:
        return None

    def _mean(ordered: list[tuple[float, float, float]]) -> dict[str, float]:
        left = window
        entropy = margin = weight = 0.0
        for e, m, n in ordered:
            take = min(int(n), left)
            if take <= 0:
                break
            entropy += e * take
            margin += m * take
            weight += take
            left -= take
        return {"entropy": entropy / weight, "margin": margin / weight}

    return _mean(steps), _mean(steps[::-1])


def compute_uncertainty_auc(
    telemetry_paths: list[str],
    ladder: list[int] | None = None,
    window: int = AUC_WINDOW,
    min_group: int = AUC_MIN_GROUP,
    cap_slack: int = AUC_CAP_SLACK,
) -> dict[str, Any]:
    """Discriminative power of the entropy/margin features on this model.

    Label (docs/dynamic-reasoning-v3-analysis.md §4): a request is *positive*
    when it needed the higher rung - it passed the rung-0 cap and then closed
    naturally rather than landing on a higher cap - and *negative* when it
    closed at or before the rung-0 cap. Length is controlled by requiring both
    groups to be at least `2 * window` think tokens long, so a request's first
    and last windows are disjoint. Every feature is scored in the direction the
    escalation rule assumes (high entropy / low margin = "still working"), so
    an AUC below 0.5 means the rule's premise is backwards on this model.

    Args:
        telemetry_paths: JSONL sinks written by `VLLM_EFFORT_TELEMETRY`.
        ladder: the rung caps the traffic ran with.
        window: think tokens per end window.
        min_group: smallest usable group; below it the result is inconclusive.
        cap_slack: tokens below a higher cap that still count as landing on it.

    Returns:
        The `auc` block stored in the sketch file.
    """
    ladder = list(ladder or DEFAULT_LADDER)
    cap = ladder[0]
    steps: dict[str, list[tuple[float, float, float]]] = {}
    for path in telemetry_paths:
        for rec in iter_jsonl(path):
            n_rows = rec.get("n_rows") or 0
            if n_rows <= 0 or not rec.get("in_think"):
                continue
            steps.setdefault(str(rec.get("req_id", "")), []).append(
                (
                    float(rec.get("entropy", 0.0)),
                    float(rec.get("margin", 0.0)),
                    float(n_rows),
                )
            )

    features = ("entropy_first", "entropy_last", "entropy_rise")
    features += ("margin_first", "margin_last", "margin_drop")
    groups: dict[str, dict[str, list[float]]] = {
        key: {"positive": [], "negative": []} for key in features
    }
    for request in steps.values():
        think = int(sum(n for _, _, n in request))
        if think > cap:
            landed = any(0 <= higher - think <= cap_slack for higher in ladder[1:])
            label = None if landed else "positive"
        else:
            label = "negative"
        if label is None:
            continue
        windows = _window_means(request, window)
        if windows is None:
            continue
        first, last = windows
        values = {
            "entropy_first": first["entropy"],
            "entropy_last": last["entropy"],
            "entropy_rise": last["entropy"] - first["entropy"],
            "margin_first": first["margin"],
            "margin_last": last["margin"],
            "margin_drop": last["margin"] - first["margin"],
        }
        for key, value in values.items():
            groups[key][label].append(value)

    n_pos = len(groups["entropy_last"]["positive"])
    n_neg = len(groups["entropy_last"]["negative"])
    scored: dict[str, float] = {}
    for key, sides in groups.items():
        # Margin features run the other way: the rule reads a *low* margin as
        # uncertainty, so P(positive < negative) is the directional AUC.
        if key.startswith("margin"):
            auc = _rank_auc(sides["negative"], sides["positive"])
        else:
            auc = _rank_auc(sides["positive"], sides["negative"])
        if auc is not None:
            scored[key] = auc
    usable = n_pos >= min_group and n_neg >= min_group
    return {
        "uncertainty_auc": max(scored.values()) if (usable and scored) else None,
        "features": scored,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "window": window,
        "cap": cap,
        "min_think_tokens": 2 * window,
        "min_group": min_group,
        "inconclusive": not usable,
    }


def format_auc(auc: dict[str, Any] | None) -> str:
    """Human-readable summary of an `auc` block."""
    if not auc:
        return "uncertainty AUC: absent (features stay off under rule='length')"
    lines = []
    overall = auc.get("uncertainty_auc")
    if overall is None:
        lines.append(
            "uncertainty AUC: inconclusive "
            f"(positives={auc.get('n_positive')} negatives={auc.get('n_negative')}, "
            f"need {auc.get('min_group')} of each)"
        )
    else:
        lines.append(
            f"uncertainty AUC: {overall:.3f} "
            f"(positives={auc.get('n_positive')} negatives={auc.get('n_negative')})"
        )
    for key, value in sorted(auc.get("features", {}).items()):
        lines.append(f"  {key:<14} {value:.3f}")
    return "\n".join(lines)


def summarise(path: str) -> str:
    """Human-readable percentile table of a sketch file."""
    sketches = SignalSketches(min_samples=1, path=path)
    sketches.load()
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.85, 0.92, 0.95, 0.99)
    lines = [f"model: {sketches.model}", ""]
    header = "signal".ljust(12) + "count".rjust(10)
    header += "".join(f"{'p' + str(int(q * 100)):>9}" for q in quantiles)
    lines.append(header)
    for key, digest in sketches.digests.items():
        if digest.count <= 0:
            lines.append(f"{key.ljust(12)}{0:>10}  (cold)")
            continue
        row = key.ljust(12) + f"{digest.count:>10.0f}"
        row += "".join(f"{digest.quantile(q):>9.4f}" for q in quantiles)
        lines.append(row)
    lines.append("")
    lines.append(format_auc(sketches.auc))
    return "\n".join(lines)


def _post_chat(
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "dynamic",
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "vllm_xargs": {"effort_telemetry": True},
        }
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_prompts(
    base_url: str,
    model: str,
    prompts: list[str],
    api_key: str | None,
    max_tokens: int,
    timeout: float,
) -> int:
    """Send the prompt set as dynamic-effort requests; return the failure count."""
    failures = 0
    for i, prompt in enumerate(prompts, 1):
        try:
            payload = _post_chat(base_url, api_key, model, prompt, max_tokens, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"[{i}/{len(prompts)}] FAILED: {exc}", file=sys.stderr)
            failures += 1
            continue
        usage = payload.get("usage", {})
        effort = (payload.get("choices") or [{}])[0].get("effort")
        print(
            f"[{i}/{len(prompts)}] completion={usage.get('completion_tokens')} "
            f"effort={effort}"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="drive a running server with a prompt set")
    run.add_argument("--base-url", default="http://localhost:8012/v1")
    run.add_argument("--model", required=True)
    run.add_argument("--prompts", help="file with one prompt per line")
    run.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    run.add_argument("--max-tokens", type=int, default=4096)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--repeat", type=int, default=1)

    build = sub.add_parser("build", help="fold a telemetry sink into a sketch file")
    build.add_argument("--telemetry", nargs="+", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--model")
    build.add_argument("--min-samples", type=int, default=2048)
    build.add_argument("--compression", type=float, default=100.0)
    build.add_argument("--include-all-rows", action="store_true")
    build.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        default=DEFAULT_LADDER,
        help="rung caps the traffic ran with; labels the AUC groups",
    )
    build.add_argument(
        "--no-auc",
        action="store_true",
        help="skip the uncertainty-feature AUC pass",
    )

    show = sub.add_parser("show", help="print a sketch file's percentiles")
    show.add_argument("--sketch", required=True)

    args = parser.parse_args(argv)

    if args.command == "run":
        prompts = DEFAULT_PROMPTS
        if args.prompts:
            with open(args.prompts, encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
        prompts = prompts * max(args.repeat, 1)
        failures = run_prompts(
            args.base_url,
            args.model,
            prompts,
            args.api_key,
            args.max_tokens,
            args.timeout,
        )
        print(f"{len(prompts) - failures}/{len(prompts)} requests completed")
        return 1 if failures else 0

    if args.command == "build":
        counts = build_sketches(
            args.telemetry,
            args.out,
            args.model,
            args.min_samples,
            args.compression,
            args.include_all_rows,
            args.ladder,
            not args.no_auc,
        )
        print(f"wrote {args.out}")
        for key, count in counts.items():
            warm = "warm" if count >= args.min_samples else "COLD"
            print(f"  {key:<12} {count:>10.0f}  {warm}")
        if not args.no_auc:
            written = SignalSketches(min_samples=1, path=args.out)
            written.load()
            print(format_auc(written.auc))
        return 0

    print(summarise(args.sketch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
