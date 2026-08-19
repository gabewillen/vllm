# ruff: noqa: E501, SIM115 - prompt literals are verbatim from the earlier goal runs
"""Same prompt set at any concurrency; reports per-request decode tok/s,
aggregate output tok/s, TTFT and the spec-decode acceptance length.

    bench_concurrency.py <port> <concurrency> [max_tokens] [repeats]

The prompt set is the eight-prompt mix used by the earlier goal runs
(ss_bench2.py): two reasoning, two prose, three code, one extraction. At
concurrency N the set is cycled, so every arm sees the same prompts in the same
order and the only variable is the engine config. Requests are greedy
(temperature 0) so a spec-decode arm and a no-spec arm produce the same tokens.

Acceptance length = 1 + accepted/drafts = mean tokens committed per target
verify step; mean draft length = draft_tokens/drafts. Both come from the
/metrics deltas across the run, so they cover exactly these requests.
"""

import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8013"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 512
REPEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
KEY = os.environ.get("VLLM_API_KEY", "")
URL = f"http://localhost:{PORT}/v1/chat/completions"

# Same two source files the earlier runs pasted into the code prompts, so the
# prompts stay byte-identical across goal runs.
SRC = "/shared/vllm/.venv-qwen38/lib/python3.12/site-packages/vllm"
CODE = open(f"{SRC}/v1/kv_offload/tiering/async_lookup.py").read()
CODE2 = open(f"{SRC}/v1/worker/ubatch_utils.py").read()[:6000]

PROMPTS = [
    (
        "reason-math",
        "A train leaves city A at 9:00 going 80 km/h; another leaves city B (300 km away) at 9:30 going 100 km/h toward A. When and where do they meet? Then generalize to symbolic speeds v1, v2, gap D, delay t.",
        True,
    ),
    (
        "reason-sys",
        "A ZFS pool on 2 NVMe drives shows 12 GiB ARC but the box has 120 GB RAM; explain what limits ARC, whether raising it helps a KV-cache file store of 55 MB files, and give a recommendation.",
        True,
    ),
    (
        "prose-essay",
        "Write a detailed essay about the history of container shipping.",
        False,
    ),
    (
        "prose-story",
        "Write a short story about a lighthouse keeper who discovers the light is being used to send messages.",
        False,
    ),
    (
        "code-write",
        "Write a Python module implementing an LRU cache with TTL expiry, thread safety, and statistics; include docstrings and a small test suite.",
        False,
    ),
    (
        "code-edit",
        "Here is a Python file:\n\n```python\n"
        + CODE
        + "\n```\n\nRewrite the ENTIRE file, changing only: rename `_pending_results` to `_result_queue` everywhere and add a `size()` method returning len(self._lookup_state). Output the full file in one code block, no commentary.",
        False,
    ),
    (
        "code-explain",
        "Explain what this code does, function by function, then list three possible bugs:\n\n```python\n"
        + CODE2
        + "\n```",
        False,
    ),
    (
        "json-extract",
        "Extract every product name, price and quantity from this order email as a JSON array of objects with keys name, price, qty. Email: 'Hi, please send 3x Widget Pro at $19.99, 12 units of the Gizmo Mini ($4.50 each), and one Deluxe Stand for $89. Also 2 packs of Cable Set B, $7.25/pack. Thanks!'",
        False,
    ),
]


def metrics():
    text = urllib.request.urlopen(f"http://localhost:{PORT}/metrics").read().decode()

    def total(name):
        pattern = rf"^vllm:{name}_total\{{[^}}]*\}} ([0-9.e+]+)$"
        return sum(float(m.group(1)) for m in re.finditer(pattern, text, re.M))

    return (
        total("spec_decode_num_accepted_tokens"),
        total("spec_decode_num_drafts"),
        total("spec_decode_num_draft_tokens"),
    )


def one(slot, out):
    name, prompt, think = PROMPTS[slot % len(PROMPTS)]
    body = json.dumps(
        {
            "model": "Qwen3.8-27B",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": think},
        }
    ).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    t0 = time.time()
    first = None
    usage = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning")):
                    first = time.time()
    except Exception as exc:  # noqa: BLE001 - bench client, report and continue
        err = repr(exc)[:120]
    end = time.time()
    out[slot] = (
        name,
        err,
        (first - t0) if first else None,
        (usage or {}).get("completion_tokens", 0),
        (end - first) if first else None,
    )


def main():
    a0, d0, dt0 = metrics()
    results = [None] * (CONC * REPEATS)
    wall0 = time.time()
    for r in range(REPEATS):
        threads = [
            threading.Thread(target=one, args=(r * CONC + i, results))
            for i in range(CONC)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    wall = time.time() - wall0
    a1, d1, dt1 = metrics()

    ok = [r for r in results if r and r[1] is None and r[3] > 0]
    if not ok:
        print("no successful requests:", [r[1] for r in results if r][:3])
        return 1
    per_req = [r[3] / r[4] for r in ok]
    total_out = sum(r[3] for r in ok)
    drafts = d1 - d0
    print(
        f"conc={CONC} repeats={REPEATS} max_tokens={MAX_TOKENS}: "
        f"ok={len(ok)}/{len(results)} wall={wall:.1f}s "
        f"aggregate={total_out / wall:.0f} tok/s "
        f"per-request decode median={statistics.median(per_req):.1f} "
        f"mean={statistics.mean(per_req):.1f} tok/s "
        f"ttft median={statistics.median(r[2] for r in ok):.2f}s "
        f"max={max(r[2] for r in ok):.2f}s"
    )
    if drafts > 0:
        print(
            f"  spec: acceptance_length={1 + (a1 - a0) / drafts:.2f} tok/verify  "
            f"mean_draft_length={(dt1 - dt0) / drafts:.2f}  "
            f"per-token acceptance={(a1 - a0) / (dt1 - dt0):.3f}  "
            f"drafts={drafts:.0f}"
        )
    else:
        print("  spec: no drafts (no-spec arm)")
    for r in ok:
        print(
            f"  {r[0]:13s} ttft={r[2]:5.2f}s decode={r[3] / r[4]:6.1f} tok/s ({r[3]} tok)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
