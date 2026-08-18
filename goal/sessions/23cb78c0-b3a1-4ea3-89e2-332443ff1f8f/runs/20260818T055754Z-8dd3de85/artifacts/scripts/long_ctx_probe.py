"""Long-context gate probe: send a ~target-token prompt with a needle at the
start, verify the model retrieves it. Uses the server's /tokenize endpoint to
hit the token target within tolerance.

Usage: /shared/vllm/.venv-qwen38/bin/python long_ctx_probe.py [--port 8012] [--target-tokens 200000]
"""

import argparse
import json
import os
import sys
import time
import urllib.request


def post(url: str, payload: dict, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['VLLM_API_KEY']}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--target-tokens", type=int, default=200_000)
    ap.add_argument("--salt", default="", help="prefix text to defeat prefix caching")
    ap.add_argument("--model", default="Qwen3.8-27B")
    args = ap.parse_args()
    base = f"http://localhost:{args.port}"

    needle = "The secret launch code is 736251."
    filler = (
        "Telemetry snapshot: subsystem nominal, latency within budget, "
        "queue depth stable, cache hit ratio steady. "
    )
    question = "\n\nWhat is the secret launch code? Answer with the number only."

    # Calibrate filler token cost via /tokenize, then scale.
    probe_block = filler * 50
    n = post(f"{base}/tokenize", {"prompt": probe_block}, 60)["count"]
    per_block = n / 50
    blocks = int((args.target_tokens - 200) / per_block)
    prefix = f"[session {args.salt}] " if args.salt else ""
    prompt = prefix + needle + "\n" + filler * blocks + question
    total = post(f"{base}/tokenize", {"prompt": prompt}, 300)["count"]
    print(f"prompt tokens: {total}")

    t0 = time.time()
    out = post(
        f"{base}/v1/completions",
        {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 32,
            "temperature": 0,
        },
        timeout=3600,
    )
    dt = time.time() - t0
    text = out["choices"][0]["text"]
    usage = out.get("usage", {})
    ok = "736251" in text
    print(f"latency: {dt:.1f}s  usage: {usage}")
    print(f"completion: {text!r}")
    print(f"needle retrieved: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
