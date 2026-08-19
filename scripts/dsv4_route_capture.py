# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive a coding-agent-like workload through a DSV4 server to populate
the expert-tier activation histogram (--expert-tier-activation-hist-path).

Feeds real source files as review prompts (prefill-heavy, some decode),
mirroring agentic coding traffic. Histograms are dumped by the server at
shutdown; this client just generates representative routing.
"""

import argparse
import time

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    for path in args.files:
        text = open(path, errors="replace").read()
        ids = tok.encode(
            f"Review this file ({path}) and explain its design:\n\n" + text,
            add_special_tokens=False,
        )[: args.prompt_tokens]
        prompt = tok.decode(ids)
        t0 = time.perf_counter()
        r = requests.post(
            f"{args.base_url}/v1/completions",
            json={
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.7,
            },
            timeout=3600,
        )
        r.raise_for_status()
        usage = r.json().get("usage", {})
        print(
            f"{path}: prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')} "
            f"took={time.perf_counter() - t0:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
