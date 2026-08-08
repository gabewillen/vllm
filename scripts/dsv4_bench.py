#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Benchmark client for the DeepSeek-V4-Flash tiered-expert serving project.

Measures TTFT and decode throughput via streaming /v1/completions, emitting
one JSON line per run plus a summary. Prompt is synthesized to a target token
count with the model's own tokenizer.

Example:
    .venv-omni/bin/python scripts/dsv4_bench.py --base-url http://127.0.0.1:8010 \
        --tokenizer /shared/vllm/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/<sha> \
        --prompt-tokens 4096 --max-tokens 256 --runs 3 --label approach0 \
        --out /path/to/artifacts/bench_approach0.jsonl
"""

import argparse
import json
import time

import requests

FILLER = (
    "The quick brown fox jumps over the lazy dog while contemplating "
    "distributed systems, memory hierarchies, and the economics of "
    "speculative execution in modern inference engines. "
)


def build_prompt(tokenizer_dir: str, target_tokens: int) -> tuple[str, int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    filler_ids = tok.encode(FILLER, add_special_tokens=False)
    reps = max(1, target_tokens // len(filler_ids) + 1)
    ids = (filler_ids * reps)[: max(1, target_tokens - 8)]
    text = tok.decode(ids)
    return text, len(tok.encode(text))


def run_once(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    usage = {}
    with requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=7200,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("choices") and obj["choices"][0].get("text"):
                n_chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    completion = usage.get("completion_tokens", n_chunks)
    decode_time = total - (ttft or 0.0)
    return {
        "ttft_s": round(ttft or -1, 3),
        "total_s": round(total, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "decode_tps": round(completion / decode_time, 3) if decode_time > 0 else None,
        "prefill_tps": (
            round(usage.get("prompt_tokens", 0) / ttft, 1) if ttft else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompt, actual = build_prompt(args.tokenizer, args.prompt_tokens)
    print(f"prompt tokens (tokenizer-measured): {actual}")
    results = []
    for i in range(args.runs):
        res = run_once(args.base_url, args.model, prompt, args.max_tokens)
        res.update({"run": i, "label": args.label, "target_prompt_tokens": actual})
        results.append(res)
        print(json.dumps(res))
    decode = [r["decode_tps"] for r in results[1:] or results if r["decode_tps"]]
    summary = {
        "label": args.label,
        "prompt_tokens": actual,
        "runs": args.runs,
        "decode_tps_best": max(decode) if decode else None,
        "decode_tps_mean": round(sum(decode) / len(decode), 3) if decode else None,
        "ttft_s_last": results[-1]["ttft_s"],
    }
    print("SUMMARY " + json.dumps(summary))
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")


if __name__ == "__main__":
    main()
