#!/usr/bin/env python3
"""Coding smoke eval for Ling-3.0-flash-fp4 served via vLLM OpenAI API.

Sends 6 coding tasks to /v1/chat/completions (default port 8011), prints each
response with TTFT / total time / tokens-per-second, and appends a JSONL
transcript record per task to --out.

Stdlib only; run with any python3:
    /shared/vllm/.venv-ling/bin/python /shared/vllm/scripts/ling3_coding_eval.py \
        --out /data/artifacts/ling3_coding_eval.jsonl
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BUGGY_SNIPPET = '''\
import heapq
import time


class Job:
    def __init__(self, name, priority, payload):
        self.name = name
        self.priority = priority
        self.payload = payload
        self.submitted_at = time.monotonic()

    def run(self):
        return f"ran {self.name} with {self.payload!r}"


class JobScheduler:
    """Runs jobs in priority order (lower number = more urgent)."""

    def __init__(self):
        self._heap = []
        self._results = {}

    def submit(self, job):
        heapq.heappush(self._heap, (job.priority, job))

    def pending(self):
        return len(self._heap)

    def run_next(self):
        if not self._heap:
            raise RuntimeError("no jobs pending")
        _, job = heapq.heappop(self._heap)
        self._results[job.name] = job.run()
        return job.name

    def run_all(self):
        while self._heap:
            self.run_next()
        return dict(self._results)


if __name__ == "__main__":
    sched = JobScheduler()
    sched.submit(Job("index", 2, {"shard": 1}))
    sched.submit(Job("compact", 1, {"level": 0}))
    sched.submit(Job("flush", 1, {"memtable": "m3"}))
    print(sched.run_all())
'''

DIFF_SNIPPET = '''\
--- a/cache/evictor.py
+++ b/cache/evictor.py
@@ -12,18 +12,17 @@ class SizeBoundedCache:
     def _evict_until_fits(self, incoming_size):
-        # Walk entries oldest-first and drop until there is room.
-        while self._total_bytes + incoming_size > self.capacity_bytes:
-            key = next(iter(self._store))
-            entry = self._store[key]
-            self._total_bytes -= entry.size_bytes
-            del self._store[key]
-            self._on_evict(key, entry)
+        # Walk entries oldest-first and drop until there is room.
+        while self._total_bytes + incoming_size > self.capacity_bytes:
+            key = next(iter(self._store))
+            del self._store[key]
+            entry = self._store[key]
+            self._total_bytes -= entry.size_bytes
+            self._on_evict(key, entry)

     def put(self, key, value, size_bytes):
         if size_bytes > self.capacity_bytes:
             raise ValueError("object larger than cache")
         self._evict_until_fits(size_bytes)
-        self._store[key] = Entry(value, size_bytes)
-        self._total_bytes += size_bytes
+        self._total_bytes += size_bytes
+        self._store[key] = Entry(value, size_bytes)
'''

CALLBACK_JS = '''\
function deployService(name, cb) {
  fetchManifest(name, function (err, manifest) {
    if (err) return cb(err);
    buildImage(manifest, function (err, image) {
      if (err) return cb(err);
      pushImage(image, function (err, digest) {
        if (err) return cb(err);
        rolloutDeployment(name, digest, function (err, status) {
          if (err) {
            rollback(name, function (rbErr) {
              cb(rbErr || err);
            });
          } else {
            notifySlack("deployed " + name, function () {
              cb(null, status);
            });
          }
        });
      });
    });
  });
}
'''

TASKS = [
    {
        "id": "rate_limiter",
        "title": "Implement a rate limiter class with tests",
        "prompt": (
            "Implement a thread-safe token-bucket rate limiter class in Python "
            "with methods `allow() -> bool` and `wait_time() -> float`, "
            "configurable capacity and refill rate. Include unit tests "
            "(unittest or pytest) covering burst behaviour, refill over time "
            "(mock the clock), and thread safety."
        ),
    },
    {
        "id": "fix_subtle_bug",
        "title": "Fix a subtle bug in a Python snippet",
        "prompt": (
            "The following job scheduler crashes intermittently in production "
            "with a TypeError, but only under certain workloads. Find the bug, "
            "explain exactly when it triggers, and provide a minimal fix:\n\n"
            "```python\n" + BUGGY_SNIPPET + "```"
        ),
    },
    {
        "id": "sql_query",
        "title": "Write a SQL query for a described schema",
        "prompt": (
            "Schema: `users(id, name, created_at)`, "
            "`orders(id, user_id, total_cents, status, placed_at)`, "
            "`order_items(id, order_id, product_id, qty, unit_cents)`, "
            "`products(id, sku, category)`. "
            "Write a PostgreSQL query returning, for each of the last 6 "
            "complete calendar months, the month, the number of distinct "
            "paying users (users with at least one order with "
            "status = 'paid' placed that month), and the top product "
            "category by paid revenue that month (revenue = qty * unit_cents "
            "over items of paid orders). One row per month, oldest first. "
            "Explain any tie-breaking you choose."
        ),
    },
    {
        "id": "diff_review",
        "title": "Explain a git diff and spot the bug in it",
        "prompt": (
            "Explain what this refactoring diff changes in behaviour, then "
            "identify any bug it introduces and give the corrected code:\n\n"
            "```diff\n" + DIFF_SNIPPET + "```"
        ),
    },
    {
        "id": "react_component",
        "title": "Implement a small React component",
        "prompt": (
            "Write a small React function component `<DebouncedSearchBox>` "
            "(TypeScript, hooks) that takes `onSearch(query: string)` and "
            "`delayMs` props, debounces input changes, cancels pending "
            "callbacks on unmount, shows a spinner while a search is "
            "in flight, and is accessible (label + aria attributes). "
            "Include a short usage example."
        ),
    },
    {
        "id": "callbacks_to_async",
        "title": "Refactor nested callbacks to async/await",
        "prompt": (
            "Refactor this callback-based JavaScript to modern async/await "
            "with equivalent error and rollback semantics (assume the helper "
            "functions gain promise-returning variants or are wrapped with "
            "util.promisify). Preserve the behaviour where a failed rollout "
            "triggers a rollback and the rollback error, if any, takes "
            "precedence:\n\n```js\n" + CALLBACK_JS + "```"
        ),
    },
]


def http_json(url, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def discover_model(base_url, timeout):
    models = http_json(f"{base_url}/models", timeout=timeout)
    ids = [m["id"] for m in models.get("data", [])]
    if not ids:
        raise RuntimeError("server reported no models")
    return ids[0]


def stream_chat(base_url, payload, timeout):
    """POST a streaming chat completion; return (record, ttft, total)."""
    payload = dict(payload, stream=True,
                   stream_options={"include_usage": True})
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    reasoning, content, finish, usage = [], [], None, None
    t0 = time.monotonic()
    ttft = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                rpiece = delta.get("reasoning_content") or delta.get("reasoning")
                if (piece or rpiece) and ttft is None:
                    ttft = time.monotonic() - t0
                if piece:
                    content.append(piece)
                if rpiece:
                    reasoning.append(rpiece)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    total = time.monotonic() - t0
    record = {
        "reasoning_content": "".join(reasoning),
        "content": "".join(content),
        "finish_reason": finish,
        "usage": usage,
    }
    return record, ttft, total


def blocking_chat(base_url, payload, timeout):
    t0 = time.monotonic()
    resp = http_json(f"{base_url}/chat/completions", payload, timeout)
    total = time.monotonic() - t0
    msg = resp["choices"][0]["message"]
    record = {
        "reasoning_content": msg.get("reasoning_content") or "",
        "content": msg.get("content") or "",
        "finish_reason": resp["choices"][0].get("finish_reason"),
        "usage": resp.get("usage"),
    }
    return record, None, total


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default=None,
                    help="model name; default: first entry of /v1/models")
    ap.add_argument("--out", required=True, help="JSONL transcript path")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="generous: thinking mode is on by default")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--no-thinking", action="store_true",
                    help="pass chat_template_kwargs.enable_thinking=false")
    ap.add_argument("--tasks", default=None,
                    help="comma-separated task ids to run (default: all)")
    args = ap.parse_args()

    model = args.model or discover_model(args.base_url, args.timeout)
    print(f"model: {model}  base_url: {args.base_url}", flush=True)

    wanted = set(args.tasks.split(",")) if args.tasks else None
    tasks = [t for t in TASKS if wanted is None or t["id"] in wanted]
    if not tasks:
        sys.exit(f"no tasks matched {args.tasks!r}")

    results = []
    with open(args.out, "a") as out:
        for i, task in enumerate(tasks, 1):
            print(f"\n=== [{i}/{len(tasks)}] {task['id']}: {task['title']} ===",
                  flush=True)
            messages = [{"role": "user", "content": task["prompt"]}]
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,  # vLLM extension, accepted top-level
            }
            if args.no_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            err = None
            record, ttft, total = {}, None, None
            try:
                if args.no_stream:
                    record, ttft, total = blocking_chat(
                        args.base_url, payload, args.timeout)
                else:
                    record, ttft, total = stream_chat(
                        args.base_url, payload, args.timeout)
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, RuntimeError, json.JSONDecodeError) as e:
                body = ""
                if isinstance(e, urllib.error.HTTPError):
                    body = e.read().decode("utf-8", "replace")[:500]
                err = f"{type(e).__name__}: {e} {body}".strip()
                print(f"ERROR: {err}", flush=True)

            usage = record.get("usage") or {}
            ctok = usage.get("completion_tokens")
            gen_tps = None
            if ctok and total:
                gen_time = (total - ttft) if ttft else total
                gen_tps = ctok / gen_time if gen_time > 0 else None
            reasoning = record.get("reasoning_content", "")
            if reasoning:
                head = reasoning[:400].replace("\n", " ")
                print(f"--- reasoning ({len(reasoning)} chars, head): "
                      f"{head}...", flush=True)
            print(f"--- response:\n{record.get('content', '')}", flush=True)
            print(f"--- timing: ttft={ttft:.2f}s " if ttft else
                  "--- timing: ttft=n/a ", end="")
            print(f"total={total:.2f}s " if total else "total=n/a ", end="")
            print(f"completion_tokens={ctok} "
                  f"gen_tok/s={gen_tps:.1f}" if gen_tps else
                  f"completion_tokens={ctok}", flush=True)

            entry = {
                "task_id": task["id"],
                "title": task["title"],
                "model": model,
                "messages": messages,
                "reasoning_content": reasoning,
                "content": record.get("content", ""),
                "finish_reason": record.get("finish_reason"),
                "usage": usage or None,
                "timing": {"ttft_s": ttft, "total_s": total,
                           "gen_tok_per_s": gen_tps},
                "error": err,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out.flush()
            results.append(entry)

    ok = sum(1 for r in results if not r["error"])
    print(f"\n=== done: {ok}/{len(results)} tasks succeeded; "
          f"transcript appended to {args.out} ===", flush=True)
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
