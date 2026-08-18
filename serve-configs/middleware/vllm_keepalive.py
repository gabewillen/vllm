# SPDX-License-Identifier: Apache-2.0
"""Keep-alive ASGI middleware for vLLM behind an edge proxy with a hard
"no response bytes for N seconds" timeout (Cloudflare: 100 s -> HTTP 524).

vLLM emits nothing until the first token, so a long uncached prefill (a
~200k-token prompt takes 130-150 s on 4x L4) is killed at the edge even when
the client streams. This middleware keeps bytes flowing while the engine
works, in a way every client parser ignores:

* text/event-stream responses: SSE comment lines (``: keepalive``) whenever
  no bytes have been sent for ``ping_interval`` seconds. Comment lines are
  ignored by the SSE spec and by the OpenAI SDKs' decoders.
* Responses the app has not started after ``json_commit_after`` seconds are
  committed early as ``200``: ``text/event-stream`` with comment pings when
  the request body has ``"stream": true`` (vLLM only sends the SSE headers
  with the first token, so a queued streaming request looks idle too), else
  ``application/json`` with whitespace pings (JSON parsers skip leading
  whitespace). Only the status line is lost - if the app later fails, the
  client gets a 200 with the error body instead of a 4xx/5xx (logged as a
  warning; after an SSE commit the JSON error body is wrapped as one
  ``data:`` event so the stream stays well-formed). This is strictly better
  than the certain 524.
* Telemetry: OTEL counters ``vllm.keepalive.pings`` /
  ``vllm.keepalive.early_commits`` (label content_type = sse|json) and the
  same as prometheus ``vllm_keepalive_*_total`` on vLLM's ``/metrics``.

Enable with ``--middleware vllm_keepalive.KeepAliveMiddleware`` and put this
directory on PYTHONPATH. Env overrides: VLLM_KEEPALIVE_PING_INTERVAL (s),
VLLM_KEEPALIVE_JSON_COMMIT_AFTER (s), VLLM_KEEPALIVE_PATH_PREFIX.
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from opentelemetry import metrics as otel_metrics
from prometheus_client import Counter

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

logger = logging.getLogger("vllm.keepalive")

_SSE_PING = b": keepalive\n\n"
_JSON_PING = b"\n"
# Bytes of request body kept while looking for "stream": true; larger bodies
# (a full 262k-token chat is ~2 MiB) fall back to the JSON commit path.
_MAX_PEEK_BYTES = 8 * 1024 * 1024

# Telemetry: OTEL counters (exported when the operator wires an OTEL metrics
# SDK) mirrored to the prometheus registry vLLM already serves at /metrics.
# Labels are bounded: content_type in {sse, json}.
_meter = otel_metrics.get_meter("vllm.keepalive")
_otel_pings = _meter.create_counter(
    "vllm.keepalive.pings", description="keep-alive bytes sent while idle"
)
_otel_early_commits = _meter.create_counter(
    "vllm.keepalive.early_commits",
    description="responses committed before the app started them",
)
_prom_pings = Counter(
    "vllm_keepalive_pings_total", "keep-alive pings sent", ["content_type"]
)
_prom_early_commits = Counter(
    "vllm_keepalive_early_commits_total",
    "responses committed before the app started them",
    ["content_type"],
)


def _body_requests_stream(body: bytes) -> bool:
    """True when a JSON request body asks for a streaming response."""
    if len(body) > _MAX_PEEK_BYTES:
        logger.debug("keepalive: body peek skipped (%d bytes)", len(body))
        return False
    try:
        return bool(json.loads(body).get("stream"))
    except (ValueError, AttributeError):
        # Not JSON, or JSON that is not an object: no streaming request.
        logger.debug("keepalive: body peek found no stream flag")
        return False


class KeepAliveMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        ping_interval: float | None = None,
        json_commit_after: float | None = None,
        path_prefix: str | None = None,
    ) -> None:
        self.app = app
        self.ping_interval = float(
            ping_interval or os.environ.get("VLLM_KEEPALIVE_PING_INTERVAL", 15)
        )
        self.json_commit_after = float(
            json_commit_after
            or os.environ.get("VLLM_KEEPALIVE_JSON_COMMIT_AFTER", 40)
        )
        self.path_prefix = path_prefix or os.environ.get(
            "VLLM_KEEPALIVE_PATH_PREFIX", "/v1/"
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or not scope.get("path", "").startswith(self.path_prefix)
        ):
            await self.app(scope, receive, send)
            return
        await _KeepAliveResponder(self, scope, receive, send).run()


class _KeepAliveResponder:
    def __init__(
        self, cfg: KeepAliveMiddleware, scope: Scope, receive: Receive, send: Send
    ) -> None:
        self.cfg = cfg
        self.scope = scope
        self.receive = receive
        self.send = send
        self._body = bytearray()
        self.expect_sse = False  # request asked for a streaming response
        self.wrap_body_as_sse = False  # app sent non-SSE after an SSE commit
        self._sse_wrap_started = False
        self.lock = asyncio.Lock()
        self.started = False  # response.start delivered to the client
        self.early_committed = False  # we sent the start ourselves
        self.is_sse = False
        self.finished = False
        self.last_bytes_at = time.monotonic()
        self.t0 = time.monotonic()
        self.pinger: asyncio.Task | None = None
        self.n_pings = 0

    async def _receive(self) -> Message:
        # Peek at the request body: a streaming request must be committed as
        # SSE (not JSON) if the app is slow to start its response.
        message = await self.receive()
        if message["type"] != "http.request":
            return message
        if len(self._body) <= _MAX_PEEK_BYTES:
            self._body += message.get("body", b"")
        if not message.get("more_body", False):
            self.expect_sse = _body_requests_stream(bytes(self._body))
            self._body = bytearray()
        return message

    async def run(self) -> None:
        self.pinger = asyncio.create_task(self._ping_loop())
        try:
            await self.cfg.app(self.scope, self._receive, self._send)
        finally:
            self.finished = True
            self.pinger.cancel()
            if self.n_pings:
                logger.info(
                    "keepalive: %s pings (%s) for %s, %.1fs to completion",
                    self.n_pings,
                    "sse" if self.is_sse else "json",
                    self.scope.get("path"),
                    time.monotonic() - self.t0,
                )

    async def _send(self, message: Message) -> None:
        async with self.lock:
            if message["type"] == "http.response.start":
                headers = message.get("headers") or []
                ctype = b""
                for k, v in headers:
                    if k.lower() == b"content-type":
                        ctype = v.lower()
                        break
                if self.early_committed:
                    # We already committed 200 (json or SSE); only the body
                    # can follow. A non-SSE body after an SSE commit is
                    # wrapped as one SSE data event so SSE parsers still see
                    # a well-formed stream (the status line is lost either
                    # way; logged for the operator).
                    self.wrap_body_as_sse = self.is_sse and not ctype.startswith(
                        b"text/event-stream"
                    )
                    if message.get("status", 200) >= 300:
                        logger.warning(
                            "keepalive: app responded %s after early commit "
                            "on %s; client will see 200 with the error body",
                            message.get("status"),
                            self.scope.get("path"),
                        )
                    return
                self.is_sse = ctype.startswith(b"text/event-stream")
                self.started = True
                await self.send(message)
                self.last_bytes_at = time.monotonic()
                return
            if message["type"] == "http.response.body":
                if self.wrap_body_as_sse:
                    message = dict(message)
                    body = bytes(message.get("body", b""))
                    if not self._sse_wrap_started:
                        body = b"data: " + body
                        self._sse_wrap_started = True
                    if not message.get("more_body", False):
                        body += b"\n\n"
                    message["body"] = body
                await self.send(message)
                self.last_bytes_at = time.monotonic()
                if not message.get("more_body", False):
                    self.finished = True
                return
            await self.send(message)

    async def _ping_loop(self) -> None:
        interval = self.cfg.ping_interval
        try:
            while not self.finished:
                await asyncio.sleep(1.0)
                now = time.monotonic()
                async with self.lock:
                    if self.finished:
                        return
                    if not self.started:
                        if now - self.t0 < self.cfg.json_commit_after:
                            continue
                        # No response yet: commit early with the content type
                        # the client asked for (SSE for stream=true, else JSON).
                        ctype = (
                            b"text/event-stream" if self.expect_sse
                            else b"application/json"
                        )
                        await self.send(
                            {
                                "type": "http.response.start",
                                "status": 200,
                                "headers": [
                                    (b"content-type", ctype),
                                    (b"cache-control", b"no-cache"),
                                    (b"x-vllm-keepalive", b"early-commit"),
                                ],
                            }
                        )
                        self.started = True
                        self.early_committed = True
                        self.is_sse = self.expect_sse
                        label = "sse" if self.is_sse else "json"
                        _otel_early_commits.add(1, {"content_type": label})
                        _prom_early_commits.labels(content_type=label).inc()
                        await self._ping(_SSE_PING if self.is_sse else _JSON_PING)
                        continue
                    if now - self.last_bytes_at >= interval:
                        await self._ping(_SSE_PING if self.is_sse else _JSON_PING)
        except asyncio.CancelledError:
            pass
        except Exception:  # client went away or send failed
            logger.debug("keepalive: ping loop ended", exc_info=True)

    async def _ping(self, payload: bytes) -> None:
        await self.send(
            {"type": "http.response.body", "body": payload, "more_body": True}
        )
        self.last_bytes_at = time.monotonic()
        self.n_pings += 1
        label = "sse" if self.is_sse else "json"
        _otel_pings.add(1, {"content_type": label})
        _prom_pings.labels(content_type=label).inc()
