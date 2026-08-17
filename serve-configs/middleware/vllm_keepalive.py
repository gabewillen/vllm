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
* JSON (non-streaming) responses: if the app has not started its response
  after ``json_commit_after`` seconds, commit ``200 application/json``
  early and send whitespace pings; JSON parsers skip leading whitespace.
  Only the status line is lost - if the app later fails, the client gets a
  200 with the error JSON body instead of a 4xx/5xx (logged as a warning).
  This is strictly better than the certain 524.

Enable with ``--middleware vllm_keepalive.KeepAliveMiddleware`` and put this
directory on PYTHONPATH. Env overrides: VLLM_KEEPALIVE_PING_INTERVAL (s),
VLLM_KEEPALIVE_JSON_COMMIT_AFTER (s), VLLM_KEEPALIVE_PATH_PREFIX.
"""

import asyncio
import logging
import os
import time

logger = logging.getLogger("vllm.keepalive")

_SSE_PING = b": keepalive\n\n"
_JSON_PING = b"\n"


class KeepAliveMiddleware:
    def __init__(
        self,
        app,
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

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or not scope.get("path", "").startswith(self.path_prefix)
        ):
            await self.app(scope, receive, send)
            return
        await _KeepAliveResponder(self, scope, receive, send).run()


class _KeepAliveResponder:
    def __init__(self, cfg: KeepAliveMiddleware, scope, receive, send) -> None:
        self.cfg = cfg
        self.scope = scope
        self.receive = receive
        self.send = send
        self.lock = asyncio.Lock()
        self.started = False  # response.start delivered to the client
        self.early_committed = False  # we sent the start ourselves
        self.is_sse = False
        self.finished = False
        self.last_bytes_at = time.monotonic()
        self.t0 = time.monotonic()
        self.pinger: asyncio.Task | None = None
        self.n_pings = 0

    async def run(self) -> None:
        self.pinger = asyncio.create_task(self._ping_loop())
        try:
            await self.cfg.app(self.scope, self.receive, self._send)
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

    async def _send(self, message) -> None:
        async with self.lock:
            if message["type"] == "http.response.start":
                headers = message.get("headers") or []
                ctype = b""
                for k, v in headers:
                    if k.lower() == b"content-type":
                        ctype = v.lower()
                        break
                if self.early_committed:
                    # We already committed 200/json; only the body can follow.
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
                        # No response yet: assume JSON, commit early.
                        await self.send(
                            {
                                "type": "http.response.start",
                                "status": 200,
                                "headers": [
                                    (b"content-type", b"application/json"),
                                    (b"cache-control", b"no-cache"),
                                    (b"x-vllm-keepalive", b"early-commit"),
                                ],
                            }
                        )
                        self.started = True
                        self.early_committed = True
                        self.is_sse = False
                        await self._ping(_JSON_PING)
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
