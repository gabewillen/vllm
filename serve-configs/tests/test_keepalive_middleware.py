# SPDX-License-Identifier: Apache-2.0
"""ASGI behavior of serve-configs/middleware/vllm_keepalive.py."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "middleware"))
import vllm_keepalive  # noqa: E402


def _scope(path="/v1/chat/completions"):
    return {"type": "http", "method": "POST", "path": path}


def _receive_for(body: bytes):
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            await asyncio.sleep(3600)
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def _collect():
    out = []

    async def send(message):
        out.append(message)

    return out, send


def _slow_app(delay, ctype=b"application/json", body=b'{"ok":true}', status=200):
    async def app(scope, receive, send):
        await receive()
        await asyncio.sleep(delay)
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", ctype)]})
        await send({"type": "http.response.body", "body": body,
                    "more_body": False})

    return app


def _run(app, body, ping=0.05, commit=0.2):
    mw = vllm_keepalive.KeepAliveMiddleware(app, ping_interval=ping,
                                            json_commit_after=commit)
    out, send = _collect()
    asyncio.run(mw(_scope(), _receive_for(body), send))
    return out


def test_body_peek():
    assert vllm_keepalive._body_requests_stream(b'{"stream": true}') is True
    assert vllm_keepalive._body_requests_stream(b'{"stream": false}') is False
    assert vllm_keepalive._body_requests_stream(b"not json") is False
    assert vllm_keepalive._body_requests_stream(b"[1,2]") is False
    pad = b"a" * (vllm_keepalive._MAX_PEEK_BYTES + 1)
    big = b'{"stream": true, "x": "' + pad + b'"}'
    assert vllm_keepalive._body_requests_stream(big) is False


def test_stream_request_early_commits_as_sse():
    out = _run(_slow_app(1.5, ctype=b"text/event-stream", body=b"data: x\n\n"),
               json.dumps({"stream": True}).encode())
    start = out[0]
    assert start["type"] == "http.response.start"
    assert dict(start["headers"])[b"content-type"] == b"text/event-stream"
    assert dict(start["headers"])[b"x-vllm-keepalive"] == b"early-commit"
    bodies = [m["body"] for m in out[1:] if m["type"] == "http.response.body"]
    assert bodies[0] == vllm_keepalive._SSE_PING
    assert bodies[-1] == b"data: x\n\n"
    # the app's own response.start was swallowed (already committed)
    assert sum(m["type"] == "http.response.start" for m in out) == 1


def test_json_request_early_commits_as_json():
    out = _run(_slow_app(1.5), json.dumps({"stream": False}).encode())
    assert dict(out[0]["headers"])[b"content-type"] == b"application/json"
    bodies = [m["body"] for m in out[1:] if m["type"] == "http.response.body"]
    assert bodies[0] == vllm_keepalive._JSON_PING
    assert bodies[-1] == b'{"ok":true}'


def test_fast_response_is_untouched():
    out = _run(_slow_app(0.0), json.dumps({"stream": True}).encode())
    assert [m["type"] for m in out] == ["http.response.start", "http.response.body"]
    assert dict(out[0]["headers"])[b"content-type"] == b"application/json"


def test_json_error_after_sse_commit_is_wrapped_as_sse_event():
    out = _run(_slow_app(1.5, body=b'{"error":"boom"}', status=500),
               json.dumps({"stream": True}).encode())
    assert dict(out[0]["headers"])[b"content-type"] == b"text/event-stream"
    bodies = [m["body"] for m in out[1:] if m["type"] == "http.response.body"]
    assert bodies[-1] == b'data: {"error":"boom"}\n\n'


def test_idle_stream_gets_comment_pings():
    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})
        await asyncio.sleep(2.5)  # the ping loop polls once per second
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n",
                    "more_body": False})

    out = _run(app, json.dumps({"stream": True}).encode(), ping=0.3, commit=60)
    pings = [m for m in out if m.get("body") == vllm_keepalive._SSE_PING]
    assert len(pings) >= 2


def test_non_v1_paths_bypass():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    mw = vllm_keepalive.KeepAliveMiddleware(app)
    out, send = _collect()
    asyncio.run(mw(_scope("/health"), _receive_for(b""), send))
    assert out[0]["type"] == "http.response.start"


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
def test_v1_paths_are_wrapped(path):
    mw = vllm_keepalive.KeepAliveMiddleware(_slow_app(0.0))
    out, send = _collect()
    asyncio.run(mw(_scope(path), _receive_for(b"{}"), send))
    assert out[-1]["body"] == b'{"ok":true}'
