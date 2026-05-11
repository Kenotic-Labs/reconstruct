"""
Kenotic MCP server — HTTP / Streamable-HTTP transport.

Implements the MCP Streamable HTTP transport (spec 2025-03-26):

  POST /mcp       — JSON-RPC request/batch. Returns JSON or SSE based on Accept.
  GET  /mcp       — SSE channel for server-initiated notifications (heartbeat).
  DELETE /mcp     — terminate a session.
  GET  /healthz   — unauthenticated liveness probe.

Auth is a static Bearer token provided via --token or KENOTIC_MCP_TOKEN env.
Session id is minted on `initialize` and returned in the `Mcp-Session-Id`
response header. All subsequent /mcp requests must echo it back.

Run:
    python -m mcp.http_server --token $KENOTIC_MCP_TOKEN \
        --host 127.0.0.1 --port 7130
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware

from mcp.tools import DB_PATH, TOOLS, ToolError, dispatch


log = logging.getLogger("kenotic.mcp.http")


# ── Module-level auth / session state ────────────────────────────

# The server is a singleton per process; stash auth token here so
# FastAPI dependencies can read it without threading through the app.
_AUTH_TOKEN: Optional[str] = None

# session_id -> {"created": monotonic_ts, "last_seen": monotonic_ts}
_SESSIONS: Dict[str, Dict[str, Any]] = {}

# Idle expiry for a minted session. Without this, _SESSIONS grows
# unboundedly over long uptimes (one entry per client reconnect).
_SESSION_IDLE_TTL_SECONDS = 24 * 60 * 60  # 24h

# Hard cap as a belt-and-braces bound against resource exhaustion.
_SESSION_MAX_ENTRIES = 1024


def _prune_sessions(now: Optional[float] = None) -> None:
    """Drop sessions idle longer than TTL and enforce the hard cap."""
    now = now if now is not None else time.monotonic()
    stale = [sid for sid, s in _SESSIONS.items()
             if now - s.get("last_seen", s.get("created", now)) > _SESSION_IDLE_TTL_SECONDS]
    for sid in stale:
        _SESSIONS.pop(sid, None)
    if len(_SESSIONS) > _SESSION_MAX_ENTRIES:
        # Evict oldest by last_seen until under cap.
        ordered = sorted(_SESSIONS.items(),
                         key=lambda kv: kv[1].get("last_seen", 0))
        for sid, _ in ordered[: len(_SESSIONS) - _SESSION_MAX_ENTRIES]:
            _SESSIONS.pop(sid, None)


def _touch_session(sid: str) -> None:
    s = _SESSIONS.get(sid)
    if s is not None:
        s["last_seen"] = time.monotonic()

# Origins the browser will send that we accept. Native MCP hosts
# typically omit Origin entirely, which we permit.
_ORIGIN_ALLOW_PREFIXES = (
    "http://127.0.0.1",
    "http://localhost",
)
_ORIGIN_ALLOW_EXACT = {"null"}


# ── Helpers ──────────────────────────────────────────────────────

def _envelope_result(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _envelope_error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def _is_notification(msg: Dict[str, Any]) -> bool:
    return "id" not in msg


def _handle_one(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Execute a single JSON-RPC message. Returns the response envelope,
    or None for notifications."""
    req_id = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params", {}) or {}
    try:
        result = dispatch(method, params)
    except ToolError as te:
        if _is_notification(msg):
            return None
        return _envelope_error(req_id, te.code, te.message)
    except Exception:
        # Log the full traceback server-side; return an opaque error to
        # the client. Exception strings can leak file paths, library
        # internals, or caller data — unsafe once exposed via Funnel.
        log.exception("mcp.dispatch failed method=%s", method)
        if _is_notification(msg):
            return None
        return _envelope_error(req_id, -32000, "Internal error")

    if _is_notification(msg):
        return None
    return _envelope_result(req_id, result if result is not None else {})


# ── Auth dependency ──────────────────────────────────────────────

def require_bearer(authorization: Optional[str] = Header(default=None)) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "missing_bearer"},
        )
    supplied = authorization.split(" ", 1)[1].strip()
    expected = _AUTH_TOKEN or ""
    # Constant-time compare
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_bearer"},
        )


def _check_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is None:
        return  # native apps — permitted
    if origin in _ORIGIN_ALLOW_EXACT:
        return
    for prefix in _ORIGIN_ALLOW_PREFIXES:
        if origin.startswith(prefix):
            return
    raise HTTPException(
        status_code=400,
        detail={"error": "origin_rejected", "origin": origin},
    )


# ── Access log middleware (redacts Authorization) ────────────────

class RedactedAccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # method + path + status only. Never log headers.
        log.info(
            "%s %s -> %s",
            request.method,
            request.url.path,
            response.status_code,
        )
        return response


# ── App factory ──────────────────────────────────────────────────

def build_app(token: str) -> FastAPI:
    global _AUTH_TOKEN
    _AUTH_TOKEN = token

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # Pre-load engines so the first tool call isn't multi-second cold.
        try:
            from sdk import Kenotic
            k = Kenotic(user_id=0, db_path=DB_PATH)
            k._engines()
            log.info("[kenotic-mcp-http] engines pre-warmed")
        except Exception as e:
            log.warning("[kenotic-mcp-http] warmup skipped: %s", e)
        yield

    app = FastAPI(title="Kenotic MCP (HTTP)", version="0.1.0", lifespan=_lifespan)
    app.add_middleware(RedactedAccessLogMiddleware)

    # ── Exception handler to emit JSON body matching spec ──

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):
        body = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=body)

    # ── /healthz (unauthenticated) ──────────────────────────

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "tools": len(TOOLS)}

    # ── POST /mcp ───────────────────────────────────────────

    @app.post("/mcp")
    async def mcp_post(
        request: Request,
        _auth: None = Depends(require_bearer),
        accept: Optional[str] = Header(default=None),
        mcp_session_id: Optional[str] = Header(default=None, alias="Mcp-Session-Id"),
    ):
        _check_origin(request)

        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail={"error": "empty_body"})
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail={"error": "parse_error", "detail": str(e)})

        is_batch = isinstance(payload, list)
        messages: List[Dict[str, Any]] = payload if is_batch else [payload]

        # Detect initialize to mint a session id
        is_initialize = any(
            isinstance(m, dict) and m.get("method") == "initialize" for m in messages
        )

        # Enforce session id on every non-initialize request, unconditionally.
        # Without this, a fresh process with empty _SESSIONS would accept
        # tools/call without an initialize handshake — spec violation and
        # lets callers skip session binding entirely.
        _prune_sessions()
        if not is_initialize:
            if not mcp_session_id or mcp_session_id not in _SESSIONS:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "invalid_session"},
                )
            _touch_session(mcp_session_id)

        # Run each message. Tool handlers are sync/CPU-bound (spaCy), so run
        # in the default threadpool to keep the event loop responsive.
        responses: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                responses.append(_envelope_error(None, -32600, "Invalid Request"))
                continue
            resp = await run_in_threadpool(_handle_one, msg)
            if resp is not None:
                responses.append(resp)

        # Pure notification batch → 202 Accepted, no body
        if not responses:
            return Response(status_code=202)

        out: Union[Dict[str, Any], List[Dict[str, Any]]] = responses if is_batch else responses[0]

        # Mint / expose session id on initialize
        headers: Dict[str, str] = {}
        if is_initialize:
            sid = uuid.uuid4().hex
            now = time.monotonic()
            _SESSIONS[sid] = {"created": now, "last_seen": now}
            headers["Mcp-Session-Id"] = sid

        wants_sse = accept is not None and "text/event-stream" in accept.lower()
        if wants_sse:
            async def _gen():
                yield {"event": "message", "data": json.dumps(out)}
            return EventSourceResponse(_gen(), headers=headers)

        return JSONResponse(content=out, headers=headers)

    # ── GET /mcp — SSE heartbeat for server-initiated notifs ─

    @app.get("/mcp")
    async def mcp_get(
        request: Request,
        _auth: None = Depends(require_bearer),
        mcp_session_id: Optional[str] = Header(default=None, alias="Mcp-Session-Id"),
    ):
        _check_origin(request)
        if not mcp_session_id or mcp_session_id not in _SESSIONS:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_session"},
            )

        async def _heartbeat():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    # SSE comment frame — keep-alive, invisible to clients
                    yield {"event": "ping", "data": ""}
                    await asyncio.sleep(15)
            except asyncio.CancelledError:
                return

        return EventSourceResponse(_heartbeat())

    # ── DELETE /mcp — terminate session ─────────────────────

    @app.delete("/mcp")
    async def mcp_delete(
        request: Request,
        _auth: None = Depends(require_bearer),
        mcp_session_id: Optional[str] = Header(default=None, alias="Mcp-Session-Id"),
    ):
        _check_origin(request)
        if not mcp_session_id or mcp_session_id not in _SESSIONS:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_session"},
            )
        _SESSIONS.pop(mcp_session_id, None)
        return Response(status_code=204)

    return app


# ── CLI ──────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m mcp.http_server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=7130, type=int)
    p.add_argument(
        "--token",
        default=None,
        help="Static bearer token. Falls back to KENOTIC_MCP_TOKEN env.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    token = args.token or os.environ.get("KENOTIC_MCP_TOKEN")
    if not token:
        print(
            "error: no bearer token provided. Use --token or set KENOTIC_MCP_TOKEN.",
            file=sys.stderr,
        )
        return 2

    import uvicorn
    app = build_app(token)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        loop="asyncio",
        access_log=False,  # our middleware handles access logs, without headers
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
