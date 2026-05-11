"""
Auth / transport tests for the MCP HTTP/Streamable-HTTP server.

These tests do NOT exercise real tool handlers — they monkeypatch
`mcp.tools.dispatch` where needed so we don't pay T5 + MiniLM load cost.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from mcp import http_server as hs
from mcp import tools as mcp_tools


TEST_TOKEN = "test-token-abc"


@pytest.fixture
def client(monkeypatch):
    # Reset session state between tests
    hs._SESSIONS.clear()
    # Build app without running uvicorn, without warmup side-effects:
    # TestClient triggers startup events, so stub the warmup to be a no-op
    # by preventing SDK import.
    app = hs.build_app(TEST_TOKEN)
    with TestClient(app) as c:
        yield c


def _auth():
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


def _initialize(client) -> str:
    """Run the MCP initialize handshake and return the minted session id."""
    r = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
        },
        headers={**_auth(), "Accept": "application/json"},
    )
    assert r.status_code == 200, r.text
    sid = r.headers.get("Mcp-Session-Id")
    assert sid, "initialize must mint a Mcp-Session-Id"
    return sid


def _session(client):
    return {"Mcp-Session-Id": _initialize(client)}


def test_missing_auth_returns_401(client):
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert r.status_code == 401
    assert r.json() == {"error": "missing_bearer"}


def test_bad_bearer_returns_401(client):
    r = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Authorization": "Bearer nope"},
    )
    assert r.status_code == 401
    assert r.json() == {"error": "invalid_bearer"}


def test_good_bearer_tools_list_returns_6(client, monkeypatch):
    sid = _session(client)
    r = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={**_auth(), "Accept": "application/json", **sid},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 1
    assert "result" in body
    tools = body["result"]["tools"]
    assert len(tools) == 6
    names = {t["name"] for t in tools}
    assert names == {
        "memory.ingest", "memory.retrieve", "memory.reconstruct",
        "memory.forget", "memory.show", "memory.trace",
    }


def test_healthz_unauthenticated_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "tools": 6}


def test_notification_returns_202(client):
    # JSON-RPC notification: no "id" field. Sent by the client after
    # initialize, so it must carry the minted session id.
    sid = _session(client)
    r = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers={**_auth(), "Accept": "application/json", **sid},
    )
    assert r.status_code == 202
    assert r.content == b""


def test_missing_session_returns_400(client):
    r = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={**_auth(), "Accept": "application/json"},
    )
    assert r.status_code == 400
    assert r.json().get("error") == "invalid_session"


def test_get_mcp_requires_session(client):
    r = client.get("/mcp", headers={**_auth()})
    assert r.status_code == 400


def test_delete_mcp_requires_session(client):
    r = client.delete("/mcp", headers={**_auth()})
    assert r.status_code == 400


def test_delete_mcp_with_session_returns_204(client):
    sid = _session(client)
    r = client.delete("/mcp", headers={**_auth(), **sid})
    assert r.status_code == 204


def test_session_idle_ttl_evicts_stale(client, monkeypatch):
    """Sessions older than _SESSION_IDLE_TTL_SECONDS must be pruned on
    the next request. Without this, _SESSIONS grows unboundedly over a
    long-running Funnel-exposed process."""
    import time as _time
    sid = _initialize(client)
    # Backdate the session past its TTL.
    hs._SESSIONS[sid]["last_seen"] = _time.monotonic() - (hs._SESSION_IDLE_TTL_SECONDS + 60)
    hs._SESSIONS[sid]["created"] = hs._SESSIONS[sid]["last_seen"]

    # Next POST with that session id must be rejected (pruned).
    r = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={**_auth(), "Accept": "application/json", "Mcp-Session-Id": sid},
    )
    assert r.status_code == 400
    assert sid not in hs._SESSIONS


def test_internal_errors_are_redacted(client, monkeypatch):
    """Exception strings from tool handlers must not leak to the
    client — they can contain file paths, caller args, or library
    internals. Once exposed via Funnel, that's an info-disclosure vuln."""
    from mcp import tools as mcp_tools

    def _boom(method, params):
        raise RuntimeError("SECRET /home/sam/kenotic.db path leak")

    monkeypatch.setattr(hs, "dispatch", _boom)
    sid = _session(client)
    r = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 7, "method": "tools/list"},
        headers={**_auth(), "Accept": "application/json", **sid},
    )
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == -32000
    # Redacted: the raw exception string must NOT appear in the response.
    assert "SECRET" not in json.dumps(body)
    assert "/home/sam" not in json.dumps(body)


def test_origin_rebind_rejected(client):
    r = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={
            **_auth(),
            "Accept": "application/json",
            "Origin": "https://evil.example.com",
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert body.get("error") == "origin_rejected"
