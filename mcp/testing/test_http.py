"""
HTTP transport tests for the MCP server.
Starts the server on a test port, runs requests, shuts down.

Usage:
    python mcp/testing/test_http.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

TEST_DB = os.path.join(os.path.dirname(__file__), "test_memory.db")
TEST_PORT = 7131  # different from production 7130
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"


def post_json(path: str, body: dict, headers: dict = None) -> tuple:
    """POST JSON and return (status_code, response_dict, response_headers)."""
    data = json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=hdrs, method="POST")
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


def test_healthz():
    """GET /healthz returns 200."""
    resp = urllib.request.urlopen(f"{BASE_URL}/healthz")
    data = json.loads(resp.read())
    assert data["status"] == "ok", f"Bad healthz: {data}"
    print(f"PASS: healthz (tools={data['tools']})")


def test_initialize():
    """POST /mcp with initialize returns session id."""
    status, data, headers = post_json("/mcp", {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {},
    })
    assert status == 200, f"Bad status: {status}"
    assert "Mcp-Session-Id" in headers or "mcp-session-id" in headers, f"No session id: {headers.keys()}"
    sid = headers.get("Mcp-Session-Id") or headers.get("mcp-session-id")
    result = data.get("result", {})
    assert result.get("protocolVersion") == "2025-03-26"
    print(f"PASS: initialize (session={sid[:8]}...)")
    return sid


def test_tools_list(session_id: str):
    """POST /mcp with tools/list."""
    status, data, _ = post_json("/mcp", {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }, headers={"Mcp-Session-Id": session_id})
    assert status == 200
    tools = data["result"]["tools"]
    names = [t["name"] for t in tools]
    assert "reconstruct_ingest" in names
    print(f"PASS: tools/list ({len(names)} tools)")


def test_tool_call(session_id: str):
    """POST /mcp with tools/call reconstruct_retrieve."""
    status, data, _ = post_json("/mcp", {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "reconstruct_retrieve",
            "arguments": {"query": "What does Sam want to learn about?"},
        },
    }, headers={"Mcp-Session-Id": session_id})
    assert status == 200, f"Bad status: {status}, data: {data}"
    content = data.get("result", {}).get("content", [])
    assert len(content) > 0, f"Empty content: {data}"
    text = content[0].get("text", "")
    print(f"PASS: tools/call retrieve -> {text[:80]}...")


def test_no_session_rejected():
    """Request without session id after initialize is rejected."""
    status, data, _ = post_json("/mcp", {
        "jsonrpc": "2.0",
        "id": 99,
        "method": "tools/list",
        "params": {},
    })
    assert status == 400, f"Expected 400 got {status}"
    print(f"PASS: no-session rejected (400)")


def main():
    print(f"Testing HTTP transport against: {TEST_DB}")
    if not os.path.exists(TEST_DB):
        print(f"ERROR: {TEST_DB} not found. Run seed_db.py first.")
        return 1

    # Start server as subprocess
    env = os.environ.copy()
    env["KENOTIC_DB_PATH"] = TEST_DB
    server = subprocess.Popen(
        [sys.executable, "-m", "mcp.http_server", "--port", str(TEST_PORT)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for startup
    for i in range(15):
        try:
            urllib.request.urlopen(f"{BASE_URL}/healthz")
            break
        except Exception:
            time.sleep(1)
    else:
        print("ERROR: Server did not start in 15s")
        server.kill()
        return 1

    try:
        passed = 0
        failed = 0

        tests_no_session = [test_healthz, test_no_session_rejected]
        for t in tests_no_session:
            try:
                t()
                passed += 1
            except Exception as e:
                print(f"FAIL: {t.__name__} -> {e}")
                failed += 1

        # Initialize gets session, then pass it to subsequent tests
        try:
            sid = test_initialize()
            passed += 1
        except Exception as e:
            print(f"FAIL: test_initialize -> {e}")
            server.kill()
            return 1

        tests_with_session = [test_tools_list, test_tool_call]
        for t in tests_with_session:
            try:
                t(sid)
                passed += 1
            except Exception as e:
                print(f"FAIL: {t.__name__} -> {e}")
                failed += 1

        print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
        return 1 if failed else 0

    finally:
        server.kill()
        server.wait()
        print("Server stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
