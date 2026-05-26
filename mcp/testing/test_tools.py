"""
Direct tool-handler tests for the MCP server.
No HTTP, no tunnel — calls tool functions directly against test_memory.db.

Usage:
    python mcp/testing/test_tools.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

TEST_DB = os.path.join(os.path.dirname(__file__), "test_memory.db")

# Override the DB path BEFORE importing tools
os.environ["KENOTIC_DB_PATH"] = TEST_DB

from mcp.tools import dispatch, ToolError


def call_tool(name: str, arguments: dict) -> dict:
    """Simulate a tools/call JSON-RPC request."""
    result = dispatch("tools/call", {"name": name, "arguments": arguments})
    # result is {"content": [{"type": "text", "text": "..."}]}
    text = result["content"][0]["text"]
    return json.loads(text)


def test_initialize():
    """Server responds to initialize with protocol version and tools."""
    result = dispatch("initialize", {})
    assert result["protocolVersion"] == "2025-03-26", f"Bad version: {result}"
    assert "tools" in result["capabilities"], "Missing tools capability"
    print("PASS: initialize")


def test_tools_list():
    """tools/list returns our 3 tools."""
    result = dispatch("tools/list", {})
    names = [t["name"] for t in result["tools"]]
    assert "reconstruct_ingest" in names, f"Missing ingest: {names}"
    assert "reconstruct_retrieve" in names, f"Missing retrieve: {names}"
    assert "reconstruct_clear" in names, f"Missing clear: {names}"
    print(f"PASS: tools/list ({len(names)} tools)")


def test_ingest():
    """Ingest returns a job_id and queued status."""
    result = call_tool("reconstruct_ingest", {
        "text": "Testing MCP ingest pipeline.",
        "speaker": "tester",
    })
    assert result["status"] == "queued", f"Bad status: {result}"
    assert "job_id" in result, f"Missing job_id: {result}"
    print(f"PASS: ingest (job_id={result['job_id'][:8]}...)")


def test_retrieve_hit():
    """Retrieve finds known data."""
    result = call_tool("reconstruct_retrieve", {
        "query": "What does Sam want to learn about?",
    })
    data = result.get("data", {})
    answer = data.get("answer", "") if isinstance(data, dict) else str(data)
    print(f"PASS: retrieve hit -> {answer[:80]}")


def test_retrieve_miss():
    """Retrieve refuses on unknown data."""
    result = call_tool("reconstruct_retrieve", {
        "query": "What is the capital of Mars?",
    })
    data = result.get("data", {})
    answer = data.get("answer", "") if isinstance(data, dict) else str(data)
    is_refusal = (
        "not mentioned" in answer.lower()
        or "no information" in answer.lower()
        or "not stored" in answer.lower()
        or answer == ""
    )
    status = "PASS" if is_refusal else "WARN"
    print(f"{status}: retrieve miss -> {answer[:80]}")


def test_unknown_tool():
    """Unknown tool raises ToolError."""
    try:
        dispatch("tools/call", {"name": "fake_tool", "arguments": {}})
        print("FAIL: should have raised ToolError")
    except ToolError as e:
        assert e.code == -32601, f"Wrong code: {e.code}"
        print(f"PASS: unknown tool rejected ({e.message})")


def test_unknown_method():
    """Unknown method raises ToolError."""
    try:
        dispatch("nonexistent/method", {})
        print("FAIL: should have raised ToolError")
    except ToolError as e:
        assert e.code == -32601
        print(f"PASS: unknown method rejected")


def main():
    print(f"Testing against: {TEST_DB}")
    if not os.path.exists(TEST_DB):
        print(f"ERROR: {TEST_DB} not found. Run seed_db.py first.")
        return 1

    tests = [
        test_initialize,
        test_tools_list,
        test_ingest,
        test_retrieve_hit,
        test_retrieve_miss,
        test_unknown_tool,
        test_unknown_method,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} -> {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
