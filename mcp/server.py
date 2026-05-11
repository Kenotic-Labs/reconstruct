"""
Kenotic MCP server — stdio JSON-RPC transport.

This module is a thin stdio loop over `mcp.tools.dispatch`. All tool
handlers and the method-dispatch logic live in `mcp.tools` so the
HTTP/Streamable-HTTP transport (`mcp.http_server`) can share them.

Usage from an MCP host config (Claude Desktop mcp.json):

    {
      "mcpServers": {
        "kenotic": {
          "command": "py",
          "args": ["-3.10", "-m", "mcp.server"],
          "env": {"KENOTIC_DB_PATH": "C:/path/to/kenotic.db"}
        }
      }
    }
"""
from __future__ import annotations

import json
import sys

from mcp.tools import DB_PATH, TOOLS, ToolError, dispatch


def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _envelope_result(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _envelope_error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def main():
    print(f"[kenotic-mcp] server started, db={DB_PATH}", file=sys.stderr)
    print(f"[kenotic-mcp] registered tools: {list(TOOLS.keys())}", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _send(_envelope_error(None, -32700, f"Parse error: {e}"))
            continue

        req_id = req.get("id")
        is_notification = "id" not in req
        method = req.get("method", "")
        params = req.get("params", {}) or {}

        try:
            result = dispatch(method, params)
        except ToolError as te:
            if not is_notification:
                _send(_envelope_error(req_id, te.code, te.message))
            continue
        except Exception as e:  # pragma: no cover — defensive
            if not is_notification:
                _send(_envelope_error(req_id, -32000, f"Internal error: {e}"))
            continue

        # Notifications produce no response.
        if is_notification:
            continue
        # Valid method with None result (e.g. lifecycle notifications sent
        # as requests) — respond with empty object for JSON-RPC compliance.
        _send(_envelope_result(req_id, result if result is not None else {}))


if __name__ == "__main__":
    main()
