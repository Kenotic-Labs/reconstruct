"""
Reconstruct MCP server — stdio JSON-RPC transport.

This module is a thin stdio loop over `mcp.tools.dispatch`. All tool
handlers and the method-dispatch logic live in `mcp.tools` so the
HTTP/Streamable-HTTP transport (`mcp.http_server`) can share them.

Usage — add to Claude Desktop config (claude_desktop_config.json):

    {
      "mcpServers": {
        "reconstruct": {
          "command": "reconstruct",
          "args": ["serve"]
        }
      }
    }
"""
from __future__ import annotations

import json
import sys

from mcp.tools import DB_PATH, TOOLS, ToolError, dispatch, tool_ingest


def _send(obj):
    # Write as bytes to the raw stdout buffer — bypasses Python's
    # text-mode buffering which can deadlock on Windows pipes.
    data = json.dumps(obj).encode("utf-8") + b"\n"
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _envelope_result(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _envelope_error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def _sync_ingest(args):
    """Run ingest synchronously — no async queue, no daemon thread.

    On Windows, pipe-based stdin reads hold the GIL at the C level,
    starving ALL daemon threads. The async ingest worker (which runs in
    a daemon thread) never gets to process queued jobs.

    For stdio transport, we bypass the async queue entirely and run
    ingest inline. The MCP host waits a few seconds — acceptable for
    a synchronous pipe protocol. The HTTP transport still uses the
    async path (no GIL starvation with asyncio).
    """
    from sdk import Kenotic
    k = Kenotic(user_id=0, db_path=DB_PATH)
    n = k.ingest(
        text=args["text"],
        source_timestamp=args.get("source_timestamp"),
        speaker=args.get("speaker"),
        confidence=args.get("confidence", 0.9),
        model_response=args.get("model_response"),
        llm_id=args.get("llm_id"),
    )
    return {"edges_stored": n, "status": "done"}


def main():
    print(f"[reconstruct] server started, db={DB_PATH}", file=sys.stderr)
    print(f"[reconstruct] registered tools: {list(TOOLS.keys())}", file=sys.stderr)

    # Use readline() — NOT `for line in sys.stdin:`.
    # The iterator form on Windows reads ahead and buffers internally,
    # causing subsequent lines to be consumed before prior responses
    # are flushed. readline() processes one line at a time.
    while True:
        line = sys.stdin.readline()
        if not line:
            break
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

        # Intercept ingest calls — run synchronously instead of async.
        # On Windows, the async daemon thread is starved by pipe stdin.
        if (method == "tools/call"
                and params.get("name") == "reconstruct.ingest"):
            try:
                tool_args = params.get("arguments", {}) or {}
                result_data = _sync_ingest(tool_args)
                result = {
                    "content": [{"type": "text", "text": json.dumps(result_data)}],
                }
                if not is_notification:
                    _send(_envelope_result(req_id, result))
            except Exception as e:
                if not is_notification:
                    _send(_envelope_error(req_id, -32000, f"Ingest error: {e}"))
            continue

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

        if is_notification:
            continue
        _send(_envelope_result(req_id, result if result is not None else {}))


if __name__ == "__main__":
    main()
