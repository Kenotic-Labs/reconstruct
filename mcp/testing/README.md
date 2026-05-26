# MCP Testing

Isolated testing for the Reconstruct MCP server. Uses a local copy of the DB
so tests never touch `~/.kenotic/memory.db`.

## Files

| File | Purpose |
|------|---------|
| `test_memory.db` | Snapshot of production DB (gitignored) |
| `test_tools.py` | Direct tool-handler tests (no HTTP, no tunnel) |
| `test_http.py` | HTTP transport tests against a local server |
| `test_tunnel.py` | End-to-end tunnel test (cloudflared) |
| `seed_db.py` | Seed a fresh test DB with known data |

## Quick start

```bash
# Run tool-level tests (no server needed)
python mcp/testing/test_tools.py

# Run HTTP-level tests (starts/stops server automatically)
python mcp/testing/test_http.py

# Seed a fresh DB with known test data
python mcp/testing/seed_db.py
```

## Architecture

```
ChatGPT Desktop
      |
      | HTTPS (cloudflared tunnel)
      v
mcp/http_server.py   (FastAPI, /mcp endpoint)
      |
      | JSON-RPC dispatch
      v
mcp/tools.py         (tool_ingest, tool_reconstruct, tool_clear)
      |
      | Kenotic SDK
      v
sdk.py -> app/engines/*  (grammar, memory, reconstruction, temporal)
      |
      v
~/.kenotic/memory.db  (production)
mcp/testing/test_memory.db  (tests)
```
