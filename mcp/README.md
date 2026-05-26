# Kenotic Continuity Memory -- MCP + REST API

Continuity memory for any AI client. One server exposes both MCP (stdio and HTTP transports) and a REST API, so Claude, ChatGPT, Cursor, and custom integrations all connect to the same memory.

---

## Quickstart

### Claude Desktop (stdio MCP)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "reconstruct": {
      "command": "reconstruct",
      "args": ["serve"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add reconstruct -- reconstruct serve
```

### Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "reconstruct": {
      "command": "reconstruct",
      "args": ["serve"]
    }
  }
}
```

### ChatGPT (Custom GPT Actions)

ChatGPT does not support MCP. Use the REST API instead:

1. Start the server with a token:
   ```bash
   python -m mcp.http_server --host 0.0.0.0 --port 7130 --token YOUR_SECRET
   ```
2. Export the OpenAPI spec:
   ```bash
   python scripts/export_openapi.py --url https://your-host:7130 --stdout
   ```
3. In the GPT editor, go to **Configure > Actions > Create new action**.
4. Paste the OpenAPI JSON output.
5. Under **Authentication**, select **API Key**, set the header to `Authorization`, and enter `Bearer YOUR_SECRET`.

### Any Model with Tool Calling

Point your HTTP client at `http://localhost:7130/api/v1`. All endpoints accept and return JSON. Pass `Authorization: Bearer YOUR_TOKEN` if auth is enabled.

---

## MCP Tools (3)

These are the tools exposed to MCP clients:

| Tool | Description |
|------|-------------|
| `reconstruct_ingest` | Store a message into continuity memory. Call on every turn. |
| `reconstruct_retrieve` | Answer a question from stored memory. Returns a verified answer or refuses. |
| `reconstruct_clear` | Clear memories by entity, or clear everything. |

---

## REST Endpoints (6)

All endpoints are mounted at `/api/v1`. Auth is required when a token is configured.

### POST /api/v1/store

Store a message into continuity memory.

```bash
curl -X POST http://localhost:7130/api/v1/store \
  -H "Content-Type: application/json" \
  -d '{"text": "I have a meeting with Alex on Friday at 2pm", "speaker": "Sam"}'
```

### POST /api/v1/retrieve

Answer a factual question from memory.

```bash
curl -X POST http://localhost:7130/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "When is my meeting with Alex?"}'
```

### POST /api/v1/reconstruct

Reconstruct the living state of a situation or person.

```bash
curl -X POST http://localhost:7130/api/v1/reconstruct \
  -H "Content-Type: application/json" \
  -d '{"query": "What is going on with my job search?"}'
```

### POST /api/v1/check_proactive

Get proactive insights -- upcoming events, unresolved situations, recurring patterns.

```bash
curl -X POST http://localhost:7130/api/v1/check_proactive
```

### POST /api/v1/forget

Delete memories by entity, time range, source, or triple ID.

```bash
curl -X POST http://localhost:7130/api/v1/forget \
  -H "Content-Type: application/json" \
  -d '{"by": "entity", "scope": "Alex"}'
```

### GET /api/v1/profile

Get the user's adaptation profile derived from stored memory.

```bash
curl http://localhost:7130/api/v1/profile
```

---

## Running the Server

```bash
# HTTP transport (REST + MCP on one server)
python -m mcp.http_server --host 127.0.0.1 --port 7130

# With auth
python -m mcp.http_server --token YOUR_SECRET

# With TLS
python -m mcp.http_server --token YOUR_SECRET --ssl-certfile cert.pem --ssl-keyfile key.pem

# stdio transport (for Claude Desktop / Claude Code)
reconstruct serve
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `7130` | Port |
| `--token` | none | Static bearer token (or set `KENOTIC_MCP_TOKEN` env) |
| `--ssl-certfile` | none | Path to SSL certificate PEM |
| `--ssl-keyfile` | none | Path to SSL private key PEM |

---

## Auth

Authentication uses a static bearer token. Set it with `--token` or the `KENOTIC_MCP_TOKEN` environment variable.

When no token is configured, auth is disabled. The server will force-bind to `127.0.0.1` if no token is set and a non-localhost host is requested, preventing unauthenticated network exposure.

When a token is configured, every request must include the header:

```
Authorization: Bearer YOUR_TOKEN
```

---

## OpenAPI Export

Generate the OpenAPI spec for use with ChatGPT Actions, Gemini, or any OpenAPI-compatible platform:

```bash
# Write to file
python scripts/export_openapi.py --url https://your-host:7130

# Print to stdout
python scripts/export_openapi.py --url https://your-host:7130 --stdout
```

The output is a standard OpenAPI 3.1 JSON document with 6 operations. Paste it directly into GPT Actions or any tool schema consumer.
