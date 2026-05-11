"""
kenotic-mcp — the continuity layer as an MCP server.

Exposes the SDK as MCP tools for host integrations (Claude Desktop,
Claude Code, any MCP-compatible client). Each tool is a thin JSON-RPC
wrapper over sdk.Kenotic.

Tools registered:
  memory.ingest       — "Add this text to continuity memory"
  memory.retrieve     — "Answer a question from continuity memory"
  memory.reconstruct  — "Produce a structured Situation for a query"

Run with:
    py -3.10 -m mcp.server
"""
__version__ = "0.1.0"
