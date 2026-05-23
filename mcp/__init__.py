"""
Reconstruct from Kenotic — the continuity layer for AI.

Cross-host memory and reconstruction via MCP. Any AI tool that
supports MCP connects to one local server. Understanding carries
across hosts, sessions, and time.

    reconstruct serve           stdio transport (Claude Desktop, Code)
    reconstruct serve --http    HTTP transport (web, custom apps)

Tools:
  reconstruct_ingest    — store text into continuity memory (every turn)
  reconstruct_retrieve  — answer a question from stored memory
  reconstruct_clear     — clear memories by entity or everything
"""
__version__ = "0.1.0"
