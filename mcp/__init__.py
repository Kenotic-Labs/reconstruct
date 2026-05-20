"""
Reconstruct from Kenotic — the continuity layer for AI.

Cross-host memory and reconstruction via MCP. Any AI tool that
supports MCP connects to one local server. Understanding carries
across hosts, sessions, and time.

    reconstruct serve           stdio transport (Claude Desktop, Code)
    reconstruct serve --http    HTTP transport (web, custom apps)

Tools registered:
  memory.ingest       — store text into continuity memory
  memory.retrieve     — answer a question from memory
  memory.reconstruct  — reconstruct a situation from memory
  memory.process      — unified entry: classify intent, route internally
"""
__version__ = "0.1.0"
