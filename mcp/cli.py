"""
Reconstruct CLI — the product entry point.

    reconstruct serve           Start the MCP server (stdio transport)
    reconstruct serve --http    Start the MCP server (HTTP transport)
    reconstruct version         Print version info
"""
from __future__ import annotations

import argparse
import sys


def _cmd_serve(args: argparse.Namespace) -> int:
    if args.http:
        from mcp.http_server import main as http_main
        argv = ["--host", args.host, "--port", str(args.port)]
        if args.token:
            argv += ["--token", args.token]
        return http_main(argv) or 0
    else:
        from mcp.server import main as stdio_main
        stdio_main()
        return 0


def _cmd_version(args: argparse.Namespace) -> int:
    from mcp import __version__
    print(f"Reconstruct from Kenotic v{__version__}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reconstruct",
        description="Reconstruct from Kenotic — the continuity layer for AI.",
    )
    sub = parser.add_subparsers(dest="command")

    # reconstruct serve
    serve = sub.add_parser("serve", help="Start the MCP server")
    serve.add_argument("--http", action="store_true",
                       help="Use HTTP/Streamable-HTTP transport instead of stdio")
    serve.add_argument("--host", default="127.0.0.1",
                       help="HTTP bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=7130,
                       help="HTTP port (default: 7130)")
    serve.add_argument("--token", default=None,
                       help="Bearer token for HTTP transport (or set KENOTIC_MCP_TOKEN)")

    # reconstruct version
    sub.add_parser("version", help="Print version info")

    parsed = parser.parse_args(argv)

    if parsed.command == "serve":
        return _cmd_serve(parsed)
    elif parsed.command == "version":
        return _cmd_version(parsed)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
