"""
Export the OpenAPI spec for the Kenotic REST API.

The output JSON can be pasted directly into:
  - ChatGPT Custom GPT Actions
  - Gemini tool schemas
  - Any platform that consumes OpenAPI definitions

Usage:
    python scripts/export_openapi.py                     # writes openapi.json
    python scripts/export_openapi.py --url https://x.com # override server URL
    python scripts/export_openapi.py --stdout             # print to stdout
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:7130", help="Server base URL")
    p.add_argument("--stdout", action="store_true", help="Print to stdout instead of file")
    p.add_argument("-o", "--output", default="openapi.json", help="Output file path")
    args = p.parse_args()

    from mcp.http_server import build_app
    app = build_app(token="")
    spec = app.openapi()

    # Set the server URL for the target platform
    spec["servers"] = [{"url": args.url}]

    # Remove security schemes — GPT Actions configures auth separately
    spec.pop("securityDefinitions", None)
    if "components" in spec:
        spec["components"].pop("securitySchemes", None)

    out = json.dumps(spec, indent=2)

    if args.stdout:
        print(out)
    else:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"Wrote {len(out)} bytes to {args.output}", file=sys.stderr)
        print(f"Endpoints: {len([p for p in spec.get('paths', {})])}", file=sys.stderr)


if __name__ == "__main__":
    main()
