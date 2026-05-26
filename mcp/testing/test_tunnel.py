"""
End-to-end tunnel test — verifies the full path ChatGPT Desktop uses.
Starts HTTP server + cloudflared tunnel, runs MCP handshake through HTTPS.

Usage:
    python mcp/testing/test_tunnel.py
"""
import json
import os
import re
import subprocess
import ssl
import sys
import time
import urllib.request
import urllib.error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

TEST_DB = os.path.join(os.path.dirname(__file__), "test_memory.db")
TEST_PORT = 7132
CLOUDFLARED = r"C:\Program Files (x86)\cloudflared\cloudflared.exe"

# SSL context that accepts any cert (tunnel terminates TLS at Cloudflare)
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def post_json(base_url: str, path: str, body: dict, headers: dict = None) -> tuple:
    data = json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=hdrs, method="POST")
    try:
        resp = urllib.request.urlopen(req, context=_SSL_CTX)
        return resp.status, json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


def main():
    print(f"Testing full tunnel path against: {TEST_DB}")
    if not os.path.exists(TEST_DB):
        print(f"ERROR: {TEST_DB} not found. Run seed_db.py first.")
        return 1
    if not os.path.exists(CLOUDFLARED):
        print(f"ERROR: cloudflared not found at {CLOUDFLARED}")
        return 1

    # 1. Start HTTP server
    env = os.environ.copy()
    env["KENOTIC_DB_PATH"] = TEST_DB
    server = subprocess.Popen(
        [sys.executable, "-m", "mcp.http_server", "--port", str(TEST_PORT)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    for i in range(15):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{TEST_PORT}/healthz")
            break
        except Exception:
            time.sleep(1)
    else:
        print("ERROR: Server did not start")
        server.kill()
        return 1
    print(f"Server running on port {TEST_PORT}")

    # 2. Start cloudflared tunnel
    tunnel = subprocess.Popen(
        [CLOUDFLARED, "tunnel", "--url", f"http://localhost:{TEST_PORT}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    tunnel_url = None
    deadline = time.time() + 20
    while time.time() < deadline:
        line = tunnel.stdout.readline().decode(errors="replace")
        match = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", line)
        if match:
            tunnel_url = match.group(1)
            break
    if not tunnel_url:
        print("ERROR: Could not get tunnel URL")
        tunnel.kill()
        server.kill()
        return 1
    print(f"Tunnel: {tunnel_url}")

    try:
        # 3. Test through tunnel
        # Healthz
        resp = urllib.request.urlopen(f"{tunnel_url}/healthz", context=_SSL_CTX)
        data = json.loads(resp.read())
        assert data["status"] == "ok"
        print(f"PASS: healthz through tunnel")

        # Initialize
        status, data, headers = post_json(tunnel_url, "/mcp", {
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        assert status == 200
        sid = headers.get("Mcp-Session-Id") or headers.get("mcp-session-id")
        assert sid, f"No session id: {headers.keys()}"
        print(f"PASS: initialize through tunnel (session={sid[:8]}...)")

        # Retrieve
        status, data, _ = post_json(tunnel_url, "/mcp", {
            "jsonrpc": "2.0", "id": 2,
            "method": "tools/call",
            "params": {
                "name": "reconstruct_retrieve",
                "arguments": {"query": "What does Sam want to learn about?"},
            },
        }, headers={"Mcp-Session-Id": sid})
        assert status == 200
        content = data.get("result", {}).get("content", [{}])
        text = content[0].get("text", "") if content else ""
        print(f"PASS: retrieve through tunnel -> {text[:80]}...")

        print(f"\nTunnel URL for ChatGPT Desktop: {tunnel_url}/mcp")
        print("Press Ctrl+C to stop...")
        # Keep tunnel alive for manual testing
        tunnel.wait()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tunnel.kill()
        tunnel.wait()
        server.kill()
        server.wait()
        print("Stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
