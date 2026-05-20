"""
Full stdio end-to-end integration test.

Pipes JSON-RPC requests to `reconstruct serve` via stdin, reads all
responses from stdout. Same protocol Claude Desktop uses.
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = str(Path(__file__).resolve().parent.parent)


def main():
    db_fd, db = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)

    requests = [
        # Handshake
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-03-26", "capabilities": {}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},

        # Day 1: ingest on "GPT"
        {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {
            "name": "reconstruct.ingest",
            "arguments": {"text": "I just got a new job at Google starting next month.", "speaker": "Sam"}}},
        {"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {
            "name": "reconstruct.ingest",
            "arguments": {"text": "I am feeling really nervous about the move to California.", "speaker": "Sam"}}},
        {"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": {
            "name": "reconstruct.ingest",
            "arguments": {"text": "My annual salary at Google is one hundred and eighty five thousand dollars.", "speaker": "Sam"}}},
        {"jsonrpc": "2.0", "id": 13, "method": "tools/call", "params": {
            "name": "reconstruct.ingest",
            "arguments": {"text": "My friend Joseph recently got promoted to senior engineer at his company.", "speaker": "Sam"}}},
        {"jsonrpc": "2.0", "id": 14, "method": "tools/call", "params": {
            "name": "reconstruct.ingest",
            "arguments": {"text": "My partner Glenda is helping me pack up the apartment for the move.", "speaker": "Sam"}}},
        {"jsonrpc": "2.0", "id": 15, "method": "tools/call", "params": {
            "name": "reconstruct.ingest",
            "arguments": {"text": "I recently started liking parrots because they are amazing birds.", "speaker": "Sam"}}},

        # Day 2: reconstruct on "Claude" (never told any of this)
        {"jsonrpc": "2.0", "id": 20, "method": "tools/call", "params": {
            "name": "reconstruct", "arguments": {"query": "Where does Sam work?"}}},
        {"jsonrpc": "2.0", "id": 21, "method": "tools/call", "params": {
            "name": "reconstruct", "arguments": {"query": "How is Sam feeling?"}}},
        {"jsonrpc": "2.0", "id": 22, "method": "tools/call", "params": {
            "name": "reconstruct", "arguments": {"query": "What is Sam's salary?"}}},
        {"jsonrpc": "2.0", "id": 23, "method": "tools/call", "params": {
            "name": "reconstruct", "arguments": {"query": "What happened with Joseph?"}}},
        {"jsonrpc": "2.0", "id": 24, "method": "tools/call", "params": {
            "name": "reconstruct", "arguments": {"query": "Who is Sam's partner?"}}},
        {"jsonrpc": "2.0", "id": 25, "method": "tools/call", "params": {
            "name": "reconstruct", "arguments": {"query": "How does Sam feel about parrots?"}}},

        # Cat 5: adversarial (should REFUSE)
        {"jsonrpc": "2.0", "id": 30, "method": "tools/call", "params": {
            "name": "reconstruct", "arguments": {"query": "What is Caroline's salary?"}}},
        {"jsonrpc": "2.0", "id": 31, "method": "tools/call", "params": {
            "name": "reconstruct", "arguments": {"query": "Where does Joseph work?"}}},
        {"jsonrpc": "2.0", "id": 32, "method": "tools/call", "params": {
            "name": "reconstruct", "arguments": {"query": "What car does Sam drive?"}}},
    ]

    input_text = "\n".join(json.dumps(r) for r in requests) + "\n"

    proc = subprocess.run(
        [sys.executable, "-u", "-m", "mcp.server"],
        input=input_text, capture_output=True, text=True, timeout=300,
        env={**os.environ, "KENOTIC_DB_PATH": db},
        cwd=PROJECT,
    )

    # Parse responses
    responses = {}
    for line in proc.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if "id" in r:
                responses[r["id"]] = r
        except json.JSONDecodeError:
            pass

    # DB check
    conn = sqlite3.connect(db)
    edge_count = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE tombstoned_at IS NULL"
    ).fetchone()[0]
    conn.close()

    # ── Report ────────────────────────────────────────────────
    print("=" * 60)
    print("STDIO END-TO-END INTEGRATION TEST")
    print("=" * 60)
    print()

    # Handshake
    info = responses.get(1, {}).get("result", {}).get("serverInfo", {})
    tools = [t["name"] for t in responses.get(2, {}).get("result", {}).get("tools", [])]
    print(f"1. Server:  {info.get('name')} v{info.get('version')}")
    print(f"2. Tools:   {tools}")
    print()

    # Ingest
    total_stored = 0
    for rid in range(10, 16):
        r = responses.get(rid)
        if r:
            content = json.loads(r["result"]["content"][0]["text"])
            total_stored += content.get("edges_stored", 0)
    print(f"3. Ingest:  {total_stored} edges stored ({edge_count} in DB)")
    print()

    # Reconstruct
    print("4. Reconstruct (Claude was never told any of this)")
    print()

    query_expect = {
        20: ("Where does Sam work?", False, "google"),
        21: ("How is Sam feeling?", False, "nervous"),
        22: ("What is Sam's salary?", False, "185"),
        23: ("What happened with Joseph?", False, "promot"),
        24: ("Who is Sam's partner?", False, "glenda"),
        25: ("How does Sam feel about parrots?", False, None),
        30: ("What is Caroline's salary?", True, None),
        31: ("Where does Joseph work?", True, None),
        32: ("What car does Sam drive?", True, None),
    }

    cat4_pass = cat4_fail = cat5_pass = cat5_fail = 0

    for rid in sorted(query_expect):
        query, expect_refusal, expect_word = query_expect[rid]
        r = responses.get(rid)
        if not r:
            print(f"   [MISS] {query}")
            continue

        data = json.loads(r["result"]["content"][0]["text"]).get("data", {})
        answer = data.get("narrative", "") or data.get("text", "")
        is_refusal = "not mentioned" in answer.lower()

        ok = True
        detail = ""
        if expect_refusal:
            if is_refusal:
                cat5_pass += 1
            else:
                cat5_fail += 1
                ok = False
                detail = f"should refuse, got: {answer[:60]}"
        else:
            if is_refusal:
                cat4_fail += 1
                ok = False
                detail = "Cat 4 gap"
            elif expect_word and expect_word.lower() not in answer.lower():
                cat4_fail += 1
                ok = False
                detail = f'expected "{expect_word}" in: {answer[:60]}'
            else:
                cat4_pass += 1

        tag = "PASS" if ok else "FAIL"
        print(f"   [{tag}] {query}")
        if is_refusal:
            print(f"          -> REFUSED")
        else:
            print(f"          -> {answer[:100]}")
        if detail:
            print(f"          !! {detail}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print(f"  Handshake:      {'PASS' if info.get('name') == 'reconstruct' else 'FAIL'}")
    print(f"  Tools:          {'PASS' if len(tools) == 3 else 'FAIL'} ({len(tools)} tools)")
    print(f"  Ingest:         {'PASS' if total_stored > 0 else 'FAIL'} ({total_stored} stored)")
    print(f"  Cat 5 (safety): {cat5_pass}/{cat5_pass + cat5_fail}")
    print(f"  Cat 4 (recall): {cat4_pass}/{cat4_pass + cat4_fail}")
    print(f"  Responses:      {len(responses)}/{len(requests)}")
    print("=" * 60)

    os.unlink(db)


if __name__ == "__main__":
    main()
