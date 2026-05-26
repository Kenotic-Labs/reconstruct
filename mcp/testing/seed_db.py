"""
Seed a test DB with known data for MCP testing.

Usage:
    python mcp/testing/seed_db.py              # seeds mcp/testing/test_memory.db
    python mcp/testing/seed_db.py path/to.db   # seeds a custom path
"""
import os
import sys

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

DEFAULT_DB = os.path.join(os.path.dirname(__file__), "test_memory.db")


SEED_CONVERSATIONS = [
    # Simple facts
    ("Sam wants to learn about retrieval systems.", "user"),
    ("How does code know what is right or wrong when it doesn't have context?", "user"),
    ("The key takeaway is building context as a filter layer around deterministic lookups rather than replacing determinism.", "assistant"),
    ("Sam asked how to add context to deterministic retrievals.", "user"),
    # Entities + relationships
    ("I adopted a dog named Kobe. He is 2 years old and loves swimming.", "Sam"),
    ("I live in Detroit and work remotely for a startup in Toronto.", "Sam"),
    ("Arjun twisted his ankle so we had to cut the trip short.", "Sam"),
    ("I stayed at the Moraine Inn when I visited Banff last summer.", "Sam"),
    ("I hiked Sulphur Mountain on Tuesday.", "Sam"),
]


def seed(db_path: str) -> None:
    from sdk import Kenotic

    # Remove old DB if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed old {db_path}")

    k = Kenotic(user_id=0, db_path=db_path)
    for text, speaker in SEED_CONVERSATIONS:
        k.ingest(text=text, speaker=speaker)
        print(f"  ingested: {text[:60]}...")

    # Verify
    import sqlite3
    conn = sqlite3.connect(db_path)
    for table in ["edges", "relationships"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"\n{table}: {count} rows")
        except Exception:
            pass

    # Sample edges
    try:
        rows = conn.execute(
            "SELECT subject, predicate, object FROM edges LIMIT 10"
        ).fetchall()
        print("\nSample edges:")
        for s, p, o in rows:
            print(f"  {s} | {p} | {o}")
    except Exception:
        pass

    conn.close()
    print(f"\nSeeded {db_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    seed(path)
