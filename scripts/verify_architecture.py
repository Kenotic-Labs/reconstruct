#!/usr/bin/env python3
"""
Standalone architecture verifier for the NURA Continuity Layer.

Initializes a temp DB, runs kenoticArchitectureV1 (all 14 engines),
and prints a formatted scoreboard grouped by engine. Exit code 0 if
all pass, 1 if any fail.

Root cause for this rewrite: the old runner imported
verify_architecture_completeness from app.engines.memory which only
covered 36 checks across 5 of 14 engines. The new module
(architecture_verifier.py) covers all 14 engines with ~100 checks
including live DB table verification.

Usage:
    python scripts/verify_architecture.py
"""
from __future__ import annotations

import os
import sys
import tempfile

# Ensure the repo root is on sys.path so app.* and sdk.* imports resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    # ── Bootstrap: point engines at a disposable temp DB ────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = tmp.name
    tmp.close()

    os.environ.setdefault("RAYA_EMBED_DEVICE", "cpu")

    from config.settings import settings
    settings.sqlite_path = tmp_path

    # Initialize the schema so import-time side effects don't crash.
    import sqlite3
    from app.db.models import MIGRATIONS, run_schema_upgrades

    conn = sqlite3.connect(tmp_path)
    conn.executescript(MIGRATIONS)
    run_schema_upgrades(conn)
    # Ensure sequence_number column exists (SDK adds this separately).
    try:
        conn.execute(
            "ALTER TABLE relationships ADD COLUMN sequence_number INTEGER"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_seq "
            "ON relationships(user_id, sequence_number)"
        )
    except Exception:
        pass
    conn.commit()
    conn.close()

    # ── Run kenoticArchitectureV1 (all 14 engines + live DB) ───────
    from app.engines.architecture_verifier import (
        print_scoreboard,
        verify_kenotic_architecture_v1,
    )

    results = verify_kenotic_architecture_v1(db_path=tmp_path)
    exit_code = print_scoreboard(results)

    # ── Cleanup ─────────────────────────────────────────────────────
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
