#!/usr/bin/env python3
# -*- coding: cp1252 -*-
"""
backfill_cluster_ids.py -- one-shot idempotent backfill for the
9-axis set-op pipeline.

For every relationship row where cluster_id IS NULL, call
TemporalEngine.cluster(user_id, source_text) and write the resulting
cluster handle back to the row. Safe to re-run: rows already labeled
are skipped.

Usage:
    py -3.10 scripts/backfill_cluster_ids.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/Nura/Env/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import get_db_context
from app.db.models import run_schema_upgrades
from app.engines.memory import get_memory_engine
from app.engines.temporal import get_temporal_engine


def main() -> int:
    # Ensure schema is current (idempotent).
    with get_db_context() as conn:
        try:
            run_schema_upgrades(conn)
            conn.commit()
        except Exception:
            pass

    mem = get_memory_engine()
    temp = get_temporal_engine()

    scanned = 0
    labeled = 0
    skipped = 0

    with get_db_context() as conn:
        rows = conn.execute(
            """SELECT id, user_id, subject, predicate, object, source_text
               FROM relationships
               WHERE cluster_id IS NULL
               ORDER BY id ASC"""
        ).fetchall()

    for r in rows:
        scanned += 1
        rel_id = r["id"]
        user_id = r["user_id"]
        src = r["source_text"] or f"{r['subject']} {r['predicate']} {r['object']}"
        try:
            clust = temp.cluster(user_id, src)
            members = [m for m in (clust.member_relationship_ids or []) if m]
            members.append(rel_id)
            cluster_key = f"c_{min(members)}"
        except Exception:
            cluster_key = f"c_{rel_id}"

        with get_db_context() as conn:
            try:
                conn.execute(
                    "UPDATE relationships SET cluster_id = ? WHERE id = ? AND cluster_id IS NULL",
                    (cluster_key, rel_id),
                )
                conn.commit()
                labeled += 1
            except Exception:
                skipped += 1

    print(f"[backfill] scanned={scanned} labeled={labeled} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
