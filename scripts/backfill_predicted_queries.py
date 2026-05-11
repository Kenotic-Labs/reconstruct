#!/usr/bin/env python3
# -*- coding: cp1252 -*-
"""
backfill_predicted_queries.py -- idempotent backfill for the write-path
foundation (2026-04-14).

For every relationship row where any of the following is true:
  - subject_type IS NULL
  - object_type  IS NULL (or legacy 'unknown')
  - entity_type  IS NULL on the subject/object entities
  - no predicted_queries row exists for this relationship_id

...re-run the write-path foundation helpers (type resolver + predicted
query generator). Already fully-labeled rows are skipped. Progress is
logged every 100 rows.

Usage:
    py -3.10 scripts/backfill_predicted_queries.py
    py -3.10 scripts/backfill_predicted_queries.py --dry-run
    py -3.10 scripts/backfill_predicted_queries.py --user-id 42 --limit 500
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/Nura/Env/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import get_db_context
from app.db.models import run_schema_upgrades


def row_needs_backfill(conn, row) -> bool:
    rel_id = row["id"]
    if not row["subject_type"]:
        return True
    if not row["object_type"] or row["object_type"] == "unknown":
        return True
    pq = conn.execute(
        "SELECT 1 FROM predicted_queries WHERE relationship_id = ? LIMIT 1",
        (rel_id,),
    ).fetchone()
    if not pq:
        return True
    return False


def backfill_row(conn, row) -> bool:
    """Return True if the row was successfully backfilled."""
    from app.engines.type_resolver import resolve_entity_type
    from app.engines.predicted_queries import generate_predicted_queries

    rel_id = row["id"]
    user_id = row["user_id"]
    subject = row["subject"]
    predicate = row["predicate"]
    object_ = row["object"]
    source_text = row["source_text"] or f"{subject} {predicate.replace('_',' ')} {object_}"

    # Types.
    s_type = row["subject_type"] or resolve_entity_type(subject)
    o_type = row["object_type"] if (row["object_type"] and row["object_type"] != "unknown") else resolve_entity_type(object_)

    try:
        conn.execute(
            "UPDATE relationships SET subject_type = ?, object_type = ? WHERE id = ?",
            (s_type, o_type, rel_id),
        )
    except Exception:
        pass

    for name, typ in ((subject, s_type), (object_, o_type)):
        if not name or name.lower() in ("user", "i", "me", "myself"):
            continue
        try:
            conn.execute(
                """UPDATE entities SET entity_type = ?
                   WHERE user_id = ? AND LOWER(name) = LOWER(?)
                     AND (entity_type IS NULL OR entity_type = 'unknown' OR entity_type = '')""",
                (typ, user_id, name),
            )
        except Exception:
            pass

    # Predicted queries.
    existing = conn.execute(
        "SELECT 1 FROM predicted_queries WHERE relationship_id = ? LIMIT 1",
        (rel_id,),
    ).fetchone()
    if existing:
        return True

    try:
        pairs = generate_predicted_queries(subject, predicate, object_, s_type, o_type)
    except Exception:
        pairs = []

    for question, emb in pairs:
        try:
            conn.execute(
                """INSERT INTO predicted_queries
                     (relationship_id, user_id, predicted_question, answer_text,
                      answer_subject, question_embedding, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (rel_id, user_id, question, source_text, subject,
                 emb.tobytes() if emb is not None else None, 0.9),
            )
        except Exception:
            continue
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--user-id", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with get_db_context() as conn:
        try:
            run_schema_upgrades(conn)
            conn.commit()
        except Exception:
            pass

    sql = "SELECT * FROM relationships"
    params = []
    if args.user_id is not None:
        sql += " WHERE user_id = ?"
        params.append(args.user_id)
    sql += " ORDER BY id ASC"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"

    with get_db_context() as conn:
        rows = conn.execute(sql, params).fetchall()

    scanned = 0
    needed = 0
    done = 0
    skipped = 0

    for r in rows:
        scanned += 1
        with get_db_context() as conn:
            try:
                if not row_needs_backfill(conn, r):
                    skipped += 1
                    continue
                needed += 1
                if args.dry_run:
                    continue
                if backfill_row(conn, r):
                    conn.commit()
                    done += 1
            except Exception as e:
                print(f"[backfill] row {r['id']} failed: {e}")

        if scanned % 100 == 0:
            print(f"[backfill] scanned={scanned} needed={needed} done={done} skipped={skipped}")

    print(f"[backfill] DONE scanned={scanned} needed={needed} done={done} skipped={skipped} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
