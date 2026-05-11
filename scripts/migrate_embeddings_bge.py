#!/usr/bin/env python3
"""
migrate_embeddings_bge.py -- re-embed all edge_embedding and
question_embedding blobs after model swap to BAAI/bge-small-en-v1.5.

The semantic space differs between all-MiniLM-L6-v2 and bge-small-en-v1.5
so every stored embedding must be recomputed. Dimension stays 384.

Usage:
    py -3.10 scripts/migrate_embeddings_bge.py
    py -3.10 scripts/migrate_embeddings_bge.py --db "Memory Storage/nura.db"
    py -3.10 scripts/migrate_embeddings_bge.py --dry-run
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/Nura/Env/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "D:/Nura/Env/hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/Nura/Env/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3
import numpy as np
from app.vector.embedder import embed_text


BATCH_SIZE = 500


def _to_blob(arr: np.ndarray) -> bytes:
    return arr.astype("float32").tobytes()


def migrate_edge_embeddings(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Re-embed all non-tombstoned relationships rows."""
    rows = conn.execute(
        """SELECT id, subject, predicate, object
           FROM relationships
           WHERE tombstoned_at IS NULL"""
    ).fetchall()

    total = len(rows)
    print(f"[migrate] relationships: {total} rows to re-embed")
    updated = 0
    t0 = time.perf_counter()

    for i, row in enumerate(rows):
        rid, subj, pred, obj = row
        pred_natural = pred.replace("_", " ")
        text = f"{subj} {pred_natural} {obj}"
        emb = embed_text(text)
        if not dry_run:
            conn.execute(
                "UPDATE relationships SET edge_embedding = ? WHERE id = ?",
                (_to_blob(emb), rid),
            )
        updated += 1
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{total}] {elapsed:.1f}s")
        if (i + 1) % BATCH_SIZE == 0 and not dry_run:
            conn.commit()

    if not dry_run:
        conn.commit()
    elapsed = time.perf_counter() - t0
    print(f"[migrate] relationships done: {updated} rows, {elapsed:.1f}s")
    return updated


def migrate_predicted_queries(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Re-embed all predicted_queries rows."""
    rows = conn.execute(
        "SELECT id, predicted_question FROM predicted_queries"
    ).fetchall()

    total = len(rows)
    print(f"[migrate] predicted_queries: {total} rows to re-embed")
    updated = 0
    t0 = time.perf_counter()

    for i, row in enumerate(rows):
        qid, question = row
        emb = embed_text(question)
        if not dry_run:
            conn.execute(
                "UPDATE predicted_queries SET question_embedding = ? WHERE id = ?",
                (_to_blob(emb), qid),
            )
        updated += 1
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{total}] {elapsed:.1f}s")
        if (i + 1) % BATCH_SIZE == 0 and not dry_run:
            conn.commit()

    if not dry_run:
        conn.commit()
    elapsed = time.perf_counter() - t0
    print(f"[migrate] predicted_queries done: {updated} rows, {elapsed:.1f}s")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Re-embed all vectors after model swap")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (default: settings.sqlite_path)")
    parser.add_argument("--dry-run", action="store_true", help="Count rows without writing")
    args = parser.parse_args()

    from config.settings import settings

    db_path = args.db or settings.sqlite_path
    print(f"[migrate] DB: {db_path}")
    print(f"[migrate] Model: {settings.embedding_model}")
    print(f"[migrate] Dry run: {args.dry_run}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = None  # plain tuples

    t_start = time.perf_counter()
    n_edges = migrate_edge_embeddings(conn, args.dry_run)
    n_pqs = migrate_predicted_queries(conn, args.dry_run)
    t_total = time.perf_counter() - t_start

    conn.close()
    print(f"\n[migrate] COMPLETE: {n_edges} edges + {n_pqs} queries, {t_total:.1f}s total")


if __name__ == "__main__":
    main()
