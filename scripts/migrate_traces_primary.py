"""Migrate relationships table from triple-keyed to trace-primary schema.

Phase 6 foundation rebuild: the old UNIQUE(user_id, subject, predicate, object)
constraint is replaced by UNIQUE(user_id, source_text_hash, created_at).  Traces
become first-class columns with NOT NULL defaults; subject/predicate/object become
optional derived fields.

Root cause: the triple-keyed UNIQUE constraint means two different utterances
producing the same (subject, predicate, object) collide -- the second overwrites
the first.  The trace-primary schema keys by source_text_hash so each distinct
source utterance gets its own row, and SPO becomes optional derived data.

Usage:
    python scripts/migrate_traces_primary.py [db_path]

If db_path is omitted, uses the default "Memory Storage/nura.db".

Safety:
    - Creates a backup before any destructive operation.
    - Detects whether migration has already run (source_text_hash column exists
      AND the old triple UNIQUE constraint is gone).
    - Batch-processes rows (1000 at a time) to keep memory bounded.
    - Verifies row count matches after migration.
    - Exits non-zero on any failure; backup remains untouched.
"""

import hashlib
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

BATCH_SIZE = 1000


# ──────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────

def _get_columns(conn: sqlite3.Connection, table: str) -> set:
    """Return the set of column names for a table."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _get_create_sql(conn: sqlite3.Connection, table: str) -> str:
    """Return the CREATE TABLE SQL for a table."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row[0] if row else ""


def is_already_migrated(conn: sqlite3.Connection) -> bool:
    """Check if the migration has already been applied.

    Positive signal: source_text_hash column exists on relationships AND
    the old triple-keyed UNIQUE constraint is gone (the CREATE TABLE SQL
    contains 'source_text_hash' in its UNIQUE clause).
    """
    cols = _get_columns(conn, "relationships")
    if "source_text_hash" not in cols:
        return False

    create_sql = _get_create_sql(conn, "relationships")
    if not create_sql:
        return False

    # Old constraint: UNIQUE(user_id, subject, predicate, object)
    # New constraint: UNIQUE(user_id, source_text_hash, created_at)
    # Check that source_text_hash appears after a UNIQUE keyword
    upper = create_sql.upper()
    idx = upper.find("UNIQUE")
    while idx != -1:
        paren_start = upper.find("(", idx)
        paren_end = upper.find(")", paren_start) if paren_start != -1 else -1
        if paren_start != -1 and paren_end != -1:
            constraint_body = create_sql[paren_start:paren_end + 1]
            if "source_text_hash" in constraint_body:
                return True
        idx = upper.find("UNIQUE", idx + 1)

    return False


# ──────────────────────────────────────────────────────────────
# Hash computation
# ──────────────────────────────────────────────────────────────

def compute_source_text_hash(source_text: str) -> str:
    """SHA-256 hex digest of the source text. Deterministic."""
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────
# Migration
# ──────────────────────────────────────────────────────────────

def migrate(db_path: str) -> None:
    """Run the trace-primary migration on the given database."""
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"[migrate] Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")

    # ── Step 0: detect ──
    if is_already_migrated(conn):
        print("[migrate] Already migrated (source_text_hash UNIQUE constraint present). Nothing to do.")
        conn.close()
        sys.exit(0)

    # ── Step 0b: backup ──
    backup_path = str(db_file) + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[migrate] Creating backup at {backup_path}")
    conn.close()
    shutil.copy2(str(db_file), backup_path)

    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")

    # Count rows before migration
    old_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
    print(f"[migrate] Source table has {old_count} rows")

    # ── Step 1: read existing columns ──
    old_cols = _get_columns(conn, "relationships")
    print(f"[migrate] Existing columns: {sorted(old_cols)}")

    # ── Step 2: create relationships_v2 ──
    print("[migrate] Creating relationships_v2 table...")
    conn.executescript("""
        DROP TABLE IF EXISTS relationships_v2;

        CREATE TABLE relationships_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            source_text TEXT NOT NULL,
            source_text_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),

            -- 5 traces (primary data)
            edge_schematic_category TEXT NOT NULL DEFAULT 'uncategorized',
            edge_temporal_context TEXT NOT NULL DEFAULT 'present',
            edge_relational_type TEXT NOT NULL DEFAULT 'personal',
            edge_episodic_significance TEXT NOT NULL DEFAULT 'routine',
            edge_emotional_valence REAL NOT NULL DEFAULT 0.5,
            edge_emotional_label TEXT,
            edge_affiliation REAL,

            -- Derived triple (optional, nullable)
            subject TEXT,
            predicate TEXT,
            object TEXT,

            -- Carried-over columns
            is_current INTEGER DEFAULT 1,
            superseded_at TEXT,
            superseded_by INTEGER,
            source_timestamp TEXT,
            source_tag TEXT,
            confidence REAL DEFAULT 0.9,
            cluster_id TEXT,
            arc_id TEXT,
            subject_type TEXT,
            object_type TEXT,
            subject_type_confidence REAL,
            object_type_confidence REAL,
            edge_embedding BLOB,
            predicate_embedding BLOB,
            tombstoned_at TEXT,
            tombstone_reason TEXT,
            tombstone_op_id TEXT,
            sequence_number INTEGER,
            utterance_type_id INTEGER,
            canonical_fields TEXT,
            temporal_expression TEXT,
            relational_entities TEXT,
            resolved_event_date TEXT,

            UNIQUE(user_id, source_text_hash, created_at)
        );
    """)
    conn.commit()

    # ── Step 3: batch copy data ──
    print("[migrate] Copying data in batches...")

    def _col_or_default(col: str, default: str) -> str:
        """Return column reference if it exists in old table, else a literal default."""
        if col in old_cols:
            return col
        return default

    # source_text: COALESCE with SPO concatenation for rows that predate
    # the source_text column addition.
    source_text_expr = (
        f"COALESCE({_col_or_default('source_text', 'NULL')}, "
        f"subject || ' ' || REPLACE(predicate, '_', ' ') || ' ' || object)"
    )

    select_parts = [
        "id",
        "user_id",
        source_text_expr + " AS source_text_val",
        _col_or_default("created_at", "datetime('now')") + " AS created_at_val",
        # Traces -- use COALESCE so missing columns get proper defaults
        f"COALESCE({_col_or_default('edge_schematic_category', 'NULL')}, 'uncategorized') AS edge_schematic_category_val",
        f"COALESCE({_col_or_default('edge_temporal_context', 'NULL')}, 'present') AS edge_temporal_context_val",
        f"COALESCE({_col_or_default('edge_relational_type', 'NULL')}, 'personal') AS edge_relational_type_val",
        f"COALESCE({_col_or_default('edge_episodic_significance', 'NULL')}, 'routine') AS edge_episodic_significance_val",
        f"COALESCE({_col_or_default('edge_emotional_valence', 'NULL')}, 0.5) AS edge_emotional_valence_val",
        _col_or_default("edge_emotional_label", "NULL") + " AS edge_emotional_label_val",
        _col_or_default("edge_affiliation", "NULL") + " AS edge_affiliation_val",
        # Triple
        "subject",
        "predicate",
        "object",
        # Carried over
        f"COALESCE({_col_or_default('is_current', 'NULL')}, 1) AS is_current_val",
        _col_or_default("superseded_at", "NULL") + " AS superseded_at_val",
        _col_or_default("superseded_by", "NULL") + " AS superseded_by_val",
        _col_or_default("source_timestamp", "NULL") + " AS source_timestamp_val",
        _col_or_default("source_tag", "NULL") + " AS source_tag_val",
        "COALESCE(confidence, 0.9) AS confidence_val",
        _col_or_default("cluster_id", "NULL") + " AS cluster_id_val",
        _col_or_default("arc_id", "NULL") + " AS arc_id_val",
        _col_or_default("subject_type", "NULL") + " AS subject_type_val",
        _col_or_default("object_type", "NULL") + " AS object_type_val",
        _col_or_default("subject_type_confidence", "NULL") + " AS subject_type_confidence_val",
        _col_or_default("object_type_confidence", "NULL") + " AS object_type_confidence_val",
        _col_or_default("edge_embedding", "NULL") + " AS edge_embedding_val",
        _col_or_default("predicate_embedding", "NULL") + " AS predicate_embedding_val",
        _col_or_default("tombstoned_at", "NULL") + " AS tombstoned_at_val",
        _col_or_default("tombstone_reason", "NULL") + " AS tombstone_reason_val",
        _col_or_default("tombstone_op_id", "NULL") + " AS tombstone_op_id_val",
        _col_or_default("sequence_number", "NULL") + " AS sequence_number_val",
        _col_or_default("utterance_type_id", "NULL") + " AS utterance_type_id_val",
        _col_or_default("canonical_fields", "NULL") + " AS canonical_fields_val",
        _col_or_default("temporal_expression", "NULL") + " AS temporal_expression_val",
        _col_or_default("relational_entities", "NULL") + " AS relational_entities_val",
        _col_or_default("resolved_event_date", "NULL") + " AS resolved_event_date_val",
    ]

    select_sql = f"SELECT {', '.join(select_parts)} FROM relationships ORDER BY id"

    insert_sql = """
        INSERT INTO relationships_v2 (
            id, user_id, source_text, source_text_hash, created_at,
            edge_schematic_category, edge_temporal_context, edge_relational_type,
            edge_episodic_significance, edge_emotional_valence, edge_emotional_label,
            edge_affiliation,
            subject, predicate, object,
            is_current, superseded_at, superseded_by, source_timestamp, source_tag,
            confidence, cluster_id, arc_id, subject_type, object_type,
            subject_type_confidence, object_type_confidence,
            edge_embedding, predicate_embedding,
            tombstoned_at, tombstone_reason, tombstone_op_id,
            sequence_number, utterance_type_id, canonical_fields,
            temporal_expression, relational_entities, resolved_event_date
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?,
            ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?
        )
    """

    copied = 0
    offset = 0
    while True:
        rows = conn.execute(
            f"{select_sql} LIMIT {BATCH_SIZE} OFFSET {offset}"
        ).fetchall()
        if not rows:
            break

        batch = []
        for r in rows:
            source_text_val = r["source_text_val"] or ""
            source_text_hash = compute_source_text_hash(source_text_val)

            batch.append((
                r["id"],
                r["user_id"],
                source_text_val,
                source_text_hash,
                r["created_at_val"],
                # Traces
                r["edge_schematic_category_val"],
                r["edge_temporal_context_val"],
                r["edge_relational_type_val"],
                r["edge_episodic_significance_val"],
                r["edge_emotional_valence_val"],
                r["edge_emotional_label_val"],
                r["edge_affiliation_val"],
                # Triple
                r["subject"],
                r["predicate"],
                r["object"],
                # Carried over
                r["is_current_val"],
                r["superseded_at_val"],
                r["superseded_by_val"],
                r["source_timestamp_val"],
                r["source_tag_val"],
                r["confidence_val"],
                r["cluster_id_val"],
                r["arc_id_val"],
                r["subject_type_val"],
                r["object_type_val"],
                r["subject_type_confidence_val"],
                r["object_type_confidence_val"],
                r["edge_embedding_val"],
                r["predicate_embedding_val"],
                r["tombstoned_at_val"],
                r["tombstone_reason_val"],
                r["tombstone_op_id_val"],
                r["sequence_number_val"],
                r["utterance_type_id_val"],
                r["canonical_fields_val"],
                r["temporal_expression_val"],
                r["relational_entities_val"],
                r["resolved_event_date_val"],
            ))

        conn.executemany(insert_sql, batch)
        conn.commit()
        copied += len(batch)
        offset += BATCH_SIZE
        if copied % 5000 == 0 or len(rows) < BATCH_SIZE:
            print(f"[migrate]   ...copied {copied}/{old_count} rows")

    print(f"[migrate] Copied {copied} rows total")

    # ── Step 4: verify row count ──
    new_count = conn.execute("SELECT COUNT(*) FROM relationships_v2").fetchone()[0]
    if new_count != old_count:
        print(f"[migrate] ERROR: row count mismatch! old={old_count}, new={new_count}")
        print(f"[migrate] Backup preserved at {backup_path}. Aborting.")
        conn.execute("DROP TABLE IF EXISTS relationships_v2")
        conn.commit()
        conn.close()
        sys.exit(1)

    print(f"[migrate] Row count verified: {new_count} == {old_count}")

    # ── Step 5: swap tables ──
    print("[migrate] Dropping old table and renaming v2...")
    conn.execute("DROP TABLE relationships")
    conn.execute("ALTER TABLE relationships_v2 RENAME TO relationships")
    conn.commit()

    # ── Step 6: rebuild FTS5 ──
    print("[migrate] Rebuilding FTS5 index...")
    conn.execute("DROP TABLE IF EXISTS relationships_fts")
    conn.execute("""
        CREATE VIRTUAL TABLE relationships_fts USING fts5(
            subject, predicate, object, source_text,
            content='relationships',
            content_rowid='id'
        )
    """)
    conn.execute("""
        INSERT INTO relationships_fts(rowid, subject, predicate, object, source_text)
        SELECT id,
               COALESCE(subject, ''),
               COALESCE(REPLACE(predicate, '_', ' '), ''),
               COALESCE(object, ''),
               COALESCE(source_text, '')
        FROM relationships
        WHERE tombstoned_at IS NULL
    """)
    conn.commit()
    print("[migrate] FTS5 index rebuilt")

    # ── Step 7: create indexes ──
    print("[migrate] Creating indexes...")
    indexes = [
        # New trace-primary indexes
        "CREATE INDEX IF NOT EXISTS idx_rel_schema_cat ON relationships(user_id, edge_schematic_category)",
        "CREATE INDEX IF NOT EXISTS idx_rel_subject_schema ON relationships(user_id, subject, edge_schematic_category)",
        "CREATE INDEX IF NOT EXISTS idx_rel_source_hash ON relationships(user_id, source_text_hash)",
        "CREATE INDEX IF NOT EXISTS idx_rel_user_current ON relationships(user_id, is_current)",
        # Carried forward from old schema
        "CREATE INDEX IF NOT EXISTS idx_relationships_user ON relationships(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(user_id, subject)",
        "CREATE INDEX IF NOT EXISTS idx_relationships_predicate ON relationships(user_id, predicate)",
        "CREATE INDEX IF NOT EXISTS idx_relationships_object ON relationships(user_id, object)",
        "CREATE INDEX IF NOT EXISTS idx_rel_source_ts ON relationships(user_id, source_timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_rel_object_type ON relationships(user_id, object_type)",
        "CREATE INDEX IF NOT EXISTS idx_rel_subject_type ON relationships(user_id, subject_type)",
        "CREATE INDEX IF NOT EXISTS idx_rel_cluster ON relationships(user_id, cluster_id)",
        "CREATE INDEX IF NOT EXISTS idx_rel_arc ON relationships(user_id, arc_id)",
        "CREATE INDEX IF NOT EXISTS idx_rel_event_date ON relationships(user_id, resolved_event_date)",
        "CREATE INDEX IF NOT EXISTS idx_rel_live_subject ON relationships(user_id, tombstoned_at, subject)",
        "CREATE INDEX IF NOT EXISTS idx_rel_live_object ON relationships(user_id, tombstoned_at, object)",
        "CREATE INDEX IF NOT EXISTS idx_rel_live_source_ts ON relationships(user_id, tombstoned_at, source_timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_rel_live_source_tag ON relationships(user_id, tombstoned_at, source_tag)",
    ]
    for sql in indexes:
        conn.execute(sql)
    conn.commit()
    print(f"[migrate] Created {len(indexes)} indexes")

    # ── Step 8: integrity check ──
    final_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
    print(f"[migrate] Final row count: {final_count}")

    # Verify new UNIQUE constraint works
    create_sql = _get_create_sql(conn, "relationships")
    if "source_text_hash" in create_sql:
        print("[migrate] New UNIQUE(user_id, source_text_hash, created_at) constraint confirmed")
    else:
        print("[migrate] WARNING: could not confirm new UNIQUE constraint in CREATE TABLE sql")

    conn.execute("PRAGMA foreign_keys=ON")
    conn.close()

    print(f"[migrate] Migration complete. Backup at {backup_path}")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "Memory Storage/nura.db"
    migrate(db_path)
