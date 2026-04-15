# SQLite migrations (kept simple for v1). Use Alembic later if needed.

MIGRATIONS = """
-- =============================================================================
-- EPISODES TABLE (Episodic Memory) - Day-to-day conversations, DOES decay
-- =============================================================================
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    memory_type TEXT CHECK(memory_type IN ('episodic','semantic','summary')) NOT NULL,
    importance REAL DEFAULT 0.5,
    embedding BLOB,
    created_at TEXT DEFAULT (datetime('now')),
    last_accessed_at TEXT,
    temporal_tags TEXT,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_user_time ON memories(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(user_id, importance DESC);

-- =============================================================================
-- FACTS TABLE (Semantic Memory) - Personal truths, ONE value per key, NO decay
-- Key-value pairs that get UPDATED, not accumulated
-- =============================================================================
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.9,
    last_confirmed_at TEXT DEFAULT (datetime('now')),
    first_learned_at TEXT DEFAULT (datetime('now')),
    provenance_memory_id INTEGER,
    embedding BLOB,
    history TEXT,  -- JSON array of previous values for contradiction tracking
    UNIQUE(user_id, key)
);

CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(user_id, key);

-- =============================================================================
-- MILESTONES TABLE (Life Events) - Timestamped events, NO decay, permanent
-- Deaths, marriages, graduations, major life changes
-- =============================================================================
CREATE TABLE IF NOT EXISTS milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_date TEXT,
    description TEXT NOT NULL,
    confidence REAL DEFAULT 0.9,
    created_at TEXT DEFAULT (datetime('now')),
    provenance_memory_id INTEGER,
    embedding BLOB,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_milestones_user ON milestones(user_id);
CREATE INDEX IF NOT EXISTS idx_milestones_type ON milestones(user_id, event_type);
CREATE INDEX IF NOT EXISTS idx_milestones_date ON milestones(user_id, event_date);

CREATE TABLE IF NOT EXISTS adaptation_profiles (
    user_id INTEGER PRIMARY KEY,
    warmth REAL DEFAULT 0.5,
    formality REAL DEFAULT 0.5,
    initiative REAL DEFAULT 0.5,
    check_in_frequency REAL DEFAULT 0.5,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS relationship_metrics (
    user_id INTEGER NOT NULL,
    week INTEGER NOT NULL,
    relationship_depth REAL,
    disclosure_avg REAL,
    emotional_events INTEGER,
    return_rate REAL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, week)
);

CREATE TABLE IF NOT EXISTS temporal_patterns (
    user_id INTEGER NOT NULL,
    pattern_type TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    detected_at TEXT DEFAULT (datetime('now')),
    example_memory_ids TEXT
);

CREATE INDEX IF NOT EXISTS idx_temporal_patterns_lookup ON temporal_patterns(user_id, pattern_type);

-- =============================================================================
-- PROACTIVE COOLDOWN (Rate limiting for proactive outreach)
-- Persists ask counts and last ask time across sessions
-- =============================================================================
CREATE TABLE IF NOT EXISTS proactive_cooldown (
    user_id INTEGER PRIMARY KEY,
    last_asked_at TEXT,
    asks_today INTEGER DEFAULT 0,
    asks_date TEXT,  -- Date for asks_today (resets daily)
    updated_at TEXT DEFAULT (datetime('now'))
);

-- =============================================================================
-- ENTITIES TABLE (Graph nodes) - People, places, things mentioned by user
-- =============================================================================
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,           -- "Emily", "FitLife", "Jake"
    entity_type TEXT,             -- "person", "place", "organization", "thing"
    attributes TEXT,              -- JSON: {"status": "pregnant", "occupation": "trainer"}
    first_mentioned_at TEXT DEFAULT (datetime('now')),
    last_mentioned_at TEXT DEFAULT (datetime('now')),
    mention_count INTEGER DEFAULT 1,
    embedding BLOB,
    UNIQUE(user_id, name)
);

CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(user_id, name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(user_id, entity_type);

-- =============================================================================
-- RELATIONSHIPS TABLE (Graph edges) - How entities relate to user and each other
-- =============================================================================
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    subject TEXT NOT NULL,        -- "user" or entity name
    predicate TEXT NOT NULL,      -- "sister", "trainer", "works_at", "married_to"
    object TEXT NOT NULL,         -- Entity name or value
    confidence REAL DEFAULT 0.9,
    first_learned_at TEXT DEFAULT (datetime('now')),
    last_confirmed_at TEXT DEFAULT (datetime('now')),
    provenance_memory_id INTEGER,
    UNIQUE(user_id, subject, predicate, object)
);

CREATE INDEX IF NOT EXISTS idx_relationships_user ON relationships(user_id);
CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(user_id, subject);
CREATE INDEX IF NOT EXISTS idx_relationships_predicate ON relationships(user_id, predicate);
CREATE INDEX IF NOT EXISTS idx_relationships_object ON relationships(user_id, object);

-- =============================================================================
-- MEMORY TRACES TABLE (Distributed Trace Convergence Memory)
-- Decomposed trace metadata per relationship triple
-- =============================================================================
CREATE TABLE IF NOT EXISTS memory_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relationship_id INTEGER UNIQUE,
    user_id INTEGER NOT NULL,
    -- Emotional trace (3 dimensions)
    valence REAL DEFAULT 0.5,
    affiliation REAL DEFAULT 0.5,
    emotional_intensity REAL,
    -- Relational trace (3 dimensions)
    relational_type TEXT,
    relational_proximity REAL DEFAULT 0.5,
    relational_valence REAL DEFAULT 0.5,
    -- Episodic trace (2 dimensions)
    episodic_significance TEXT DEFAULT 'routine',
    episodic_narrative_position TEXT DEFAULT 'ongoing',
    -- Temporal trace
    temporal_context TEXT CHECK(temporal_context IN ('past','present','future','ongoing')),
    -- Schematic trace
    schema_category TEXT,
    schema_confidence REAL,
    -- Rehearsal
    access_count INTEGER DEFAULT 0,
    last_accessed_at TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (relationship_id) REFERENCES relationships(id)
);

CREATE INDEX IF NOT EXISTS idx_memory_traces_user ON memory_traces(user_id);

-- =============================================================================
-- PREDICTED QUERIES TABLE (Distributed Trace Convergence Memory)
-- Pre-computed query-answer fingerprints written at extraction time
-- =============================================================================
CREATE TABLE IF NOT EXISTS predicted_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relationship_id INTEGER,
    user_id INTEGER NOT NULL,
    predicted_question TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    answer_subject TEXT,
    question_embedding BLOB NOT NULL,
    confidence REAL DEFAULT 0.9,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (relationship_id) REFERENCES relationships(id)
);

CREATE INDEX IF NOT EXISTS idx_predicted_queries_user ON predicted_queries(user_id);

-- =============================================================================
-- SCHEMA UPGRADES (for existing databases)
-- These are safe to run multiple times
-- =============================================================================

-- Add missing columns to facts table (for older databases)
-- SQLite doesn't support IF NOT EXISTS for columns, so we use a workaround
"""

# Additional migration to add missing columns
SCHEMA_UPGRADES = """
-- Upgrade facts table if columns are missing
-- These will fail silently if columns already exist
"""

def run_schema_upgrades(conn) -> None:
    """Add missing columns to existing databases."""
    # Check facts table columns
    cursor = conn.execute("PRAGMA table_info(facts)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    upgrades = []
    if "history" not in existing_columns:
        upgrades.append("ALTER TABLE facts ADD COLUMN history TEXT")
    if "first_learned_at" not in existing_columns:
        upgrades.append("ALTER TABLE facts ADD COLUMN first_learned_at TEXT DEFAULT (datetime('now'))")
    if "embedding" not in existing_columns:
        upgrades.append("ALTER TABLE facts ADD COLUMN embedding BLOB")

    for sql in upgrades:
        try:
            conn.execute(sql)
            print(f"[DB] Applied: {sql[:50]}...")
        except Exception:
            pass

    if upgrades:
        conn.commit()

    # --- Grammar Engine: object_type column on relationships ---
    rel_cursor = conn.execute("PRAGMA table_info(relationships)")
    rel_columns = {row[1] for row in rel_cursor.fetchall()}

    if "object_type" not in rel_columns:
        try:
            conn.execute("ALTER TABLE relationships ADD COLUMN object_type TEXT DEFAULT 'unknown'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_object_type ON relationships(user_id, object_type)")
            conn.commit()
            print("[DB] Applied: ALTER TABLE relationships ADD COLUMN object_type")
        except Exception:
            pass

    # --- 594 Equation System: utterance_type_id + canonical_fields on relationships ---
    if "utterance_type_id" not in rel_columns:
        try:
            conn.execute("ALTER TABLE relationships ADD COLUMN utterance_type_id INTEGER")
            conn.commit()
            print("[DB] Applied: ALTER TABLE relationships ADD COLUMN utterance_type_id")
        except Exception:
            pass

    if "canonical_fields" not in rel_columns:
        try:
            conn.execute("ALTER TABLE relationships ADD COLUMN canonical_fields TEXT")
            conn.commit()
            print("[DB] Applied: ALTER TABLE relationships ADD COLUMN canonical_fields")
        except Exception:
            pass

    # --- Supersession columns on relationships (update handling) ---
    if "is_current" not in rel_columns:
        try:
            conn.execute("ALTER TABLE relationships ADD COLUMN is_current INTEGER DEFAULT 1")
            conn.commit()
            print("[DB] Applied: ALTER TABLE relationships ADD COLUMN is_current")
        except Exception:
            pass

    if "superseded_at" not in rel_columns:
        try:
            conn.execute("ALTER TABLE relationships ADD COLUMN superseded_at TEXT")
            conn.commit()
            print("[DB] Applied: ALTER TABLE relationships ADD COLUMN superseded_at")
        except Exception:
            pass

    if "superseded_by" not in rel_columns:
        try:
            conn.execute("ALTER TABLE relationships ADD COLUMN superseded_by INTEGER")
            conn.commit()
            print("[DB] Applied: ALTER TABLE relationships ADD COLUMN superseded_by")
        except Exception:
            pass

    # --- Situation linking: situation_id on relationships ---
    if "situation_id" not in rel_columns:
        try:
            conn.execute("ALTER TABLE relationships ADD COLUMN situation_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_situation ON relationships(user_id, situation_id)")
            conn.commit()
            print("[DB] Applied: ALTER TABLE relationships ADD COLUMN situation_id")
        except Exception:
            pass

    # --- Source timestamp: when the source utterance occurred ---
    # Used by temporal engine to resolve "when did X happen?" queries.
    # For Raya: this is the wall-clock time the user said it.
    # For LOCOMO/SDK: this is the session_date_time from the source data.
    if "source_timestamp" not in rel_columns:
        try:
            conn.execute("ALTER TABLE relationships ADD COLUMN source_timestamp TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source_ts ON relationships(user_id, source_timestamp)")
            conn.commit()
            print("[DB] Applied: ALTER TABLE relationships ADD COLUMN source_timestamp")
        except Exception:
            pass

    # --- T5 SRL: continuous dimension columns on memory_traces ---
    trace_cursor = conn.execute("PRAGMA table_info(memory_traces)")
    trace_columns = {row[1] for row in trace_cursor.fetchall()}

    t5_upgrades = []
    for col in ("valence", "affiliation"):
        if col not in trace_columns:
            t5_upgrades.append(f"ALTER TABLE memory_traces ADD COLUMN {col} REAL DEFAULT 0.5")

    # Migrate existing categorical emotional_valence → continuous valence
    migrate_valence = "emotional_valence" in trace_columns and "valence" not in trace_columns

    for sql in t5_upgrades:
        try:
            conn.execute(sql)
            print(f"[DB] Applied: {sql[:60]}...")
        except Exception:
            pass

    if t5_upgrades:
        conn.commit()

    # Backfill: convert old categorical values to continuous
    if migrate_valence:
        try:
            conn.execute("""UPDATE memory_traces SET valence = CASE
                WHEN emotional_valence = 'positive' THEN 0.8
                WHEN emotional_valence = 'negative' THEN 0.2
                WHEN emotional_valence = 'mixed' THEN 0.5
                ELSE 0.5 END
                WHERE valence = 0.5 AND emotional_valence IS NOT NULL""")
            conn.commit()
            print("[DB] Applied: backfill emotional_valence → valence")
        except Exception:
            pass

    # --- DTCM 5-trace upgrade: relational + episodic columns ---
    new_trace_cols = {
        "relational_type": "TEXT",
        "relational_proximity": "REAL DEFAULT 0.5",
        "relational_valence": "REAL DEFAULT 0.5",
        "episodic_significance": "TEXT DEFAULT 'routine'",
        "episodic_narrative_position": "TEXT DEFAULT 'ongoing'",
    }
    for col, col_type in new_trace_cols.items():
        if col not in trace_columns:
            try:
                conn.execute(f"ALTER TABLE memory_traces ADD COLUMN {col} {col_type}")
                conn.commit()
                print(f"[DB] Applied: ALTER TABLE memory_traces ADD COLUMN {col}")
            except Exception:
                pass

    # --- Filter → Complete migration (2026-04-12): trace columns
    # and edge_embedding live DIRECTLY on the relationships row.
    # memory_traces is kept for backward-compat read but no longer
    # written. predicted_queries is no longer used — edge_embedding
    # is the access path. ---
    edge_trace_cols = {
        "edge_embedding": "BLOB",
        "edge_emotional_valence": "REAL DEFAULT 0.5",
        "edge_emotional_label": "TEXT",
        "edge_schematic_category": "TEXT",
        "edge_episodic_significance": "TEXT",
        "edge_relational_type": "TEXT",
        "edge_temporal_context": "TEXT",
    }
    for col, col_type in edge_trace_cols.items():
        if col not in rel_columns:
            try:
                conn.execute(f"ALTER TABLE relationships ADD COLUMN {col} {col_type}")
                conn.commit()
                print(f"[DB] Applied: ALTER TABLE relationships ADD COLUMN {col}")
            except Exception:
                pass
