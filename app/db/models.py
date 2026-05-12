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
-- RELATIONSHIPS TABLE (Graph edges) - Trace-primary schema (2026-04-25)
-- source_text_hash is the dedup key; subject/predicate/object are optional
-- derived fields. 5 trace columns carry NOT NULL defaults (same sentinel
-- values already used in memory.py _write_edge_traces() at lines 867-869
-- and in scripts/migrate_traces_primary.py v2 table at lines 151-156).
-- Root cause: UNIQUE(user_id, subject, predicate, object) caused two
-- distinct utterances producing the same triple to collide, losing trace
-- data from the first. source_text_hash dedup preserves each utterance.
-- =============================================================================
CREATE TABLE IF NOT EXISTS edges (
    -- 1. Core Edge
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    source_text TEXT NOT NULL DEFAULT '',
    source_text_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),

    -- 2. Five Traces
    edge_schematic_category TEXT NOT NULL DEFAULT 'uncategorized',
    edge_temporal_context TEXT NOT NULL DEFAULT 'present',
    edge_relational_type TEXT NOT NULL DEFAULT 'personal',
    edge_episodic_significance TEXT NOT NULL DEFAULT 'routine',
    edge_emotional_valence REAL NOT NULL DEFAULT 0.5,
    edge_emotional_label TEXT,

    -- 3. Temporal Truth State
    source_timestamp TEXT,
    temporal_expression TEXT,
    resolved_event_date TEXT,
    is_historical INTEGER DEFAULT 0,
    is_current INTEGER DEFAULT 1,
    superseded_at TEXT,
    superseded_by INTEGER,

    -- 4. Lifecycle
    tombstoned_at TEXT,
    first_learned_at TEXT DEFAULT (datetime('now')),
    last_confirmed_at TEXT DEFAULT (datetime('now')),

    -- 5. Structure / Reconstruction
    cluster_id TEXT,
    arc_id TEXT,
    sequence_number INTEGER,
    relational_entities TEXT,

    -- 6. Retrieval Aids
    edge_embedding BLOB,
    predicate_embedding BLOB,
    subject_type TEXT,
    object_type TEXT,
    edge_negated INTEGER DEFAULT 0,
    edge_mood TEXT DEFAULT 'indicative',
    episodic_fact TEXT,
    emotional_target TEXT,

    -- 7. Full Object + Context (write-time enrichment)
    object_full TEXT,
    context_entity TEXT,

    -- 8. Predicted Queries (write-time)
    pq_1 TEXT,
    pq_2 TEXT,
    pq_3 TEXT,
    pq_4 TEXT,

    -- 8. Verified Queries (read-time, passed verification loop)
    vq_1 TEXT,
    vq_2 TEXT,

    UNIQUE(user_id, source_text_hash, created_at)
);

CREATE INDEX IF NOT EXISTS idx_edges_user ON edges(user_id);
CREATE INDEX IF NOT EXISTS idx_edges_subject ON edges(user_id, subject);
CREATE INDEX IF NOT EXISTS idx_edges_predicate ON edges(user_id, predicate);
CREATE INDEX IF NOT EXISTS idx_edges_object ON edges(user_id, object);
CREATE INDEX IF NOT EXISTS idx_edges_schema_cat ON edges(user_id, edge_schematic_category);
CREATE INDEX IF NOT EXISTS idx_edges_subject_schema ON edges(user_id, subject, edge_schematic_category);
CREATE INDEX IF NOT EXISTS idx_edges_source_hash ON edges(user_id, source_text_hash);
CREATE INDEX IF NOT EXISTS idx_edges_is_current ON edges(user_id, is_current);
CREATE INDEX IF NOT EXISTS idx_edges_seq ON edges(user_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_edges_arc ON edges(user_id, arc_id);
CREATE INDEX IF NOT EXISTS idx_edges_cluster ON edges(user_id, cluster_id);
CREATE INDEX IF NOT EXISTS idx_edges_resolved_date ON edges(user_id, resolved_event_date);

-- Full-text search on edges (SPO + source_text + predicted/verified queries)
CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5(
    subject, predicate, object, source_text,
    pq_1, pq_2, pq_3, pq_4, vq_1, vq_2,
    content='edges', content_rowid='id'
);

-- =============================================================================
-- EDGE_EXTRACTION — How the parser interpreted the sentence.
-- One row per edge. Read path never touches this table.
-- =============================================================================
CREATE TABLE IF NOT EXISTS edge_extraction (
    edge_id INTEGER PRIMARY KEY REFERENCES edges(id),

    -- 7. Grammar / Extraction Metadata
    edge_negated INTEGER DEFAULT 0,
    edge_mood TEXT DEFAULT 'indicative',
    episodic_fact TEXT,
    emotional_target TEXT,
    extraction_rule TEXT,
    canonical_fields TEXT,

    -- 8. Derived Social Signal
    edge_affiliation REAL,

    -- 9. Provenance / Operational
    provenance_memory_id INTEGER,
    tombstone_reason TEXT,
    tombstone_op_id INTEGER,
    source_tag TEXT,

    -- 10. Weak / Future
    confidence REAL DEFAULT 0.9,
    utterance_type_id INTEGER,
    subject_type_confidence REAL,
    object_type_confidence REAL
);

-- =============================================================================
-- MEMORY TRACES TABLE (Distributed Trace Convergence Memory)
-- DEPRECATED: table retained for backward-compat reads; get_traces() removed 2026-04-24.
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
    FOREIGN KEY (relationship_id) REFERENCES edges(id)
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
    FOREIGN KEY (relationship_id) REFERENCES edges(id)
);

CREATE INDEX IF NOT EXISTS idx_predicted_queries_user ON predicted_queries(user_id);

-- =============================================================================
-- ARCS TABLE (Story arcs for proactive engine)
-- =============================================================================
CREATE TABLE IF NOT EXISTS arcs (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    topic TEXT NOT NULL,
    topic_embedding BLOB,
    start_edge_id INTEGER,
    emotional_baseline REAL,
    status TEXT DEFAULT 'open',
    created_at TEXT DEFAULT (datetime('now')),
    last_checked_at TEXT DEFAULT (datetime('now')),
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_arcs_user_status ON arcs(user_id, status);
CREATE INDEX IF NOT EXISTS idx_arcs_last_checked ON arcs(user_id, last_checked_at);

-- =============================================================================
-- TIMERS TABLE (Short-term reminders for proactive engine)
-- =============================================================================
CREATE TABLE IF NOT EXISTS timers (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    fire_at TEXT NOT NULL,
    callback_type TEXT,
    payload TEXT,
    fired INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_timers_user ON timers(user_id, fired);
CREATE INDEX IF NOT EXISTS idx_timers_fire ON timers(fire_at, fired);

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

    # --- Grammar Engine: object_type column on edges ---
    rel_cursor = conn.execute("PRAGMA table_info(edges)")
    rel_columns = {row[1] for row in rel_cursor.fetchall()}

    if "object_type" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN object_type TEXT DEFAULT 'unknown'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_object_type ON edges(user_id, object_type)")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN object_type")
        except Exception:
            pass

    # --- 594 Equation System: utterance_type_id + canonical_fields on edges ---
    if "utterance_type_id" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN utterance_type_id INTEGER")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN utterance_type_id")
        except Exception:
            pass

    if "canonical_fields" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN canonical_fields TEXT")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN canonical_fields")
        except Exception:
            pass

    # --- Supersession columns on edges (update handling) ---
    if "is_current" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN is_current INTEGER DEFAULT 1")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN is_current")
        except Exception:
            pass

    if "superseded_at" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN superseded_at TEXT")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN superseded_at")
        except Exception:
            pass

    if "superseded_by" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN superseded_by INTEGER")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN superseded_by")
        except Exception:
            pass

    # --- situation_id: DEPRECATED (2026-04-23). Column was never populated.
    # Existing databases retain the column for backward compat but it is
    # no longer added to new databases. No code reads or writes it. ---

    # --- Source timestamp: when the source utterance occurred ---
    # Used by temporal engine to resolve "when did X happen?" queries.
    # For Raya: this is the wall-clock time the user said it.
    # For LOCOMO/SDK: this is the session_date_time from the source data.
    if "source_timestamp" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN source_timestamp TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source_ts ON edges(user_id, source_timestamp)")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN source_timestamp")
        except Exception:
            pass

    # --- Continuous dimension columns on memory_traces (legacy T5 SRL naming, kept for schema compat) ---
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
    # and edge_embedding live DIRECTLY on the edges row.
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
        "edge_affiliation": "REAL DEFAULT 0.5",
    }
    for col, col_type in edge_trace_cols.items():
        if col not in rel_columns:
            try:
                conn.execute(f"ALTER TABLE edges ADD COLUMN {col} {col_type}")
                conn.commit()
                print(f"[DB] Applied: ALTER TABLE edges ADD COLUMN {col}")
            except Exception:
                pass

    # --- Predicate embedding cache (2026-04-24): store the embedded
    # predicate at write time so retrieval can skip embed_text() per edge.
    if "predicate_embedding" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN predicate_embedding BLOB")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN predicate_embedding")
        except Exception:
            pass

    # --- Forget/Show migration (2026-04-14): tombstone columns +
    # source provenance on edges, plus live-query indexes. ---
    forget_cols = {
        "tombstoned_at": "TEXT",
        "tombstone_reason": "TEXT",
        "tombstone_op_id": "TEXT",
        "source_text": "TEXT",
        "source_tag": "TEXT",
    }
    for col, col_type in forget_cols.items():
        if col not in rel_columns:
            try:
                conn.execute(f"ALTER TABLE edges ADD COLUMN {col} {col_type}")
                conn.commit()
                print(f"[DB] Applied: ALTER TABLE edges ADD COLUMN {col}")
            except Exception:
                pass

    forget_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_rel_live_subject "
        "ON edges(user_id, tombstoned_at, subject)",
        "CREATE INDEX IF NOT EXISTS idx_rel_live_object "
        "ON edges(user_id, tombstoned_at, object)",
        "CREATE INDEX IF NOT EXISTS idx_rel_live_source_ts "
        "ON edges(user_id, tombstoned_at, source_timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_rel_live_source_tag "
        "ON edges(user_id, tombstoned_at, source_tag)",
    ]
    for sql in forget_indexes:
        try:
            conn.execute(sql)
            conn.commit()
        except Exception:
            pass

    # --- Set-op 9-axis pipeline (2026-04-14): cluster_id + arc_id.
    # cluster_id is persisted at write time via TemporalEngine.cluster.
    # arc_id stays NULL (placeholder for multi-session arc grouping).
    if "cluster_id" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN cluster_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_cluster ON edges(user_id, cluster_id)")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN cluster_id")
        except Exception:
            pass

    if "arc_id" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN arc_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_arc ON edges(user_id, arc_id)")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN arc_id")
        except Exception:
            pass

    # --- Write-path foundation (2026-04-14): entity-type tagging on the
    # triple. object_type already exists from an earlier migration; add
    # subject_type on edges and ensure entity_type is indexed on
    # entities. Values drawn from the closed vocabulary resolved by
    # app.engines.type_resolver: PERSON | ORG | LOCATION | TIME | EVENT |
    # QUANTITY | WORK_OF_ART | PRODUCT | GENERIC.
    if "subject_type" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN subject_type TEXT")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN subject_type")
        except Exception:
            pass

    # --- Resolved event date (2026-04-25): the actual date of the event
    # described in the source text, resolved from DATE/TIME NER spans
    # against source_timestamp. Distinct from source_timestamp (session
    # wall-clock) — this is what the user said happened WHEN.
    # Root cause: "when" queries found the right edge but returned no
    # date because source_timestamp is session time, not event time.
    if "resolved_event_date" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN resolved_event_date TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_event_date ON edges(user_id, resolved_event_date)")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN resolved_event_date")
        except Exception:
            pass

    # --- Grammar engine decomposition columns (2026-04-25): persist
    # temporal_expression and relational_entities from TraceDecomposition.
    if "temporal_expression" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN temporal_expression TEXT")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN temporal_expression")
        except Exception:
            pass

    if "relational_entities" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN relational_entities TEXT")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN relational_entities")
        except Exception:
            pass

    # --- Type confidence columns (2026-04-24): signals whether NER and
    # WordNet agreed on the entity type. HIGH = agreement or single-source,
    # LOW = disagreement (type is uncertain). Used by retrieval as a soft
    # tiebreaker — low-confidence types don't gate, only nudge.
    if "subject_type_confidence" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN subject_type_confidence TEXT DEFAULT 'high'")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN subject_type_confidence")
        except Exception:
            pass
    if "object_type_confidence" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN object_type_confidence TEXT DEFAULT 'high'")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN object_type_confidence")
        except Exception:
            pass

    # Ensure object_type index exists with the canonical name expected by
    # the coherence gate (older migration used a DEFAULT 'unknown' and a
    # different index; we add the canonical one idempotently).
    for sql in (
        "CREATE INDEX IF NOT EXISTS idx_rel_object_type ON edges(user_id, object_type)",
        "CREATE INDEX IF NOT EXISTS idx_rel_subject_type ON edges(user_id, subject_type)",
    ):
        try:
            conn.execute(sql)
            conn.commit()
        except Exception:
            pass

    # --- sequence_number column (2026-04-25): narrative-time axis.
    # Root cause: memory.py lines 747-755 writes sequence_number via
    # UPDATE but the column was never formally added via ALTER TABLE or
    # included in the base CREATE TABLE. Retrieval.py references it in
    # ORDER BY clauses (lines 774, 956, 1183, 1309, 1351, 1396, 1439).
    # On databases created from MIGRATIONS alone, the column does not
    # exist and those ORDER BY clauses silently get NULLs.
    if "sequence_number" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN sequence_number INTEGER")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN sequence_number")
        except Exception:
            pass

    # --- Negation column (2026-04-27): grammar engine sets negated=True
    # on TraceDecomposition but _write_edge_traces never persisted it.
    if "edge_negated" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN edge_negated INTEGER DEFAULT 0")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN edge_negated")
        except Exception:
            pass

    # --- Mood column (2026-04-28): stores "indicative", "interrogative",
    # "imperative", "conditional" from grammar engine TraceDecomposition.
    # Retrieval filters out non-indicative edges so imposed facts from
    # questions/commands don't leak as answers.
    if "edge_mood" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN edge_mood TEXT DEFAULT 'indicative'")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN edge_mood")
        except Exception:
            pass

    # --- is_historical column (2026-04-29): grammar engine sets
    # is_historical=True for past-tense facts (e.g. "I used to work at
    # Google"). Retrieval can distinguish current vs historical facts.
    if "is_historical" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN is_historical INTEGER DEFAULT 0")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN is_historical")
        except Exception:
            pass

    # --- episodic_fact column (2026-04-29): normalized sentence-level
    # fact from TraceDecomposition. The canonical natural-language form
    # of what was stored, independent of triple decomposition.
    if "episodic_fact" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN episodic_fact TEXT")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN episodic_fact")
        except Exception:
            pass

    # --- emotional_target column (2026-04-29): what the emotion is
    # about (e.g. "the job interview" when user says "I'm nervous about
    # the job interview"). From TraceDecomposition.emotional_target.
    if "emotional_target" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN emotional_target TEXT")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN emotional_target")
        except Exception:
            pass

    # --- extraction_rule column (2026-04-29): provenance trace showing
    # which grammar rule produced this edge (trace|imposed|
    # free_indirect_speech etc.). From TraceDecomposition.extraction_rule.
    if "extraction_rule" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN extraction_rule TEXT")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN extraction_rule")
        except Exception:
            pass

    # --- source_text_hash column (Phase 6, 2026-04-25): SHA-256 of
    # source_text. Root cause: the full trace-primary migration
    # (scripts/migrate_traces_primary.py) replaces the UNIQUE constraint,
    # but memory.py store() needs to start writing hashes before the full
    # migration runs. This column-add is the incremental bridge so
    # store() can populate the hash on each INSERT.
    if "source_text_hash" not in rel_columns:
        try:
            conn.execute("ALTER TABLE edges ADD COLUMN source_text_hash TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source_hash ON edges(user_id, source_text_hash)")
            conn.commit()
            print("[DB] Applied: ALTER TABLE edges ADD COLUMN source_text_hash")
        except Exception:
            pass

    # entities.entity_type already exists in the base CREATE TABLE; ensure
    # its index is named idx_entity_type for the coherence gate.
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(user_id, entity_type)"
        )
        conn.commit()
    except Exception:
        pass

    # --- FTS5 full-text search on edges (BM25 hybrid retrieval) ---
    # Content-sync FTS5 virtual table: external content points to the
    # edges table. Queries use BM25 ranking over the concatenated
    # subject + predicate + object + source_text. Zero new dependencies —
    # FTS5 is built into SQLite 3.9+ (Python 3.10 ships 3.37+).
    #
    # Migration (2026-04-24): added source_text as 4th FTS column so BM25
    # can match the raw user utterance. Existing 3-column FTS tables are
    # detected and rebuilt automatically.
    _fts_needs_rebuild = False
    try:
        # Detect whether the FTS table exists and has the expected columns.
        # FTS5 content-sync tables expose columns via PRAGMA; if source_text
        # is missing we must drop + recreate.
        _fts_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(edges_fts)").fetchall()
        }
        if _fts_cols and "source_text" not in _fts_cols:
            conn.execute("DROP TABLE IF EXISTS edges_fts")
            conn.commit()
            _fts_needs_rebuild = True
            print("[DB] Dropped old 3-column edges_fts for source_text migration")
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts
            USING fts5(
                subject, predicate, object, source_text,
                content='edges',
                content_rowid='id'
            )
        """)
        conn.commit()
        print("[DB] Applied: CREATE VIRTUAL TABLE edges_fts (FTS5, 4-col)")
    except Exception:
        pass

    # Populate FTS5 index for any existing rows not yet indexed.
    # This is idempotent: INSERT OR IGNORE semantics via FTS5's
    # content-sync mechanism. For content-sync tables, we rebuild
    # if the table is empty (fresh migration) or after a schema rebuild.
    try:
        fts_count = conn.execute(
            "SELECT COUNT(*) FROM edges_fts"
        ).fetchone()[0]
        if fts_count == 0 or _fts_needs_rebuild:
            # Clear any stale rows from a partial state before full backfill.
            if _fts_needs_rebuild and fts_count > 0:
                conn.execute(
                    "INSERT INTO edges_fts(edges_fts) VALUES('delete-all')"
                )
            conn.execute("""
                INSERT INTO edges_fts(rowid, subject, predicate, object, source_text)
                SELECT id, subject, REPLACE(predicate, '_', ' '), object,
                       COALESCE(source_text, '')
                FROM edges
                WHERE tombstoned_at IS NULL
            """)
            conn.commit()
            print("[DB] Applied: backfill edges_fts from existing rows")
    except Exception:
        pass
