# db-models

Source: [app/db/models.py](/D:/Nura/Code/nura_living_memory_code/app/db/models.py)

```text
0001: # SQLite migrations (kept simple for v1). Use Alembic later if needed.
0002: 
0003: MIGRATIONS = """
0004: -- =============================================================================
0005: -- EPISODES TABLE (Episodic Memory) - Day-to-day conversations, DOES decay
0006: -- =============================================================================
0007: CREATE TABLE IF NOT EXISTS memories (
0008:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0009:     user_id INTEGER NOT NULL,
0010:     content TEXT NOT NULL,
0011:     memory_type TEXT CHECK(memory_type IN ('episodic','semantic','summary')) NOT NULL,
0012:     importance REAL DEFAULT 0.5,
0013:     embedding BLOB,
0014:     created_at TEXT DEFAULT (datetime('now')),
0015:     last_accessed_at TEXT,
0016:     temporal_tags TEXT,
0017:     metadata TEXT
0018: );
0019: 
0020: CREATE INDEX IF NOT EXISTS idx_memories_user_time ON memories(user_id, created_at);
0021: CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(user_id, memory_type);
0022: CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(user_id, importance DESC);
0023: 
0024: -- =============================================================================
0025: -- FACTS TABLE (Semantic Memory) - Personal truths, ONE value per key, NO decay
0026: -- Key-value pairs that get UPDATED, not accumulated
0027: -- =============================================================================
0028: CREATE TABLE IF NOT EXISTS facts (
0029:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0030:     user_id INTEGER NOT NULL,
0031:     key TEXT NOT NULL,
0032:     value TEXT NOT NULL,
0033:     confidence REAL DEFAULT 0.9,
0034:     last_confirmed_at TEXT DEFAULT (datetime('now')),
0035:     first_learned_at TEXT DEFAULT (datetime('now')),
0036:     provenance_memory_id INTEGER,
0037:     embedding BLOB,
0038:     history TEXT,  -- JSON array of previous values for contradiction tracking
0039:     UNIQUE(user_id, key)
0040: );
0041: 
0042: CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
0043: CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(user_id, key);
0044: 
0045: -- =============================================================================
0046: -- MILESTONES TABLE (Life Events) - Timestamped events, NO decay, permanent
0047: -- Deaths, marriages, graduations, major life changes
0048: -- =============================================================================
0049: CREATE TABLE IF NOT EXISTS milestones (
0050:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0051:     user_id INTEGER NOT NULL,
0052:     event_type TEXT NOT NULL,
0053:     event_date TEXT,
0054:     description TEXT NOT NULL,
0055:     confidence REAL DEFAULT 0.9,
0056:     created_at TEXT DEFAULT (datetime('now')),
0057:     provenance_memory_id INTEGER,
0058:     embedding BLOB,
0059:     metadata TEXT
0060: );
0061: 
0062: CREATE INDEX IF NOT EXISTS idx_milestones_user ON milestones(user_id);
0063: CREATE INDEX IF NOT EXISTS idx_milestones_type ON milestones(user_id, event_type);
0064: CREATE INDEX IF NOT EXISTS idx_milestones_date ON milestones(user_id, event_date);
0065: 
0066: CREATE TABLE IF NOT EXISTS adaptation_profiles (
0067:     user_id INTEGER PRIMARY KEY,
0068:     warmth REAL DEFAULT 0.5,
0069:     formality REAL DEFAULT 0.5,
0070:     initiative REAL DEFAULT 0.5,
0071:     check_in_frequency REAL DEFAULT 0.5,
0072:     updated_at TEXT DEFAULT (datetime('now'))
0073: );
0074: 
0075: CREATE TABLE IF NOT EXISTS relationship_metrics (
0076:     user_id INTEGER NOT NULL,
0077:     week INTEGER NOT NULL,
0078:     relationship_depth REAL,
0079:     disclosure_avg REAL,
0080:     emotional_events INTEGER,
0081:     return_rate REAL,
0082:     created_at TEXT DEFAULT (datetime('now')),
0083:     PRIMARY KEY (user_id, week)
0084: );
0085: 
0086: CREATE TABLE IF NOT EXISTS temporal_patterns (
0087:     user_id INTEGER NOT NULL,
0088:     pattern_type TEXT NOT NULL,
0089:     confidence REAL DEFAULT 0.5,
0090:     detected_at TEXT DEFAULT (datetime('now')),
0091:     example_memory_ids TEXT
0092: );
0093: 
0094: CREATE INDEX IF NOT EXISTS idx_temporal_patterns_lookup ON temporal_patterns(user_id, pattern_type);
0095: 
0096: -- =============================================================================
0097: -- PROACTIVE COOLDOWN (Rate limiting for proactive outreach)
0098: -- Persists ask counts and last ask time across sessions
0099: -- =============================================================================
0100: CREATE TABLE IF NOT EXISTS proactive_cooldown (
0101:     user_id INTEGER PRIMARY KEY,
0102:     last_asked_at TEXT,
0103:     asks_today INTEGER DEFAULT 0,
0104:     asks_date TEXT,  -- Date for asks_today (resets daily)
0105:     updated_at TEXT DEFAULT (datetime('now'))
0106: );
0107: 
0108: -- =============================================================================
0109: -- ENTITIES TABLE (Graph nodes) - People, places, things mentioned by user
0110: -- =============================================================================
0111: CREATE TABLE IF NOT EXISTS entities (
0112:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0113:     user_id INTEGER NOT NULL,
0114:     name TEXT NOT NULL,           -- "Emily", "FitLife", "Jake"
0115:     entity_type TEXT,             -- "person", "place", "organization", "thing"
0116:     attributes TEXT,              -- JSON: {"status": "pregnant", "occupation": "trainer"}
0117:     first_mentioned_at TEXT DEFAULT (datetime('now')),
0118:     last_mentioned_at TEXT DEFAULT (datetime('now')),
0119:     mention_count INTEGER DEFAULT 1,
0120:     embedding BLOB,
0121:     UNIQUE(user_id, name)
0122: );
0123: 
0124: CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id);
0125: CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(user_id, name);
0126: CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(user_id, entity_type);
0127: 
0128: -- =============================================================================
0129: -- RELATIONSHIPS TABLE (Graph edges) - Trace-primary schema (2026-04-25)
0130: -- source_text_hash is the dedup key; subject/predicate/object are optional
0131: -- derived fields. 5 trace columns carry NOT NULL defaults (same sentinel
0132: -- values already used in memory.py _write_edge_traces() at lines 867-869
0133: -- and in scripts/migrate_traces_primary.py v2 table at lines 151-156).
0134: -- Root cause: UNIQUE(user_id, subject, predicate, object) caused two
0135: -- distinct utterances producing the same triple to collide, losing trace
0136: -- data from the first. source_text_hash dedup preserves each utterance.
0137: -- =============================================================================
0138: CREATE TABLE IF NOT EXISTS relationships (
0139:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0140:     user_id INTEGER NOT NULL,
0141:     source_text TEXT NOT NULL DEFAULT '',
0142:     source_text_hash TEXT NOT NULL DEFAULT '',
0143:     created_at TEXT DEFAULT (datetime('now')),
0144: 
0145:     -- 5 traces (primary data)
0146:     edge_schematic_category TEXT NOT NULL DEFAULT 'uncategorized',
0147:     edge_temporal_context TEXT NOT NULL DEFAULT 'present',
0148:     edge_relational_type TEXT NOT NULL DEFAULT 'personal',
0149:     edge_episodic_significance TEXT NOT NULL DEFAULT 'routine',
0150:     edge_emotional_valence REAL NOT NULL DEFAULT 0.5,
0151:     edge_emotional_label TEXT,
0152:     edge_affiliation REAL,
0153: 
0154:     -- Derived triple
0155:     subject TEXT,
0156:     predicate TEXT,
0157:     object TEXT,
0158: 
0159:     -- Type resolution (grammar engine + type_resolver)
0160:     subject_type TEXT,
0161:     object_type TEXT,
0162:     subject_type_confidence REAL,
0163:     object_type_confidence REAL,
0164: 
0165:     -- Temporal (temporal engine)
0166:     source_timestamp TEXT,
0167:     temporal_expression TEXT,
0168:     resolved_event_date TEXT,
0169:     is_historical INTEGER DEFAULT 0,
0170:     is_current INTEGER DEFAULT 1,
0171:     superseded_at TEXT,
0172:     superseded_by INTEGER,
0173:     tombstoned_at TEXT,
0174:     tombstone_reason TEXT,
0175:     tombstone_op_id INTEGER,
0176: 
0177:     -- Embeddings (MiniLM)
0178:     edge_embedding BLOB,
0179:     predicate_embedding BLOB,
0180: 
0181:     -- Structure (memory engine)
0182:     cluster_id TEXT,
0183:     arc_id TEXT,
0184:     sequence_number INTEGER,
0185:     utterance_type_id INTEGER,
0186:     source_tag TEXT,
0187:     relational_entities TEXT,
0188: 
0189:     -- Grammar traces
0190:     edge_negated INTEGER DEFAULT 0,
0191:     edge_mood TEXT DEFAULT 'indicative',
0192:     episodic_fact TEXT,
0193:     emotional_target TEXT,
0194:     extraction_rule TEXT,
0195:     canonical_fields TEXT,
0196: 
0197:     -- Metadata
0198:     confidence REAL DEFAULT 0.9,
0199:     first_learned_at TEXT DEFAULT (datetime('now')),
0200:     last_confirmed_at TEXT DEFAULT (datetime('now')),
0201:     provenance_memory_id INTEGER,
0202:     UNIQUE(user_id, source_text_hash, created_at)
0203: );
0204: 
0205: CREATE INDEX IF NOT EXISTS idx_relationships_user ON relationships(user_id);
0206: CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(user_id, subject);
0207: CREATE INDEX IF NOT EXISTS idx_relationships_predicate ON relationships(user_id, predicate);
0208: CREATE INDEX IF NOT EXISTS idx_relationships_object ON relationships(user_id, object);
0209: CREATE INDEX IF NOT EXISTS idx_rel_schema_cat ON relationships(user_id, edge_schematic_category);
0210: CREATE INDEX IF NOT EXISTS idx_rel_subject_schema ON relationships(user_id, subject, edge_schematic_category);
0211: CREATE INDEX IF NOT EXISTS idx_rel_source_hash ON relationships(user_id, source_text_hash);
0212: CREATE INDEX IF NOT EXISTS idx_rel_is_current ON relationships(user_id, is_current);
0213: CREATE INDEX IF NOT EXISTS idx_rel_seq ON relationships(user_id, sequence_number);
0214: CREATE INDEX IF NOT EXISTS idx_rel_arc ON relationships(user_id, arc_id);
0215: CREATE INDEX IF NOT EXISTS idx_rel_cluster ON relationships(user_id, cluster_id);
0216: CREATE INDEX IF NOT EXISTS idx_rel_resolved_date ON relationships(user_id, resolved_event_date);
0217: CREATE INDEX IF NOT EXISTS idx_rel_tombstoned ON relationships(user_id, tombstoned_at);
0218: 
0219: -- Full-text search on relationships (subject, predicate, object, source_text)
0220: CREATE VIRTUAL TABLE IF NOT EXISTS relationships_fts USING fts5(
0221:     subject, predicate, object, source_text,
0222:     content='relationships', content_rowid='id'
0223: );
0224: 
0225: -- =============================================================================
0226: -- MEMORY TRACES TABLE (Distributed Trace Convergence Memory)
0227: -- DEPRECATED: table retained for backward-compat reads; get_traces() removed 2026-04-24.
0228: -- Decomposed trace metadata per relationship triple
0229: -- =============================================================================
0230: CREATE TABLE IF NOT EXISTS memory_traces (
0231:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0232:     relationship_id INTEGER UNIQUE,
0233:     user_id INTEGER NOT NULL,
0234:     -- Emotional trace (3 dimensions)
0235:     valence REAL DEFAULT 0.5,
0236:     affiliation REAL DEFAULT 0.5,
0237:     emotional_intensity REAL,
0238:     -- Relational trace (3 dimensions)
0239:     relational_type TEXT,
0240:     relational_proximity REAL DEFAULT 0.5,
0241:     relational_valence REAL DEFAULT 0.5,
0242:     -- Episodic trace (2 dimensions)
0243:     episodic_significance TEXT DEFAULT 'routine',
0244:     episodic_narrative_position TEXT DEFAULT 'ongoing',
0245:     -- Temporal trace
0246:     temporal_context TEXT CHECK(temporal_context IN ('past','present','future','ongoing')),
0247:     -- Schematic trace
0248:     schema_category TEXT,
0249:     schema_confidence REAL,
0250:     -- Rehearsal
0251:     access_count INTEGER DEFAULT 0,
0252:     last_accessed_at TEXT,
0253:     created_at TEXT DEFAULT (datetime('now')),
0254:     FOREIGN KEY (relationship_id) REFERENCES relationships(id)
0255: );
0256: 
0257: CREATE INDEX IF NOT EXISTS idx_memory_traces_user ON memory_traces(user_id);
0258: 
0259: -- =============================================================================
0260: -- PREDICTED QUERIES TABLE (Distributed Trace Convergence Memory)
0261: -- Pre-computed query-answer fingerprints written at extraction time
0262: -- =============================================================================
0263: CREATE TABLE IF NOT EXISTS predicted_queries (
0264:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0265:     relationship_id INTEGER,
0266:     user_id INTEGER NOT NULL,
0267:     predicted_question TEXT NOT NULL,
0268:     answer_text TEXT NOT NULL,
0269:     answer_subject TEXT,
0270:     question_embedding BLOB NOT NULL,
0271:     confidence REAL DEFAULT 0.9,
0272:     created_at TEXT DEFAULT (datetime('now')),
0273:     FOREIGN KEY (relationship_id) REFERENCES relationships(id)
0274: );
0275: 
0276: CREATE INDEX IF NOT EXISTS idx_predicted_queries_user ON predicted_queries(user_id);
0277: 
0278: -- =============================================================================
0279: -- ARCS TABLE (Story arcs for proactive engine)
0280: -- =============================================================================
0281: CREATE TABLE IF NOT EXISTS arcs (
0282:     id TEXT PRIMARY KEY,
0283:     user_id INTEGER NOT NULL,
0284:     topic TEXT NOT NULL,
0285:     topic_embedding BLOB,
0286:     start_edge_id INTEGER,
0287:     emotional_baseline REAL,
0288:     status TEXT DEFAULT 'open',
0289:     created_at TEXT DEFAULT (datetime('now')),
0290:     last_checked_at TEXT DEFAULT (datetime('now')),
0291:     resolved_at TEXT
0292: );
0293: 
0294: CREATE INDEX IF NOT EXISTS idx_arcs_user_status ON arcs(user_id, status);
0295: CREATE INDEX IF NOT EXISTS idx_arcs_last_checked ON arcs(user_id, last_checked_at);
0296: 
0297: -- =============================================================================
0298: -- TIMERS TABLE (Short-term reminders for proactive engine)
0299: -- =============================================================================
0300: CREATE TABLE IF NOT EXISTS timers (
0301:     id TEXT PRIMARY KEY,
0302:     user_id INTEGER NOT NULL,
0303:     fire_at TEXT NOT NULL,
0304:     callback_type TEXT,
0305:     payload TEXT,
0306:     fired INTEGER DEFAULT 0,
0307:     created_at TEXT DEFAULT (datetime('now'))
0308: );
0309: 
0310: CREATE INDEX IF NOT EXISTS idx_timers_user ON timers(user_id, fired);
0311: CREATE INDEX IF NOT EXISTS idx_timers_fire ON timers(fire_at, fired);
0312: 
0313: -- =============================================================================
0314: -- SCHEMA UPGRADES (for existing databases)
0315: -- These are safe to run multiple times
0316: -- =============================================================================
0317: 
0318: -- Add missing columns to facts table (for older databases)
0319: -- SQLite doesn't support IF NOT EXISTS for columns, so we use a workaround
0320: """
0321: 
0322: # Additional migration to add missing columns
0323: SCHEMA_UPGRADES = """
0324: -- Upgrade facts table if columns are missing
0325: -- These will fail silently if columns already exist
0326: """
0327: 
0328: def run_schema_upgrades(conn) -> None:
0329:     """Add missing columns to existing databases."""
0330:     # Check facts table columns
0331:     cursor = conn.execute("PRAGMA table_info(facts)")
0332:     existing_columns = {row[1] for row in cursor.fetchall()}
0333: 
0334:     upgrades = []
0335:     if "history" not in existing_columns:
0336:         upgrades.append("ALTER TABLE facts ADD COLUMN history TEXT")
0337:     if "first_learned_at" not in existing_columns:
0338:         upgrades.append("ALTER TABLE facts ADD COLUMN first_learned_at TEXT DEFAULT (datetime('now'))")
0339:     if "embedding" not in existing_columns:
0340:         upgrades.append("ALTER TABLE facts ADD COLUMN embedding BLOB")
0341: 
0342:     for sql in upgrades:
0343:         try:
0344:             conn.execute(sql)
0345:             print(f"[DB] Applied: {sql[:50]}...")
0346:         except Exception:
0347:             pass
0348: 
0349:     if upgrades:
0350:         conn.commit()
0351: 
0352:     # --- Grammar Engine: object_type column on relationships ---
0353:     rel_cursor = conn.execute("PRAGMA table_info(relationships)")
0354:     rel_columns = {row[1] for row in rel_cursor.fetchall()}
0355: 
0356:     if "object_type" not in rel_columns:
0357:         try:
0358:             conn.execute("ALTER TABLE relationships ADD COLUMN object_type TEXT DEFAULT 'unknown'")
0359:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_object_type ON relationships(user_id, object_type)")
0360:             conn.commit()
0361:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN object_type")
0362:         except Exception:
0363:             pass
0364: 
0365:     # --- 594 Equation System: utterance_type_id + canonical_fields on relationships ---
0366:     if "utterance_type_id" not in rel_columns:
0367:         try:
0368:             conn.execute("ALTER TABLE relationships ADD COLUMN utterance_type_id INTEGER")
0369:             conn.commit()
0370:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN utterance_type_id")
0371:         except Exception:
0372:             pass
0373: 
0374:     if "canonical_fields" not in rel_columns:
0375:         try:
0376:             conn.execute("ALTER TABLE relationships ADD COLUMN canonical_fields TEXT")
0377:             conn.commit()
0378:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN canonical_fields")
0379:         except Exception:
0380:             pass
0381: 
0382:     # --- Supersession columns on relationships (update handling) ---
0383:     if "is_current" not in rel_columns:
0384:         try:
0385:             conn.execute("ALTER TABLE relationships ADD COLUMN is_current INTEGER DEFAULT 1")
0386:             conn.commit()
0387:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN is_current")
0388:         except Exception:
0389:             pass
0390: 
0391:     if "superseded_at" not in rel_columns:
0392:         try:
0393:             conn.execute("ALTER TABLE relationships ADD COLUMN superseded_at TEXT")
0394:             conn.commit()
0395:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN superseded_at")
0396:         except Exception:
0397:             pass
0398: 
0399:     if "superseded_by" not in rel_columns:
0400:         try:
0401:             conn.execute("ALTER TABLE relationships ADD COLUMN superseded_by INTEGER")
0402:             conn.commit()
0403:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN superseded_by")
0404:         except Exception:
0405:             pass
0406: 
0407:     # --- situation_id: DEPRECATED (2026-04-23). Column was never populated.
0408:     # Existing databases retain the column for backward compat but it is
0409:     # no longer added to new databases. No code reads or writes it. ---
0410: 
0411:     # --- Source timestamp: when the source utterance occurred ---
0412:     # Used by temporal engine to resolve "when did X happen?" queries.
0413:     # For Raya: this is the wall-clock time the user said it.
0414:     # For LOCOMO/SDK: this is the session_date_time from the source data.
0415:     if "source_timestamp" not in rel_columns:
0416:         try:
0417:             conn.execute("ALTER TABLE relationships ADD COLUMN source_timestamp TEXT")
0418:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source_ts ON relationships(user_id, source_timestamp)")
0419:             conn.commit()
0420:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN source_timestamp")
0421:         except Exception:
0422:             pass
0423: 
0424:     # --- Continuous dimension columns on memory_traces (legacy T5 SRL naming, kept for schema compat) ---
0425:     trace_cursor = conn.execute("PRAGMA table_info(memory_traces)")
0426:     trace_columns = {row[1] for row in trace_cursor.fetchall()}
0427: 
0428:     t5_upgrades = []
0429:     for col in ("valence", "affiliation"):
0430:         if col not in trace_columns:
0431:             t5_upgrades.append(f"ALTER TABLE memory_traces ADD COLUMN {col} REAL DEFAULT 0.5")
0432: 
0433:     # Migrate existing categorical emotional_valence â†’ continuous valence
0434:     migrate_valence = "emotional_valence" in trace_columns and "valence" not in trace_columns
0435: 
0436:     for sql in t5_upgrades:
0437:         try:
0438:             conn.execute(sql)
0439:             print(f"[DB] Applied: {sql[:60]}...")
0440:         except Exception:
0441:             pass
0442: 
0443:     if t5_upgrades:
0444:         conn.commit()
0445: 
0446:     # Backfill: convert old categorical values to continuous
0447:     if migrate_valence:
0448:         try:
0449:             conn.execute("""UPDATE memory_traces SET valence = CASE
0450:                 WHEN emotional_valence = 'positive' THEN 0.8
0451:                 WHEN emotional_valence = 'negative' THEN 0.2
0452:                 WHEN emotional_valence = 'mixed' THEN 0.5
0453:                 ELSE 0.5 END
0454:                 WHERE valence = 0.5 AND emotional_valence IS NOT NULL""")
0455:             conn.commit()
0456:             print("[DB] Applied: backfill emotional_valence â†’ valence")
0457:         except Exception:
0458:             pass
0459: 
0460:     # --- DTCM 5-trace upgrade: relational + episodic columns ---
0461:     new_trace_cols = {
0462:         "relational_type": "TEXT",
0463:         "relational_proximity": "REAL DEFAULT 0.5",
0464:         "relational_valence": "REAL DEFAULT 0.5",
0465:         "episodic_significance": "TEXT DEFAULT 'routine'",
0466:         "episodic_narrative_position": "TEXT DEFAULT 'ongoing'",
0467:     }
0468:     for col, col_type in new_trace_cols.items():
0469:         if col not in trace_columns:
0470:             try:
0471:                 conn.execute(f"ALTER TABLE memory_traces ADD COLUMN {col} {col_type}")
0472:                 conn.commit()
0473:                 print(f"[DB] Applied: ALTER TABLE memory_traces ADD COLUMN {col}")
0474:             except Exception:
0475:                 pass
0476: 
0477:     # --- Filter â†’ Complete migration (2026-04-12): trace columns
0478:     # and edge_embedding live DIRECTLY on the relationships row.
0479:     # memory_traces is kept for backward-compat read but no longer
0480:     # written. predicted_queries is no longer used â€” edge_embedding
0481:     # is the access path. ---
0482:     edge_trace_cols = {
0483:         "edge_embedding": "BLOB",
0484:         "edge_emotional_valence": "REAL DEFAULT 0.5",
0485:         "edge_emotional_label": "TEXT",
0486:         "edge_schematic_category": "TEXT",
0487:         "edge_episodic_significance": "TEXT",
0488:         "edge_relational_type": "TEXT",
0489:         "edge_temporal_context": "TEXT",
0490:         "edge_affiliation": "REAL DEFAULT 0.5",
0491:     }
0492:     for col, col_type in edge_trace_cols.items():
0493:         if col not in rel_columns:
0494:             try:
0495:                 conn.execute(f"ALTER TABLE relationships ADD COLUMN {col} {col_type}")
0496:                 conn.commit()
0497:                 print(f"[DB] Applied: ALTER TABLE relationships ADD COLUMN {col}")
0498:             except Exception:
0499:                 pass
0500: 
0501:     # --- Predicate embedding cache (2026-04-24): store the embedded
0502:     # predicate at write time so retrieval can skip embed_text() per edge.
0503:     if "predicate_embedding" not in rel_columns:
0504:         try:
0505:             conn.execute("ALTER TABLE relationships ADD COLUMN predicate_embedding BLOB")
0506:             conn.commit()
0507:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN predicate_embedding")
0508:         except Exception:
0509:             pass
0510: 
0511:     # --- Forget/Show migration (2026-04-14): tombstone columns +
0512:     # source provenance on relationships, plus live-query indexes. ---
0513:     forget_cols = {
0514:         "tombstoned_at": "TEXT",
0515:         "tombstone_reason": "TEXT",
0516:         "tombstone_op_id": "TEXT",
0517:         "source_text": "TEXT",
0518:         "source_tag": "TEXT",
0519:     }
0520:     for col, col_type in forget_cols.items():
0521:         if col not in rel_columns:
0522:             try:
0523:                 conn.execute(f"ALTER TABLE relationships ADD COLUMN {col} {col_type}")
0524:                 conn.commit()
0525:                 print(f"[DB] Applied: ALTER TABLE relationships ADD COLUMN {col}")
0526:             except Exception:
0527:                 pass
0528: 
0529:     forget_indexes = [
0530:         "CREATE INDEX IF NOT EXISTS idx_rel_live_subject "
0531:         "ON relationships(user_id, tombstoned_at, subject)",
0532:         "CREATE INDEX IF NOT EXISTS idx_rel_live_object "
0533:         "ON relationships(user_id, tombstoned_at, object)",
0534:         "CREATE INDEX IF NOT EXISTS idx_rel_live_source_ts "
0535:         "ON relationships(user_id, tombstoned_at, source_timestamp)",
0536:         "CREATE INDEX IF NOT EXISTS idx_rel_live_source_tag "
0537:         "ON relationships(user_id, tombstoned_at, source_tag)",
0538:     ]
0539:     for sql in forget_indexes:
0540:         try:
0541:             conn.execute(sql)
0542:             conn.commit()
0543:         except Exception:
0544:             pass
0545: 
0546:     # --- Set-op 9-axis pipeline (2026-04-14): cluster_id + arc_id.
0547:     # cluster_id is persisted at write time via TemporalEngine.cluster.
0548:     # arc_id stays NULL (placeholder for multi-session arc grouping).
0549:     if "cluster_id" not in rel_columns:
0550:         try:
0551:             conn.execute("ALTER TABLE relationships ADD COLUMN cluster_id TEXT")
0552:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_cluster ON relationships(user_id, cluster_id)")
0553:             conn.commit()
0554:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN cluster_id")
0555:         except Exception:
0556:             pass
0557: 
0558:     if "arc_id" not in rel_columns:
0559:         try:
0560:             conn.execute("ALTER TABLE relationships ADD COLUMN arc_id TEXT")
0561:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_arc ON relationships(user_id, arc_id)")
0562:             conn.commit()
0563:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN arc_id")
0564:         except Exception:
0565:             pass
0566: 
0567:     # --- Write-path foundation (2026-04-14): entity-type tagging on the
0568:     # triple. object_type already exists from an earlier migration; add
0569:     # subject_type on relationships and ensure entity_type is indexed on
0570:     # entities. Values drawn from the closed vocabulary resolved by
0571:     # app.engines.type_resolver: PERSON | ORG | LOCATION | TIME | EVENT |
0572:     # QUANTITY | WORK_OF_ART | PRODUCT | GENERIC.
0573:     if "subject_type" not in rel_columns:
0574:         try:
0575:             conn.execute("ALTER TABLE relationships ADD COLUMN subject_type TEXT")
0576:             conn.commit()
0577:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN subject_type")
0578:         except Exception:
0579:             pass
0580: 
0581:     # --- Resolved event date (2026-04-25): the actual date of the event
0582:     # described in the source text, resolved from DATE/TIME NER spans
0583:     # against source_timestamp. Distinct from source_timestamp (session
0584:     # wall-clock) â€” this is what the user said happened WHEN.
0585:     # Root cause: "when" queries found the right edge but returned no
0586:     # date because source_timestamp is session time, not event time.
0587:     if "resolved_event_date" not in rel_columns:
0588:         try:
0589:             conn.execute("ALTER TABLE relationships ADD COLUMN resolved_event_date TEXT")
0590:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_event_date ON relationships(user_id, resolved_event_date)")
0591:             conn.commit()
0592:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN resolved_event_date")
0593:         except Exception:
0594:             pass
0595: 
0596:     # --- Grammar engine decomposition columns (2026-04-25): persist
0597:     # temporal_expression and relational_entities from TraceDecomposition.
0598:     if "temporal_expression" not in rel_columns:
0599:         try:
0600:             conn.execute("ALTER TABLE relationships ADD COLUMN temporal_expression TEXT")
0601:             conn.commit()
0602:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN temporal_expression")
0603:         except Exception:
0604:             pass
0605: 
0606:     if "relational_entities" not in rel_columns:
0607:         try:
0608:             conn.execute("ALTER TABLE relationships ADD COLUMN relational_entities TEXT")
0609:             conn.commit()
0610:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN relational_entities")
0611:         except Exception:
0612:             pass
0613: 
0614:     # --- Type confidence columns (2026-04-24): signals whether NER and
0615:     # WordNet agreed on the entity type. HIGH = agreement or single-source,
0616:     # LOW = disagreement (type is uncertain). Used by retrieval as a soft
0617:     # tiebreaker â€” low-confidence types don't gate, only nudge.
0618:     if "subject_type_confidence" not in rel_columns:
0619:         try:
0620:             conn.execute("ALTER TABLE relationships ADD COLUMN subject_type_confidence TEXT DEFAULT 'high'")
0621:             conn.commit()
0622:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN subject_type_confidence")
0623:         except Exception:
0624:             pass
0625:     if "object_type_confidence" not in rel_columns:
0626:         try:
0627:             conn.execute("ALTER TABLE relationships ADD COLUMN object_type_confidence TEXT DEFAULT 'high'")
0628:             conn.commit()
0629:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN object_type_confidence")
0630:         except Exception:
0631:             pass
0632: 
0633:     # Ensure object_type index exists with the canonical name expected by
0634:     # the coherence gate (older migration used a DEFAULT 'unknown' and a
0635:     # different index; we add the canonical one idempotently).
0636:     for sql in (
0637:         "CREATE INDEX IF NOT EXISTS idx_rel_object_type ON relationships(user_id, object_type)",
0638:         "CREATE INDEX IF NOT EXISTS idx_rel_subject_type ON relationships(user_id, subject_type)",
0639:     ):
0640:         try:
0641:             conn.execute(sql)
0642:             conn.commit()
0643:         except Exception:
0644:             pass
0645: 
0646:     # --- sequence_number column (2026-04-25): narrative-time axis.
0647:     # Root cause: memory.py lines 747-755 writes sequence_number via
0648:     # UPDATE but the column was never formally added via ALTER TABLE or
0649:     # included in the base CREATE TABLE. Retrieval.py references it in
0650:     # ORDER BY clauses (lines 774, 956, 1183, 1309, 1351, 1396, 1439).
0651:     # On databases created from MIGRATIONS alone, the column does not
0652:     # exist and those ORDER BY clauses silently get NULLs.
0653:     if "sequence_number" not in rel_columns:
0654:         try:
0655:             conn.execute("ALTER TABLE relationships ADD COLUMN sequence_number INTEGER")
0656:             conn.commit()
0657:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN sequence_number")
0658:         except Exception:
0659:             pass
0660: 
0661:     # --- Negation column (2026-04-27): grammar engine sets negated=True
0662:     # on TraceDecomposition but _write_edge_traces never persisted it.
0663:     if "edge_negated" not in rel_columns:
0664:         try:
0665:             conn.execute("ALTER TABLE relationships ADD COLUMN edge_negated INTEGER DEFAULT 0")
0666:             conn.commit()
0667:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN edge_negated")
0668:         except Exception:
0669:             pass
0670: 
0671:     # --- Mood column (2026-04-28): stores "indicative", "interrogative",
0672:     # "imperative", "conditional" from grammar engine TraceDecomposition.
0673:     # Retrieval filters out non-indicative edges so imposed facts from
0674:     # questions/commands don't leak as answers.
0675:     if "edge_mood" not in rel_columns:
0676:         try:
0677:             conn.execute("ALTER TABLE relationships ADD COLUMN edge_mood TEXT DEFAULT 'indicative'")
0678:             conn.commit()
0679:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN edge_mood")
0680:         except Exception:
0681:             pass
0682: 
0683:     # --- is_historical column (2026-04-29): grammar engine sets
0684:     # is_historical=True for past-tense facts (e.g. "I used to work at
0685:     # Google"). Retrieval can distinguish current vs historical facts.
0686:     if "is_historical" not in rel_columns:
0687:         try:
0688:             conn.execute("ALTER TABLE relationships ADD COLUMN is_historical INTEGER DEFAULT 0")
0689:             conn.commit()
0690:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN is_historical")
0691:         except Exception:
0692:             pass
0693: 
0694:     # --- episodic_fact column (2026-04-29): normalized sentence-level
0695:     # fact from TraceDecomposition. The canonical natural-language form
0696:     # of what was stored, independent of triple decomposition.
0697:     if "episodic_fact" not in rel_columns:
0698:         try:
0699:             conn.execute("ALTER TABLE relationships ADD COLUMN episodic_fact TEXT")
0700:             conn.commit()
0701:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN episodic_fact")
0702:         except Exception:
0703:             pass
0704: 
0705:     # --- emotional_target column (2026-04-29): what the emotion is
0706:     # about (e.g. "the job interview" when user says "I'm nervous about
0707:     # the job interview"). From TraceDecomposition.emotional_target.
0708:     if "emotional_target" not in rel_columns:
0709:         try:
0710:             conn.execute("ALTER TABLE relationships ADD COLUMN emotional_target TEXT")
0711:             conn.commit()
0712:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN emotional_target")
0713:         except Exception:
0714:             pass
0715: 
0716:     # --- extraction_rule column (2026-04-29): provenance trace showing
0717:     # which grammar rule produced this edge (trace|imposed|
0718:     # free_indirect_speech etc.). From TraceDecomposition.extraction_rule.
0719:     if "extraction_rule" not in rel_columns:
0720:         try:
0721:             conn.execute("ALTER TABLE relationships ADD COLUMN extraction_rule TEXT")
0722:             conn.commit()
0723:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN extraction_rule")
0724:         except Exception:
0725:             pass
0726: 
0727:     # --- source_text_hash column (Phase 6, 2026-04-25): SHA-256 of
0728:     # source_text. Root cause: the full trace-primary migration
0729:     # (scripts/migrate_traces_primary.py) replaces the UNIQUE constraint,
0730:     # but memory.py store() needs to start writing hashes before the full
0731:     # migration runs. This column-add is the incremental bridge so
0732:     # store() can populate the hash on each INSERT.
0733:     if "source_text_hash" not in rel_columns:
0734:         try:
0735:             conn.execute("ALTER TABLE relationships ADD COLUMN source_text_hash TEXT")
0736:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source_hash ON relationships(user_id, source_text_hash)")
0737:             conn.commit()
0738:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN source_text_hash")
0739:         except Exception:
0740:             pass
0741: 
0742:     # entities.entity_type already exists in the base CREATE TABLE; ensure
0743:     # its index is named idx_entity_type for the coherence gate.
0744:     try:
0745:         conn.execute(
0746:             "CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(user_id, entity_type)"
0747:         )
0748:         conn.commit()
0749:     except Exception:
0750:         pass
0751: 
0752:     # --- FTS5 full-text search on relationships (BM25 hybrid retrieval) ---
0753:     # Content-sync FTS5 virtual table: external content points to the
0754:     # relationships table. Queries use BM25 ranking over the concatenated
0755:     # subject + predicate + object + source_text. Zero new dependencies â€”
0756:     # FTS5 is built into SQLite 3.9+ (Python 3.10 ships 3.37+).
0757:     #
0758:     # Migration (2026-04-24): added source_text as 4th FTS column so BM25
0759:     # can match the raw user utterance. Existing 3-column FTS tables are
0760:     # detected and rebuilt automatically.
0761:     _fts_needs_rebuild = False
0762:     try:
0763:         # Detect whether the FTS table exists and has the expected columns.
0764:         # FTS5 content-sync tables expose columns via PRAGMA; if source_text
0765:         # is missing we must drop + recreate.
0766:         _fts_cols = {
0767:             row[1]
0768:             for row in conn.execute("PRAGMA table_info(relationships_fts)").fetchall()
0769:         }
0770:         if _fts_cols and "source_text" not in _fts_cols:
0771:             conn.execute("DROP TABLE IF EXISTS relationships_fts")
0772:             conn.commit()
0773:             _fts_needs_rebuild = True
0774:             print("[DB] Dropped old 3-column relationships_fts for source_text migration")
0775:     except Exception:
0776:         pass
0777: 
0778:     try:
0779:         conn.execute("""
0780:             CREATE VIRTUAL TABLE IF NOT EXISTS relationships_fts
0781:             USING fts5(
0782:                 subject, predicate, object, source_text,
0783:                 content='relationships',
0784:                 content_rowid='id'
0785:             )
0786:         """)
0787:         conn.commit()
0788:         print("[DB] Applied: CREATE VIRTUAL TABLE relationships_fts (FTS5, 4-col)")
0789:     except Exception:
0790:         pass
0791: 
0792:     # Populate FTS5 index for any existing rows not yet indexed.
0793:     # This is idempotent: INSERT OR IGNORE semantics via FTS5's
0794:     # content-sync mechanism. For content-sync tables, we rebuild
0795:     # if the table is empty (fresh migration) or after a schema rebuild.
0796:     try:
0797:         fts_count = conn.execute(
0798:             "SELECT COUNT(*) FROM relationships_fts"
0799:         ).fetchone()[0]
0800:         if fts_count == 0 or _fts_needs_rebuild:
0801:             # Clear any stale rows from a partial state before full backfill.
0802:             if _fts_needs_rebuild and fts_count > 0:
0803:                 conn.execute(
0804:                     "INSERT INTO relationships_fts(relationships_fts) VALUES('delete-all')"
0805:                 )
0806:             conn.execute("""
0807:                 INSERT INTO relationships_fts(rowid, subject, predicate, object, source_text)
0808:                 SELECT id, subject, REPLACE(predicate, '_', ' '), object,
0809:                        COALESCE(source_text, '')
0810:                 FROM relationships
0811:                 WHERE tombstoned_at IS NULL
0812:             """)
0813:             conn.commit()
0814:             print("[DB] Applied: backfill relationships_fts from existing rows")
0815:     except Exception:
0816:         pass
```
