# ============================================================================
# TRACE-PRIMARY MEMORY ENGINE (2026-04-29 three-phase rewrite)
#
# Three-phase store(): PREPARE (pure computation) -> WRITE (one SQL) -> SIDE-EFFECTS
# No cosine anchors. No triple validation gate.
# If the grammar engine produced a trace decomposition, store it.
# ============================================================================
"""
MemoryEngine -- trace-primary store + all derived views.

Three-phase store path:
  Phase 1 PREPARE: Extract all column values into a flat dict (no DB access)
  Phase 2 WRITE:   One INSERT or UPDATE for the relationship row + FTS5
  Phase 3 SIDE-EFFECTS: facts/milestones, predicted queries, clustering, arcs

Public interface:
    store()                 -> three-phase write + side-effects
    ingest_text()           -> grammar_engine.process() -> store() per decomp
    get_relationships()     -> read triples by any combination of fields
    get_entity()            -> read an entity row by name
    entity_link()           -> resolve a mention to an entity via cosine
    supersede()             -> supersession hook called by TemporalEngine
    summarize()             -> natural-language summary
    forget_by_triple_id()   -> tombstone single triple
    forget_by_entity()      -> tombstone all triples for entity
    forget_by_time_range()  -> tombstone triples in time window
    forget_by_source()      -> tombstone triples by source_tag
    list_by_facet()         -> faceted listing (time/entity/source/trace)
    clean()                 -> singular cleanup path
    extract()               -> singular extraction path
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.db.session import get_db_context
from app.utils.cosine import cosine_sim
from app.vector.embedder import embed_text

log = logging.getLogger(__name__)


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class Entity:
    name: str
    entity_type: str = "unknown"
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    mention_count: int = 1


@dataclass
class Relationship:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.9
    id: Optional[int] = None
    is_current: bool = True
    sequence_number: Optional[int] = None


@dataclass
class Traces:
    valence: float = 0.5
    affiliation: float = 0.5
    emotional_intensity: Optional[float] = None
    relational_type: Optional[str] = None
    relational_proximity: float = 0.5
    relational_valence: float = 0.5
    episodic_significance: str = "routine"
    episodic_narrative_position: str = "ongoing"
    temporal_context: Optional[str] = None
    schema_category: Optional[str] = None
    schema_confidence: float = 0.0


# =============================================================================
# MEMORY ENGINE
# =============================================================================

class MemoryEngine:
    """Trace-primary engine owning the relationships store.

    Grammar engine is the canonical extraction path. No T5, no cosine
    anchors. Edge embeddings and predicted queries are computed at
    write time (best-effort, fail-open).
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Singular ingestion path
    # ------------------------------------------------------------------

    def _run_ingestion_path(
        self,
        text: str,
        speaker: Optional[str] = None,
    ) -> Tuple[str, List[Tuple[str, str, str, Any]], Any]:
        """The one write-path entry: cleanup -> grammar -> SPO/decomp rows.

        This is the only ingestion pipeline. `clean()`, `extract()`, and
        `ingest_text()` all route through it so KenoticV1 cannot drift onto
        a different extraction path.
        """
        if not text or not text.strip():
            return "", [], None

        from app.engines.sentence_model import cleanup as _cleanup
        from app.engines import grammar_engine as _grammar_eng

        cleaned = _cleanup(text.strip(), speaker=speaker)
        grammar_result = _grammar_eng.process(cleaned, speaker=speaker)

        rows: List[Tuple[str, str, str, Any]] = []
        decomps = grammar_result.trace_decompositions or []
        for idx, triple in enumerate(grammar_result.triples):
            decomp = decomps[idx] if idx < len(decomps) else None
            rows.append((triple.subject, triple.predicate, triple.object, decomp))

        for idx in range(len(grammar_result.triples), len(decomps)):
            decomp = decomps[idx]
            rows.append(
                (
                    getattr(decomp, "subject", ""),
                    getattr(decomp, "predicate", ""),
                    getattr(decomp, "object", ""),
                    decomp,
                )
            )

        final_text = grammar_result.resolved_text or cleaned
        return final_text, rows, grammar_result

    def clean(self, text: str) -> str:
        """Run the singular cleanup path used by live ingestion."""
        if not text or not text.strip():
            return text or ""
        from app.engines.sentence_model import cleanup as _cleanup
        return _cleanup(text.strip())

    def extract(self, text: str) -> List[Tuple[str, str, str, bool]]:
        """Run the singular extraction path used by live ingestion."""
        _, rows, _ = self._run_ingestion_path(text)
        extracted: List[Tuple[str, str, str, bool]] = []
        for s, p, o, decomp in rows:
            extracted.append(
                (
                    s,
                    p,
                    o,
                    bool(getattr(decomp, "is_historical", False)) if decomp else False,
                )
            )
        return extracted

    # ------------------------------------------------------------------
    # ingest_text -- singular cleanup -> grammar -> store
    # ------------------------------------------------------------------

    def ingest_text(
        self,
        user_id: int,
        text: str,
        source_timestamp: Optional[str] = None,
        speaker: Optional[str] = None,
        speaker_is_user: bool = True,
        confidence: float = 0.9,
        source_tag: Optional[str] = None,
    ) -> int:
        """End-to-end write-path entry for raw text.

        1. Grammar engine produces trace decompositions
        2. Speaker resolution (I/me/myself -> speaker name)
        3. Preserve speaker identity in decomp.relational_subject
        4. For each decomposition -> store()

        No _validate_triple(). If grammar produced it, store it.

        Returns the number of rows actually stored.
        """
        if not text or not text.strip():
            return 0

        cleaned, triples_with_decomp, _grammar_result = self._run_ingestion_path(
            text,
            speaker=speaker,
        )

        # -- Speaker resolution --
        def _resolve(tok: str) -> str:
            if not tok:
                return tok
            lower = tok.strip().lower()
            if lower in ("user", "i", "me", "myself"):
                return speaker if speaker else "user"
            return tok

        count = 0
        for s, p, o, decomp in triples_with_decomp:
            resolved_s = _resolve(s)
            resolved_o = _resolve(o)
            # Preserve speaker identity in trace decomposition
            if decomp is not None and speaker:
                if not getattr(decomp, 'relational_subject', None) or \
                   getattr(decomp, 'relational_subject', '') == 'user':
                    decomp.relational_subject = speaker

            has_spo = bool(resolved_s and p and resolved_o)

            # Use the decomposition's sentence-level source_text, not
            # the full cleaned turn. Each sentence gets its own hash
            # so multi-sentence turns produce multiple edges, not one.
            decomp_src = getattr(decomp, 'source_text', '') if decomp else ''
            # FIX 1: Wire utterance_type from grammar decomposition to store
            _utt_type = None
            if decomp is not None:
                _utt_type = getattr(decomp, 'utterance_type', None)

            rel_id = self.store(
                user_id=user_id,
                trace_decomposition=decomp,
                source_text=decomp_src or cleaned,
                confidence=confidence,
                source_timestamp=source_timestamp,
                source_tag=source_tag,
                utterance_type_id=_utt_type,
                subject=resolved_s if has_spo else None,
                predicate=p if has_spo else None,
                object=resolved_o if has_spo else None,
            )
            if rel_id:
                count += 1
        return count

    # ------------------------------------------------------------------
    # store -- three phases: PREPARE -> WRITE -> SIDE-EFFECTS
    # ------------------------------------------------------------------

    def store(
        self,
        user_id: int,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        source_text: str = "",
        confidence: float = 0.9,
        utterance_type_id: Optional[int] = None,
        source_timestamp: Optional[str] = None,
        source_tag: Optional[str] = None,
        trace_decomposition: Any = None,
    ) -> int:
        """Three-phase store: PREPARE -> WRITE -> SIDE-EFFECTS.

        Phase 1 PREPARE: Extract all column values from TraceDecomposition
                         into a flat dict. Pure computation, no DB access.
        Phase 2 WRITE:   One INSERT or UPDATE for the relationship row,
                         plus FTS5 sync. ~2-4 SQL statements total.
        Phase 3 SIDE-EFFECTS: Entity types + affiliation (one combined
                         UPDATE), facts/milestones, predicted queries,
                         event date, supersession, clustering, arcs.

        Returns: relationship_id (0 on failure).
        """
        td = trace_decomposition

        # Fill S/P/O from decomposition if not provided directly
        if td is not None:
            subject = subject or getattr(td, 'subject', '') or ''
            predicate = predicate or getattr(td, 'predicate', '') or ''
            object = object or getattr(td, 'object', '') or ''
            if not source_text and getattr(td, 'source_text', ''):
                source_text = td.source_text

        subject = (subject or "").strip()
        predicate = (predicate or "").strip().lower().replace(" ", "_")
        object = (object or "").strip()

        has_triple = bool(subject and predicate and object)
        has_source = bool(source_text and source_text.strip())
        if not has_triple and not has_source:
            return 0

        if not has_source and has_triple:
            source_text = f"{subject} {predicate.replace('_', ' ')} {object}"

        source_text_hash = hashlib.sha256(
            source_text.encode('utf-8')
        ).hexdigest()

        # ── Phase 1: PREPARE (pure computation, no DB) ──
        row = self._prepare_row(
            user_id, td, subject, predicate, object, source_text,
            confidence, utterance_type_id, source_timestamp, source_tag,
            source_text_hash,
        )

        try:
            with get_db_context() as conn:
                # ── Phase 2: WRITE (one row write + FTS5 sync) ──
                rel_id = self._write_row(conn, row, user_id, source_text_hash)
                if not rel_id:
                    return 0

                # ── Phase 3: SIDE-EFFECTS ──

                # 3-pre: Populate entities (before affiliation reads them)
                self._upsert_entities(conn, user_id, td)

                # 3a: Entity type + affiliation (read DB, write back as ONE update)
                _side_updates: Dict[str, Any] = {}
                try:
                    _speaker = getattr(td, 'relational_subject', None) or subject or ''
                    _aff = self._compute_affiliation(conn, user_id, td, _speaker)
                    if _aff is not None:
                        _side_updates["edge_affiliation"] = round(_aff, 4)
                except Exception:
                    pass
                try:
                    for col, name in [("subject_type", subject), ("object_type", object)]:
                        if not name:
                            continue
                        ent_row = conn.execute(
                            "SELECT entity_type FROM entities WHERE user_id = ? AND name = ? LIMIT 1",
                            (user_id, name),
                        ).fetchone()
                        if ent_row and ent_row["entity_type"]:
                            _side_updates[col] = ent_row["entity_type"]
                except Exception:
                    pass
                if _side_updates:
                    _cols = ", ".join(f"{k} = ?" for k in _side_updates)
                    _vals: List[Any] = list(_side_updates.values()) + [rel_id]
                    try:
                        conn.execute(
                            f"UPDATE relationships SET {_cols} WHERE id = ?",
                            tuple(_vals),
                        )
                    except Exception:
                        pass

                # 3b: Tier routing (separate tables)
                tier = row.get("_tier", "episode")
                if tier == "fact" and td is not None:
                    self._upsert_fact(conn, user_id, rel_id, td, confidence)
                elif tier == "milestone" and td is not None:
                    self._append_milestone(conn, user_id, rel_id, td, confidence)

                # 3c: Predicted queries (separate table)
                if has_triple:
                    try:
                        from app.engines.predicted_queries import generate_predicted_queries as _gen_pqs
                        _pqs = _gen_pqs(subject, predicate, object)
                        answer_text = source_text or f"{subject} {predicate.replace('_', ' ')} {object}"
                        for q_text, q_emb in _pqs:
                            conn.execute(
                                """INSERT INTO predicted_queries
                                     (relationship_id, user_id, predicted_question,
                                      answer_text, answer_subject, question_embedding,
                                      confidence)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (rel_id, user_id, q_text, answer_text, subject,
                                 q_emb.tobytes(), confidence),
                            )
                    except Exception:
                        pass

                # 3d: Event date resolution
                self._resolve_event_date(
                    conn, rel_id, source_text,
                    source_timestamp, trace_decomposition,
                )

                # 3e: Supersession detection
                try:
                    from app.engines.temporal import get_temporal_engine
                    get_temporal_engine().detect_supersession(
                        user_id, source_text or "", rel_id, None
                    )
                except Exception:
                    pass

                # 3f: Cluster assignment + arc membership
                self._assign_cluster(conn, user_id, rel_id, td)
                try:
                    _cr = conn.execute(
                        "SELECT cluster_id FROM relationships WHERE id = ?",
                        (rel_id,),
                    ).fetchone()
                    _cid = _cr["cluster_id"] if _cr else None
                    # Arc detection via temporal engine (single owner)
                    from app.engines.temporal import get_temporal_engine
                    _arc_id = get_temporal_engine().detect_arcs(user_id, rel_id, _cid)
                    if _arc_id:
                        conn.execute(
                            "UPDATE relationships SET arc_id = ? WHERE id = ?",
                            (_arc_id, rel_id),
                        )
                except Exception as e:
                    log.warning("arc assignment failed for rel_id=%s: %s", rel_id, e)

                conn.commit()
                return rel_id
        except Exception as e:
            try:
                conn.rollback()  # type: ignore[possibly-undefined]
            except Exception:
                pass  # conn may not be bound if get_db_context() itself failed
            log.error("store failed on (%s, %s, %s): %s", subject, predicate, object, e)
            return 0

    # ------------------------------------------------------------------
    # Phase 1: _prepare_row -- pure computation, no DB
    # ------------------------------------------------------------------

    def _prepare_row(
        self,
        user_id: int,
        td: Any,
        subject: str,
        predicate: str,
        object: str,
        source_text: str,
        confidence: float,
        utterance_type_id: Optional[int],
        source_timestamp: Optional[str],
        source_tag: Optional[str],
        source_text_hash: str,
    ) -> Dict[str, Any]:
        """Prepare all column values for one relationship row.

        Pure computation -- no database access. Returns a flat dict
        of column->value. Internal keys (prefixed with '_') carry
        metadata used by later phases but are NOT written to SQL.
        """
        row: Dict[str, Any] = {}

        # Core fields
        row["user_id"] = user_id
        row["subject"] = subject or None
        row["predicate"] = predicate or None
        row["object"] = object or None
        row["source_text"] = source_text
        row["source_text_hash"] = source_text_hash
        row["confidence"] = confidence
        row["utterance_type_id"] = utterance_type_id
        row["source_timestamp"] = source_timestamp
        row["source_tag"] = source_tag

        # 5 DTCM trace dimensions (from TraceDecomposition)
        if td is not None:
            _ev = getattr(td, 'emotional_valence', None)
            row["edge_emotional_valence"] = round(
                float(_ev if _ev is not None else 0.5), 4
            )
            row["edge_emotional_label"] = getattr(td, 'emotional_state', None)
            row["edge_schematic_category"] = (
                getattr(td, 'schematic_category', None) or 'uncategorized'
            )
            row["edge_episodic_significance"] = (
                getattr(td, 'episodic_significance', None) or 'routine'
            )
            row["edge_relational_type"] = (
                getattr(td, 'relational_type', None) or 'personal'
            )
            row["edge_temporal_context"] = (
                getattr(td, 'temporal_direction', None) or 'present'
            )
            row["edge_negated"] = 1 if getattr(td, 'negated', False) else 0
            row["edge_mood"] = getattr(td, 'mood', 'indicative') or 'indicative'

            # Extended trace fields
            row["is_historical"] = 1 if getattr(td, 'is_historical', False) else 0
            row["episodic_fact"] = getattr(td, 'episodic_fact', None) or None
            row["emotional_target"] = getattr(td, 'emotional_target', None) or None
            row["extraction_rule"] = getattr(td, 'extraction_rule', None) or None

            # Temporal expression + relational entities
            row["temporal_expression"] = getattr(td, 'temporal_expression', None)
            _rel_subj = getattr(td, 'relational_subject', None)
            _rel_ents = getattr(td, 'relational_entities', None) or []
            if isinstance(_rel_ents, list) and _rel_subj and _rel_subj.lower() != 'user':
                if _rel_subj not in _rel_ents:
                    _rel_ents = list(_rel_ents) + [_rel_subj]
            row["relational_entities"] = json.dumps(_rel_ents) if _rel_ents else None

        # Embeddings (computed here, not in separate steps)
        try:
            from app.vector.embedder import embed_text as _embed
            if source_text:
                row["edge_embedding"] = _embed(source_text).tobytes()
            if predicate:
                row["predicate_embedding"] = _embed(
                    predicate.replace("_", " ")
                ).tobytes()
        except Exception:
            pass  # embedder unavailable

        # Tier classification (pure logic, no DB)
        row["_tier"] = self._classify_tier(td)

        return row

    # ------------------------------------------------------------------
    # Phase 2: _write_row -- one INSERT or UPDATE + FTS5
    # ------------------------------------------------------------------

    def _write_row(
        self,
        conn: sqlite3.Connection,
        row: Dict[str, Any],
        user_id: int,
        source_text_hash: str,
    ) -> int:
        """Write one relationship row. Dedup on source_text_hash.

        UPDATE path: one SELECT (dedup+tombstone check) + one UPDATE + FTS5 sync.
        INSERT path: one INSERT + FTS5 sync + one sequence_number UPDATE.

        Returns relationship_id (0 on failure).
        """
        # ── Dedup check ──
        existing = None
        if source_text_hash:
            try:
                existing = conn.execute(
                    "SELECT id, tombstoned_at, subject, predicate, object, source_text "
                    "FROM relationships WHERE user_id = ? AND source_text_hash = ?",
                    (user_id, source_text_hash),
                ).fetchone()
            except Exception:
                existing = None  # fail-open: column not yet migrated

        if existing:
            rel_id: int = existing["id"]

            # Tombstone guard: never revive a forgotten row
            if existing["tombstoned_at"] is not None:
                return rel_id

            # Capture old values for FTS5 delete
            old_subj = existing["subject"] or ""
            old_pred = (existing["predicate"] or "").replace("_", " ")
            old_obj = existing["object"] or ""
            old_src = existing["source_text"] or ""

            # ── Build ONE UPDATE with all columns ──
            update_cols: List[str] = []
            update_vals: List[Any] = []

            # Always update these
            update_cols.append("confidence = MAX(confidence, ?)")
            update_vals.append(row.get("confidence", 0.9))
            update_cols.append("last_confirmed_at = datetime('now')")
            update_cols.append("is_current = 1")

            # COALESCE for S/P/O (keep existing if new is NULL)
            update_cols.append("subject = COALESCE(?, subject)")
            update_vals.append(row.get("subject"))
            update_cols.append("predicate = COALESCE(?, predicate)")
            update_vals.append(row.get("predicate"))
            update_cols.append("object = COALESCE(?, object)")
            update_vals.append(row.get("object"))

            # Source text (always overwrite if provided)
            if row.get("source_text"):
                update_cols.append("source_text = ?")
                update_vals.append(row["source_text"])
            if row.get("source_tag"):
                update_cols.append("source_tag = COALESCE(source_tag, ?)")
                update_vals.append(row["source_tag"])

            # All trace + embedding columns in one pass
            _trace_cols = (
                "edge_emotional_valence", "edge_emotional_label",
                "edge_schematic_category", "edge_episodic_significance",
                "edge_relational_type", "edge_temporal_context",
                "edge_negated", "edge_mood",
                "is_historical", "episodic_fact", "emotional_target",
                "extraction_rule", "temporal_expression", "relational_entities",
                "edge_embedding", "predicate_embedding",
            )
            for col in _trace_cols:
                if col in row and row[col] is not None:
                    update_cols.append(f"{col} = ?")
                    update_vals.append(row[col])

            update_vals.append(rel_id)
            sql = f"UPDATE relationships SET {', '.join(update_cols)} WHERE id = ?"

            try:
                conn.execute(sql, tuple(update_vals))
            except sqlite3.OperationalError as e:
                # Some columns may not exist yet in old DBs -- fall back to core
                log.warning("full update failed, falling back to core: %s", e)
                conn.execute(
                    """UPDATE relationships SET
                         confidence = MAX(confidence, ?),
                         last_confirmed_at = datetime('now'),
                         is_current = 1,
                         subject = COALESCE(?, subject),
                         predicate = COALESCE(?, predicate),
                         object = COALESCE(?, object)
                       WHERE id = ?""",
                    (row.get("confidence", 0.9), row.get("subject"),
                     row.get("predicate"), row.get("object"), rel_id),
                )

            # FTS5 sync (delete old + insert new)
            try:
                conn.execute(
                    "INSERT INTO relationships_fts(relationships_fts, rowid, "
                    "subject, predicate, object, source_text) "
                    "VALUES('delete', ?, ?, ?, ?, ?)",
                    (rel_id, old_subj, old_pred, old_obj, old_src),
                )
                conn.execute(
                    """INSERT INTO relationships_fts
                         (rowid, subject, predicate, object, source_text)
                       VALUES (?, ?, ?, ?, ?)""",
                    (rel_id, row.get("subject") or "",
                     (row.get("predicate") or "").replace("_", " "),
                     row.get("object") or "", row.get("source_text") or ""),
                )
            except Exception:
                pass  # FTS5 table may not exist yet

            return rel_id

        # ── INSERT new row ──
        # Build column list dynamically from what's in row
        insert_cols: List[str] = ["user_id", "is_current"]
        insert_vals: List[Any] = [user_id, 1]

        for col, val in row.items():
            if col.startswith("_"):  # skip internal keys like _tier
                continue
            if val is not None and col != "user_id":
                insert_cols.append(col)
                insert_vals.append(val)

        placeholders = ", ".join(["?"] * len(insert_vals))
        col_names = ", ".join(insert_cols)

        try:
            cur = conn.execute(
                f"INSERT INTO relationships ({col_names}) VALUES ({placeholders})",
                tuple(insert_vals),
            )
        except sqlite3.OperationalError:
            # Fall back to core columns only for old DBs
            cur = conn.execute(
                """INSERT INTO relationships
                     (user_id, subject, predicate, object, confidence,
                      utterance_type_id, source_timestamp, is_current,
                      source_text, source_tag, source_text_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (user_id, row.get("subject"), row.get("predicate"),
                 row.get("object"), row.get("confidence", 0.9),
                 row.get("utterance_type_id"), row.get("source_timestamp"),
                 row.get("source_text"), row.get("source_tag"),
                 row.get("source_text_hash")),
            )

        rel_id = cur.lastrowid

        # FTS5 sync
        try:
            conn.execute(
                """INSERT INTO relationships_fts
                     (rowid, subject, predicate, object, source_text)
                   VALUES (?, ?, ?, ?, ?)""",
                (rel_id, row.get("subject") or "",
                 (row.get("predicate") or "").replace("_", " "),
                 row.get("object") or "", row.get("source_text") or ""),
            )
        except Exception:
            pass

        # Sequence number
        try:
            conn.execute(
                "UPDATE relationships SET sequence_number = ? WHERE id = ?",
                (rel_id, rel_id),
            )
        except Exception:
            pass

        return rel_id

    # ------------------------------------------------------------------
    # _compute_affiliation -- in-group/out-group scoring
    # ------------------------------------------------------------------

    def _compute_affiliation(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        td: Any,
        speaker: str,
    ) -> Optional[float]:
        """Compute edge_affiliation: how personally connected a fact is to the speaker.

        1.0 = speaker's own life fact
        0.8 = well-known entity (mention_count >= 5)
        0.6 = mentioned before (mention_count >= 2)
        0.4 = first mention but named
        0.2 = unknown entity
        0.1 = no entities at all (no personal connection)

        Returns MAX across all entity scores (one close entity makes the fact personal).
        """
        if td is None:
            return None

        # If relational_subject IS the speaker -> speaker's own fact
        rel_subj = getattr(td, 'relational_subject', None) or ''
        if rel_subj:
            subj_lower = rel_subj.strip().lower()
            speaker_lower = (speaker or '').strip().lower()
            if subj_lower in ('user', 'i', 'me', 'myself') or (
                speaker_lower and subj_lower == speaker_lower
            ):
                return 1.0

        # Gather candidate entities from decomposition
        candidates: List[str] = []
        for ent in (getattr(td, 'relational_entities', None) or []):
            if ent and ent.strip().lower() not in ('', 'user'):
                candidates.append(ent.strip())
        for attr in ('subject', 'object'):
            val = getattr(td, attr, None)
            if val and val.strip().lower() not in ('', 'user'):
                candidates.append(val.strip())

        # Deduplicate (case-insensitive) while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for c in candidates:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)

        if not unique:
            return 0.1  # no entities -> no personal connection

        # Score each entity by mention_count
        scores: List[float] = []
        for name in unique:
            row = None
            try:
                row = conn.execute(
                    "SELECT mention_count FROM entities WHERE user_id = ? AND name = ? LIMIT 1",
                    (user_id, name),
                ).fetchone()
            except Exception:
                pass  # entities table may not exist

            if row is None:
                scores.append(0.2)
            else:
                mc = row["mention_count"] or 1
                if mc >= 5:
                    scores.append(0.8)
                elif mc >= 2:
                    scores.append(0.6)
                else:
                    scores.append(0.4)

        return max(scores)

    # ------------------------------------------------------------------
    # Entity population -- upsert entities from trace decomposition
    # ------------------------------------------------------------------

    def _upsert_entities(self, conn: sqlite3.Connection, user_id: str, td: Any) -> None:
        """Upsert entities from TraceDecomposition into the entities table.

        Only NER-labeled entities (already filtered by grammar engine).
        Title-case normalized to prevent duplicate rows from case differences.
        """
        if td is None:
            return

        # Gather entities ONLY from relational_entities (already NER-filtered
        # by grammar_engine._extract_relational). Subject/object fields carry
        # arbitrary phrases ("Really Nervous About The Interview") that are
        # not proper entities.
        names: set[str] = set()
        rel_ents = getattr(td, 'relational_entities', None) or []
        if isinstance(rel_ents, list):
            for e in rel_ents:
                if isinstance(e, str) and e.strip():
                    if e.strip().lower() not in ('user', 'i', 'me', 'myself'):
                        names.add(e.strip().title())

        for name in names:
            try:
                existing = conn.execute(
                    "SELECT id, mention_count FROM entities WHERE user_id = ? AND name = ?",
                    (user_id, name),
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE entities SET mention_count = mention_count + 1 WHERE id = ?",
                        (existing["id"],),
                    )
                else:
                    conn.execute(
                        "INSERT INTO entities (user_id, name, mention_count) VALUES (?, ?, 1)",
                        (user_id, name),
                    )
            except Exception:
                pass  # fail-open

    # ------------------------------------------------------------------
    # Tier routing -- classify edge as fact/milestone/episode
    # ------------------------------------------------------------------

    def _classify_tier(self, td: Any) -> str:
        """Classify a TraceDecomposition into fact/milestone/episode tier.

        Significance alone determines tier — no schema allow-list.
        The grammar engine's significance classifier already did the work
        of determining whether this is a persisting state or a one-time event.
        """
        if td is None:
            return "episode"
        sig = getattr(td, 'episodic_significance', 'routine') or 'routine'

        if sig == "milestone":
            return "milestone"
        if sig == "stative":
            return "fact"
        return "episode"

    # ------------------------------------------------------------------
    # _classify_supersession_type -- structural signal, no word lists
    # ------------------------------------------------------------------

    def _classify_supersession_type(self, td: Any) -> str:
        """Classify how a fact was superseded: update, correction, or reversal.

        Uses signals already present on the TraceDecomposition:
        - correction: grammar engine flagged via extraction_rule
        - reversal:   decomposition carries negated=True
        - update:     natural temporal progression (default)
        """
        if td is None:
            return "update"

        extraction_rule = getattr(td, 'extraction_rule', '') or ''
        if 'correction' in extraction_rule.lower():
            return "correction"

        negated = getattr(td, 'negated', False)
        if negated:
            return "reversal"

        return "update"

    # ------------------------------------------------------------------
    # _upsert_fact -- semantic key dedup with history chain
    # ------------------------------------------------------------------

    def _upsert_fact(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        rel_id: int,
        td: Any,
        confidence: float,
    ) -> None:
        """Upsert a fact row keyed on schema::VerbClass::subject.

        Root cause addressed: the old key schema::predicate::subject used
        surface predicates, so "work_at" and "start_at" created separate
        fact rows for the same career fact. Now uses WordNet open-vocabulary
        hypernym classification (classify_verb_class) to group synonymous
        verbs under one VerbClass label. The verb lemma is extracted from
        compound predicates by splitting on underscore before lookup.

        If the value changed, the old value is appended to the history
        JSON array for contradiction tracking. If the value is the same,
        only last_confirmed_at is bumped.
        """
        from app.engines.grammar_engine import classify_verb_class

        schema = getattr(td, 'schematic_category', 'uncategorized') or 'uncategorized'
        predicate = getattr(td, 'predicate', '') or ''
        subject = getattr(td, 'subject', '') or ''
        value = getattr(td, 'object', '') or ''

        # Extract verb lemma from compound predicates (e.g., "work_at" -> "work")
        # then classify via WordNet hypernym closure (open-vocabulary, not a word list)
        verb_lemma = predicate.split('_')[0] if predicate else ''
        verb_class = classify_verb_class(verb_lemma).name if verb_lemma else 'UNKNOWN'
        key = f"{schema}::{verb_class}::{subject}"
        if not key or not value:
            return

        try:
            existing = conn.execute(
                "SELECT id, value, history FROM facts WHERE user_id = ? AND key = ?",
                (user_id, key),
            ).fetchone()

            if existing:
                old_value = existing["value"] or ""
                if old_value != value:
                    # Value changed -- append old to history chain
                    history_raw = existing["history"] or "[]"
                    try:
                        history: List[Dict[str, str]] = json.loads(history_raw)
                    except (json.JSONDecodeError, TypeError):
                        history = []
                    history.append({
                        "value": old_value,
                        "at": datetime.now(timezone.utc).isoformat(),
                        "type": self._classify_supersession_type(td),
                    })
                    conn.execute(
                        """UPDATE facts SET
                             value = ?,
                             confidence = ?,
                             last_confirmed_at = datetime('now'),
                             provenance_memory_id = ?,
                             history = ?
                           WHERE id = ?""",
                        (value, confidence, rel_id,
                         json.dumps(history), existing["id"]),
                    )
                else:
                    # Same value -- just bump confirmation timestamp
                    conn.execute(
                        "UPDATE facts SET last_confirmed_at = datetime('now') WHERE id = ?",
                        (existing["id"],),
                    )
            else:
                # New fact
                conn.execute(
                    """INSERT INTO facts
                         (user_id, key, value, confidence,
                          provenance_memory_id)
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, key, value, confidence, rel_id),
                )
        except Exception as e:
            log.warning("_upsert_fact failed for key=%s: %s", key, e)

    # ------------------------------------------------------------------
    # _append_milestone -- dedup on exact description text
    # ------------------------------------------------------------------

    def _append_milestone(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        rel_id: int,
        td: Any,
        confidence: float,
    ) -> None:
        """Insert a milestone row if no exact-text duplicate exists.

        Event date is left NULL; _resolve_event_date fills it later.
        """
        description = (
            getattr(td, 'source_text', '') or
            getattr(td, 'episodic_fact', '') or ''
        ).strip()
        event_type = getattr(td, 'schematic_category', '') or 'life_event'

        if not description:
            return

        try:
            dup = conn.execute(
                "SELECT id FROM milestones WHERE user_id = ? AND description = ?",
                (user_id, description),
            ).fetchone()
            if dup:
                return  # exact duplicate -- skip

            conn.execute(
                """INSERT INTO milestones
                     (user_id, event_type, description, confidence,
                      provenance_memory_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, event_type, description, confidence, rel_id),
            )
        except Exception as e:
            log.warning("_append_milestone failed: %s", e)

    # ------------------------------------------------------------------
    # _resolve_event_date -- 4-tier waterfall (extracted from old store)
    # ------------------------------------------------------------------

    def _resolve_event_date(
        self,
        conn: sqlite3.Connection,
        rel_id: int,
        source_text: str,
        source_timestamp: Optional[str],
        trace_decomposition: Any,
    ) -> None:
        """Resolve event date via 4-tier waterfall and write to row."""
        td = trace_decomposition
        try:
            from app.engines.temporal import get_temporal_engine
            _te_date = get_temporal_engine()

            # Tier 1: resolve from source text
            _resolved = _te_date.resolve_event_date(
                source_text or "", source_timestamp
            )

            # Tier 2: grammar engine temporal_expression
            if not _resolved and td is not None:
                _temp_expr = getattr(td, 'temporal_expression', None)
                if _temp_expr:
                    _resolved = _te_date.resolve_event_date(
                        _temp_expr, source_timestamp
                    )

            # Tier 3: grammar engine pre-resolved date
            if not _resolved and td is not None:
                _temp_resolved = getattr(td, 'temporal_resolved', None)
                if _temp_resolved:
                    _resolved = _temp_resolved

            # Tier 4: source_timestamp (session time)
            if not _resolved and source_timestamp:
                _resolved = source_timestamp

            if _resolved:
                conn.execute(
                    "UPDATE relationships SET resolved_event_date = ? WHERE id = ?",
                    (_resolved, rel_id),
                )
        except Exception:
            pass  # fail-open: date resolution is observational

    # ------------------------------------------------------------------
    # _assign_cluster -- bind related edges via shared entities
    # ------------------------------------------------------------------

    def _assign_cluster(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        rel_id: int,
        td: Any,
    ) -> None:
        """Assign cluster_id to a newly written edge via entity overlap.

        Algorithm:
          1. Build entity set from subject, object, and relational_entities
          2. Find existing clusters that share at least one entity
          3. No match   -> new cluster (str(rel_id))
             One match  -> join that cluster
             N matches  -> merge all into smallest cluster_id, then join
          4. Write cluster_id to the new edge

        Fail-open: entire method is wrapped in try/except so a clustering
        bug never blocks the store path.
        """
        try:
            # -- 1. Build entity set for this edge --
            entities: set[str] = set()

            # From subject/object columns (already written to the row)
            try:
                row = conn.execute(
                    "SELECT subject, object, relational_entities FROM relationships WHERE id = ?",
                    (rel_id,),
                ).fetchone()
                if row:
                    if row["subject"]:
                        entities.add(row["subject"])
                    if row["object"]:
                        entities.add(row["object"])
                    if row["relational_entities"]:
                        try:
                            rel_ents = json.loads(row["relational_entities"])
                            if isinstance(rel_ents, list):
                                for e in rel_ents:
                                    if isinstance(e, str) and e.strip():
                                        entities.add(e.strip())
                        except (json.JSONDecodeError, TypeError):
                            pass
            except Exception:
                pass  # relational_entities column may not exist

            # Fallback: pull from trace decomposition if DB read missed
            if td is not None:
                for attr in ('subject', 'object'):
                    val = getattr(td, attr, None)
                    if val and isinstance(val, str) and val.strip():
                        entities.add(val.strip())
                td_ents = getattr(td, 'relational_entities', None) or []
                if isinstance(td_ents, list):
                    for e in td_ents:
                        if isinstance(e, str) and e.strip():
                            entities.add(e.strip())

            # -- 2. Filter noise --
            entities.discard("")
            entities.discard("user")

            if not entities:
                return  # no clustering signal

            # -- 3. Find existing clusters sharing at least one entity --
            entity_list = list(entities)
            placeholders = ",".join("?" for _ in entity_list)

            # Build LIKE clauses for relational_entities JSON.
            # Each entity gets: relational_entities LIKE '%"entity"%'
            # The quotes prevent substring false positives ("Sam" won't
            # match "Samantha" because the JSON stores ["Sam"]).
            like_clauses = " OR ".join(
                "relational_entities LIKE ?" for _ in entity_list
            )
            like_vals = [f'%"{e}"%' for e in entity_list]

            existing_clusters = conn.execute(
                f"""SELECT DISTINCT cluster_id FROM relationships
                    WHERE user_id = ? AND cluster_id IS NOT NULL AND cluster_id != ''
                    AND id != ?
                    AND (subject IN ({placeholders})
                         OR object IN ({placeholders})
                         OR ({like_clauses}))""",
                [user_id, rel_id] + entity_list + entity_list + like_vals,
            ).fetchall()

            cluster_ids = [r["cluster_id"] for r in existing_clusters if r["cluster_id"]]

            if not cluster_ids:
                # -- 4a. No match -> new cluster --
                chosen = str(rel_id)
            elif len(cluster_ids) == 1:
                # -- 4b. One match -> join --
                chosen = cluster_ids[0]
            else:
                # -- 4c. Multiple matches -> merge into smallest --
                chosen = min(cluster_ids, key=lambda cid: int(cid) if cid.isdigit() else float('inf'))
                old_ids = [cid for cid in cluster_ids if cid != chosen]
                if old_ids:
                    merge_placeholders = ",".join("?" for _ in old_ids)
                    conn.execute(
                        f"""UPDATE relationships SET cluster_id = ?
                            WHERE user_id = ? AND cluster_id IN ({merge_placeholders})""",
                        [chosen, user_id] + old_ids,
                    )

            # -- 5. Write cluster_id to the new edge --
            conn.execute(
                "UPDATE relationships SET cluster_id = ? WHERE id = ?",
                (chosen, rel_id),
            )

        except Exception as e:
            log.warning("_assign_cluster failed for rel_id=%s: %s", rel_id, e)

    # ------------------------------------------------------------------
    # _assign_arc -- match new edge's cluster to open arcs
    # ------------------------------------------------------------------

    def _assign_arc(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        rel_id: int,
        cluster_id: Optional[str],
    ) -> None:
        """Assign arc_id to a newly written edge by matching its cluster
        to open arcs whose start_edge shares the same cluster_id.

        Fail-open: entire method is wrapped in try/except so an arc
        assignment bug never blocks the store path.
        """
        try:
            if not cluster_id:
                return  # can't match without a cluster

            # Find an open arc whose start_edge belongs to the same cluster
            arc_row = conn.execute(
                """SELECT a.id FROM arcs a
                   JOIN relationships r ON r.id = a.start_edge_id
                   WHERE a.user_id = ? AND a.status = 'open'
                     AND r.cluster_id = ?
                   LIMIT 1""",
                (user_id, cluster_id),
            ).fetchone()

            if not arc_row:
                return  # no matching open arc

            arc_id: str = arc_row["id"] if isinstance(arc_row, sqlite3.Row) else arc_row[0]

            # Write arc_id to the new edge
            conn.execute(
                "UPDATE relationships SET arc_id = ? WHERE id = ?",
                (arc_id, rel_id),
            )

            # Touch the arc's last_checked_at
            conn.execute(
                "UPDATE arcs SET last_checked_at = datetime('now') WHERE id = ?",
                (arc_id,),
            )

            log.debug("_assign_arc: rel_id=%s -> arc_id=%s", rel_id, arc_id)

        except Exception as e:
            log.warning("_assign_arc failed for rel_id=%s: %s", rel_id, e)

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def get_relationships(
        self,
        user_id: int,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        only_current: bool = True,
    ) -> List[Relationship]:
        conditions = ["user_id = ?"]
        params: List[Any] = [user_id]
        if only_current:
            conditions.append("COALESCE(is_current, 1) = 1")
            conditions.append("tombstoned_at IS NULL")
        if subject:
            conditions.append("LOWER(subject) = LOWER(?)")
            params.append(subject)
        if predicate:
            conditions.append("LOWER(predicate) = LOWER(?)")
            params.append(predicate)
        if object:
            conditions.append("LOWER(object) = LOWER(?)")
            params.append(object)

        sql = (
            f"SELECT id, subject, predicate, object, confidence, is_current "
            f"FROM relationships WHERE {' AND '.join(conditions)} "
            f"ORDER BY id DESC"
        )
        with get_db_context() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                Relationship(
                    id=r["id"],
                    subject=r["subject"],
                    predicate=r["predicate"],
                    object=r["object"],
                    confidence=r["confidence"] or 0.9,
                    is_current=bool(r["is_current"] if r["is_current"] is not None else 1),
                )
                for r in rows
            ]

    def get_entity(self, user_id: int, name: str) -> Optional[Entity]:
        with get_db_context() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE user_id = ? AND LOWER(name) = LOWER(?)",
                (user_id, name),
            ).fetchone()
        if not row:
            return None
        emb = None
        if row["embedding"]:
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            except Exception:
                emb = None
        attrs: Dict[str, Any] = {}
        try:
            attrs = json.loads(row["attributes"] or "{}")
        except Exception:
            pass
        return Entity(
            name=row["name"],
            entity_type=row["entity_type"] or "unknown",
            attributes=attrs,
            embedding=emb,
            mention_count=row["mention_count"] or 1,
        )

    # ------------------------------------------------------------------
    # Neural entity linker -- pure cosine, no regex, no stop-word list
    # ------------------------------------------------------------------

    def entity_link(
        self,
        user_id: int,
        mention_text: str,
    ) -> Optional[Entity]:
        """Resolve a free-text mention to a canonical entity via cosine
        over entities.embedding for this user."""
        if not mention_text or not mention_text.strip():
            return None

        try:
            query_emb = embed_text(mention_text.strip())
        except Exception:
            return None

        with get_db_context() as conn:
            rows = conn.execute(
                """SELECT id, name, entity_type, attributes, embedding, mention_count
                   FROM entities WHERE user_id = ? AND embedding IS NOT NULL""",
                (user_id,),
            ).fetchall()

        if not rows:
            return None

        sims: List[Tuple[float, sqlite3.Row]] = []
        for r in rows:
            try:
                emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
                sims.append((cosine_sim(query_emb, emb), r))
            except Exception:
                continue

        if not sims:
            return None

        sims.sort(key=lambda x: -x[0])
        best_sim, best_row = sims[0]

        # Adaptive population threshold
        scores = np.array([s for s, _ in sims], dtype=np.float32)
        if len(scores) >= 3:
            mean = float(scores.mean())
            std = float(scores.std())
            threshold = mean + std
            if best_sim < threshold:
                return None

        attrs: Dict[str, Any] = {}
        try:
            attrs = json.loads(best_row["attributes"] or "{}")
        except Exception:
            pass
        emb = None
        try:
            emb = np.frombuffer(best_row["embedding"], dtype=np.float32).copy()
        except Exception:
            pass
        return Entity(
            name=best_row["name"],
            entity_type=best_row["entity_type"] or "unknown",
            attributes=attrs,
            embedding=emb,
            mention_count=best_row["mention_count"] or 1,
        )

    # ------------------------------------------------------------------
    # Supersession hook (called by TemporalEngine)
    # ------------------------------------------------------------------

    def supersede(self, relationship_id: int, superseded_by: int) -> None:
        """Mark a prior relationship as superseded by a newer one."""
        try:
            with get_db_context() as conn:
                conn.execute(
                    """UPDATE relationships SET
                         is_current = 0,
                         superseded_at = datetime('now'),
                         superseded_by = ?
                       WHERE id = ?""",
                    (superseded_by, relationship_id),
                )
                # Clean predicted_queries for superseded edge (legacy table)
                try:
                    conn.execute(
                        "DELETE FROM predicted_queries WHERE relationship_id = ?",
                        (relationship_id,),
                    )
                except sqlite3.OperationalError:
                    pass  # table may not exist
                conn.commit()
        except Exception as e:
            log.error("supersede failed: %s", e)

    # ------------------------------------------------------------------
    # Summarize
    # ------------------------------------------------------------------

    def summarize(self, user_id: int, entity: Optional[str] = None) -> str:
        """Natural-language summary of entity state."""
        rels = (
            self.get_relationships(user_id, subject=entity)
            if entity
            else self.get_relationships(user_id, subject="user")
        )
        if not rels:
            return ""
        lines = [
            f"{r.subject} {r.predicate.replace('_', ' ')} {r.object}"
            for r in rels[:10]
        ]
        return ". ".join(lines) + "."

    # ------------------------------------------------------------------
    # Forget (soft tombstone) + faceted show
    # ------------------------------------------------------------------

    def _append_forget_audit(
        self, conn: sqlite3.Connection, user_id: int, op_id: str,
        reason: str, count: int,
    ) -> None:
        """Append an audit log row into the memories table."""
        try:
            conn.execute(
                """INSERT INTO memories (user_id, content, memory_type, importance)
                   VALUES (?, ?, 'summary', 0.9)""",
                (user_id, f"forget op {op_id}: reason={reason}, n={count}"),
            )
        except Exception as e:
            log.warning("forget audit write failed: %s", e)

    def _tombstone_rows(
        self, conn: sqlite3.Connection, user_id: int,
        ids: List[int], reason: str, op_id: str,
    ) -> int:
        """Flip tombstone flags on a set of relationship rows."""
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        cur = conn.execute(
            f"""UPDATE relationships SET
                  tombstoned_at = datetime('now'),
                  tombstone_reason = ?,
                  tombstone_op_id = ?
                WHERE user_id = ? AND tombstoned_at IS NULL
                  AND id IN ({placeholders})""",
            [reason, op_id, user_id] + list(ids),
        )
        return cur.rowcount or 0

    def forget_by_triple_id(self, user_id: int, triple_id: int) -> int:
        """Tombstone a single triple by its relationship id."""
        from uuid import uuid4
        op_id = uuid4().hex
        reason = f"by_triple_id:{int(triple_id)}"
        try:
            with get_db_context() as conn:
                row = conn.execute(
                    """SELECT id FROM relationships
                       WHERE user_id = ? AND id = ?
                         AND tombstoned_at IS NULL
                         AND COALESCE(is_current, 1) = 1""",
                    (user_id, int(triple_id)),
                ).fetchone()
                if not row:
                    return 0
                count = self._tombstone_rows(
                    conn, user_id, [row["id"]], reason, op_id
                )
                if count:
                    self._append_forget_audit(conn, user_id, op_id, reason, count)
                conn.commit()
                return count
        except Exception as e:
            log.error("forget_by_triple_id failed: %s", e)
            return 0

    def forget_by_entity(self, user_id: int, entity_name: str) -> int:
        """Tombstone every live triple where the entity appears as subject
        or object (case-insensitive exact match)."""
        from uuid import uuid4
        if not entity_name or not entity_name.strip():
            return 0
        name = entity_name.strip()
        op_id = uuid4().hex
        reason = f"by_entity:{name}"
        try:
            with get_db_context() as conn:
                rows = conn.execute(
                    """SELECT id FROM relationships
                       WHERE user_id = ? AND tombstoned_at IS NULL
                         AND COALESCE(is_current, 1) = 1
                         AND (LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?))""",
                    (user_id, name, name),
                ).fetchall()
                ids = [r["id"] for r in rows]
                count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
                if count:
                    self._append_forget_audit(conn, user_id, op_id, reason, count)
                conn.commit()
                return count
        except Exception as e:
            log.error("forget_by_entity failed: %s", e)
            return 0

    def forget_by_time_range(
        self, user_id: int, start_iso: str, end_iso: str,
    ) -> int:
        """Tombstone every live triple whose source_timestamp falls
        within [start_iso, end_iso]."""
        from uuid import uuid4
        op_id = uuid4().hex
        reason = f"by_time_range:{start_iso}..{end_iso}"
        try:
            with get_db_context() as conn:
                rows = conn.execute(
                    """SELECT id FROM relationships
                       WHERE user_id = ? AND tombstoned_at IS NULL
                         AND COALESCE(is_current, 1) = 1
                         AND source_timestamp IS NOT NULL
                         AND source_timestamp >= ?
                         AND source_timestamp <= ?""",
                    (user_id, start_iso, end_iso),
                ).fetchall()
                ids = [r["id"] for r in rows]
                count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
                if count:
                    self._append_forget_audit(conn, user_id, op_id, reason, count)
                conn.commit()
                return count
        except Exception as e:
            log.error("forget_by_time_range failed: %s", e)
            return 0

    def forget_by_source(self, user_id: int, source_tag: str) -> int:
        """Tombstone every live triple whose source_tag exactly matches."""
        from uuid import uuid4
        if source_tag is None:
            return 0
        op_id = uuid4().hex
        reason = f"by_source:{source_tag}"
        try:
            with get_db_context() as conn:
                rows = conn.execute(
                    """SELECT id FROM relationships
                       WHERE user_id = ? AND tombstoned_at IS NULL
                         AND COALESCE(is_current, 1) = 1
                         AND source_tag = ?""",
                    (user_id, source_tag),
                ).fetchall()
                ids = [r["id"] for r in rows]
                count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
                if count:
                    self._append_forget_audit(conn, user_id, op_id, reason, count)
                conn.commit()
                return count
        except Exception as e:
            log.error("forget_by_source failed: %s", e)
            return 0

    # ------------------------------------------------------------------
    # Faceted listing
    # ------------------------------------------------------------------

    def list_by_facet(
        self,
        user_id: int,
        facet: str,
        value: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return live triples matching a facet view.

        Facets: 'time', 'entity', 'source', 'trace'.
        """
        if facet not in ("time", "entity", "source", "trace"):
            raise ValueError(f"Unknown facet: {facet}")
        limit = max(1, min(int(limit or 100), 1000))

        base_cols = (
            "id, subject, predicate, object, "
            "source_timestamp, source_tag, confidence, source_text"
        )
        where = [
            "user_id = ?", "tombstoned_at IS NULL",
            "COALESCE(is_current, 1) = 1",
        ]
        params: List[Any] = [user_id]
        order = "id DESC"

        if facet == "time":
            if value:
                where.append("source_timestamp LIKE ?")
                params.append(f"{value}%")
            order = (
                "CASE WHEN source_timestamp IS NULL THEN 1 ELSE 0 END, "
                "source_timestamp DESC, id DESC"
            )
        elif facet == "entity":
            if not value:
                return []
            where.append(
                "(LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?))"
            )
            params.extend([value, value])
        elif facet == "source":
            if value is None:
                return []
            where.append("source_tag = ?")
            params.append(value)
        elif facet == "trace":
            if value:
                where.append("edge_schematic_category = ?")
                params.append(value)

        sql = (
            f"SELECT {base_cols} FROM relationships "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY {order} LIMIT ?"
        )
        params.append(limit)

        try:
            with get_db_context() as conn:
                rows = conn.execute(sql, params).fetchall()
        except Exception as e:
            log.error("list_by_facet failed: %s", e)
            return []

        return [
            {
                "id": r["id"],
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "source_timestamp": r["source_timestamp"],
                "source_tag": r["source_tag"],
                "confidence": r["confidence"] or 0.9,
                "source_text": r["source_text"],
            }
            for r in rows
        ]


# =============================================================================
# Module-level singleton
# =============================================================================

_singleton: Optional[MemoryEngine] = None


def get_memory_engine() -> MemoryEngine:
    global _singleton
    if _singleton is None:
        _singleton = MemoryEngine()
    return _singleton
