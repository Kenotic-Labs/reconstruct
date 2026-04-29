# ============================================================================
# TRACE-PRIMARY MEMORY ENGINE (2026-04-27 rewrite)
#
# 4-step store(): upsert row -> write grammar traces -> resolve date -> supersede
# No cosine anchors. No embeddings at write time. No triple validation gate.
# If the grammar engine produced a trace decomposition, store it.
# ============================================================================
"""
MemoryEngine -- trace-primary store + all derived views.

Public interface:
    store()                 -> upsert row + write traces + date + supersede
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
    clean()                 -> backward compat stub
    extract()               -> backward compat stub
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
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
    anchors, no embedding computation at write time.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Backward compat stubs
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Legacy cleanup passthrough. Retained for API compatibility."""
        if not text or not text.strip():
            return text or ""
        return text.strip()

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        """Legacy extraction stub. Retained for API compatibility."""
        return []

    # ------------------------------------------------------------------
    # ingest_text -- simplified: grammar -> store, no validation gate
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

        # -- Extraction via grammar engine --
        triples_with_decomp: List[Tuple[str, str, str, Any]] = []
        cleaned = text.strip()

        try:
            from app.engines.sentence_model import cleanup as _cleanup
            cleaned = _cleanup(cleaned, speaker=speaker)
        except Exception:
            pass  # fail-open: use raw text if CoEdit unavailable

        try:
            from app.engines import grammar_engine as _grammar_eng
            grammar_result = _grammar_eng.process(cleaned, speaker=speaker)
            decomps = grammar_result.trace_decompositions or []
            for idx, t in enumerate(grammar_result.triples):
                decomp = decomps[idx] if idx < len(decomps) else None
                triples_with_decomp.append(
                    (t.subject, t.predicate, t.object, decomp)
                )
            # Decompositions beyond len(triples) are trace-only edges
            for idx in range(len(grammar_result.triples), len(decomps)):
                decomp = decomps[idx]
                triples_with_decomp.append(
                    (
                        getattr(decomp, 'subject', ''),
                        getattr(decomp, 'predicate', ''),
                        getattr(decomp, 'object', ''),
                        decomp,
                    )
                )
            if grammar_result.resolved_text:
                cleaned = grammar_result.resolved_text
        except (ImportError, Exception):
            # Fallback: no decompositions available
            cleaned = self.clean(text)

        # -- Speaker resolution --
        def _resolve(tok: str) -> str:
            if not tok:
                return tok
            lower = tok.strip().lower()
            if lower in ("user", "i", "me", "myself"):
                return speaker if speaker else "user"
            return tok

        def _resolve_possessives_in_object(obj_text: str) -> str:
            """Replace first-person possessives in objects:
            'my identity' -> 'Caroline's identity' (when speaker=Caroline)"""
            if not obj_text or not speaker:
                return obj_text
            import re
            # Replace leading "my " with "speaker's "
            resolved = re.sub(
                r'\bmy\b', f"{speaker}'s", obj_text, flags=re.IGNORECASE
            )
            # Replace "myself" with speaker name
            resolved = re.sub(
                r'\bmyself\b', speaker, resolved, flags=re.IGNORECASE
            )
            return resolved

        count = 0
        for s, p, o, decomp in triples_with_decomp:
            resolved_s = _resolve(s)
            resolved_o = _resolve(o)
            # Also resolve possessives in the object
            if resolved_o and speaker:
                resolved_o = _resolve_possessives_in_object(resolved_o)

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
            rel_id = self.store(
                user_id=user_id,
                trace_decomposition=decomp,
                source_text=decomp_src or cleaned,
                confidence=confidence,
                source_timestamp=source_timestamp,
                source_tag=source_tag,
                subject=resolved_s if has_spo else None,
                predicate=p if has_spo else None,
                object=resolved_o if has_spo else None,
            )
            if rel_id:
                count += 1
        return count

    # ------------------------------------------------------------------
    # store -- 4 steps, not 11
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
        """Trace-primary store.

        4 steps:
          1. Upsert relationship row (source_text_hash dedup)
          2. Write trace columns from grammar decomposition
          3. Resolve event date (4-tier waterfall)
          4. Supersession detection

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

        try:
            with get_db_context() as conn:
                # Step 1: Upsert relationship row (dedup on source_text_hash)
                rel_id = self._upsert_relationship_row(
                    conn, user_id, subject, predicate, object,
                    confidence, utterance_type_id, source_timestamp,
                    source_text=source_text, source_tag=source_tag,
                    source_text_hash=source_text_hash,
                )
                if not rel_id:
                    return 0

                # Step 2: Write edge traces from grammar decomposition
                self._write_edge_traces(conn, rel_id, trace_decomposition)

                # Step 3: Resolve event date (4-tier waterfall)
                self._resolve_event_date(
                    conn, rel_id, source_text,
                    source_timestamp, trace_decomposition,
                )

                # Step 4: Supersession detection
                try:
                    from app.engines.temporal import get_temporal_engine
                    _te = get_temporal_engine()
                    _te.detect_supersession(
                        user_id, source_text or "", rel_id, None
                    )
                except Exception:
                    pass

                conn.commit()
                return rel_id
        except Exception as e:
            log.error("store failed on (%s, %s, %s): %s", subject, predicate, object, e)
            return 0

    # ------------------------------------------------------------------
    # _upsert_relationship_row -- kept as-is (source_text_hash dedup, FTS5)
    # ------------------------------------------------------------------

    def _upsert_relationship_row(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        subject: str,
        predicate: str,
        object: str,
        confidence: float,
        utterance_type_id: Optional[int],
        source_timestamp: Optional[str],
        source_text: Optional[str] = None,
        source_tag: Optional[str] = None,
        source_text_hash: Optional[str] = None,
    ) -> int:
        """Insert or update a relationship. Dedup on source_text_hash."""
        # Dedup: check for existing row with same source_text_hash
        existing = None
        if source_text_hash:
            try:
                existing = conn.execute(
                    """SELECT id FROM relationships
                       WHERE user_id = ? AND source_text_hash = ?""",
                    (user_id, source_text_hash),
                ).fetchone()
            except Exception:
                existing = None  # fail-open: column not yet migrated

        if existing:
            rel_id = existing["id"]
            conn.execute(
                """UPDATE relationships SET
                     confidence = MAX(confidence, ?),
                     last_confirmed_at = datetime('now'),
                     is_current = 1,
                     tombstoned_at = NULL,
                     tombstone_reason = NULL,
                     tombstone_op_id = NULL,
                     subject = COALESCE(?, subject),
                     predicate = COALESCE(?, predicate),
                     object = COALESCE(?, object)
                   WHERE id = ?""",
                (confidence, subject or None, predicate or None,
                 object or None, rel_id),
            )
            try:
                if source_text:
                    conn.execute(
                        "UPDATE relationships SET source_text = ? WHERE id = ?",
                        (source_text, rel_id),
                    )
                if source_tag:
                    conn.execute(
                        "UPDATE relationships SET source_tag = COALESCE(source_tag, ?) WHERE id = ?",
                        (source_tag, rel_id),
                    )
            except sqlite3.OperationalError:
                pass
            return rel_id

        # Insert new row
        try:
            cur = conn.execute(
                """INSERT INTO relationships
                     (user_id, subject, predicate, object, confidence,
                      utterance_type_id, source_timestamp, is_current,
                      source_text, source_tag, source_text_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (user_id, subject or None, predicate or None,
                 object or None, confidence,
                 utterance_type_id, source_timestamp,
                 source_text, source_tag, source_text_hash),
            )
        except sqlite3.OperationalError:
            # Columns not yet migrated -- fall back to pre-migration insert
            cur = conn.execute(
                """INSERT INTO relationships
                     (user_id, subject, predicate, object, confidence,
                      utterance_type_id, source_timestamp, is_current)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                (user_id, subject or None, predicate or None,
                 object or None, confidence,
                 utterance_type_id, source_timestamp),
            )
        rel_id = cur.lastrowid

        # Sync to relationships_fts for BM25 retrieval
        try:
            conn.execute(
                """INSERT INTO relationships_fts
                     (rowid, subject, predicate, object, source_text)
                   VALUES (?, ?, ?, ?, ?)""",
                (rel_id, subject or "", (predicate or "").replace("_", " "),
                 object or "", source_text or ""),
            )
        except (sqlite3.OperationalError, Exception):
            pass

        # Assign sequence_number = relationship_id (monotonic)
        try:
            conn.execute(
                "UPDATE relationships SET sequence_number = ? WHERE id = ?",
                (rel_id, rel_id),
            )
        except sqlite3.OperationalError:
            pass

        return rel_id

    # ------------------------------------------------------------------
    # _write_edge_traces -- simple, ~30 lines
    # No cosine. No anchors. No embeddings. Grammar already did the work.
    # ------------------------------------------------------------------

    def _write_edge_traces(
        self,
        conn: sqlite3.Connection,
        rel_id: int,
        trace_decomposition: Any,
    ) -> None:
        """Write grammar engine traces directly to relationship columns."""
        td = trace_decomposition
        if td is None:
            return

        try:
            conn.execute(
                """UPDATE relationships SET
                     edge_emotional_valence = ?,
                     edge_emotional_label = ?,
                     edge_schematic_category = ?,
                     edge_episodic_significance = ?,
                     edge_relational_type = ?,
                     edge_temporal_context = ?,
                     edge_negated = ?,
                     edge_mood = ?
                   WHERE id = ?""",
                (
                    round(float(getattr(td, 'emotional_valence', 0.5) or 0.5), 4),
                    getattr(td, 'emotional_state', None),
                    getattr(td, 'schematic_category', None) or 'uncategorized',
                    getattr(td, 'episodic_significance', None) or 'routine',
                    getattr(td, 'relational_type', None) or 'personal',
                    getattr(td, 'temporal_direction', None) or 'present',
                    1 if getattr(td, 'negated', False) else 0,
                    getattr(td, 'mood', 'indicative') or 'indicative',
                    rel_id,
                ),
            )
        except Exception as e:
            log.warning("edge-trace write failed: %s", e)

        # Write temporal_expression and relational_entities
        _temp_expr = getattr(td, 'temporal_expression', None)
        _rel_subj = getattr(td, 'relational_subject', None)
        _rel_ents = getattr(td, 'relational_entities', None) or []
        if isinstance(_rel_ents, list) and _rel_subj and _rel_subj.lower() != 'user':
            if _rel_subj not in _rel_ents:
                _rel_ents = list(_rel_ents) + [_rel_subj]
        _rel_json = json.dumps(_rel_ents) if _rel_ents else None

        if _temp_expr or _rel_json:
            try:
                conn.execute(
                    """UPDATE relationships SET
                         temporal_expression = COALESCE(?, temporal_expression),
                         relational_entities = COALESCE(?, relational_entities)
                       WHERE id = ?""",
                    (_temp_expr, _rel_json, rel_id),
                )
            except Exception:
                pass

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
