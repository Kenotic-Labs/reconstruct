# ============================================================================
# MEMORY ENGINE — The writer. Takes grammar + temporal output, writes to DB.
#
# Grammar engine produces meaning (5 traces, SPO, PQs, types).
# Temporal engine produces time truth (event dates, supersession, arcs).
# Memory engine computes embeddings + hash, writes one row to `edges`,
# one row to `edge_extraction`, and runs side-effects.
# ============================================================================
"""
MemoryEngine -- the sole DB writer for the continuity layer.

Write path:
    ingest_text()  -> grammar_engine.process() -> per decomp:
      1. PREPARE: extract edges columns + edge_extraction columns
      2. WRITE:   INSERT into edges + edge_extraction + FTS5
      3. SIDE-EFFECTS: entities, event date, supersession, clustering, arcs

Public interface:
    store()                 -> three-phase write
    ingest_text()           -> grammar_engine.process() -> store() per decomp
    get_relationships()     -> read edges by any combination of fields
    get_entity()            -> read an entity row by name
    entity_link()           -> resolve a mention to an entity via cosine
    supersede()             -> supersession hook called by TemporalEngine
    summarize()             -> natural-language summary
    forget_by_triple_id()   -> tombstone single edge
    forget_by_entity()      -> tombstone all edges for entity
    forget_by_time_range()  -> tombstone edges in time window
    forget_by_source()      -> tombstone edges by source_tag
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
# Entry / Exit checks
# =============================================================================

_ENTRY_CHECKED = False


class MemoryEntryError(RuntimeError):
    """Raised when memory engine's dependencies are not available."""
    pass


def _check_entry():
    """Verify grammar, temporal, DB, and reconstruction are importable. Runs once."""
    global _ENTRY_CHECKED
    if _ENTRY_CHECKED:
        return
    missing = []
    try:
        from app.engines import grammar_engine  # noqa: F401
        if not hasattr(grammar_engine, 'process'):
            missing.append("grammar_engine.process")
    except ImportError:
        missing.append("grammar_engine")
    try:
        from app.engines import temporal  # noqa: F401
        if not hasattr(temporal, 'get_temporal_engine'):
            missing.append("temporal.get_temporal_engine")
    except ImportError:
        missing.append("temporal")
    try:
        from app.db.session import get_db_context as _test_db  # noqa: F401
    except ImportError:
        missing.append("db.session")
    try:
        from app.engines import reconstruction  # noqa: F401
        if not hasattr(reconstruction, 'reconstruct'):
            missing.append("reconstruction.reconstruct")
    except ImportError:
        missing.append("reconstruction")
    if missing:
        raise MemoryEntryError(
            f"memory engine entry check failed — missing: {', '.join(missing)}"
        )
    _ENTRY_CHECKED = True


def check_exit(edge_id: int) -> bool:
    """Validate that store() produced a valid edge."""
    return isinstance(edge_id, int) and edge_id > 0


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
    id: int
    subject: str
    predicate: str
    object: str
    confidence: float = 0.9
    is_current: bool = True


# =============================================================================
# MEMORY ENGINE
# =============================================================================

class MemoryEngine:
    """The sole DB writer. Grammar engine and temporal engine produce.
    Memory engine takes their output and writes to edges + edge_extraction."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Singular ingestion path
    # ------------------------------------------------------------------

    @dataclass
    class IngestionResult:
        """Output of _run_ingestion_path — everything both engines produce."""
        cleaned_text: str = ""
        rows: list = field(default_factory=list)
        grammar_result: Any = None
        # Temporal engine outputs (None = NA, engine had nothing to say)
        resolved_event_date: Optional[str] = None
        temporal_expression: Optional[str] = None

    def _run_ingestion_path(
        self,
        text: str,
        speaker: Optional[str] = None,
        listener: Optional[str] = None,
        source_timestamp: Optional[str] = None,
    ) -> "MemoryEngine.IngestionResult":
        """The one write-path entry.

        ingestion.cleanup() → clean text
                                ↓           ↓
                          grammar_engine   temporal_engine
                           (traces/SPO)    (event date)
                                ↓           ↓
                          IngestionResult (merged)

        Both engines receive the same clean text.
        If an engine has nothing to return, its fields are None (NA).
        Memory engine waits for both before writing.
        """
        result = self.IngestionResult()

        if not text or not text.strip():
            return result

        from app.engines.ingestion import cleanup as _cleanup
        from app.engines import grammar_engine as _grammar_eng

        # ── Ingestion: messy text → clean English ──
        cleaned = _cleanup(text.strip(), speaker=speaker)
        if not cleaned:
            return result
        result.cleaned_text = cleaned

        # ── Grammar engine: clean text → traces, SPO, decompositions ──
        grammar_result = _grammar_eng.process(
            cleaned, speaker=speaker, listener=listener or "user",
        )
        result.grammar_result = grammar_result

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
        result.rows = rows

        # ── Temporal engine: clean text → event date, temporal expression ──
        try:
            from app.engines.temporal import get_temporal_engine
            _te = get_temporal_engine()

            # Resolve event date from clean text
            _resolved = _te.resolve_event_date(cleaned, source_timestamp)

            # Fallback: grammar engine's temporal_expression
            if not _resolved and decomps:
                for d in decomps:
                    _temp_expr = getattr(d, 'temporal_expression', None)
                    if _temp_expr:
                        _resolved = _te.resolve_event_date(
                            _temp_expr, source_timestamp
                        )
                        if _resolved:
                            break
                    _temp_resolved = getattr(d, 'temporal_resolved', None)
                    if _temp_resolved:
                        _resolved = _temp_resolved
                        break

            # Final fallback: source_timestamp (session time)
            if not _resolved and source_timestamp:
                _resolved = source_timestamp

            result.resolved_event_date = _resolved

            # Extract temporal_expression from clean text via spaCy NER.
            # This is temporal engine's job, not grammar engine's.
            try:
                from app.engines.grammar_engine import _get_nlp
                _te_doc = _get_nlp()(cleaned)
                for ent in _te_doc.ents:
                    if ent.label_ in ("DATE", "TIME"):
                        result.temporal_expression = ent.text
                        break
            except Exception:
                pass
        except Exception as e:
            log.warning("temporal engine failed during ingestion: %s", e)
            # NA — temporal has nothing, memory writes without it

        return result

    def clean(self, text: str) -> str:
        """Run the singular cleanup path used by live ingestion."""
        if not text or not text.strip():
            return text or ""
        from app.engines.ingestion import cleanup as _cleanup
        return _cleanup(text.strip())

    def extract(self, text: str) -> List[Tuple[str, str, str, bool]]:
        """Run the singular extraction path used by live ingestion."""
        result = self._run_ingestion_path(text)
        extracted: List[Tuple[str, str, str, bool]] = []
        for s, p, o, decomp in result.rows:
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
        listener: Optional[str] = None,
        speaker_is_user: bool = True,
        confidence: float = 0.9,
        source_tag: Optional[str] = None,
    ) -> int:
        """End-to-end write-path entry for raw text.

        1. Grammar engine produces trace decompositions
        2. Speaker resolution (I/me/myself -> speaker name)
        3. Preserve speaker identity in decomp.relational_subject
        4. For each decomposition -> store()

        Returns the number of rows actually stored.
        """
        if not text or not text.strip():
            return 0

        ingestion = self._run_ingestion_path(
            text,
            speaker=speaker,
            listener=listener,
            source_timestamp=source_timestamp,
        )
        cleaned = ingestion.cleaned_text
        triples_with_decomp = ingestion.rows
        _grammar_result = ingestion.grammar_result

        if not cleaned:
            return 0

        from app.engines.grammar_engine import _get_nlp

        # Pronouns already resolved by grammar_engine.process().
        # No manual _resolve() needed.

        # Gate: skip non-storable decompositions.
        # Backchannels, questions → keep only imposed facts.
        # Commands/imperatives → skip entirely (not facts).
        if _grammar_result is not None and hasattr(_grammar_result, 'classification'):
            _cls = _grammar_result.classification
            if _cls.is_command:
                triples_with_decomp = []  # "take a look", "keep up" — not facts
            elif _cls.is_backchannel or _cls.is_question:
                triples_with_decomp = [
                    (s, p, o, d) for s, p, o, d in triples_with_decomp
                    if d is not None and getattr(d, 'extraction_rule', '') and
                    'imposed' in getattr(d, 'extraction_rule', '')
                ]

        count = 0
        for s, p, o, decomp in triples_with_decomp:
            resolved_s = s
            resolved_o = o

            # Skip edges with no object — (Subject, predicate, None) is useless
            if not resolved_o and not (getattr(decomp, 'object', '') if decomp else ''):
                continue

            # Skip edges where object contains the subject — grammar leak
            # e.g. (Caroline, love, Caroline is keen on counseling...)
            if (resolved_s and resolved_o
                    and resolved_s.lower() in resolved_o.lower()
                    and len(resolved_o) > len(resolved_s) + 10):
                continue

            # Content filter: skip edges where object is just pronouns/
            # function words, too short, or same as subject/speaker.
            obj_to_check = resolved_o or (getattr(decomp, 'object', '') if decomp else '')
            if obj_to_check:
                try:
                    from app.engines.grammar_engine import _get_nlp
                    _doc = _get_nlp()(obj_to_check.strip())
                    _CONTENT_POS = frozenset({"NOUN", "PROPN", "NUM", "ADJ", "VERB"})
                    _obj_lower = obj_to_check.strip().lower()
                    # Skip indefinite pronouns — spaCy POS=PRON or DET
                    if len(_doc) <= 2 and all(
                        tok.pos_ in ("PRON", "DET", "ADV", "ADP") for tok in _doc
                    ):
                        continue
                    # Skip very short objects (< 3 chars)
                    if len(_obj_lower) < 3:
                        continue
                    # Skip objects that are just the speaker or subject name
                    if speaker and _obj_lower == speaker.lower():
                        continue
                    if resolved_s and _obj_lower == resolved_s.lower():
                        continue
                    # Skip single-token ADJ/ADV objects — commentary, not facts.
                    # "great", "awesome", "glad" are not storable facts.
                    # Multi-word objects with nouns ARE facts.
                    if len(_doc) == 1 and _doc[0].pos_ in ("ADJ", "ADV", "INTJ"):
                        continue
                    # Skip 2-token DET+ADJ objects ("the best", "so cool")
                    if (len(_doc) == 2 and _doc[0].pos_ in ("DET", "ADV")
                            and _doc[1].pos_ in ("ADJ", "ADV")):
                        continue
                    if not any(tok.pos_ in _CONTENT_POS for tok in _doc):
                        continue
                except Exception:
                    pass

            # Preserve speaker identity in trace decomposition
            if decomp is not None and speaker:
                _rs = getattr(decomp, 'relational_subject', None) or ''
                _rs_low = _rs.lower()
                # Check if relational_subject is a pronoun/placeholder via spaCy
                _rs_is_placeholder = False
                if not _rs or _rs_low == "" or _rs_low == "user":
                    _rs_is_placeholder = True
                else:
                    _rs_doc = _get_nlp()(_rs_low)
                    if _rs_doc and len(_rs_doc) == 1 and _rs_doc[0].pos_ in ("PRON", "DET", "ADV"):
                        _rs_is_placeholder = True
                if _rs_is_placeholder:
                    decomp.relational_subject = speaker
                _ents = getattr(decomp, 'relational_entities', None) or []
                if isinstance(_ents, list) and speaker not in _ents:
                    decomp.relational_entities = list(_ents) + [speaker]

            has_spo = bool(resolved_s and p and resolved_o)
            decomp_src = getattr(decomp, 'source_text', '') if decomp else ''

            # Pronouns already resolved by grammar_engine.process() before
            # trace extraction. No duplicate resolve needed.
            _final_src = decomp_src or cleaned

            rel_id = self.store(
                user_id=user_id,
                trace_decomposition=decomp,
                source_text=_final_src,
                source_timestamp=source_timestamp,
                source_tag=source_tag,
                subject=resolved_s if has_spo else None,
                predicate=p if has_spo else None,
                object=resolved_o if has_spo else None,
                resolved_event_date=ingestion.resolved_event_date,
                temporal_expression=ingestion.temporal_expression,
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
        source_timestamp: Optional[str] = None,
        source_tag: Optional[str] = None,
        trace_decomposition: Any = None,
        resolved_event_date: Optional[str] = None,
        temporal_expression: Optional[str] = None,
    ) -> int:
        """Three-phase store: PREPARE -> WRITE -> SIDE-EFFECTS.

        Returns: edge id (0 on failure).
        """
        _check_entry()
        td = trace_decomposition
        from app.engines.grammar_engine import _get_nlp

        # Fill S/P/O from decomposition if not provided directly
        if td is not None:
            subject = subject or getattr(td, 'subject', '') or ''
            predicate = predicate or getattr(td, 'predicate', '') or ''
            object = object or getattr(td, 'object', '') or ''
            if not source_text and getattr(td, 'source_text', ''):
                # Pronouns already resolved by grammar_engine.process()
                source_text = td.source_text

        subject = (subject or "").strip()
        predicate = (predicate or "").strip().lower().replace(" ", "_")
        object = (object or "").strip()

        # ── Edge quality gate: filter garbage triples before storage ──
        # Reject edges with pronoun/determiner subjects, None objects,
        # or subject==object (grammar engine artifacts).
        # Reject subjects that are pronouns/placeholders — via spaCy POS.
        # 3rd person pronouns (it, he, she, they) → skip edge entirely
        # (we can't resolve them without coreference).
        # Non-pronoun garbage (determiners, adverbs) → also skip.
        _subj_low = subject.lower()
        _subj_is_garbage = False
        if _subj_low:
            _subj_doc = _get_nlp()(_subj_low)
            if _subj_doc and len(_subj_doc) == 1:
                _tok = _subj_doc[0]
                if _tok.pos_ in ("DET", "ADV", "SCONJ"):
                    _subj_is_garbage = True
                elif _tok.pos_ == "PRON":
                    # 3rd person pronouns → can't resolve, skip
                    _person = _tok.morph.get("Person", [""])[0]
                    if _person == "3":
                        subject = ""  # skip — no coreference
                    else:
                        _subj_is_garbage = True
        if _subj_is_garbage:
            subject = ""  # will fail has_triple check below
        # Strip leading determiners from subject via spaCy POS
        if subject:
            _subj_doc2 = _get_nlp()(subject)
            if _subj_doc2 and len(_subj_doc2) > 1 and _subj_doc2[0].pos_ == "DET":
                subject = "".join(tok.text_with_ws for tok in _subj_doc2[1:]).strip()
        # Reject subject==object
        if subject and object and subject.lower() == object.lower():
            object = ""

        has_triple = bool(subject and predicate and object)
        has_source = bool(source_text and source_text.strip())
        if not has_triple and not has_source:
            return 0
        # Skip very short source texts — these are backchannels,
        # greetings, and fragments that don't carry factual content.
        # "Mel !", "Good to see you!", "Caroline !" → no useful edges.
        if source_text and len(source_text.strip()) < 15 and not object:
            return 0

        if not has_source and has_triple:
            source_text = f"{subject} {predicate.replace('_', ' ')} {object}"

        source_text_hash = hashlib.sha256(
            source_text.encode('utf-8')
        ).hexdigest()

        # ── Phase 1: PREPARE ──
        edge_row, extraction_row = self._prepare_rows(
            user_id, td, subject, predicate, object, source_text,
            source_timestamp, source_tag, source_text_hash,
            resolved_event_date=resolved_event_date,
            temporal_expression=temporal_expression,
        )

        try:
            with get_db_context() as conn:
                # ── Phase 2: WRITE ──
                edge_id = self._write_edge(conn, edge_row, user_id, source_text_hash)
                if not edge_id:
                    return 0

                # Write extraction metadata
                self._write_extraction(conn, edge_id, extraction_row)

                # ── Phase 3: SIDE-EFFECTS ──

                # 3a: Populate entities
                self._upsert_entities(conn, user_id, td)

                # 3b: Entity type resolution (read entities, write back to edge)
                _type_updates: Dict[str, Any] = {}
                try:
                    for col, name in [("subject_type", subject), ("object_type", object)]:
                        if not name:
                            continue
                        ent_row = conn.execute(
                            "SELECT entity_type FROM entities WHERE user_id = ? AND name = ? LIMIT 1",
                            (user_id, name),
                        ).fetchone()
                        if ent_row and ent_row["entity_type"]:
                            _type_updates[col] = ent_row["entity_type"]
                except Exception:
                    pass
                if _type_updates:
                    _cols = ", ".join(f"{k} = ?" for k in _type_updates)
                    _vals: List[Any] = list(_type_updates.values()) + [edge_id]
                    try:
                        conn.execute(
                            f"UPDATE edges SET {_cols} WHERE id = ?",
                            tuple(_vals),
                        )
                    except Exception:
                        pass

                # 3c: Affiliation (needs entities populated first)
                try:
                    _speaker = getattr(td, 'relational_subject', None) or subject or ''
                    _aff = self._compute_affiliation(conn, user_id, td, _speaker)
                    if _aff is not None:
                        conn.execute(
                            "UPDATE edge_extraction SET edge_affiliation = ? WHERE edge_id = ?",
                            (round(_aff, 4), edge_id),
                        )
                except Exception:
                    pass

                # 3d: Predicted queries (onto the edge row)
                if has_triple:
                    try:
                        from app.engines.grammar_engine import generate_predicted_queries as _gen_pqs
                        _pqs = _gen_pqs(subject, predicate, object)
                        _pq_updates: Dict[str, str] = {}
                        for i, (q_text, _q_emb) in enumerate(_pqs[:4]):
                            _pq_updates[f"pq_{i+1}"] = q_text
                        if _pq_updates:
                            _cols = ", ".join(f"{k} = ?" for k in _pq_updates)
                            _vals = list(_pq_updates.values()) + [edge_id]
                            conn.execute(
                                f"UPDATE edges SET {_cols} WHERE id = ?",
                                tuple(_vals),
                            )
                    except Exception:
                        pass

                # 3e: Tier routing (facts/milestones side tables)
                tier = self._classify_tier(td)
                if tier == "fact" and td is not None:
                    self._upsert_fact(conn, user_id, edge_id, td)
                elif tier == "milestone" and td is not None:
                    self._append_milestone(conn, user_id, edge_id, td)

                # 3f: Event date — already resolved in _run_ingestion_path,
                # written in Phase 1 PREPARE. No side-effect needed.

                # 3g: Supersession detection (temporal engine)
                try:
                    from app.engines.temporal import get_temporal_engine
                    get_temporal_engine().detect_supersession(
                        user_id, source_text or "", edge_id, None
                    )
                except Exception:
                    pass

                # 3h: Cluster assignment + arc membership
                self._assign_cluster(conn, user_id, edge_id, td)
                try:
                    _cr = conn.execute(
                        "SELECT cluster_id FROM edges WHERE id = ?",
                        (edge_id,),
                    ).fetchone()
                    _cid = _cr["cluster_id"] if _cr else None
                    from app.engines.temporal import get_temporal_engine
                    _arc_id = get_temporal_engine().detect_arcs(user_id, edge_id, _cid)
                    if _arc_id:
                        conn.execute(
                            "UPDATE edges SET arc_id = ? WHERE id = ?",
                            (_arc_id, edge_id),
                        )
                except Exception as e:
                    log.warning("arc assignment failed for edge_id=%s: %s", edge_id, e)

                conn.commit()
                return edge_id
        except Exception as e:
            try:
                conn.rollback()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            log.error("store failed on (%s, %s, %s): %s", subject, predicate, object, e)
            return 0

    # ------------------------------------------------------------------
    # Phase 1: _prepare_rows -- pure computation, no DB
    # ------------------------------------------------------------------

    def _prepare_rows(
        self,
        user_id: int,
        td: Any,
        subject: str,
        predicate: str,
        object: str,
        source_text: str,
        source_timestamp: Optional[str],
        source_tag: Optional[str],
        source_text_hash: str,
        resolved_event_date: Optional[str] = None,
        temporal_expression: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare column values for edges + edge_extraction.

        Pure computation -- no database access. Returns two dicts:
        (edge_row, extraction_row).
        """
        edge: Dict[str, Any] = {}
        extraction: Dict[str, Any] = {}

        # ── edges: Core Edge ──
        edge["user_id"] = user_id
        edge["subject"] = subject or None
        edge["predicate"] = predicate or None
        edge["object"] = object or None
        edge["source_text"] = source_text
        edge["source_text_hash"] = source_text_hash
        edge["source_timestamp"] = source_timestamp
        if resolved_event_date:
            edge["resolved_event_date"] = resolved_event_date

        # ── edges: Five Traces (from grammar engine) ──
        if td is not None:
            _ev = getattr(td, 'emotional_valence', None)
            edge["edge_emotional_valence"] = round(
                float(_ev if _ev is not None else 0.5), 4
            )
            edge["edge_emotional_label"] = getattr(td, 'emotional_state', None)
            edge["edge_schematic_category"] = (
                getattr(td, 'schematic_category', None) or 'uncategorized'
            )
            edge["edge_episodic_significance"] = (
                getattr(td, 'episodic_significance', None) or 'routine'
            )
            _grammar_rel_type = getattr(td, 'relational_type', None)
            if _grammar_rel_type and _grammar_rel_type != 'personal':
                edge["edge_relational_type"] = _grammar_rel_type
            else:
                _schema = edge.get("edge_schematic_category", "uncategorized")
                _SCHEMA_TO_RELTYPE = {
                    "career": "professional",
                    "finance": "professional",
                    "education": "professional",
                    "family": "familial",
                    "social": "social",
                    "hobby": "social",
                    "experience": "social",
                    "health": "personal",
                    "housing": "personal",
                    "identity": "personal",
                    "planning": "personal",
                    "uncategorized": "personal",
                }
                edge["edge_relational_type"] = _SCHEMA_TO_RELTYPE.get(
                    _schema, "personal"
                )
            edge["edge_temporal_context"] = (
                getattr(td, 'temporal_direction', None) or 'present'
            )

            # Temporal state
            edge["is_historical"] = 1 if getattr(td, 'is_historical', False) else 0

            # Temporal expression from temporal engine (not grammar engine)
            edge["temporal_expression"] = temporal_expression or getattr(td, 'temporal_expression', None)
            edge["episodic_fact"] = getattr(td, 'episodic_fact', None) or None
            _rel_subj = getattr(td, 'relational_subject', None)
            _rel_ents = getattr(td, 'relational_entities', None) or []
            if isinstance(_rel_ents, list) and _rel_subj and _rel_subj.lower() != 'user':
                if _rel_subj not in _rel_ents:
                    _rel_ents = list(_rel_ents) + [_rel_subj]
            edge["relational_entities"] = json.dumps(_rel_ents) if _rel_ents else None

        # ── edges: Embeddings (memory engine computes — storage concern) ──
        try:
            from app.vector.embedder import embed_text as _embed
            if source_text:
                edge["edge_embedding"] = _embed(source_text).tobytes()
            if predicate:
                edge["predicate_embedding"] = _embed(
                    predicate.replace("_", " ")
                ).tobytes()
        except Exception:
            pass

        # ── edge_extraction: Grammar metadata ──
        if td is not None:
            extraction["edge_negated"] = 1 if getattr(td, 'negated', False) else 0
            extraction["edge_mood"] = getattr(td, 'mood', 'indicative') or 'indicative'
            extraction["episodic_fact"] = getattr(td, 'episodic_fact', None) or None
            extraction["emotional_target"] = getattr(td, 'emotional_target', None) or None
            extraction["extraction_rule"] = getattr(td, 'extraction_rule', None) or None

        # ── edge_extraction: Provenance ──
        if source_tag:
            extraction["source_tag"] = source_tag

        # ── edge_extraction: Weak/Future ──
        extraction["utterance_type_id"] = getattr(td, 'utterance_type', None) if td else None

        # Internal metadata (not written to SQL)
        edge["_tier"] = self._classify_tier(td)

        return edge, extraction

    # ------------------------------------------------------------------
    # Phase 2: _write_edge -- INSERT into edges + FTS5
    # ------------------------------------------------------------------

    def _write_edge(
        self,
        conn: sqlite3.Connection,
        row: Dict[str, Any],
        user_id: int,
        source_text_hash: str,
    ) -> int:
        """Write one edge row. Dedup on source_text_hash.

        Returns edge id (0 on failure).
        """
        # ── Dedup check ──
        existing = None
        if source_text_hash:
            try:
                existing = conn.execute(
                    "SELECT id, tombstoned_at, subject, predicate, object, source_text "
                    "FROM edges WHERE user_id = ? AND source_text_hash = ?",
                    (user_id, source_text_hash),
                ).fetchone()
            except Exception:
                existing = None

        if existing:
            edge_id: int = existing["id"]

            # Tombstone guard: never revive a forgotten row
            if existing["tombstoned_at"] is not None:
                return edge_id

            # Capture old values for FTS5 delete
            old_subj = existing["subject"] or ""
            old_pred = (existing["predicate"] or "").replace("_", " ")
            old_obj = existing["object"] or ""
            old_src = existing["source_text"] or ""

            # ── Build ONE UPDATE ──
            update_cols: List[str] = []
            update_vals: List[Any] = []

            update_cols.append("last_confirmed_at = datetime('now')")
            update_cols.append("is_current = 1")

            # COALESCE for S/P/O
            update_cols.append("subject = COALESCE(?, subject)")
            update_vals.append(row.get("subject"))
            update_cols.append("predicate = COALESCE(?, predicate)")
            update_vals.append(row.get("predicate"))
            update_cols.append("object = COALESCE(?, object)")
            update_vals.append(row.get("object"))

            if row.get("source_text"):
                update_cols.append("source_text = ?")
                update_vals.append(row["source_text"])

            # Trace + embedding columns
            _trace_cols = (
                "edge_emotional_valence", "edge_emotional_label",
                "edge_schematic_category", "edge_episodic_significance",
                "edge_relational_type", "edge_temporal_context",
                "is_historical", "temporal_expression", "relational_entities",
                "edge_embedding", "predicate_embedding",
            )
            for col in _trace_cols:
                if col in row and row[col] is not None:
                    update_cols.append(f"{col} = ?")
                    update_vals.append(row[col])

            update_vals.append(edge_id)
            sql = f"UPDATE edges SET {', '.join(update_cols)} WHERE id = ?"
            conn.execute(sql, tuple(update_vals))

            # FTS5 sync
            try:
                conn.execute(
                    "INSERT INTO edges_fts(edges_fts, rowid, "
                    "subject, predicate, object, source_text) "
                    "VALUES('delete', ?, ?, ?, ?, ?)",
                    (edge_id, old_subj, old_pred, old_obj, old_src),
                )
                conn.execute(
                    """INSERT INTO edges_fts
                         (rowid, subject, predicate, object, source_text)
                       VALUES (?, ?, ?, ?, ?)""",
                    (edge_id, row.get("subject") or "",
                     (row.get("predicate") or "").replace("_", " "),
                     row.get("object") or "", row.get("source_text") or ""),
                )
            except Exception:
                pass

            return edge_id

        # ── INSERT new row ──
        insert_cols: List[str] = ["user_id", "is_current"]
        insert_vals: List[Any] = [user_id, 1]

        for col, val in row.items():
            if col.startswith("_"):
                continue
            if val is not None and col != "user_id":
                insert_cols.append(col)
                insert_vals.append(val)

        placeholders = ", ".join(["?"] * len(insert_vals))
        col_names = ", ".join(insert_cols)

        cur = conn.execute(
            f"INSERT INTO edges ({col_names}) VALUES ({placeholders})",
            tuple(insert_vals),
        )
        edge_id = cur.lastrowid

        # FTS5 sync
        try:
            conn.execute(
                """INSERT INTO edges_fts
                     (rowid, subject, predicate, object, source_text)
                   VALUES (?, ?, ?, ?, ?)""",
                (edge_id, row.get("subject") or "",
                 (row.get("predicate") or "").replace("_", " "),
                 row.get("object") or "", row.get("source_text") or ""),
            )
        except Exception:
            pass

        # Sequence number
        try:
            conn.execute(
                "UPDATE edges SET sequence_number = ? WHERE id = ?",
                (edge_id, edge_id),
            )
        except Exception:
            pass

        return edge_id

    # ------------------------------------------------------------------
    # _write_extraction -- INSERT into edge_extraction
    # ------------------------------------------------------------------

    def _write_extraction(
        self,
        conn: sqlite3.Connection,
        edge_id: int,
        extraction: Dict[str, Any],
    ) -> None:
        """Write one edge_extraction row. Always INSERT OR REPLACE."""
        extraction["edge_id"] = edge_id

        cols = [k for k, v in extraction.items() if v is not None]
        vals = [extraction[k] for k in cols]

        if not cols:
            return

        placeholders = ", ".join(["?"] * len(vals))
        col_names = ", ".join(cols)
        try:
            conn.execute(
                f"INSERT OR REPLACE INTO edge_extraction ({col_names}) VALUES ({placeholders})",
                tuple(vals),
            )
        except Exception:
            pass

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
        """Compute edge_affiliation: how personally connected a fact is.

        1.0 = speaker's own life fact
        0.8 = well-known entity (mention_count >= 5)
        0.6 = mentioned before (mention_count >= 2)
        0.4 = first mention but named
        0.2 = unknown entity
        0.1 = no entities at all
        """
        if td is None:
            return None

        rel_subj = getattr(td, 'relational_subject', None) or ''
        if rel_subj:
            subj_lower = rel_subj.strip().lower()
            speaker_lower = (speaker or '').strip().lower()
            if subj_lower in ('user', 'i', 'me', 'myself') or (
                speaker_lower and subj_lower == speaker_lower
            ):
                return 1.0

        candidates: List[str] = []
        for ent in (getattr(td, 'relational_entities', None) or []):
            if ent and ent.strip().lower() not in ('', 'user'):
                candidates.append(ent.strip())
        for attr in ('subject', 'object'):
            val = getattr(td, attr, None)
            if val and val.strip().lower() not in ('', 'user'):
                candidates.append(val.strip())

        seen: set[str] = set()
        unique: List[str] = []
        for c in candidates:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)

        if not unique:
            return 0.1

        scores: List[float] = []
        for name in unique:
            row = None
            try:
                row = conn.execute(
                    "SELECT mention_count FROM entities WHERE user_id = ? AND name = ? LIMIT 1",
                    (user_id, name),
                ).fetchone()
            except Exception:
                pass

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
    # Entity population
    # ------------------------------------------------------------------

    def _upsert_entities(self, conn: sqlite3.Connection, user_id: str, td: Any) -> None:
        """Upsert entities from TraceDecomposition into the entities table."""
        if td is None:
            return

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
                    "SELECT id, mention_count, entity_type FROM entities WHERE user_id = ? AND name = ?",
                    (user_id, name),
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE entities SET mention_count = mention_count + 1 WHERE id = ?",
                        (existing["id"],),
                    )
                    # Resolve type if not yet set
                    if not existing["entity_type"]:
                        try:
                            from app.engines.type_resolver import resolve_entity_type
                            _etype = resolve_entity_type(name)
                            if _etype and _etype != "GENERIC":
                                conn.execute(
                                    "UPDATE entities SET entity_type = ? WHERE id = ?",
                                    (_etype, existing["id"]),
                                )
                        except Exception:
                            pass
                else:
                    # Resolve entity type at first insert
                    _etype = None
                    try:
                        from app.engines.type_resolver import resolve_entity_type
                        _etype = resolve_entity_type(name)
                    except Exception:
                        pass
                    conn.execute(
                        "INSERT INTO entities (user_id, name, mention_count, entity_type) VALUES (?, ?, 1, ?)",
                        (user_id, name, _etype),
                    )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Tier routing
    # ------------------------------------------------------------------

    def _classify_tier(self, td: Any) -> str:
        """Classify a TraceDecomposition into fact/milestone/episode tier."""
        if td is None:
            return "episode"
        sig = getattr(td, 'episodic_significance', 'routine') or 'routine'
        if sig == "milestone":
            return "milestone"
        if sig == "stative":
            return "fact"
        return "episode"

    def _classify_supersession_type(self, td: Any) -> str:
        """Classify how a fact was superseded: update, correction, or reversal."""
        if td is None:
            return "update"
        extraction_rule = getattr(td, 'extraction_rule', '') or ''
        if 'correction' in extraction_rule.lower():
            return "correction"
        if getattr(td, 'negated', False):
            return "reversal"
        return "update"

    # ------------------------------------------------------------------
    # _upsert_fact
    # ------------------------------------------------------------------

    def _upsert_fact(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        edge_id: int,
        td: Any,
    ) -> None:
        """Upsert a fact row keyed on schema::VerbClass::subject."""
        from app.engines.grammar_engine import classify_verb_class

        schema = getattr(td, 'schematic_category', 'uncategorized') or 'uncategorized'
        predicate = getattr(td, 'predicate', '') or ''
        subject = getattr(td, 'subject', '') or ''
        value = getattr(td, 'object', '') or ''

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
                             last_confirmed_at = datetime('now'),
                             provenance_memory_id = ?,
                             history = ?
                           WHERE id = ?""",
                        (value, edge_id,
                         json.dumps(history), existing["id"]),
                    )
                else:
                    conn.execute(
                        "UPDATE facts SET last_confirmed_at = datetime('now') WHERE id = ?",
                        (existing["id"],),
                    )
            else:
                conn.execute(
                    """INSERT INTO facts
                         (user_id, key, value,
                          provenance_memory_id)
                       VALUES (?, ?, ?, ?)""",
                    (user_id, key, value, edge_id),
                )
        except Exception as e:
            log.warning("_upsert_fact failed for key=%s: %s", key, e)

    # ------------------------------------------------------------------
    # _append_milestone
    # ------------------------------------------------------------------

    def _append_milestone(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        edge_id: int,
        td: Any,
    ) -> None:
        """Insert a milestone row if no exact-text duplicate exists."""
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
                return

            conn.execute(
                """INSERT INTO milestones
                     (user_id, event_type, description,
                      provenance_memory_id)
                   VALUES (?, ?, ?, ?)""",
                (user_id, event_type, description, edge_id),
            )
        except Exception as e:
            log.warning("_append_milestone failed: %s", e)

    # ------------------------------------------------------------------
    # _resolve_event_date
    # ------------------------------------------------------------------

    def _resolve_event_date(
        self,
        conn: sqlite3.Connection,
        edge_id: int,
        source_text: str,
        source_timestamp: Optional[str],
        trace_decomposition: Any,
    ) -> None:
        """Resolve event date via 4-tier waterfall and write to edge."""
        td = trace_decomposition
        try:
            from app.engines.temporal import get_temporal_engine
            _te = get_temporal_engine()

            # Tier 1: resolve from source text
            _resolved = _te.resolve_event_date(
                source_text or "", source_timestamp
            )

            # Tier 2: grammar engine temporal_expression
            if not _resolved and td is not None:
                _temp_expr = getattr(td, 'temporal_expression', None)
                if _temp_expr:
                    _resolved = _te.resolve_event_date(
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
                    "UPDATE edges SET resolved_event_date = ? WHERE id = ?",
                    (_resolved, edge_id),
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # _assign_cluster
    # ------------------------------------------------------------------

    def _assign_cluster(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        edge_id: int,
        td: Any,
    ) -> None:
        """Assign cluster_id to a newly written edge via entity overlap."""
        try:
            entities: set[str] = set()

            try:
                row = conn.execute(
                    "SELECT subject, object, relational_entities FROM edges WHERE id = ?",
                    (edge_id,),
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
                pass

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

            entities.discard("")
            entities.discard("user")

            if not entities:
                return

            entity_list = list(entities)
            placeholders = ",".join("?" for _ in entity_list)
            like_clauses = " OR ".join(
                "relational_entities LIKE ?" for _ in entity_list
            )
            like_vals = [f'%"{e}"%' for e in entity_list]

            existing_clusters = conn.execute(
                f"""SELECT DISTINCT cluster_id FROM edges
                    WHERE user_id = ? AND cluster_id IS NOT NULL AND cluster_id != ''
                    AND id != ?
                    AND (subject IN ({placeholders})
                         OR object IN ({placeholders})
                         OR ({like_clauses}))""",
                [user_id, edge_id] + entity_list + entity_list + like_vals,
            ).fetchall()

            cluster_ids = [r["cluster_id"] for r in existing_clusters if r["cluster_id"]]

            if not cluster_ids:
                chosen = str(edge_id)
            elif len(cluster_ids) == 1:
                chosen = cluster_ids[0]
            else:
                chosen = min(cluster_ids, key=lambda cid: int(cid) if cid.isdigit() else float('inf'))
                old_ids = [cid for cid in cluster_ids if cid != chosen]
                if old_ids:
                    merge_placeholders = ",".join("?" for _ in old_ids)
                    conn.execute(
                        f"""UPDATE edges SET cluster_id = ?
                            WHERE user_id = ? AND cluster_id IN ({merge_placeholders})""",
                        [chosen, user_id] + old_ids,
                    )

            conn.execute(
                "UPDATE edges SET cluster_id = ? WHERE id = ?",
                (chosen, edge_id),
            )

        except Exception as e:
            log.warning("_assign_cluster failed for edge_id=%s: %s", edge_id, e)

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
            f"SELECT id, subject, predicate, object, is_current "
            f"FROM edges WHERE {' AND '.join(conditions)} "
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
    # Neural entity linker
    # ------------------------------------------------------------------

    def entity_link(
        self,
        user_id: int,
        mention_text: str,
    ) -> Optional[Entity]:
        """Resolve a free-text mention to a canonical entity via cosine."""
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
        """Mark a prior edge as superseded by a newer one."""
        try:
            with get_db_context() as conn:
                conn.execute(
                    """UPDATE edges SET
                         is_current = 0,
                         superseded_at = datetime('now'),
                         superseded_by = ?
                       WHERE id = ?""",
                    (superseded_by, relationship_id),
                )
                # Clear predicted queries on superseded edge
                try:
                    conn.execute(
                        """UPDATE edges SET
                             pq_1 = NULL, pq_2 = NULL, pq_3 = NULL, pq_4 = NULL
                           WHERE id = ?""",
                        (relationship_id,),
                    )
                except Exception:
                    pass
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

    def _tombstone_rows(
        self, conn: sqlite3.Connection, user_id: int,
        ids: List[int], reason: str, op_id: str,
    ) -> int:
        """Flip tombstone flag on edges, write reason to edge_extraction."""
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)

        # Tombstone on edges (just the timestamp)
        cur = conn.execute(
            f"""UPDATE edges SET
                  tombstoned_at = datetime('now')
                WHERE user_id = ? AND tombstoned_at IS NULL
                  AND id IN ({placeholders})""",
            [user_id] + list(ids),
        )
        count = cur.rowcount or 0

        # Write reason/op_id to edge_extraction
        for eid in ids:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO edge_extraction
                         (edge_id, tombstone_reason, tombstone_op_id)
                       VALUES (?, ?, ?)""",
                    (eid, reason, op_id),
                )
            except Exception:
                pass

        return count

    def forget_by_triple_id(self, user_id: int, triple_id: int) -> int:
        """Tombstone a single edge by its id."""
        from uuid import uuid4
        op_id = uuid4().hex
        reason = f"by_triple_id:{int(triple_id)}"
        try:
            with get_db_context() as conn:
                row = conn.execute(
                    """SELECT id FROM edges
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
                conn.commit()
                return count
        except Exception as e:
            log.error("forget_by_triple_id failed: %s", e)
            return 0

    def forget_by_entity(self, user_id: int, entity_name: str) -> int:
        """Tombstone every live edge where the entity appears."""
        from uuid import uuid4
        if not entity_name or not entity_name.strip():
            return 0
        name = entity_name.strip()
        op_id = uuid4().hex
        reason = f"by_entity:{name}"
        try:
            with get_db_context() as conn:
                rows = conn.execute(
                    """SELECT id FROM edges
                       WHERE user_id = ? AND tombstoned_at IS NULL
                         AND COALESCE(is_current, 1) = 1
                         AND (LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?))""",
                    (user_id, name, name),
                ).fetchall()
                ids = [r["id"] for r in rows]
                count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
                conn.commit()
                return count
        except Exception as e:
            log.error("forget_by_entity failed: %s", e)
            return 0

    def forget_by_time_range(
        self, user_id: int, start_iso: str, end_iso: str,
    ) -> int:
        """Tombstone every live edge in a time window."""
        from uuid import uuid4
        op_id = uuid4().hex
        reason = f"by_time_range:{start_iso}..{end_iso}"
        try:
            with get_db_context() as conn:
                rows = conn.execute(
                    """SELECT id FROM edges
                       WHERE user_id = ? AND tombstoned_at IS NULL
                         AND COALESCE(is_current, 1) = 1
                         AND source_timestamp IS NOT NULL
                         AND source_timestamp >= ?
                         AND source_timestamp <= ?""",
                    (user_id, start_iso, end_iso),
                ).fetchall()
                ids = [r["id"] for r in rows]
                count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
                conn.commit()
                return count
        except Exception as e:
            log.error("forget_by_time_range failed: %s", e)
            return 0

    def forget_by_source(self, user_id: int, source_tag: str) -> int:
        """Tombstone every live edge with matching source_tag."""
        from uuid import uuid4
        if source_tag is None:
            return 0
        op_id = uuid4().hex
        reason = f"by_source:{source_tag}"
        try:
            with get_db_context() as conn:
                # source_tag is on edge_extraction, need a JOIN
                rows = conn.execute(
                    """SELECT e.id FROM edges e
                       JOIN edge_extraction ex ON ex.edge_id = e.id
                       WHERE e.user_id = ? AND e.tombstoned_at IS NULL
                         AND COALESCE(e.is_current, 1) = 1
                         AND ex.source_tag = ?""",
                    (user_id, source_tag),
                ).fetchall()
                ids = [r["id"] for r in rows]
                count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
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
        """Return live edges matching a facet view."""
        if facet not in ("time", "entity", "source", "trace"):
            raise ValueError(f"Unknown facet: {facet}")
        limit = max(1, min(int(limit or 100), 1000))

        base_cols = (
            "e.id, e.subject, e.predicate, e.object, "
            "e.source_timestamp, e.source_text"
        )
        where = [
            "e.user_id = ?", "e.tombstoned_at IS NULL",
            "COALESCE(e.is_current, 1) = 1",
        ]
        params: List[Any] = [user_id]
        order = "e.id DESC"
        join = ""

        if facet == "time":
            if value:
                where.append("e.source_timestamp LIKE ?")
                params.append(f"{value}%")
            order = (
                "CASE WHEN e.source_timestamp IS NULL THEN 1 ELSE 0 END, "
                "e.source_timestamp DESC, e.id DESC"
            )
        elif facet == "entity":
            if not value:
                return []
            where.append(
                "(LOWER(e.subject) = LOWER(?) OR LOWER(e.object) = LOWER(?))"
            )
            params.extend([value, value])
        elif facet == "source":
            if value is None:
                return []
            join = "JOIN edge_extraction ex ON ex.edge_id = e.id"
            where.append("ex.source_tag = ?")
            params.append(value)
        elif facet == "trace":
            if value:
                where.append("e.edge_schematic_category = ?")
                params.append(value)

        sql = (
            f"SELECT {base_cols} FROM edges e {join} "
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
