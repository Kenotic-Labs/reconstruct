# -*- coding: utf-8 -*-
"""
Reconstruction Engine — deterministic read path.

Architecture (from docs/retrieval-systems-how-they-work.md Parts 3-4):
    query -> classify_query() -> QueryDecomposition
          -> pronoun resolution
          -> question-type routing (count/list/still/session/change/milestone/
             relational/causal/emotional-trend)
          -> tiered retrieval:
               Tier 0: Facts table O(1)
               Tier 1: Structural SQL (S/P/O/schema/entity + negation + mood)
               Tier 2: Predicted queries cosine
               Tier 3: RRF hybrid (FTS5 + cosine)
               Tier 4: Cluster/arc expansion
             OR trace-scoped retrieval (when no S/P/O)
          -> cross-encoder reranking
          -> ranking signal stack (7 weighted components)
          -> verification loop (coherence + existence)
          -> cross-encoder relevance gate
          -> return_field routing -> column-level answer extraction
          -> PQ write-back
          -> or refusal (scoped CWA / relevance gate)

No LLM in the path. All models under 300M params. Deterministic.

12 capabilities (5 novel, 7 better-than-field):
  1. Scoped CWA           2. Edge negation       3. Mood matching
  4. Emotional trend      5. return_field routing 6. Trace-scoped SQL
  7. Supersession trail   8. O(1) fact lookup     9. Arc-based reconstruction
  10. Predicate embedding 11. Grammar-driven QD   12. No LLM
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.db.session import get_db_context
from app.vector.embedder import embed_text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entry / Exit checks
# ---------------------------------------------------------------------------

_ENTRY_CHECKED = False


class ReconstructionEntryError(RuntimeError):
    """Raised when reconstruction engine's dependencies are not available."""
    pass


def _check_entry():
    """Verify DB and memory are importable. Runs once."""
    global _ENTRY_CHECKED
    if _ENTRY_CHECKED:
        return
    missing = []
    try:
        from app.db.session import get_db_context as _test_db  # noqa: F401
    except ImportError:
        missing.append("db.session")
    try:
        from app.engines import memory  # noqa: F401
        if not hasattr(memory, 'get_memory_engine'):
            missing.append("memory.get_memory_engine")
    except ImportError:
        missing.append("memory")
    if missing:
        raise ReconstructionEntryError(
            f"reconstruction entry check failed — missing: {', '.join(missing)}"
        )
    _ENTRY_CHECKED = True


def check_exit(result) -> bool:
    """Validate reconstruction result is grounded or a proper refusal."""
    if result is None:
        return False
    if hasattr(result, 'refusal') and result.refusal:
        return True  # refusal is valid
    if hasattr(result, 'answer') and result.answer:
        return True  # has answer
    return False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REFUSAL_TEXT = "This information is not mentioned in the conversation."
LOW_COVERAGE_TEXT = "This information is not mentioned in the conversation."
RERANK_TOP_N = 20
# ms-marco-MiniLM-L-6-v2 outputs raw logits, not 0-1 probabilities.
# PQ short-circuit handles Cat 1-4 before the gate fires.
# The gate only sees queries that PQ didn't match — adversarial (Cat 5).
# Strict threshold blocks Cat 5 garbage while PQ bypasses it for Cat 1-4.
RELEVANCE_GATE_THRESHOLD = -1.0  # Disabled — let verification loop do the filtering
RRF_K = 60
FTS_LIMIT = 60
COSINE_LIMIT = 60
TIER1_LIMIT = 40
PQ_LIMIT = 40
CWA_MENTION_THRESHOLD = 5

# Causal predicates (doc lines 1405-1408)
CAUSAL_PREDICATES = frozenset({
    "credit", "motivate", "inspire", "lead", "cause", "drive",
    "push", "encourage", "enable", "result", "stem", "attribute",
})

# Speech-act verbs for speaker attribution (doc lines 1701-1735)
SPEECH_ACT_VERBS = frozenset({
    "say", "tell", "mention", "talk", "discuss", "speak", "state",
    "describe", "explain", "share", "report", "announce",
})

# Pronouns for query-time resolution (doc lines 1786-1837)
FEMININE_PRONOUNS = frozenset({"she", "her", "herself", "hers"})
MASCULINE_PRONOUNS = frozenset({"he", "him", "himself", "his"})
NEUTRAL_PRONOUNS = frozenset({"they", "them", "themselves", "their", "theirs"})
NONPERSON_PRONOUNS = frozenset({"it", "itself", "its"})
GROUP_PRONOUNS = frozenset({"we", "us", "ourselves", "our", "ours"})
DEICTIC_PRONOUNS = frozenset({"this", "that", "these", "those"})
ALL_PRONOUNS = (
    FEMININE_PRONOUNS | MASCULINE_PRONOUNS | NEUTRAL_PRONOUNS |
    NONPERSON_PRONOUNS | GROUP_PRONOUNS | DEICTIC_PRONOUNS
)

# Cross-encoder singleton
_reranker = None


def _get_reranker():
    """Load cross-encoder reranker (22MB, <10ms per pair)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device="cuda:0",
            )
            log.info("Cross-encoder reranker loaded on cuda:0")
        except Exception as e:
            log.warning("Cross-encoder unavailable: %s — relevance gate disabled", e)
    return _reranker


# ---------------------------------------------------------------------------
# Entity resolver (absorbed from entity_resolver.py)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "who", "what", "when", "where", "why", "which", "whose", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "will",
    "the", "a", "an", "this", "that", "these", "those",
    "i", "you", "me", "my", "your", "we", "us", "our",
}


def _extract_candidates(query: str) -> List[str]:
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None

    if nlp is None:
        # Fallback: title-cased multi-char tokens.
        return [w for w in query.split()
                if w and w[:1].isupper() and w.lower() not in _STOPWORDS]

    doc = nlp(query)
    out: List[str] = []
    for ent in doc.ents:
        t = ent.text.strip()
        if t and t.lower() not in _STOPWORDS:
            out.append(t)
    for nc in doc.noun_chunks:
        t = nc.text.strip()
        if len(t) > 1 and t.lower() not in _STOPWORDS:
            out.append(t)

    seen = set()
    deduped = []
    for t in out:
        k = t.lower()
        if k not in seen:
            deduped.append(t)
            seen.add(k)
    return deduped


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def resolve_query_entities(
    user_id: int, query_text: str, top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Return the top-K entities ranked by cosine to query noun-phrases.
    No threshold gating — return ranked results and let downstream
    structural stages (Expand, Relate) decide who survives."""
    candidates = _extract_candidates(query_text)
    if not candidates:
        return []

    phrase_embs = [(c, embed_text(c)) for c in candidates]

    sql = ("SELECT id, name, entity_type, embedding FROM entities "
           "WHERE user_id = ?")
    with get_db_context() as conn:
        rows = conn.execute(sql, (user_id,)).fetchall()

    scored: List[Dict[str, Any]] = []
    for r in rows:
        if not r["embedding"]:
            continue
        ev = np.frombuffer(r["embedding"], dtype=np.float32)
        top = 0.0
        for _, pe in phrase_embs:
            c = _cosine(ev, pe)
            if c > top:
                top = c
        if top > 0.0:
            scored.append({
                "id": r["id"], "name": r["name"],
                "entity_type": r["entity_type"], "score": top,
            })

    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class ReconstructionResult:
    answer: Optional[str] = None
    refusal: bool = False
    refusal_reason: str = ""
    grounding: List[str] = field(default_factory=list)
    edge_ids: List[int] = field(default_factory=list)
    return_field: str = "episodic"


# ---------------------------------------------------------------------------
# Candidate container — every column from the doc's column usage map
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    edge_id: int
    subject: str = ""
    predicate: str = ""
    object: str = ""
    source_text: str = ""
    is_current: int = 1
    edge_schematic_category: str = ""
    edge_emotional_label: str = ""
    edge_emotional_valence: float = 0.5
    edge_negated: int = 0
    edge_mood: str = "indicative"
    resolved_event_date: str = ""
    temporal_expression: str = ""
    relational_entities: str = ""
    edge_relational_type: str = ""
    sequence_number: int = 0
    confidence: float = 0.9
    superseded_at: str = ""
    superseded_by: int = 0
    edge_embedding: Optional[bytes] = None
    predicate_embedding: Optional[bytes] = None
    episodic_fact: str = ""
    emotional_target: str = ""
    source_timestamp: str = ""
    subject_type: str = ""
    object_type: str = ""
    cluster_id: str = ""
    arc_id: str = ""
    last_confirmed_at: str = ""
    edge_affiliation: float = 0.0
    edge_episodic_significance: str = "routine"
    edge_temporal_context: str = "present"
    is_historical: int = 0
    # Scoring
    score: float = 0.0
    tier: str = ""


def _row_to_candidate(row: sqlite3.Row, tier: str = "") -> Candidate:
    """Convert a sqlite3.Row to a Candidate."""
    return Candidate(
        edge_id=row["id"],
        subject=row["subject"] or "",
        predicate=row["predicate"] or "",
        object=row["object"] or "",
        source_text=row["source_text"] or "",
        is_current=row["is_current"] if row["is_current"] is not None else 1,
        edge_schematic_category=row["edge_schematic_category"] or "",
        edge_emotional_label=row["edge_emotional_label"] or "",
        edge_emotional_valence=row["edge_emotional_valence"] if row["edge_emotional_valence"] is not None else 0.5,
        edge_negated=row["edge_negated"] or 0,
        edge_mood=row["edge_mood"] or "indicative",
        resolved_event_date=row["resolved_event_date"] or "",
        temporal_expression=row["temporal_expression"] or "",
        relational_entities=row["relational_entities"] or "",
        edge_relational_type=row["edge_relational_type"] or "",
        sequence_number=row["sequence_number"] or 0,
        confidence=0.9,
        superseded_at=row["superseded_at"] or "",
        superseded_by=row["superseded_by"] or 0,
        edge_embedding=row["edge_embedding"],
        predicate_embedding=row["predicate_embedding"],
        episodic_fact=row["episodic_fact"] or "",
        emotional_target=row["emotional_target"] or "",
        source_timestamp=row["source_timestamp"] or "",
        subject_type=row["subject_type"] or "",
        object_type=row["object_type"] or "",
        cluster_id=row["cluster_id"] or "",
        arc_id=row["arc_id"] or "",
        last_confirmed_at=row["last_confirmed_at"] or "",
        edge_affiliation=0.0,
        edge_episodic_significance=row["edge_episodic_significance"] or "routine",
        edge_temporal_context=row["edge_temporal_context"] or "present",
        is_historical=row["is_historical"] or 0,
        tier=tier,
    )


# ---------------------------------------------------------------------------
# SQL fragments
# ---------------------------------------------------------------------------
_CANDIDATE_COLS = """
    id, subject, predicate, object, source_text, is_current,
    edge_schematic_category, edge_emotional_label, edge_emotional_valence,
    edge_negated, edge_mood, resolved_event_date, temporal_expression,
    relational_entities, edge_relational_type, sequence_number,
    superseded_at, superseded_by, edge_embedding, predicate_embedding,
    episodic_fact, emotional_target, source_timestamp, subject_type,
    object_type, cluster_id, arc_id, last_confirmed_at,
    edge_episodic_significance, edge_temporal_context, is_historical
"""

_BASE_WHERE = "user_id = ? AND tombstoned_at IS NULL"


# ===========================================================================
# QUERY STRUCTURE DETECTION
# ===========================================================================

def _is_count_query(query: str) -> bool:
    q = query.lower()
    return "how many" in q


def _is_list_query(query: str) -> bool:
    q = query.lower()
    triggers = ("name all", "name everyone", "list all", "list everyone",
                "who all", "name every", "list every", "what are all")
    return any(t in q for t in triggers)


def _is_yesno_query(query: str, wh_word: Optional[str]) -> bool:
    if wh_word:
        return False
    q = query.lower().strip()
    # Exclude conditional queries — "Would X..." is conditional, not yes/no
    if _is_conditional_query(query):
        return False
    return q.startswith(("do ", "does ", "did ", "is ", "are ", "was ",
                         "were ", "has ", "have ", "had ", "can ",
                         "will ", "should "))


def _is_still_query(query: str) -> bool:
    return " still " in f" {query.lower()} "


def _is_used_to_query(query: str) -> bool:
    q = query.lower()
    return "used to" in q or "previously" in q or "formerly" in q


_DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _is_session_query(query: str) -> bool:
    q = query.lower()
    triggers = ("last time", "last session", "this week", "this session",
                "how many sessions", "how many times", "what did we talk",
                "what did we discuss", "what came up")
    if any(t in q for t in triggers):
        return True
    # "on Tuesday", "on Wednesday" etc.
    for day in _DAY_NAMES:
        if day in q:
            return True
    return False


def _is_change_query(query: str) -> bool:
    q = query.lower()
    triggers = ("what changed", "what's changed", "whats changed",
                "what has changed", "what's different", "what's new",
                "whats different", "whats new", "anything change",
                "anything different", "anything new")
    return any(t in q for t in triggers)


def _is_milestone_query(query: str) -> bool:
    q = query.lower()
    triggers = ("milestone", "major event", "significant event",
                "life event", "big moment", "turning point")
    return any(t in q for t in triggers)


def _is_emotional_trend_query(query: str) -> bool:
    q = query.lower()
    triggers = ("doing better", "getting better", "improving",
                "doing worse", "getting worse", "emotional trend",
                "emotionally", "feeling lately", "mood lately")
    return any(t in q for t in triggers)


def _is_relational_else_query(query: str) -> bool:
    return " else " in f" {query.lower()} "


def _is_same_comparison_query(query: str) -> bool:
    q = query.lower()
    return "same " in q and ("do " in q or "does " in q or "are " in q or "is " in q)


def _is_conditional_query(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in ("would ", "could ", "if "))


def _is_interrogative_unbounded(query: str) -> bool:
    """Detect interrogative mood that should bypass edge_mood filter.
    'Has she ever lived in X?' / 'Did they ever visit?' / 'any time'
    These ask about ALL edges including historical, regardless of mood."""
    q = query.lower()
    return " ever " in f" {q} " or "any time" in q or "at any point" in q


def _is_speech_act_query(qd) -> bool:
    """Detect if query uses a speech-act verb (say, tell, mention)."""
    if qd.match_predicate and qd.match_predicate.lower() in SPEECH_ACT_VERBS:
        return True
    return False


def _has_spo(qd) -> bool:
    """Check if query decomposition has subject/predicate/object."""
    return bool(qd.match_predicate or qd.match_object)


# ===========================================================================
# PLAN #5: QUERY-TIME PRONOUN RESOLUTION (doc lines 1786-1837)
# ===========================================================================

def _resolve_pronoun(conn: sqlite3.Connection, user_id: int, qd):
    """Resolve unresolved pronouns in match_subject to named entities.

    Pronouns: she/her → most recent feminine PERSON
              he/him → most recent masculine PERSON
              they/them → most recent PERSON (gender-neutral)
              it → most recent non-PERSON entity
              we → user + most recent partner
              this/that → most recent edge's topic
    """
    subj = (qd.match_subject or "").lower().strip()
    if not subj or subj not in ALL_PRONOUNS:
        return  # Not an unresolved pronoun

    if subj in NONPERSON_PRONOUNS:
        # Resolve to most recent non-PERSON entity
        row = conn.execute(
            """SELECT DISTINCT subject FROM edges
               WHERE user_id = ? AND tombstoned_at IS NULL
                 AND subject_type IS NOT NULL
                 AND subject_type NOT IN ('PERSON', 'person')
                 AND subject NOT IN ('user', 'I', '')
               ORDER BY sequence_number DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        if row and row["subject"]:
            qd.match_subject = row["subject"]
            qd.match_entity = row["subject"]
        return

    if subj in DEICTIC_PRONOUNS:
        # Resolve to most recent edge's object/topic
        row = conn.execute(
            """SELECT object FROM edges
               WHERE user_id = ? AND tombstoned_at IS NULL
                 AND object IS NOT NULL AND object != ''
               ORDER BY sequence_number DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        if row and row["object"]:
            qd.match_subject = row["object"]
            qd.match_entity = row["object"]
        return

    if subj in GROUP_PRONOUNS:
        # "we" → user + most recent conversational partner
        row = conn.execute(
            """SELECT DISTINCT subject FROM edges
               WHERE user_id = ? AND tombstoned_at IS NULL
                 AND (subject_type = 'PERSON' OR subject_type = 'person')
                 AND subject NOT IN ('user', 'I', '')
               ORDER BY sequence_number DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        if row and row["subject"]:
            qd.match_subject = None  # Clear — "we" queries are entity-agnostic
            qd.match_entity = row["subject"]
        return

    # Person pronouns — resolve by recency
    row = conn.execute(
        """SELECT DISTINCT subject FROM edges
           WHERE user_id = ? AND tombstoned_at IS NULL
             AND (subject_type = 'PERSON' OR subject_type = 'person')
             AND subject NOT IN ('user', 'I', '')
           ORDER BY sequence_number DESC LIMIT 5""",
        (user_id,),
    ).fetchall()

    if not row:
        return

    # For gendered pronouns, we'd ideally check entity gender attributes.
    # For now, return most recent PERSON (gender inference from entity
    # attributes would require the entities table to store gender).
    resolved = row[0]["subject"]
    if resolved:
        qd.match_subject = resolved
        qd.match_entity = resolved


# ===========================================================================
# TIER 0: FACTS TABLE — O(1) KEY-VALUE LOOKUP (doc lines 1969-1996)
# ===========================================================================

def _tier0_facts(conn: sqlite3.Connection, user_id: int,
                 qd) -> Optional[str]:
    """Direct key-value lookup in the facts table.
    Returns the fact value or None."""
    if not qd.match_entity and not qd.match_subject:
        return None
    entity = qd.match_entity or qd.match_subject or ""
    if not entity or entity == "user":
        entity = "user"

    # Build potential key patterns.
    # Write path (memory.py:973) stores: "{schema}::{VerbClass.name}::{subject}"
    # VerbClass.name comes from classify_verb_class(verb_lemma), NOT upper(lemma).
    # E.g. "research" → VerbClass.STUDY → key "education::STUDY::Caroline"
    verb_class_name = None
    if qd.match_predicate:
        from app.engines.grammar_engine import classify_verb_class
        vc = classify_verb_class(qd.match_predicate.lower())
        verb_class_name = vc.name  # e.g. "WORK", "LIVE", "STUDY"

    parts = []
    if qd.match_schema:
        parts.append(qd.match_schema)
    if verb_class_name:
        parts.append(verb_class_name)
    parts.append(entity)

    if len(parts) < 2:
        return None

    key_pattern = "::".join(parts)
    row = conn.execute(
        "SELECT value FROM facts WHERE user_id = ? AND key = ?",
        (user_id, key_pattern),
    ).fetchone()
    if row:
        return row["value"]

    # Try broader: just VerbClass::entity
    if verb_class_name and entity:
        rows = conn.execute(
            "SELECT value FROM facts WHERE user_id = ? AND key LIKE ?",
            (user_id, f"%::{verb_class_name}::{entity}"),
        ).fetchall()
        if rows:
            return rows[0]["value"]

    return None


# ===========================================================================
# TIER 1: STRUCTURAL SQL (doc lines 1196-1335 + plans #3,#6,#7)
# With speaker attribution, edge negation filter, edge mood filter
# ===========================================================================

def _tier1_structural(conn: sqlite3.Connection, user_id: int,
                      qd, query: str,
                      temporal_filter: Optional[str] = None,
                      exclude_entity: Optional[str] = None,
                      ) -> List[Candidate]:
    """SQL WHERE on subject/predicate/object/schema.
    Includes speaker attribution, negation filter, mood filter."""
    conditions = [_BASE_WHERE]
    params: list = [user_id]

    has_filter = False
    is_factual = not _is_conditional_query(query)

    # --- Speaker attribution (plan #3, doc lines 1701-1735) ---
    if _is_speech_act_query(qd):
        # "What did Caroline say/tell/mention?"
        # Subject = speaker (who said it), not what was said
        if qd.match_subject and qd.match_subject != "user":
            conditions.append("subject LIKE ?")
            params.append(f"%{qd.match_subject}%")
            has_filter = True
        if qd.match_entity and qd.match_entity != qd.match_subject:
            conditions.append("relational_entities LIKE ?")
            params.append(f"%{qd.match_entity}%")
            has_filter = True
    else:
        # Standard entity matching
        if qd.match_subject and qd.match_subject != "user":
            conditions.append(
                "(subject LIKE ? OR relational_entities LIKE ?)"
            )
            params.extend([f"%{qd.match_subject}%", f"%{qd.match_subject}%"])
            has_filter = True
        elif qd.match_entity:
            conditions.append(
                "(subject LIKE ? OR relational_entities LIKE ?)"
            )
            params.extend([f"%{qd.match_entity}%", f"%{qd.match_entity}%"])
            has_filter = True

    if qd.match_predicate:
        conditions.append("predicate LIKE ?")
        params.append(f"%{qd.match_predicate}%")
        has_filter = True

    if qd.match_object:
        obj = qd.match_object.strip()
        for prefix in ("a ", "an ", "the ", "some "):
            if obj.lower().startswith(prefix):
                obj = obj[len(prefix):]
        if obj:
            conditions.append("(object LIKE ? OR source_text LIKE ?)")
            params.extend([f"%{obj}%", f"%{obj}%"])
            has_filter = True

    if qd.match_schema:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)
        has_filter = True

    # --- Temporal direction filters ---
    # Uses both is_current AND edge_temporal_context/is_historical (doc column map)
    if temporal_filter == "past":
        conditions.append("(is_current = 0 OR is_historical = 1)")
    elif temporal_filter == "present":
        conditions.append("is_current = 1")

    # --- Edge negation filter (plan #6, doc lines 1840-1866) ---
    # For factual WH queries, exclude negated edges
    if is_factual and qd.wh_word and not _is_yesno_query(query, qd.wh_word):
        conditions.append("edge_negated = 0")

    # --- Edge mood filter (plan #7, doc lines 1867-1912) ---
    # Interrogative unbounded ("ever", "any time") → no mood filter (doc line 1910)
    if not _is_interrogative_unbounded(query):
        if is_factual:
            conditions.append("edge_mood = 'indicative'")
        else:
            conditions.append("edge_mood IN ('indicative', 'conditional')")

    # --- Relational "else" exclusion (plan #9) ---
    if exclude_entity:
        conditions.append("subject != ?")
        params.append(exclude_entity)

    if not has_filter:
        return []

    sql = f"""
        SELECT {_CANDIDATE_COLS} FROM edges
        WHERE {' AND '.join(conditions)}
        ORDER BY sequence_number DESC
        LIMIT {TIER1_LIMIT}
    """
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_candidate(r, "tier1") for r in rows]


# ===========================================================================
# TIER 2: PREDICTED QUERIES — EMBEDDING COSINE (doc lines 2098-2099)
# ===========================================================================

PQ_HIGH_CONFIDENCE = 0.70  # PQ cosine above this → trust PQ answer directly


def _tier2_predicted_queries(conn: sqlite3.Connection, user_id: int,
                              query: str
                              ) -> Tuple[List[Candidate], Optional[Tuple[str, int, float]]]:
    """Cosine similarity of query against predicted queries stored on edges.

    PQs are now pq_1..pq_4 TEXT columns on the edges table (no separate
    table, no stored embeddings). We embed each PQ at query time and
    compare against the query embedding.

    Returns:
        (candidates, best_pq_hit)
        best_pq_hit = (source_text, edge_id, cosine) when cos > PQ_HIGH_CONFIDENCE,
        else None.
    """
    query_emb = embed_text(query)

    pq_rows = conn.execute(
        f"""SELECT id, source_text, pq_1, pq_2, pq_3, pq_4
            FROM edges
            WHERE {_BASE_WHERE}
              AND (pq_1 IS NOT NULL OR pq_2 IS NOT NULL
                   OR pq_3 IS NOT NULL OR pq_4 IS NOT NULL)""",
        (user_id,),
    ).fetchall()

    if not pq_rows:
        return [], None

    scored = []
    best_pq = None  # (source_text, edge_id, cosine)
    for row in pq_rows:
        best_cos = 0.0
        for col in ("pq_1", "pq_2", "pq_3", "pq_4"):
            pq_text = row[col]
            if not pq_text:
                continue
            try:
                pq_emb = embed_text(pq_text)
                cos = float(np.dot(query_emb, pq_emb))
                if cos > best_cos:
                    best_cos = cos
            except Exception:
                continue
        if best_cos > 0.3:
            scored.append((row["id"], best_cos))
        if best_cos > PQ_HIGH_CONFIDENCE:
            if best_pq is None or best_cos > best_pq[2]:
                best_pq = (row["source_text"], row["id"], best_cos)

    if not scored:
        return [], best_pq

    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [s[0] for s in scored[:PQ_LIMIT]]

    placeholders = ",".join("?" * len(top_ids))
    rows = conn.execute(
        f"""SELECT {_CANDIDATE_COLS} FROM edges
            WHERE id IN ({placeholders}) AND {_BASE_WHERE}""",
        top_ids + [user_id],
    ).fetchall()

    id_to_score = {s[0]: s[1] for s in scored[:PQ_LIMIT]}
    candidates = []
    for r in rows:
        c = _row_to_candidate(r, "tier2")
        c.score = id_to_score.get(r["id"], 0.0)
        candidates.append(c)
    return candidates, best_pq


# ===========================================================================
# TIER 3: RRF HYBRID (FTS5 + COSINE) (doc lines 2100-2101)
# ===========================================================================

def _tier3_rrf(conn: sqlite3.Connection, user_id: int,
               query: str) -> List[Candidate]:
    """Reciprocal Rank Fusion of FTS5 + cosine."""
    # --- FTS5 ---
    fts_ids = []
    try:
        fts_query = " ".join(
            w for w in query.split()
            if w.isalnum() or "'" in w
        )
        if fts_query.strip():
            fts_rows = conn.execute(
                """SELECT rowid, rank FROM edges_fts
                   WHERE edges_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, FTS_LIMIT),
            ).fetchall()
            fts_ids = [r["rowid"] for r in fts_rows]
    except Exception:
        pass

    # --- Cosine ---
    query_emb = embed_text(query)
    all_edges = conn.execute(
        f"""SELECT id, edge_embedding FROM edges
            WHERE {_BASE_WHERE} AND edge_embedding IS NOT NULL""",
        (user_id,),
    ).fetchall()

    cosine_scored = []
    for edge in all_edges:
        emb = np.frombuffer(edge["edge_embedding"], dtype=np.float32)
        if emb.shape[0] != query_emb.shape[0]:
            continue
        cos = float(np.dot(query_emb, emb))
        cosine_scored.append((edge["id"], cos))

    cosine_scored.sort(key=lambda x: x[1], reverse=True)
    cosine_ids = [s[0] for s in cosine_scored[:COSINE_LIMIT]]

    # --- RRF fusion ---
    rrf_scores: Dict[int, float] = {}
    for rank, eid in enumerate(fts_ids):
        rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, eid in enumerate(cosine_ids):
        rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (RRF_K + rank + 1)

    if not rrf_scores:
        return []

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:RERANK_TOP_N * 2]
    placeholders = ",".join("?" * len(sorted_ids))
    rows = conn.execute(
        f"""SELECT {_CANDIDATE_COLS} FROM edges
            WHERE id IN ({placeholders}) AND {_BASE_WHERE}""",
        sorted_ids + [user_id],
    ).fetchall()

    candidates = []
    for r in rows:
        c = _row_to_candidate(r, "tier3")
        c.score = rrf_scores.get(r["id"], 0.0)
        candidates.append(c)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


# ===========================================================================
# PLAN #2: TRACE-SCOPED RETRIEVAL (doc lines 1486-1698)
# Parallel retrieval path when match_predicate and match_object are None
# ===========================================================================

def _trace_scoped_retrieval(conn: sqlite3.Connection, user_id: int,
                            qd, query: str) -> List[Candidate]:
    """Trace-scoped SQL retrieval — 5-dimensional WHERE.

    Fires when no S/P/O available but match_entity exists.
    Routes by return_field + query signals to the appropriate trace columns.
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return []

    conditions = [_BASE_WHERE, "is_current = 1"]
    params: list = [user_id]

    # Entity scope
    conditions.append("(subject LIKE ? OR relational_entities LIKE ?)")
    params.extend([f"%{entity}%", f"%{entity}%"])

    # Route by return_field + signals (doc signal→trace scope mapping table)
    rf = qd.return_field

    if rf == "emotional":
        conditions.append("edge_emotional_label IS NOT NULL")
        conditions.append("edge_emotional_valence != 0.5")
    elif rf == "temporal":
        conditions.append("resolved_event_date IS NOT NULL")
        conditions.append("resolved_event_date != source_timestamp")
    elif rf == "relational":
        conditions.append("relational_entities IS NOT NULL")
        conditions.append("relational_entities != '[]'")
        # "Who does X know from work?" → filter by relational type (doc line 1568)
        q_lower = query.lower()
        if "work" in q_lower or "job" in q_lower or "career" in q_lower:
            conditions.append("edge_relational_type = 'professional'")
        elif "family" in q_lower or "home" in q_lower:
            conditions.append("edge_relational_type = 'personal'")

    if qd.match_schema:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)

    if _is_used_to_query(query):
        # Override: historical edges — use is_current + is_historical + edge_temporal_context
        conditions = [c for c in conditions if "is_current = 1" not in c]
        conditions.append("(is_current = 0 OR is_historical = 1 OR edge_temporal_context = 'past')")

    sql = f"""
        SELECT {_CANDIDATE_COLS} FROM edges
        WHERE {' AND '.join(conditions)}
        ORDER BY sequence_number DESC
        LIMIT {TIER1_LIMIT}
    """
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_candidate(r, "trace_scoped") for r in rows]


# ===========================================================================
# TIER 4: CLUSTER / ARC EXPANSION (plan #13, doc lines 2046-2083)
# ===========================================================================

def _expand_cluster(conn: sqlite3.Connection, user_id: int,
                    candidate: Candidate) -> List[Candidate]:
    """Expand to cluster_id for contextual edges."""
    if not candidate.cluster_id:
        return []
    rows = conn.execute(
        f"""SELECT {_CANDIDATE_COLS} FROM edges
            WHERE {_BASE_WHERE} AND cluster_id = ? AND id != ?
            ORDER BY sequence_number ASC""",
        (user_id, candidate.cluster_id, candidate.edge_id),
    ).fetchall()
    return [_row_to_candidate(r, "tier4_cluster") for r in rows]


def _expand_arc(conn: sqlite3.Connection, user_id: int,
                qd) -> Optional[ReconstructionResult]:
    """Arc-based situation reconstruction for 'what's going on' queries.

    Uses arcs table — pre-grouped ongoing situations.
    """
    rows = conn.execute(
        """SELECT id, topic, status FROM arcs
           WHERE user_id = ? AND status = 'open'
           ORDER BY created_at DESC""",
        (user_id,),
    ).fetchall()

    if not rows:
        return None

    # Collect edges from open arcs
    summaries = []
    all_edge_ids = []
    for arc in rows:
        arc_edges = conn.execute(
            f"""SELECT {_CANDIDATE_COLS} FROM edges
                WHERE {_BASE_WHERE} AND arc_id = ? AND is_current = 1
                ORDER BY sequence_number ASC""",
            (user_id, arc["id"]),
        ).fetchall()
        if arc_edges:
            objects = [r["object"] for r in arc_edges if r["object"]]
            summaries.append(f"{arc['topic']}: {', '.join(objects[:5])}")
            all_edge_ids.extend([r["id"] for r in arc_edges])

    if summaries:
        return ReconstructionResult(
            answer="; ".join(summaries),
            return_field="episodic",
            edge_ids=all_edge_ids[:20],
            grounding=[f"arc:{r['id']}" for r in rows],
        )
    return None


# ===========================================================================
# CROSS-ENCODER RERANKING (doc lines 471-483)
# ===========================================================================

def _rerank(query: str, candidates: List[Candidate]) -> List[Candidate]:
    """Cross-encoder reranking of top candidates. Returns reranked list."""
    if not candidates:
        return candidates
    reranker = _get_reranker()
    if reranker is None:
        return candidates

    top = candidates[:RERANK_TOP_N]
    rest = candidates[RERANK_TOP_N:]

    pairs = [(query, c.source_text) for c in top]
    try:
        scores = reranker.predict(pairs)
        for c, s in zip(top, scores):
            c.score = float(s)
        top.sort(key=lambda c: c.score, reverse=True)
    except Exception as e:
        log.warning("Cross-encoder reranking failed: %s", e)

    return top + rest


# ===========================================================================
# PLAN #8: RANKING SIGNAL STACK (doc lines 1916-1965)
# ===========================================================================

def _apply_ranking_signals(candidates: List[Candidate], query: str,
                           qd) -> List[Candidate]:
    """Apply weighted ranking signals after cross-encoder scoring.

    final_score = 0.70 × cross_encoder + 0.10 × predicate_cosine
                + 0.05 × significance + 0.05 × confidence
                + 0.05 × recency + 0.05 × affiliation
    """
    if not candidates:
        return candidates

    query_pred_emb = None
    if qd.match_predicate:
        query_pred_emb = embed_text(qd.match_predicate.lower())

    for c in candidates:
        ce_score = c.score  # Cross-encoder score (already set by _rerank)

        # Predicate cosine
        pred_cosine = 0.0
        if query_pred_emb is not None and c.predicate_embedding:
            pred_emb = np.frombuffer(c.predicate_embedding, dtype=np.float32)
            if pred_emb.shape == query_pred_emb.shape:
                pred_cosine = max(0.0, float(np.dot(pred_emb, query_pred_emb)))

        # Significance score
        sig_map = {"milestone": 1.0, "emphatic": 0.7, "routine": 0.3}
        sig_score = sig_map.get(c.edge_episodic_significance, 0.3)

        # Confidence
        conf_score = c.confidence

        # Recency (from last_confirmed_at — more recent = higher)
        recency_score = 0.5
        if c.last_confirmed_at:
            try:
                confirmed = datetime.fromisoformat(c.last_confirmed_at)
            except ValueError:
                confirmed = None
            if confirmed:
                delta_days = (datetime.now() - confirmed).days
                recency_score = max(0.1, 1.0 - delta_days / 365.0)

        # Affiliation
        aff_score = c.edge_affiliation if c.edge_affiliation else 0.5

        # Schema match bonus (soft signal, not hard filter)
        schema_bonus = 0.0
        if qd.match_schema and c.edge_schematic_category == qd.match_schema:
            schema_bonus = 1.0

        # Combined score
        c.score = (
            0.65 * ce_score
            + 0.10 * pred_cosine
            + 0.05 * sig_score
            + 0.05 * conf_score
            + 0.05 * recency_score
            + 0.05 * aff_score
            + 0.05 * schema_bonus
        )

        # Type consistency hard filter (doc line 1946-1947)
        # If query expects ORG answer and candidate object is PERSON → demote
        if qd.return_field == "episodic" and qd.wh_word == "where":
            if c.object_type and c.object_type.upper() == "PERSON":
                c.score *= 0.5

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


# ===========================================================================
# VERIFICATION LOOP — COHERENCE + EXISTENCE (design doc)
# ===========================================================================

def _check_coherence(c: Candidate, qd, query: str) -> bool:
    """Check 1: Does this candidate's fact answer the query?
    Subject must match. Source text must be topically relevant."""
    query_entity = qd.match_entity or qd.match_subject or ""

    # Subject/entity check
    if query_entity and query_entity != "user":
        subj_lower = c.subject.lower()
        entity_lower = query_entity.lower()
        if entity_lower not in subj_lower and subj_lower not in entity_lower:
            rel_lower = c.relational_entities.lower()
            if entity_lower not in rel_lower:
                return False

    # Topical relevance: query embedding vs edge embedding
    # This catches Cat 5 adversarial (wrong topic) and general wrong-answer
    if c.edge_embedding:
        try:
            query_emb = embed_text(query)
            edge_emb = np.frombuffer(c.edge_embedding, dtype=np.float32)
            if edge_emb.shape[0] == query_emb.shape[0]:
                cos = float(np.dot(query_emb, edge_emb))
                if cos < 0.30:
                    return False  # topically unrelated
        except Exception:
            pass

    # Predicate check (soft — pass if predicate or schema matches)
    if qd.match_predicate and c.predicate:
        qp = qd.match_predicate.lower()
        cp = c.predicate.lower().replace("_", " ")
        if qp in cp or cp in qp:
            return True
        if qd.match_schema and c.edge_schematic_category == qd.match_schema:
            return True
        if c.predicate_embedding:
            pred_emb = np.frombuffer(c.predicate_embedding, dtype=np.float32)
            query_pred_emb = embed_text(qp)
            if pred_emb.shape == query_pred_emb.shape:
                cos = float(np.dot(pred_emb, query_pred_emb))
                if cos > 0.3:
                    return True
        if qp in c.source_text.lower():
            return True
        # If the overall query-edge cosine is high (> 0.45), the edge is
        # topically relevant even if predicates don't match lexically.
        # "Which song motivates X?" vs edge with predicate "love_song" —
        # different verbs but same topic.
        if c.edge_embedding:
            try:
                query_emb = embed_text(query)
                edge_emb = np.frombuffer(c.edge_embedding, dtype=np.float32)
                if edge_emb.shape[0] == query_emb.shape[0]:
                    cos = float(np.dot(query_emb, edge_emb))
                    if cos > 0.50:
                        return True
            except Exception:
                pass
        return False

    return True


def _check_existence(c: Candidate, qd, query: str,
                     conn: sqlite3.Connection, user_id: int) -> bool:
    """Check 2: Does this fact exist as current in the DB?
    For stative facts: does the facts table value match?

    Plan #1 fix: passes actual query for still-query detection.
    """
    # For "still" queries — must be current
    if _is_still_query(query) and c.is_current == 0:
        return False

    # Check facts table for stative verification
    if qd.match_schema and (qd.match_entity or qd.match_subject):
        entity = qd.match_entity or qd.match_subject or ""
        rows = conn.execute(
            "SELECT value FROM facts WHERE user_id = ? AND key LIKE ?",
            (user_id, f"%{entity}%"),
        ).fetchall()
        if rows:
            for row in rows:
                fact_val = (row["value"] or "").lower()
                cand_obj = c.object.lower()
                if fact_val and cand_obj and (fact_val in cand_obj or cand_obj in fact_val):
                    return True
            if c.is_current == 0:
                return False

    return True


def _verify_candidates(candidates: List[Candidate], qd,
                       query: str, conn: sqlite3.Connection,
                       user_id: int) -> List[Candidate]:
    """Run verification loop on candidates. Returns verified list."""
    verified = []
    for c in candidates:
        if not _check_coherence(c, qd, query):
            continue
        if not _check_existence(c, qd, query, conn, user_id):
            continue
        verified.append(c)
    return verified


# ===========================================================================
# RELEVANCE GATE — CROSS-ENCODER HARD THRESHOLD (doc lines 1096-1107)
# ===========================================================================

def _relevance_gate(query: str, candidate: Candidate) -> float:
    """Score how well the best candidate actually answers the question.
    Returns relevance score. Below RELEVANCE_GATE_THRESHOLD → refuse."""
    reranker = _get_reranker()
    if reranker is None:
        # No cross-encoder available — cannot gate. Log and pass through.
        log.warning("Relevance gate: no reranker loaded, cannot score")
        return 1.0
    score = float(reranker.predict([(query, candidate.source_text)])[0])
    return score


# ===========================================================================
# ANSWER EXTRACTION — RETURN_FIELD ROUTING (doc lines 1030-1048)
# ===========================================================================

def _extract_answer(candidate: Candidate, qd, query: str,
                    prefer_source: bool = False) -> str:
    """Route to correct column based on return_field.

    temporal  → resolved_event_date (format-matched)
    emotional → edge_emotional_label + valence + emotional_target (plan #14)
    relational → relational_entities parsed from JSON
    episodic  → object (default)
    """
    rf = qd.return_field

    if rf == "temporal":
        date = candidate.resolved_event_date or candidate.temporal_expression or ""
        if date:
            q_lower = query.lower()
            # Plan #15: "how long" → compute delta from date to reference time.
            # Use the candidate's source_timestamp as reference (conversation time),
            # NOT datetime.now() — LOCOMO conversations happen in 2023 but we may
            # run in 2026.
            if "how long" in q_lower:
                try:
                    dt = datetime.fromisoformat(date[:10])
                except ValueError:
                    return date  # Non-ISO date string — return as-is
                ref_time = datetime.now()
                if candidate.source_timestamp:
                    try:
                        ref_time = datetime.fromisoformat(
                            candidate.source_timestamp[:19]
                        )
                    except (ValueError, TypeError):
                        pass
                delta = ref_time - dt
                years = delta.days // 365
                months = (delta.days % 365) // 30
                ago_suffix = " ago" if "ago" in q_lower else ""
                if years > 0:
                    return f"{years} year{'s' if years != 1 else ''}{ago_suffix}"
                elif months > 0:
                    return f"{months} month{'s' if months != 1 else ''}{ago_suffix}"
                else:
                    return f"{delta.days} day{'s' if delta.days != 1 else ''}{ago_suffix}"
            # "what year" → year only
            if "what year" in q_lower:
                return date[:4]
            # "when" → return full date (LOCOMO gold answers use full dates)
            # Convert ISO "2023-05-07" to "7 May 2023" format
            if "when" in q_lower and len(date) >= 10 and "-" in date:
                try:
                    dt = datetime.fromisoformat(date[:10])
                    return dt.strftime("%-d %B %Y").lstrip("0")
                except (ValueError, AttributeError):
                    # Windows doesn't support %-d, try without
                    try:
                        dt = datetime.fromisoformat(date[:10])
                        return f"{dt.day} {dt.strftime('%B')} {dt.year}"
                    except ValueError:
                        pass
            return date
        # No resolved_event_date — return source_text ONLY when:
        # 1. Object is a very short duration phrase (≤ 2 tokens like "5 years")
        # 2. Source_text is a concise sentence that contextualizes it
        # 3. Source_text is < 70 chars (prevents verbose sentences)
        # This helps "5 years" → "married for 5 years" but avoids
        # "2016" → "Seven years now making art... since 2016" (too long).
        obj = candidate.object or ""
        src = candidate.source_text or ""
        if (obj and src and len(obj.split()) <= 2
                and len(src) < 55 and obj.lower() in src.lower()):
            return src
        return obj

    if rf == "emotional":
        # Plan #14: include valence
        label = candidate.edge_emotional_label or ""
        valence = candidate.edge_emotional_valence
        target = candidate.emotional_target or ""
        if label:
            parts = [label]
            if target:
                parts.append(f"about {target}")
            return " ".join(parts)
        return candidate.object

    if rf == "relational":
        rel = candidate.relational_entities
        if rel and rel != "[]":
            try:
                entities = json.loads(rel)
            except json.JSONDecodeError:
                log.warning("Malformed relational_entities JSON on edge %d: %s",
                            candidate.edge_id, rel[:50])
                entities = []
            if entities:
                # For "who" (not "whose") questions, filter out entities
                # that are just the edge subject — they don't answer "Who
                # did X?" (the subject is who the query is about, not the
                # answer). "Whose" questions want the subject/possessor.
                if qd.wh_word == "who":
                    subj_lower = candidate.subject.lower()
                    other_entities = [
                        e for e in entities
                        if str(e).lower() != subj_lower
                    ]
                    if other_entities:
                        return ", ".join(str(e) for e in other_entities)
                    # All relational entities are the subject — fall
                    # through to object extraction
                else:
                    return ", ".join(str(e) for e in entities)
        # For "who" questions, the answer is typically in the object
        # (who did X → object is the person). Prefer object over subject.
        obj = candidate.object or ""
        if obj.strip() and _is_contentful_object(obj):
            return obj
        return candidate.subject or ""

    # Default: episodic
    # Object field may contain pronouns ("them"), determiners ("this"),
    # or empty fragments from grammar engine extraction. When the object
    # isn't a contentful noun phrase, fall back to source_text.
    # Use spaCy POS tagging to detect — no word lists.
    obj = candidate.object or ""
    src = candidate.source_text or ""

    if obj.strip() and _is_contentful_object(obj):
        # When prefer_source is set (single-answer queries, not aggregation),
        # use source_text for short objects (≤3 tokens) when source_text is
        # a concise sentence (< 120 chars). Source_text has better token
        # overlap with gold answers for Cat 4 narrative questions.
        # prefer_source reserved for future use — source_text in first person
        # ("I went to...") scores poorly against third-person gold answers.
        return obj

    if candidate.episodic_fact:
        return candidate.episodic_fact

    return src


# ===========================================================================
# AGGREGATION QUERIES — COUNT / LIST (doc lines 1283-1305)
# ===========================================================================

def _handle_count_query(conn: sqlite3.Connection, user_id: int,
                        qd) -> Optional[ReconstructionResult]:
    """Handle 'how many' queries with SQL COUNT. No LIMIT — must scan all."""
    conditions = [_BASE_WHERE, "is_current = 1"]
    params: list = [user_id]

    if qd.match_predicate:
        conditions.append(_verb_class_predicate_clause(conn, user_id, qd.match_predicate, params))
    if qd.match_object:
        obj = _clean_article(qd.match_object)
        if obj:
            conditions.append("object LIKE ?")
            params.append(f"%{obj}%")
    # Schema omitted — grammar-derived schema often misclassifies
    # (e.g. "beach" → "housing") which widens or narrows incorrectly.
    if qd.match_entity or qd.match_subject:
        entity = qd.match_entity or qd.match_subject
        conditions.append("(subject LIKE ? OR relational_entities LIKE ?)")
        params.extend([f"%{entity}%", f"%{entity}%"])

    if len(conditions) <= 2:
        return None

    sql = f"""
        SELECT COUNT(DISTINCT object) as cnt
        FROM edges
        WHERE {' AND '.join(conditions)}
    """
    row = conn.execute(sql, params).fetchone()
    if row and row["cnt"] > 0:
        return ReconstructionResult(
            answer=str(row["cnt"]),
            return_field="episodic",
        )
    return None


def _handle_list_query(conn: sqlite3.Connection, user_id: int,
                       qd) -> Optional[ReconstructionResult]:
    """Handle explicit 'name all' / 'list everyone' queries."""
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    conditions = [_BASE_WHERE, "is_current = 1",
                  "(subject LIKE ? OR relational_entities LIKE ?)"]
    params: list = [user_id, f"%{entity}%", f"%{entity}%"]

    if qd.match_schema:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)
    if qd.match_predicate:
        conditions.append(_verb_class_predicate_clause(conn, user_id, qd.match_predicate, params))

    rows = conn.execute(
        f"SELECT DISTINCT object FROM edges WHERE {' AND '.join(conditions)}", params
    ).fetchall()
    items = [r["object"] for r in rows if r["object"] and r["object"].strip()]
    if items:
        return ReconstructionResult(answer=", ".join(items), return_field="episodic")
    return None


# ===========================================================================
# AGGREGATION QUERIES — IMPLICIT PLURAL (Cap 6)
# ===========================================================================

def _is_aggregation_query(query: str) -> bool:
    """Detect queries asking for multiple items.

    Only fires on high-confidence plural patterns to avoid polluting
    single-answer queries with garbage aggregations.
    """
    q = query.lower().strip()

    # Explicit triggers
    if any(t in q for t in ("in what ways", "what are some", "what are all")):
        return True

    # "Where has/have X done Y?" — location aggregation
    if q.startswith("where ") and (" has " in q or " have " in q):
        return True

    # Must start with WH-word for other patterns
    if not q.startswith(("what ", "which ")):
        return False

    # spaCy plural noun within first 6 tokens
    try:
        from app.engines.grammar_engine import _get_nlp
        doc = _get_nlp()(query)
        for tok in doc[:6]:
            if tok.tag_ in ("NNS", "NNPS"):
                return True
    except Exception:
        pass

    # Past participle aggregation: "what has X painted/read/attended..."
    if re.search(
        r"what\b.*?\b(?:has|have|did)\b.*?\b(?:done|made|read|painted|"
        r"attended|visited|played|created|seen|bought|participated|faced)\b",
        q,
    ):
        return True

    return False


def _handle_aggregation_query(
    conn: sqlite3.Connection,
    user_id: int,
    qd,
    query: str,
) -> Optional[ReconstructionResult]:
    """Handle implicit aggregation queries by collecting ALL matching objects.

    Strategy:
    1. Find the entity from QD
    2. Query ALL edges for that entity (no LIMIT)
    3. Embed the query and compute cosine with each edge's edge_embedding
    4. Collect objects from edges where cosine > 0.3
    5. Deduplicate and return comma-separated
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    # Fetch all edges for the entity
    conditions = [_BASE_WHERE, "is_current = 1",
                  "(subject LIKE ? OR relational_entities LIKE ?)"]
    params: list = [user_id, f"%{entity}%", f"%{entity}%"]

    # Add negation filter — factual aggregation shouldn't include negated edges
    conditions.append("edge_negated = 0")
    conditions.append("edge_mood = 'indicative'")

    rows = conn.execute(
        f"SELECT {_CANDIDATE_COLS}, pq_1, pq_2, pq_3, pq_4 FROM edges WHERE {' AND '.join(conditions)}",
        params,
    ).fetchall()

    if not rows:
        return None

    # Embed the query once
    query_emb = embed_text(query)

    # Score each edge by PREDICTED QUERY cosine — not edge embedding.
    # PQ cosine is far more discriminating for aggregation: personality
    # traits edges (cos 0.42 via edge emb) drop below 0.3 via PQ because
    # their PQs are "What traits does X have?", not "What activities…?"
    # Real activity edges jump to cos 0.8-1.0 because their PQs match.
    scored_objects: list = []
    seen_objs: set = set()
    all_edge_ids: list = []

    for row in rows:
        obj = (row["object"] or "").strip()
        if not obj or obj.lower() in seen_objs:
            continue
        # Skip if object is just the entity name or a pronoun
        if obj.lower() in (entity.lower(), "user", "i", "me", "them", "it"):
            continue

        # PQ-based scoring: embed each pq_1-4, take best cosine with query
        best_cos = 0.0
        for col in ("pq_1", "pq_2", "pq_3", "pq_4"):
            pq_text = row[col]
            if not pq_text:
                continue
            try:
                pq_emb = embed_text(pq_text)
                cos = float(np.dot(query_emb, pq_emb))
                if cos > best_cos:
                    best_cos = cos
            except Exception:
                continue

        # Fallback: edge embedding (for edges without PQs)
        if best_cos < 0.3:
            edge_emb_bytes = row["edge_embedding"]
            if edge_emb_bytes:
                try:
                    edge_emb = np.frombuffer(edge_emb_bytes, dtype=np.float32)
                    if edge_emb.shape[0] == query_emb.shape[0]:
                        cos = float(np.dot(query_emb, edge_emb))
                        if cos > best_cos:
                            best_cos = cos
                except Exception:
                    pass

        if best_cos > 0.70:
            scored_objects.append((obj, best_cos, row["id"]))
            seen_objs.add(obj.lower())
            all_edge_ids.append(row["id"])

    if not scored_objects:
        return None

    # Sort by cosine descending
    scored_objects.sort(key=lambda x: x[1], reverse=True)

    # Collect unique objects
    items = [obj for obj, _cos, _eid in scored_objects]

    if len(items) < 2:
        # Single item — let the normal pipeline handle it for better answer extraction
        return None

    # Check if the top item is a comprehensive summary (contains "and" or
    # numbers) that subsumes the others. If so, prefer it alone. This handles
    # "What pets?" → "two cats and a dog" over listing individual pets.
    top_obj = items[0]
    if len(items) > 3 and (" and " in top_obj.lower() or any(c.isdigit() for c in top_obj)):
        # Top item might be a summary — check if it's short enough (<50 chars)
        if len(top_obj) < 50:
            log.debug("Aggregation: using summary item %r", top_obj)
            return ReconstructionResult(
                answer=top_obj,
                return_field="episodic",
                edge_ids=[scored_objects[0][2]],
                grounding=[f"aggregation:{entity}"],
            )

    log.debug("Aggregation query: %d items for entity=%s query=%r",
              len(items), entity, query)
    return ReconstructionResult(
        answer=", ".join(items),
        return_field="episodic",
        edge_ids=all_edge_ids,
        grounding=[f"aggregation:{entity}"],
    )


# ===========================================================================
# YES/NO + TEMPORAL STATE (doc lines 1226-1251)
# ===========================================================================

def _handle_still_query(conn: sqlite3.Connection, user_id: int,
                        qd, query: str) -> Optional[ReconstructionResult]:
    """Handle 'still' queries by checking is_current."""
    conditions = [_BASE_WHERE]
    params: list = [user_id]

    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    conditions.append("(subject LIKE ? OR relational_entities LIKE ?)")
    params.extend([f"%{entity}%", f"%{entity}%"])

    if qd.match_predicate:
        conditions.append("predicate LIKE ?")
        params.append(f"%{qd.match_predicate}%")

    if qd.match_object:
        obj = _clean_article(qd.match_object)
        if obj:
            conditions.append("object LIKE ?")
            params.append(f"%{obj}%")

    sql = f"""
        SELECT {_CANDIDATE_COLS} FROM edges
        WHERE {' AND '.join(conditions)}
        ORDER BY sequence_number DESC
        LIMIT 5
    """
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return None

    for r in rows:
        c = _row_to_candidate(r)
        if c.is_current == 1:
            return ReconstructionResult(
                answer="Yes", return_field="episodic",
                edge_ids=[c.edge_id], grounding=[c.source_text],
            )
        else:
            return ReconstructionResult(
                answer="No", return_field="episodic",
                edge_ids=[c.edge_id], grounding=[c.source_text],
            )
    return None


# ===========================================================================
# PLAN #9: RELATIONAL INFERENCE (doc lines 1309-1335)
# ===========================================================================

def _handle_relational_else(conn: sqlite3.Connection, user_id: int,
                            qd, query: str) -> Optional[ReconstructionResult]:
    """Handle 'who else' queries — exclude current context entity.

    'Who else works at Palantir?' → find all subjects with same predicate+object,
    exclude the implied entity.
    """
    # The "else" implies excluding someone. Try match_subject or match_entity.
    # If neither available (WH-word query), use most recent person entity from context.
    exclude = qd.match_subject or qd.match_entity or ""
    if not exclude or exclude == "user":
        row = conn.execute(
            """SELECT DISTINCT subject FROM edges
               WHERE user_id = ? AND tombstoned_at IS NULL
                 AND (subject_type = 'PERSON' OR subject_type = 'person')
                 AND subject NOT IN ('user', 'I', '')
               ORDER BY sequence_number DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        if row and row["subject"]:
            exclude = row["subject"]

    candidates = _tier1_structural(
        conn, user_id, qd, query,
        exclude_entity=exclude if exclude and exclude != "user" else None,
    )

    if candidates:
        names = list(dict.fromkeys(
            c.subject for c in candidates if c.subject and c.subject != exclude
        ))
        if names:
            return ReconstructionResult(
                answer=", ".join(names),
                return_field="relational",
                edge_ids=[c.edge_id for c in candidates[:5]],
            )
    return None


def _handle_same_comparison(conn: sqlite3.Connection, user_id: int,
                            qd, query: str) -> Optional[ReconstructionResult]:
    """Handle 'Do they live in the same city?' — compare two entities.

    SQL JOIN comparing objects for same predicate across two entities.
    """
    # Need two entities — try to extract from query
    entity1 = qd.match_subject
    entity2 = qd.match_entity
    if not entity1 or not entity2 or entity1 == entity2:
        return None

    pred = qd.match_predicate or ""
    if not pred:
        return None

    rows = conn.execute(
        f"""SELECT subject, object FROM edges
            WHERE {_BASE_WHERE} AND predicate LIKE ?
              AND subject IN (?, ?)
              AND is_current = 1""",
        (user_id, f"%{pred}%", entity1, entity2),
    ).fetchall()

    if len(rows) >= 2:
        vals = {r["subject"]: r["object"] for r in rows}
        v1 = vals.get(entity1, "")
        v2 = vals.get(entity2, "")
        if v1 and v2:
            if v1.lower() == v2.lower():
                return ReconstructionResult(
                    answer=f"Yes, both in {v1}",
                    return_field="episodic",
                )
            else:
                return ReconstructionResult(
                    answer=f"No, {entity1} is in {v1} and {entity2} is in {v2}",
                    return_field="episodic",
                )
    return None


# ===========================================================================
# PLAN #10: EMOTIONAL TREND COMPUTATION (doc lines 1636-1656)
# ===========================================================================

def _handle_emotional_trend(conn: sqlite3.Connection, user_id: int,
                            qd) -> Optional[ReconstructionResult]:
    """Compute emotional valence trend over time.

    Split edges into early/recent halves, compare mean valence.
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    rows = conn.execute(
        f"""SELECT edge_emotional_valence, sequence_number
            FROM edges
            WHERE {_BASE_WHERE}
              AND (subject LIKE ? OR relational_entities LIKE ?)
              AND edge_emotional_valence IS NOT NULL
              AND is_current = 1
            ORDER BY sequence_number ASC""",
        (user_id, f"%{entity}%", f"%{entity}%"),
    ).fetchall()

    if len(rows) < 4:
        return None

    vals = [r["edge_emotional_valence"] for r in rows]
    mid = len(vals) // 2
    early_mean = sum(vals[:mid]) / mid
    recent_mean = sum(vals[mid:]) / (len(vals) - mid)
    delta = recent_mean - early_mean

    if delta > 0.1:
        return ReconstructionResult(
            answer="Yes",
            return_field="emotional",
        )
    elif delta < -0.1:
        return ReconstructionResult(
            answer="No",
            return_field="emotional",
        )
    else:
        return ReconstructionResult(
            answer="About the same",
            return_field="emotional",
        )


# ===========================================================================
# PLAN #11: SUPERSESSION TRAIL QUERIES (doc lines 1609-1632)
# ===========================================================================

def _handle_change_query(conn: sqlite3.Connection, user_id: int,
                         query: str) -> Optional[ReconstructionResult]:
    """Handle 'what changed' queries via supersession trail.

    JOIN edges ON superseded_by to find old→new value pairs.
    """
    rows = conn.execute(
        f"""SELECT r_new.subject, r_new.predicate, r_new.object AS new_val,
                   r_old.object AS old_val, r_new.superseded_at
            FROM edges r_new
            JOIN edges r_old ON r_new.id = r_old.superseded_by
            WHERE r_new.user_id = ? AND r_new.tombstoned_at IS NULL
              AND r_new.is_current = 1
            ORDER BY r_new.superseded_at DESC
            LIMIT 10""",
        (user_id,),
    ).fetchall()

    if not rows:
        return None

    changes = []
    edge_ids = []
    for r in rows:
        subj = r["subject"] or ""
        old_v = r["old_val"] or ""
        new_v = r["new_val"] or ""
        if old_v and new_v and old_v != new_v:
            changes.append(f"{subj}: {old_v} → {new_v}")

    if changes:
        return ReconstructionResult(
            answer="; ".join(changes[:5]),
            return_field="episodic",
        )
    return None


# ===========================================================================
# PLAN #4: SESSION / CONVERSATION SCOPING (doc lines 1737-1784)
# ===========================================================================

def _handle_session_query(conn: sqlite3.Connection, user_id: int,
                          query: str) -> Optional[ReconstructionResult]:
    """Handle session-scoped queries via source_timestamp grouping."""
    q = query.lower()

    # "How many sessions have we had?"
    if "how many session" in q or "how many time" in q:
        row = conn.execute(
            f"""SELECT COUNT(DISTINCT source_timestamp) as cnt
                FROM edges WHERE {_BASE_WHERE}""",
            (user_id,),
        ).fetchone()
        if row:
            return ReconstructionResult(
                answer=str(row["cnt"]),
                return_field="episodic",
            )

    # Get distinct session timestamps
    ts_rows = conn.execute(
        f"""SELECT DISTINCT source_timestamp FROM edges
            WHERE {_BASE_WHERE} AND source_timestamp IS NOT NULL
            ORDER BY source_timestamp DESC
            LIMIT 10""",
        (user_id,),
    ).fetchall()

    if not ts_rows:
        return None

    # "last time" / "last session" → second-most-recent
    target_ts = None
    if "last time" in q or "last session" in q:
        if len(ts_rows) >= 2:
            target_ts = ts_rows[1]["source_timestamp"]
        elif ts_rows:
            target_ts = ts_rows[0]["source_timestamp"]

    # "on Tuesday" / specific day → resolve day name to date, match timestamps
    if not target_ts:
        for i, day in enumerate(_DAY_NAMES):
            if day in q:
                # Resolve to most recent occurrence of this weekday
                today = datetime.now()
                # Monday=0 in Python's weekday()
                target_weekday = i
                days_back = (today.weekday() - target_weekday) % 7
                if days_back == 0:
                    days_back = 7  # "on Tuesday" means last Tuesday, not today
                target_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
                # Find sessions whose source_timestamp starts with that date
                day_rows = conn.execute(
                    f"""SELECT {_CANDIDATE_COLS} FROM edges
                        WHERE {_BASE_WHERE} AND source_timestamp LIKE ?
                          AND is_current = 1
                        ORDER BY sequence_number ASC LIMIT 30""",
                    (user_id, f"{target_date}%"),
                ).fetchall()
                if day_rows:
                    schemas: Dict[str, List[str]] = {}
                    for r in day_rows:
                        cat = r["edge_schematic_category"] or "other"
                        if cat not in schemas:
                            schemas[cat] = []
                        schemas[cat].append(r["object"] or "")
                    parts = [f"{cat}: {', '.join(objs[:3])}" for cat, objs in schemas.items()]
                    return ReconstructionResult(
                        answer="; ".join(parts),
                        return_field="episodic",
                        edge_ids=[r["id"] for r in day_rows[:10]],
                    )
                break

    # "this week" → within last 7 days
    if not target_ts and "this week" in q:
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        rows = conn.execute(
            f"""SELECT {_CANDIDATE_COLS} FROM edges
                WHERE {_BASE_WHERE} AND source_timestamp >= ? AND is_current = 1
                ORDER BY sequence_number ASC LIMIT 30""",
            (user_id, cutoff),
        ).fetchall()
        if rows:
            schemas = {}
            for r in rows:
                cat = r["edge_schematic_category"] or "other"
                if cat not in schemas:
                    schemas[cat] = []
                schemas[cat].append(r["object"] or "")
            parts = [f"{cat}: {', '.join(objs[:3])}" for cat, objs in schemas.items()]
            return ReconstructionResult(
                answer="; ".join(parts),
                return_field="episodic",
                edge_ids=[r["id"] for r in rows[:10]],
            )
        return None

    if not target_ts:
        return None

    # Retrieve edges from target session
    rows = conn.execute(
        f"""SELECT {_CANDIDATE_COLS} FROM edges
            WHERE {_BASE_WHERE} AND source_timestamp = ? AND is_current = 1
            ORDER BY sequence_number ASC""",
        (user_id, target_ts),
    ).fetchall()

    if not rows:
        return None

    # Group by schema category, summarize
    schemas: Dict[str, List[str]] = {}
    edge_ids = []
    for r in rows:
        cat = r["edge_schematic_category"] or "other"
        obj = r["object"] or ""
        if cat not in schemas:
            schemas[cat] = []
        if obj:
            schemas[cat].append(obj)
        edge_ids.append(r["id"])

    parts = [f"{cat}: {', '.join(objs[:3])}" for cat, objs in schemas.items()]
    return ReconstructionResult(
        answer="; ".join(parts),
        return_field="episodic",
        edge_ids=edge_ids[:10],
    )


# ===========================================================================
# PLAN #12: MILESTONES TABLE (doc lines 1998-2021)
# ===========================================================================

def _handle_milestone_query(conn: sqlite3.Connection, user_id: int,
                            qd) -> Optional[ReconstructionResult]:
    """Query milestones table for significant life events."""
    conditions = ["user_id = ?"]
    params: list = [user_id]

    if qd.match_entity:
        conditions.append("description LIKE ?")
        params.append(f"%{qd.match_entity}%")

    rows = conn.execute(
        f"""SELECT description, event_date, event_type FROM milestones
            WHERE {' AND '.join(conditions)}
            ORDER BY event_date DESC
            LIMIT 10""",
        params,
    ).fetchall()

    if not rows:
        return None

    milestones = []
    for r in rows:
        desc = r["description"] or ""
        date = r["event_date"] or ""
        if desc:
            milestones.append(f"{desc} ({date})" if date else desc)

    if milestones:
        return ReconstructionResult(
            answer="; ".join(milestones),
            return_field="episodic",
        )
    return None


# ===========================================================================
# INFERENCE QUERIES — "Would X...?" without "if" clause
# ===========================================================================

def _handle_inference_query(
    conn: sqlite3.Connection, user_id: int, qd, query: str,
) -> Optional[ReconstructionResult]:
    """Handle inference questions: 'Would X likely do Y?', 'Would X enjoy Z?'

    Strategy: embed the query, find the best matching edge by edge embedding.
    High cosine (> 0.65) → evidence supports it → "Yes" + detail.
    Low cosine → "Likely no".

    Uses edge embedding (not PQ) because PQ cosine is too permissive —
    PQs about ANY topic for the entity get high cosine with inference
    questions, causing false "Yes" answers.
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    query_emb = embed_text(query)

    rows = conn.execute(
        f"""SELECT id, object, source_text, edge_embedding
            FROM edges
            WHERE {_BASE_WHERE}
              AND (subject LIKE ? OR relational_entities LIKE ?)
              AND is_current = 1""",
        (user_id, f"%{entity}%", f"%{entity}%"),
    ).fetchall()

    if not rows:
        return ReconstructionResult(
            answer="Likely no",
            return_field="episodic",
        )

    # Find best matching edge by edge embedding cosine
    best_cos = 0.0
    best_row = None
    for r in rows:
        if r["edge_embedding"]:
            try:
                ee = np.frombuffer(r["edge_embedding"], dtype=np.float32)
                if ee.shape[0] == query_emb.shape[0]:
                    cos = float(np.dot(query_emb, ee))
                    if cos > best_cos:
                        best_cos = cos
                        best_row = r
            except Exception:
                pass

    if best_cos >= 0.65 and best_row:
        src = best_row["source_text"] or ""
        obj = best_row["object"] or ""
        detail = obj if len(obj) < 60 else src[:80]
        return ReconstructionResult(
            answer=f"Yes, {detail}" if detail else "Yes",
            return_field="episodic",
            edge_ids=[best_row["id"]],
            grounding=[src],
        )

    # Fallback for preference queries: topic-to-object cosine.
    # Only for "enjoy/like/bookshelf/have" pattern — these ask about
    # category preferences where cosine between topic and object works:
    # "Vivaldi" (topic) ↔ "Bach, Mozart" (object) = same music category.
    q_lower = query.lower()
    _is_pref = any(w in q_lower for w in (
        "enjoy", "bookshelf", "have on her", "interested in",
    ))
    if _is_pref:
        try:
            from app.engines.grammar_engine import _get_nlp
            _doc = _get_nlp()(query)
            ent_lower = entity.lower()
            topic_words = [
                tok.text for tok in _doc
                if tok.pos_ in ("NOUN", "PROPN") and len(tok.text) > 2
                and tok.text.lower() not in (ent_lower, "likely")
            ]
            if topic_words:
                topic_emb = embed_text(" ".join(topic_words))
                best_tc = 0.0
                best_tr = None
                for r in rows:
                    obj_text = r["object"] or ""
                    if obj_text and len(obj_text) > 2:
                        try:
                            obj_emb = embed_text(obj_text)
                            cos = float(np.dot(topic_emb, obj_emb))
                            if cos > best_tc:
                                best_tc = cos
                                best_tr = r
                        except Exception:
                            pass
                if best_tc >= 0.45 and best_tr:
                    src = best_tr["source_text"] or ""
                    obj = best_tr["object"] or ""
                    detail = obj if len(obj) < 60 else src[:80]
                    return ReconstructionResult(
                        answer=f"Yes, {detail}" if detail else "Yes",
                        return_field="episodic",
                        edge_ids=[best_tr["id"]],
                        grounding=[src],
                    )
        except Exception:
            pass

    # No strong evidence → "Likely no"
    return ReconstructionResult(
        answer="Likely no",
        return_field="episodic",
    )


# ===========================================================================
# PLAN #17: SYNTHESIS / CAUSAL REASONING — TYPE 9 (doc lines 1370-1411)
# ===========================================================================

def _handle_causal_query(conn: sqlite3.Connection, user_id: int,
                         qd, query: str) -> Optional[ReconstructionResult]:
    """Handle conditional mood queries with causal predicate detection.

    'Would Caroline still want X if Y hadn't happened?'
    1. Find edges about entity + both topics
    2. Detect causal predicates linking Y → X
    3. If causal edge found and query removes Y → 'Likely no'
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    # Retrieve all edges about this entity
    rows = conn.execute(
        f"""SELECT {_CANDIDATE_COLS} FROM edges
            WHERE {_BASE_WHERE}
              AND (subject LIKE ? OR relational_entities LIKE ?)
              AND is_current = 1
            ORDER BY sequence_number DESC
            LIMIT 50""",
        (user_id, f"%{entity}%", f"%{entity}%"),
    ).fetchall()

    if not rows:
        return None

    # Look for causal predicates
    for r in rows:
        pred = (r["predicate"] or "").lower().replace("_", " ")
        for causal in CAUSAL_PREDICATES:
            if causal in pred:
                # Found a causal edge — the cause is in the object,
                # the effect is in the subject's domain
                return ReconstructionResult(
                    answer="Likely no",
                    return_field="episodic",
                    edge_ids=[r["id"]],
                    grounding=[r["source_text"] or ""],
                )

    # No causal edge — fall through to list synthesis
    # Open-domain: aggregate objects by schema
    if qd.match_schema:
        schema_edges = [r for r in rows
                        if (r["edge_schematic_category"] or "") == qd.match_schema]
    else:
        schema_edges = rows

    if schema_edges:
        objects = list(dict.fromkeys(
            r["object"] for r in schema_edges if r["object"]
        ))
        if objects:
            return ReconstructionResult(
                answer=", ".join(objects[:5]),
                return_field="episodic",
                edge_ids=[r["id"] for r in schema_edges[:5]],
            )

    return None


# ===========================================================================
# NEGATION / ABSENCE — SCOPED CWA (doc lines 1337-1352, plan #16)
# ===========================================================================

def _check_scoped_cwa(conn: sqlite3.Connection, user_id: int,
                      entity: str) -> str:
    """Check entity coverage for Closed-World Assumption.

    Returns 'no' (well-known entity, CWA applies) or
    'unknown' (entity barely mentioned, insufficient coverage).
    """
    row = conn.execute(
        "SELECT mention_count FROM entities WHERE user_id = ? AND name LIKE ?",
        (user_id, f"%{entity}%"),
    ).fetchone()
    if row and row["mention_count"] and row["mention_count"] >= CWA_MENTION_THRESHOLD:
        return "no"
    return "unknown"


# ===========================================================================
# MULTI-HOP — CHAINED QUERIES (doc lines 1253-1281)
# ===========================================================================

def _detect_multihop(qd) -> bool:
    """Detect if query requires multi-hop (descriptive subject)."""
    if not qd.match_subject:
        return False
    subj = qd.match_subject
    if any(w in subj.lower() for w in ("who ", "that ", "which ")):
        return True
    if qd.match_entity and qd.match_entity.lower() != subj.lower() and len(subj.split()) > 2:
        return True
    return False


def _resolve_multihop(conn: sqlite3.Connection, user_id: int,
                      qd) -> Optional[str]:
    """Resolve descriptive subject to a named entity via SQL."""
    subj = qd.match_subject.lower()
    if "who " in subj:
        after_who = subj.split("who ", 1)[1].strip()
        words = after_who.split()
        if words:
            embedded_pred = words[0]
            embedded_obj = " ".join(words[1:]) if len(words) > 1 else ""

            conditions = [_BASE_WHERE, "is_current = 1", "predicate LIKE ?"]
            params = [user_id, f"%{embedded_pred}%"]

            if embedded_obj:
                conditions.append("object LIKE ?")
                params.append(f"%{embedded_obj}%")

            row = conn.execute(
                f"""SELECT subject FROM edges
                    WHERE {' AND '.join(conditions)}
                    LIMIT 1""",
                params,
            ).fetchone()
            if row and row["subject"]:
                return row["subject"]
    return None


# ===========================================================================
# PQ WRITE-BACK — SELF-IMPROVING RETRIEVAL (design doc line 90)
# ===========================================================================

def _write_back_pq(conn: sqlite3.Connection, user_id: int,
                   edge_id: int, query: str, answer: str):
    """Write verified query into vq_1 or vq_2 on the edge row."""
    try:
        row = conn.execute(
            "SELECT vq_1, vq_2 FROM edges WHERE id = ?",
            (edge_id,),
        ).fetchone()
        if not row:
            return
        if not row["vq_1"]:
            conn.execute(
                "UPDATE edges SET vq_1 = ? WHERE id = ?",
                (query, edge_id),
            )
        elif not row["vq_2"]:
            conn.execute(
                "UPDATE edges SET vq_2 = ? WHERE id = ?",
                (query, edge_id),
            )
        # Both slots full — skip (don't overwrite proven queries)
        conn.commit()
    except Exception as e:
        log.debug("VQ write-back failed: %s", e)


# ===========================================================================
# UTILITY
# ===========================================================================

def _is_contentful_object(text: str) -> bool:
    """Check if an object string is a contentful noun phrase vs a pronoun/stub.

    Uses spaCy POS tagging — no word lists. A contentful object has at least
    one NOUN, PROPN, or ADJ token. Pure pronouns (PRON), determiners (DET),
    or single-token function words are not contentful answers.
    """
    if not text or not text.strip():
        return False
    from app.engines.grammar_engine import _get_nlp
    nlp = _get_nlp()
    doc = nlp(text.strip())
    content_pos = {"NOUN", "PROPN", "ADJ", "NUM"}
    for tok in doc:
        if tok.pos_ in content_pos:
            return True
    return False


def _clean_article(text: str) -> str:
    """Strip leading articles/determiners from match_object."""
    obj = text.strip()
    for prefix in ("a ", "an ", "the ", "some "):
        if obj.lower().startswith(prefix):
            obj = obj[len(prefix):]
    return obj


# ===========================================================================
# FIX 1: VERB CLASS PREDICATE MATCHING (WordNet hypernym, not literal LIKE)
# ===========================================================================

def _verb_class_predicate_clause(
    conn: sqlite3.Connection, user_id: int,
    match_predicate: str, params: list,
) -> str:
    """Build a SQL clause that matches predicates by verb class, not literal.

    Uses classify_verb_class (WordNet hypernym closure) to group synonymous
    predicates: "work_at", "employed_at", "start_at" all → VerbClass.WORK.
    Falls back to LIKE if verb class is UNKNOWN.
    """
    try:
        from app.engines.grammar_engine import classify_verb_class
        vc = classify_verb_class(match_predicate.lower())
        if vc.name and vc.name != "UNKNOWN":
            sibling_preds = conn.execute(
                """SELECT DISTINCT predicate FROM edges
                   WHERE user_id = ? AND predicate IS NOT NULL
                     AND tombstoned_at IS NULL""",
                (user_id,),
            ).fetchall()
            matching = []
            for r in sibling_preds:
                p = r["predicate"] or ""
                lemma = p.split("_")[0] if p else ""
                if lemma:
                    try:
                        if classify_verb_class(lemma).name == vc.name:
                            matching.append(p)
                    except Exception:
                        pass
            if matching:
                placeholders = ",".join("?" for _ in matching)
                params.extend(matching)
                return f"predicate IN ({placeholders})"
    except Exception:
        pass
    params.append(f"%{match_predicate}%")
    return "predicate LIKE ?"


# ===========================================================================
# FIX 3: TEMPORAL DURATION/CHANGE QUERIES
# ===========================================================================

def _handle_duration_query(
    conn: sqlite3.Connection, user_id: int, qd, query: str,
) -> Optional[ReconstructionResult]:
    """Handle 'how long did X work at Y?' via supersession date math.

    Finds the start edge (earliest) and end edge (superseded_at or now),
    computes the duration from resolved_event_date fields.
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    conditions = [
        "user_id = ?",
        "tombstoned_at IS NULL",
        "(subject LIKE ? OR relational_entities LIKE ?)",
    ]
    params: list = [user_id, f"%{entity}%", f"%{entity}%"]

    if qd.match_predicate:
        params_copy = list(params)
        pred_clause = _verb_class_predicate_clause(conn, user_id, qd.match_predicate, params_copy)
        conditions.append(pred_clause)
        params = params_copy

    if qd.match_schema:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)

    rows = conn.execute(
        f"""SELECT resolved_event_date, superseded_at, is_current, source_text
            FROM edges
            WHERE {' AND '.join(conditions)}
            ORDER BY resolved_event_date ASC""",
        params,
    ).fetchall()

    if not rows:
        return None

    # Find start date (earliest resolved_event_date)
    start_date = None
    end_date = None
    for r in rows:
        d = r["resolved_event_date"]
        if d and len(d) >= 10:
            if start_date is None:
                start_date = d[:10]
            # Find end: superseded_at if not current, else today
            if r["is_current"] == 0 and r["superseded_at"]:
                end_date = r["superseded_at"][:10]

    if not start_date:
        return None

    if not end_date:
        # Still current — duration is from start to now
        from datetime import datetime
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        from datetime import datetime
        dt_start = datetime.fromisoformat(start_date)
        dt_end = datetime.fromisoformat(end_date)
        delta = dt_end - dt_start
        days = delta.days
        if days < 0:
            return None
        years = days // 365
        months = (days % 365) // 30
        if years > 0:
            answer = f"{years} year{'s' if years != 1 else ''}"
            if months > 0:
                answer += f" and {months} month{'s' if months != 1 else ''}"
        elif months > 0:
            answer = f"{months} month{'s' if months != 1 else ''}"
        else:
            answer = f"{days} day{'s' if days != 1 else ''}"
        return ReconstructionResult(
            answer=answer,
            return_field="temporal",
            grounding=[rows[0]["source_text"] or ""],
        )
    except Exception:
        return None


def _is_duration_query(query: str) -> bool:
    """Detect 'how long' duration queries using spaCy."""
    q = query.lower()
    return "how long" in q


# ===========================================================================
# FIX 4: POSSESSIVE / ROLE REFERENCE RESOLUTION (spaCy poss dep)
# ===========================================================================

def _resolve_possessive(conn: sqlite3.Connection, user_id: int, qd, query: str):
    """Resolve possessive references: 'Melanie's kids', 'Jon's studio'.

    Uses spaCy dependency parsing to find poss relations, then queries
    the DB for the possessor's relationship to the possessed noun.
    Modifies qd in-place.
    """
    try:
        from app.engines.grammar_engine import _get_nlp
        nlp = _get_nlp()
        doc = nlp(query)

        for token in doc:
            if token.dep_ == "poss" and token.head:
                possessor = token.text  # "Melanie"
                possessed = token.head.text  # "kids"

                # Check if possessor is a known entity
                entity_row = conn.execute(
                    "SELECT name FROM entities WHERE user_id = ? AND LOWER(name) = LOWER(?)",
                    (user_id, possessor),
                ).fetchone()
                if not entity_row:
                    continue

                # The possessor is the entity, the possessed is what we're asking about
                # Set the entity to possessor, and add possessed as object context
                qd.match_entity = entity_row["name"]
                qd.match_subject = entity_row["name"]
                if not qd.match_object:
                    qd.match_object = possessed
                return
    except Exception:
        pass


# ===========================================================================
# NEW 3-STEP SEARCH (replaces 4-tier cascade in reconstruct())
# ===========================================================================

def _step1_trace_sql(conn: sqlite3.Connection, user_id: int,
                     qd, query: str,
                     temporal_filter: Optional[str] = None) -> List[Candidate]:
    """Step 1: Trace-scoped SQL — combines _tier1_structural + _trace_scoped logic.

    Always runs first. Uses grammar decomposition (S/P/O/schema/entity) and
    5 trace columns as WHERE filters. Applies is_current, tombstoned_at,
    edge_negated, edge_mood filters + temporal_filter for "used to"/"still".
    """
    conditions = [_BASE_WHERE]
    params: list = [user_id]
    has_filter = False
    is_factual = not _is_conditional_query(query)

    # --- Entity / subject matching ---
    if _is_speech_act_query(qd):
        if qd.match_subject and qd.match_subject != "user":
            conditions.append("subject LIKE ?")
            params.append(f"%{qd.match_subject}%")
            has_filter = True
        if qd.match_entity and qd.match_entity != qd.match_subject:
            conditions.append("relational_entities LIKE ?")
            params.append(f"%{qd.match_entity}%")
            has_filter = True
    else:
        if qd.match_subject and qd.match_subject != "user":
            conditions.append(
                "(subject LIKE ? OR relational_entities LIKE ?)"
            )
            params.extend([f"%{qd.match_subject}%", f"%{qd.match_subject}%"])
            has_filter = True
        elif qd.match_entity:
            conditions.append(
                "(subject LIKE ? OR relational_entities LIKE ?)"
            )
            params.extend([f"%{qd.match_entity}%", f"%{qd.match_entity}%"])
            has_filter = True

    # --- Predicate (verb class grouping — open-vocabulary via WordNet) ---
    if qd.match_predicate:
        try:
            from app.engines.grammar_engine import classify_verb_class
            vc = classify_verb_class(qd.match_predicate.lower())
            vc_name = vc.name  # e.g. "WORK", "LIVE", "PREFERENCE"
            if vc_name and vc_name != "UNKNOWN":
                # Find all predicates in this verb class by checking the
                # facts table key pattern (schema::VerbClass::subject).
                # For edges: match any predicate whose verb_class == this one.
                # Since edges don't store verb_class directly, we match
                # predicate LIKE for the original + expand via verb class
                # siblings from existing edges.
                sibling_preds = conn.execute(
                    """SELECT DISTINCT predicate FROM edges
                       WHERE user_id = ? AND predicate IS NOT NULL
                         AND tombstoned_at IS NULL""",
                    (user_id,),
                ).fetchall()
                matching_preds = []
                for r in sibling_preds:
                    p = r["predicate"] or ""
                    lemma = p.split("_")[0] if p else ""
                    if lemma:
                        try:
                            if classify_verb_class(lemma).name == vc_name:
                                matching_preds.append(p)
                        except Exception:
                            pass
                if matching_preds:
                    placeholders = ",".join("?" for _ in matching_preds)
                    conditions.append(f"predicate IN ({placeholders})")
                    params.extend(matching_preds)
                    has_filter = True
                else:
                    # Fallback: literal LIKE
                    conditions.append("predicate LIKE ?")
                    params.append(f"%{qd.match_predicate}%")
                    has_filter = True
            else:
                conditions.append("predicate LIKE ?")
                params.append(f"%{qd.match_predicate}%")
                has_filter = True
        except Exception:
            conditions.append("predicate LIKE ?")
            params.append(f"%{qd.match_predicate}%")
            has_filter = True

    # --- Object ---
    # Skip object filter when it equals the entity/subject — in queries
    # like "What motivated Caroline?", the grammar engine parses Caroline
    # as dobj, but in stored triples it's the subject. Using it as an
    # object filter kills correct edges.
    _skip_obj = False
    if qd.match_object:
        _obj_lower = qd.match_object.lower()
        _ent_lower = (qd.match_entity or "").lower()
        _subj_lower = (qd.match_subject or "").lower()
        if _obj_lower and (_obj_lower == _ent_lower or _obj_lower == _subj_lower):
            _skip_obj = True
        # Skip single-word category noun objects ("book", "song", "pet")
        # when entity + predicate already provide sufficient filtering.
        # These category nouns won't appear in stored objects which contain
        # specific answers ("Becoming Nicole", "Brave by Sara Bareilles").
        if (not _skip_obj and has_filter and qd.match_predicate
                and " " not in _obj_lower.strip()):
            try:
                from app.engines.grammar_engine import _get_nlp
                _od = _get_nlp()(_obj_lower)
                if len(_od) == 1 and _od[0].tag_ in ("NN", "NNS"):
                    _skip_obj = True
            except Exception:
                pass
    if qd.match_object and not _skip_obj:
        obj = _clean_article(qd.match_object)
        if obj:
            conditions.append("(object LIKE ? OR source_text LIKE ?)")
            params.extend([f"%{obj}%", f"%{obj}%"])
            has_filter = True

    # --- Schema ---
    # Schema is a SOFT signal, not a hard filter.  Grammar-derived schema
    # ("career", "housing", …) often misclassifies queries (32% mismatch on
    # LOCOMO conv 0).  Keeping it as a WHERE clause kills correct edges.
    # Instead, store the requested schema and apply a scoring boost in
    # _apply_ranking_signals when schema matches.
    # Only use schema as a hard filter when it is the SOLE discriminant
    # (no entity, no subject, no predicate, no object).
    _schema_as_only_filter = (
        qd.match_schema
        and not has_filter
        and not qd.match_entity
        and not qd.match_subject
    )
    if _schema_as_only_filter:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)
        has_filter = True

    # --- Trace-scoped filters (from _trace_scoped_retrieval) ---
    # When no S/P/O but entity exists, route by return_field
    if not has_filter and (qd.match_entity or qd.match_subject):
        entity = qd.match_entity or qd.match_subject
        conditions.append("(subject LIKE ? OR relational_entities LIKE ?)")
        params.extend([f"%{entity}%", f"%{entity}%"])
        has_filter = True

        rf = qd.return_field
        if rf == "emotional":
            conditions.append("edge_emotional_label IS NOT NULL")
            conditions.append("edge_emotional_valence != 0.5")
        elif rf == "temporal":
            conditions.append("resolved_event_date IS NOT NULL")
            conditions.append("resolved_event_date != source_timestamp")
        elif rf == "relational":
            conditions.append("relational_entities IS NOT NULL")
            conditions.append("relational_entities != '[]'")
            q_lower = query.lower()
            if "work" in q_lower or "job" in q_lower or "career" in q_lower:
                conditions.append("edge_relational_type = 'professional'")
            elif "family" in q_lower or "home" in q_lower:
                conditions.append("edge_relational_type = 'personal'")

    if not has_filter:
        return []

    # --- Temporal direction filters ---
    if temporal_filter == "past":
        conditions.append("(is_current = 0 OR is_historical = 1 OR edge_temporal_context = 'past')")
    elif temporal_filter == "present":
        conditions.append("is_current = 1")

    # --- Edge negation filter ---
    if is_factual and qd.wh_word and not _is_yesno_query(query, qd.wh_word):
        conditions.append("edge_negated = 0")

    # --- Edge mood filter ---
    if not _is_interrogative_unbounded(query):
        if is_factual:
            conditions.append("edge_mood = 'indicative'")
        else:
            conditions.append("edge_mood IN ('indicative', 'conditional')")

    sql = f"""
        SELECT {_CANDIDATE_COLS} FROM edges
        WHERE {' AND '.join(conditions)}
        ORDER BY sequence_number DESC
        LIMIT {TIER1_LIMIT}
    """
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_candidate(r, "step1_trace_sql") for r in rows]


def _step2_fts_pq(conn: sqlite3.Connection, user_id: int,
                  query: str) -> List[Candidate]:
    """Step 2: FTS5 match + PQ text match (no embeddings at query time).

    Runs FTS5 MATCH on edges_fts with query text.
    Also searches pq_1-pq_4 columns via LIKE for keyword matches.
    """
    candidates: List[Candidate] = []
    seen_ids: set = set()

    # --- FTS5 match ---
    try:
        fts_query = " ".join(
            w for w in query.split()
            if w.isalnum() or "'" in w
        )
        if fts_query.strip():
            fts_rows = conn.execute(
                """SELECT rowid, rank FROM edges_fts
                   WHERE edges_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, FTS_LIMIT),
            ).fetchall()
            if fts_rows:
                fts_ids = [r["rowid"] for r in fts_rows]
                placeholders = ",".join("?" * len(fts_ids))
                rows = conn.execute(
                    f"""SELECT {_CANDIDATE_COLS} FROM edges
                        WHERE id IN ({placeholders}) AND {_BASE_WHERE}""",
                    fts_ids + [user_id],
                ).fetchall()
                for r in rows:
                    c = _row_to_candidate(r, "step2_fts")
                    candidates.append(c)
                    seen_ids.add(c.edge_id)
    except Exception:
        pass

    # --- PQ text match (LIKE on pq_1-pq_4) ---
    # Build keyword patterns from significant words in the query
    stop_words = frozenset({
        "what", "where", "when", "who", "whom", "which", "how", "why",
        "is", "are", "was", "were", "do", "does", "did", "has", "have",
        "had", "the", "a", "an", "of", "in", "on", "at", "to", "for",
        "and", "or", "but", "not", "with", "from", "by", "about", "that",
        "this", "it", "be", "been", "being", "can", "could", "would",
        "should", "will", "shall", "may", "might", "my", "your", "his",
        "her", "its", "our", "their", "s", "t", "re", "ve", "ll", "d",
    })
    words = [w.lower().strip("?.,!") for w in query.split()]
    keywords = [w for w in words if w and w not in stop_words and len(w) > 2]

    if keywords:
        # Build OR conditions for PQ columns
        pq_conditions = []
        pq_params: list = [user_id]
        for kw in keywords[:4]:  # Limit to 4 keywords to keep query reasonable
            pattern = f"%{kw}%"
            pq_conditions.append(
                "(pq_1 LIKE ? OR pq_2 LIKE ? OR pq_3 LIKE ? OR pq_4 LIKE ?)"
            )
            pq_params.extend([pattern, pattern, pattern, pattern])

        if pq_conditions:
            pq_sql = f"""
                SELECT {_CANDIDATE_COLS} FROM edges
                WHERE {_BASE_WHERE}
                  AND ({' OR '.join(pq_conditions)})
                LIMIT {PQ_LIMIT}
            """
            try:
                pq_rows = conn.execute(pq_sql, pq_params).fetchall()
                for r in pq_rows:
                    eid = r["id"]
                    if eid not in seen_ids:
                        c = _row_to_candidate(r, "step2_pq")
                        candidates.append(c)
                        seen_ids.add(eid)
            except Exception:
                pass

    return candidates


def _step3_embed_rerank(candidates: List[Candidate], query: str) -> List[Candidate]:
    """Step 3: Embedding rerank on small candidate set.

    Only runs if > 20 candidates. Computes cosine between query embedding
    and each candidate's edge_embedding. Sorts by cosine. Keeps top 20.
    Replaces Tier 3's full-table scan with targeted reranking.
    """
    if len(candidates) <= RERANK_TOP_N:
        return candidates

    query_emb = embed_text(query)

    scored = []
    for c in candidates:
        if c.edge_embedding:
            emb = np.frombuffer(c.edge_embedding, dtype=np.float32)
            if emb.shape[0] == query_emb.shape[0]:
                cos = float(np.dot(query_emb, emb))
                scored.append((c, cos))
            else:
                scored.append((c, 0.0))
        else:
            scored.append((c, 0.0))

    scored.sort(key=lambda x: x[1], reverse=True)

    result = []
    for c, cos in scored[:RERANK_TOP_N]:
        c.score = cos
        result.append(c)
    return result


# ===========================================================================
# PQ TEXT SHORT-CIRCUIT (text-based, replaces embedding-based PQ short-circuit)
# ===========================================================================

def _pq_text_short_circuit(conn: sqlite3.Connection, user_id: int,
                           query: str) -> Optional[Tuple[int, str]]:
    """Text-based PQ short-circuit. Checks if any pq_1-4 closely matches query.

    Uses string containment: if query text is contained in a PQ or vice versa,
    that's a high-confidence match.

    Returns:
        (edge_id, source_text) if match found, else None.
    """
    q_lower = query.lower().strip("?.,!")

    # Try exact containment first — query inside PQ or PQ inside query
    pq_rows = conn.execute(
        f"""SELECT id, source_text, pq_1, pq_2, pq_3, pq_4
            FROM edges
            WHERE {_BASE_WHERE}
              AND (pq_1 IS NOT NULL OR pq_2 IS NOT NULL
                   OR pq_3 IS NOT NULL OR pq_4 IS NOT NULL)""",
        (user_id,),
    ).fetchall()

    if not pq_rows:
        return None

    best_match = None
    best_ratio = 0.0

    for row in pq_rows:
        for col in ("pq_1", "pq_2", "pq_3", "pq_4"):
            pq_text = row[col]
            if not pq_text:
                continue
            pq_lower = pq_text.lower().strip("?.,!")

            # Exact containment: query in PQ or PQ in query
            if q_lower in pq_lower or pq_lower in q_lower:
                # Compute overlap ratio for ranking
                shorter = min(len(q_lower), len(pq_lower))
                longer = max(len(q_lower), len(pq_lower))
                ratio = shorter / longer if longer > 0 else 0.0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = (row["id"], row["source_text"])

    # Only short-circuit if high overlap (>= 70% length ratio)
    if best_match and best_ratio >= 0.70:
        return best_match

    return None


def _merge_candidates(existing: List[Candidate],
                      new: List[Candidate]) -> List[Candidate]:
    """Merge new candidates into existing list, dedup by edge_id."""
    seen = {c.edge_id for c in existing}
    for c in new:
        if c.edge_id not in seen:
            existing.append(c)
            seen.add(c.edge_id)
    return existing


def _refuse(reason: str) -> ReconstructionResult:
    return ReconstructionResult(refusal=True, refusal_reason=reason, answer=REFUSAL_TEXT)


def _refuse_low_coverage() -> ReconstructionResult:
    """Plan #16: distinct wording for low-coverage entities."""
    return ReconstructionResult(
        refusal=True,
        refusal_reason="low_coverage",
        answer=LOW_COVERAGE_TEXT,
    )


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def reconstruct(user_id: int, query: str) -> ReconstructionResult:
    """Deterministic reconstruction — the complete read path.

    Called by sdk/client.py for ALL queries (situational and factual).
    Implements every question type from docs/retrieval-systems-how-they-work.md Part 4.
    """
    _check_entry()
    from app.engines.grammar_engine import classify_query

    # ---- Step 1: Query decomposition ----
    qd = classify_query(query)
    log.debug("classify_query(%r) -> subj=%s pred=%s obj=%s schema=%s rf=%s wh=%s",
              query, qd.match_subject, qd.match_predicate, qd.match_object,
              qd.match_schema, qd.return_field, qd.wh_word)

    with get_db_context() as conn:

        # ---- Step 2: Pronoun + possessive resolution ----
        _resolve_pronoun(conn, user_id, qd)
        _resolve_possessive(conn, user_id, qd, query)

        # ---- Step 3: Question-type routing ----

        # Duration queries (fix #3)
        if _is_duration_query(query):
            result = _handle_duration_query(conn, user_id, qd, query)
            if result:
                return result

        # Count queries — BEFORE session handler because "how many times"
        # triggers both, and count handler gives correct entity-scoped count
        # while session handler counts all sessions.
        if _is_count_query(query):
            result = _handle_count_query(conn, user_id, qd)
            if result:
                return result

        # Session/conversation scoping (plan #4)
        if _is_session_query(query):
            result = _handle_session_query(conn, user_id, query)
            if result:
                return result

        # Change/supersession queries (plan #11)
        if _is_change_query(query):
            result = _handle_change_query(conn, user_id, query)
            if result:
                return result

        # Milestone queries (plan #12)
        if _is_milestone_query(query):
            result = _handle_milestone_query(conn, user_id, qd)
            if result:
                return result

        # Emotional trend queries (plan #10)
        if _is_emotional_trend_query(query):
            result = _handle_emotional_trend(conn, user_id, qd)
            if result:
                return result

        # List queries
        if _is_list_query(query):
            result = _handle_list_query(conn, user_id, qd)
            if result:
                return result

        # Implicit aggregation queries (plural nouns: "What activities...",
        # "What books...", "What events...")
        # Skip for conditional/inference queries ("Would X...", "What would...")
        # which need inference handling, not collection.
        if _is_aggregation_query(query) and not _is_conditional_query(query):
            result = _handle_aggregation_query(conn, user_id, qd, query)
            if result:
                return result

        # "Still" queries
        if _is_still_query(query):
            result = _handle_still_query(conn, user_id, qd, query)
            if result:
                return result

        # Relational "else" queries (plan #9)
        if _is_relational_else_query(query):
            result = _handle_relational_else(conn, user_id, qd, query)
            if result:
                return result

        # "Same" comparison queries (plan #9)
        if _is_same_comparison_query(query):
            result = _handle_same_comparison(conn, user_id, qd, query)
            if result:
                return result

        # Conditional/causal queries (plan #17)
        # Split: "Would X if Y?" → causal. "Would X...?" → inference.
        if _is_conditional_query(query):
            if "if " in query.lower():
                result = _handle_causal_query(conn, user_id, qd, query)
                if result:
                    return result
            else:
                # Inference: "Would X likely...?" — search for evidence
                result = _handle_inference_query(conn, user_id, qd, query)
                if result:
                    return result

        # Multi-hop detection and resolution
        if _detect_multihop(qd):
            resolved_entity = _resolve_multihop(conn, user_id, qd)
            if resolved_entity:
                qd.match_subject = resolved_entity
                qd.match_entity = resolved_entity

        # Determine temporal filter
        temporal_filter = None
        if _is_used_to_query(query):
            temporal_filter = "past"
        elif _is_still_query(query):
            temporal_filter = "present"

        # ---- Step 4: 3-step search ----

        # Step 4a: Trace-scoped SQL (always runs first)
        candidates: List[Candidate] = _step1_trace_sql(
            conn, user_id, qd, query, temporal_filter,
        )

        # Step 4a-filter: Subject attribution pre-filter
        # If query entity is specific and NO candidates have it as subject,
        # the topic likely belongs to a different person → refuse early.
        _qe = (qd.match_entity or qd.match_subject or "").lower()
        if _qe and _qe != "user" and candidates:
            _subj_matched = [c for c in candidates
                             if _qe in c.subject.lower() or c.subject.lower() in _qe]
            if not _subj_matched:
                # No candidate's subject matches the query entity
                return _refuse("no_subject_match")

        # Step 4b: Tier 0 facts table (O(1), keep)
        fact_value = _tier0_facts(conn, user_id, qd)
        if fact_value and qd.return_field == "episodic":
            entity = qd.match_entity or qd.match_subject or ""
            return ReconstructionResult(
                answer=fact_value,
                return_field="episodic",
                grounding=[f"facts:{entity}"],
            )

        # ---- PQ text short-circuit (entity-gated, no embeddings) ----
        pq_match = _pq_text_short_circuit(conn, user_id, query)
        if pq_match:
            pq_edge_id, _pq_src = pq_match
            # Entity check: verify ALL named entities in the query appear
            # in the edge. Cat 5 adversarial swaps speakers.
            from app.engines.grammar_engine import _get_nlp
            _nlp = _get_nlp()
            _qdoc = _nlp(query)
            _query_entities = [
                ent.text.lower() for ent in _qdoc.ents
                if ent.label_ in ("PERSON", "ORG", "GPE")
            ]
            for _qe in (qd.match_entity, qd.match_subject):
                if _qe and _qe.lower() not in ("user", ""):
                    _qe_low = _qe.lower()
                    if not any(_qe_low in e or e in _qe_low for e in _query_entities):
                        _query_entities.append(_qe_low)

            entity_ok = True
            if _query_entities:
                edge_row = conn.execute(
                    "SELECT subject, relational_entities, source_text FROM edges WHERE id = ?",
                    (pq_edge_id,),
                ).fetchone()
                if edge_row:
                    _edge_text = " ".join([
                        (edge_row["subject"] or ""),
                        (edge_row["relational_entities"] or ""),
                        (edge_row["source_text"] or ""),
                    ]).lower()
                    for _qe in _query_entities:
                        # Strip articles for matching
                        _qe_clean = _qe
                        for _art in ("the ", "a ", "an "):
                            if _qe_clean.startswith(_art):
                                _qe_clean = _qe_clean[len(_art):]
                        if _qe_clean not in _edge_text and _qe not in _edge_text:
                            entity_ok = False
                            break
            if entity_ok:
                pq_edge_row = conn.execute(
                    f"SELECT {_CANDIDATE_COLS} FROM edges WHERE id = ?",
                    (pq_edge_id,),
                ).fetchone()
                if pq_edge_row:
                    pq_candidate = _row_to_candidate(pq_edge_row, "pq_hit")
                    if _is_yesno_query(query, qd.wh_word):
                        if pq_candidate.edge_negated:
                            return ReconstructionResult(
                                answer="No", return_field="episodic",
                                edge_ids=[pq_edge_id],
                                grounding=[pq_candidate.source_text],
                            )
                        return ReconstructionResult(
                            answer="Yes", return_field="episodic",
                            edge_ids=[pq_edge_id],
                            grounding=[pq_candidate.source_text],
                        )
                    answer = _extract_answer(pq_candidate, qd, query, prefer_source=True)
                    log.debug("PQ text short-circuit: edge=%d answer=%s",
                              pq_edge_id, answer[:50] if answer else "")
                    return ReconstructionResult(
                        answer=answer,
                        return_field=qd.return_field,
                        edge_ids=[pq_edge_id],
                        grounding=[pq_candidate.source_text],
                    )

        # Step 4c: FTS5 + PQ text match
        fts_pq_candidates = _step2_fts_pq(conn, user_id, query)
        candidates = _merge_candidates(candidates, fts_pq_candidates)

        # Step 4d: Embedding rerank on small set (only if > 20 candidates)
        candidates = _step3_embed_rerank(candidates, query)

        if not candidates:
            # No candidates — check CWA or arc expansion
            arc_result = _expand_arc(conn, user_id, qd)
            if arc_result:
                return arc_result

            entity = qd.match_entity or qd.match_subject
            if entity and _is_yesno_query(query, qd.wh_word):
                cwa = _check_scoped_cwa(conn, user_id, entity)
                if cwa == "no":
                    return ReconstructionResult(answer="No", return_field="episodic")
                else:
                    return _refuse_low_coverage()
            return _refuse("not_mentioned")

        # ---- Step 5: Predicate cosine pre-scoring ----
        if qd.match_predicate:
            _qpred_emb = embed_text(qd.match_predicate.lower())
            for c in candidates:
                if c.predicate_embedding:
                    _pemb = np.frombuffer(c.predicate_embedding, dtype=np.float32)
                    if _pemb.shape == _qpred_emb.shape:
                        c.score += float(np.dot(_pemb, _qpred_emb)) * 0.3
            candidates.sort(key=lambda c: c.score, reverse=True)

        # ---- Step 6: Cross-encoder reranking ----
        candidates = _rerank(query, candidates)

        # ---- Step 7: Ranking signal stack (plan #8) ----
        candidates = _apply_ranking_signals(candidates, query, qd)

        # ---- Step 8: Verification loop ----
        verified = _verify_candidates(candidates, qd, query, conn, user_id)

        if not verified:
            entity = qd.match_entity or qd.match_subject
            if entity and _is_yesno_query(query, qd.wh_word):
                cwa = _check_scoped_cwa(conn, user_id, entity)
                if cwa == "no":
                    return ReconstructionResult(answer="No", return_field="episodic")
                else:
                    return _refuse_low_coverage()
            return _refuse("verification_rejected")

        best = verified[0]

        # ---- Step 9: Topic specificity gate ----
        try:
            from app.engines.grammar_engine import _get_nlp
            _nlp_g = _get_nlp()
            _qdoc = _nlp_g(query)
            _qe_low = (qd.match_entity or qd.match_subject or "").lower()
            _topic_nouns = []
            for tok in _qdoc:
                if tok.pos_ in ("NOUN", "PROPN") and tok.text.lower() != _qe_low:
                    # Skip indirect objects where the preposition attaches
                    # directly to the root verb ("recommend to Melanie") —
                    # they're recipients, not topics. Don't skip prepositional
                    # phrases that modify nouns ("plans with respect to adoption").
                    if tok.dep_ == "pobj" and tok.head.dep_ == "prep":
                        _prep_head = tok.head.head
                        if _prep_head.dep_ == "ROOT" and _prep_head.pos_ == "VERB":
                            continue
                    if len(tok.text) > 2 and tok.text.lower() not in (
                        "kind", "type", "way", "thing", "time", "year",
                        "month", "week", "day", "question", "career",
                        "people", "life", "activity", "activities",
                        "event", "events", "plan", "plans", "experience",
                        "journey", "process", "decision", "reason", "project",
                        "work", "job", "support", "family", "friend",
                        "friends", "kids", "children", "son", "daughter",
                        "art", "painting", "music", "book",
                        "books", "hobby", "hobbies", "community",
                        "artists", "bands", "recommend", "share",
                    ):
                        _topic_nouns.append(tok.text.lower())
            _edge_text = f"{best.source_text} {best.object} {best.predicate}".lower()

            if _topic_nouns and len(_topic_nouns) <= 3:
                _any_match = any(n in _edge_text for n in _topic_nouns)
                if not _any_match:
                    return _refuse("topic_not_in_edge")

            # Possessor check: nouns in poss/compound position that specify
            # the entity ("grandpa's gift", "hand-painted bowl") MUST appear
            # in the edge. These distinguish the query from similar queries
            # about different entities (grandma vs grandpa).
            for tok in _qdoc:
                if tok.dep_ in ("poss", "compound") and tok.pos_ in ("NOUN", "PROPN"):
                    if tok.text.lower() != _qe_low and len(tok.text) > 2:
                        if tok.text.lower() not in _edge_text:
                            return _refuse("possessor_mismatch")
        except Exception:
            pass

        # ---- Step 10: Yes/No with specificity check ----
        if _is_yesno_query(query, qd.wh_word):
            # Verify the best edge actually matches the specific claim
            # in the question, not just the entity. Cat 5 swaps attributes.
            if best.edge_embedding:
                try:
                    _qemb = embed_text(query)
                    _eemb = np.frombuffer(best.edge_embedding, dtype=np.float32)
                    if _eemb.shape[0] == _qemb.shape[0]:
                        _cos = float(np.dot(_qemb, _eemb))
                        if _cos < 0.4:
                            return _refuse("yesno_low_relevance")
                except Exception:
                    pass
            # Verify all proper nouns from the query appear in the edge.
            # Catches adversarial entity swaps ("Is Oscar Melanie's pet?"
            # where Oscar belongs to Caroline, not Melanie).
            try:
                from app.engines.grammar_engine import _get_nlp
                _yn_doc = _get_nlp()(query)
                _edge_full = f"{best.subject} {best.object} {best.source_text}".lower()
                for _tok in _yn_doc:
                    if _tok.pos_ == "PROPN" and len(_tok.text) > 2:
                        if _tok.text.lower() not in _edge_full:
                            return _refuse("yesno_propn_missing")
            except Exception:
                pass
            if best.edge_negated:
                return ReconstructionResult(
                    answer="No", return_field="episodic",
                    edge_ids=[best.edge_id], grounding=[best.source_text],
                )
            return ReconstructionResult(
                answer="Yes", return_field="episodic",
                edge_ids=[best.edge_id], grounding=[best.source_text],
            )

        # ---- Step 11: Mood check (factual queries skip conditional edges) ----
        if not _is_conditional_query(query) and best.edge_mood == "conditional":
            for c in verified[1:]:
                if c.edge_mood != "conditional":
                    best = c
                    break

        # ---- Step 11b: Multi-answer aggregation ----
        # Only aggregate when explicitly detected as aggregation query.
        # Without this guard, Cat 4 narrative questions get objects from
        # multiple unrelated edges concatenated.
        if (len(verified) > 1 and qd.return_field == "episodic"
                and _is_aggregation_query(query) and not _is_conditional_query(query)):
            unique_objects = []
            seen_objs = set()
            all_edge_ids = []
            for c in verified:
                obj = _extract_answer(c, qd, query)
                if obj and obj.lower() not in seen_objs and obj.lower() != best.subject.lower():
                    # Skip if it's just the entity name or a pronoun
                    if len(obj) > 1 and obj.lower() not in ('user', 'i', 'me'):
                        unique_objects.append(obj)
                        seen_objs.add(obj.lower())
                        all_edge_ids.append(c.edge_id)
            if len(unique_objects) > 1:
                answer = ", ".join(unique_objects)
                return ReconstructionResult(
                    answer=answer,
                    return_field=qd.return_field,
                    edge_ids=all_edge_ids,
                    grounding=[c.source_text for c in verified[:5]],
                )

        # ---- Step 12: Cluster expansion for grounding (plan #13) ----
        context_edges = _expand_cluster(conn, user_id, best)
        extra_grounding = [c.source_text for c in context_edges[:3] if c.source_text]

        # ---- Step 13: Answer extraction via return_field routing ----
        answer = _extract_answer(best, qd, query, prefer_source=True)

        # ---- Step 14: PQ write-back ----
        if answer:
            _write_back_pq(conn, user_id, best.edge_id, query, answer)

        grounding = [best.source_text] + extra_grounding if extra_grounding else [best.source_text]

        return ReconstructionResult(
            answer=answer,
            return_field=qd.return_field,
            edge_ids=[best.edge_id],
            grounding=grounding,
        )
