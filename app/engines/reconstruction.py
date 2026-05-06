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
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.db.session import get_db_context
from app.vector.embedder import embed_text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REFUSAL_TEXT = "This information is not mentioned in the conversation."
LOW_COVERAGE_TEXT = "This hasn't come up in your conversations."
RERANK_TOP_N = 20
# ms-marco-MiniLM-L-6-v2 outputs raw logits, not 0-1 probabilities.
# PQ short-circuit handles Cat 1-4 before the gate fires.
# The gate only sees queries that PQ didn't match — adversarial (Cat 5).
# Strict threshold blocks Cat 5 garbage while PQ bypasses it for Cat 1-4.
RELEVANCE_GATE_THRESHOLD = 0.25
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
        confidence=row["confidence"] if row["confidence"] is not None else 0.9,
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
        edge_affiliation=row["edge_affiliation"] if row["edge_affiliation"] is not None else 0.0,
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
    relational_entities, edge_relational_type, sequence_number, confidence,
    superseded_at, superseded_by, edge_embedding, predicate_embedding,
    episodic_fact, emotional_target, source_timestamp, subject_type,
    object_type, cluster_id, arc_id, last_confirmed_at, edge_affiliation,
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
            """SELECT DISTINCT subject FROM relationships
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
            """SELECT object FROM relationships
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
            """SELECT DISTINCT subject FROM relationships
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
        """SELECT DISTINCT subject FROM relationships
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
        try:
            from app.engines.grammar_engine import classify_verb_class
            vc = classify_verb_class(qd.match_predicate.lower())
            verb_class_name = vc.name  # e.g. "WORK", "LIVE", "STUDY"
        except Exception:
            verb_class_name = qd.match_predicate.upper()

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
        SELECT {_CANDIDATE_COLS} FROM relationships
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
    """Cosine similarity against predicted_queries embeddings.

    Returns:
        (candidates, best_pq_hit)
        best_pq_hit = (answer_text, edge_id, cosine) when cos > PQ_HIGH_CONFIDENCE,
        else None. The caller can short-circuit on a high-confidence PQ match.
    """
    query_emb = embed_text(query)

    pq_rows = conn.execute(
        """SELECT relationship_id, predicted_question, answer_text,
                  question_embedding
           FROM predicted_queries
           WHERE user_id = ?""",
        (user_id,),
    ).fetchall()

    if not pq_rows:
        return [], None

    scored = []
    best_pq = None  # (answer_text, edge_id, cosine)
    for pq in pq_rows:
        if not pq["question_embedding"]:
            continue
        pq_emb = np.frombuffer(pq["question_embedding"], dtype=np.float32)
        if pq_emb.shape[0] != query_emb.shape[0]:
            continue
        cos = float(np.dot(query_emb, pq_emb))
        if cos > 0.3:
            scored.append((pq["relationship_id"], cos))
        if cos > PQ_HIGH_CONFIDENCE:
            if best_pq is None or cos > best_pq[2]:
                best_pq = (pq["answer_text"], pq["relationship_id"], cos)

    if not scored:
        return [], best_pq

    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [s[0] for s in scored[:PQ_LIMIT]]

    placeholders = ",".join("?" * len(top_ids))
    rows = conn.execute(
        f"""SELECT {_CANDIDATE_COLS} FROM relationships
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
                """SELECT rowid, rank FROM relationships_fts
                   WHERE relationships_fts MATCH ?
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
        f"""SELECT id, edge_embedding FROM relationships
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
        f"""SELECT {_CANDIDATE_COLS} FROM relationships
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
        SELECT {_CANDIDATE_COLS} FROM relationships
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
        f"""SELECT {_CANDIDATE_COLS} FROM relationships
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
            f"""SELECT {_CANDIDATE_COLS} FROM relationships
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
        try:
            query_pred_emb = embed_text(qd.match_predicate.lower())
        except Exception:
            pass

    for c in candidates:
        ce_score = c.score  # Cross-encoder score (already set by _rerank)

        # Predicate cosine
        pred_cosine = 0.0
        if query_pred_emb is not None and c.predicate_embedding:
            try:
                pred_emb = np.frombuffer(c.predicate_embedding, dtype=np.float32)
                if pred_emb.shape == query_pred_emb.shape:
                    pred_cosine = max(0.0, float(np.dot(pred_emb, query_pred_emb)))
            except Exception:
                pass

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
                delta_days = (datetime.now() - confirmed).days
                recency_score = max(0.1, 1.0 - delta_days / 365.0)
            except Exception:
                pass

        # Affiliation
        aff_score = c.edge_affiliation if c.edge_affiliation else 0.5

        # Combined score
        c.score = (
            0.70 * ce_score
            + 0.10 * pred_cosine
            + 0.05 * sig_score
            + 0.05 * conf_score
            + 0.05 * recency_score
            + 0.05 * aff_score
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
    Subject must match. Predicate must relate."""
    query_entity = qd.match_entity or qd.match_subject or ""

    # Subject check
    if query_entity and query_entity != "user":
        subj_lower = c.subject.lower()
        entity_lower = query_entity.lower()
        if entity_lower not in subj_lower and subj_lower not in entity_lower:
            rel_lower = c.relational_entities.lower()
            if entity_lower not in rel_lower:
                return False

    # Predicate check
    if qd.match_predicate and c.predicate:
        qp = qd.match_predicate.lower()
        cp = c.predicate.lower().replace("_", " ")
        # Direct match
        if qp in cp or cp in qp:
            return True
        # Schema-level match
        if qd.match_schema and c.edge_schematic_category == qd.match_schema:
            return True
        # Predicate embedding cosine
        if c.predicate_embedding:
            try:
                pred_emb = np.frombuffer(c.predicate_embedding, dtype=np.float32)
                query_pred_emb = embed_text(qp)
                if pred_emb.shape == query_pred_emb.shape:
                    cos = float(np.dot(pred_emb, query_pred_emb))
                    if cos > 0.45:
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
        return 1.0
    try:
        score = float(reranker.predict([(query, candidate.source_text)])[0])
        return score
    except Exception:
        return 1.0


# ===========================================================================
# ANSWER EXTRACTION — RETURN_FIELD ROUTING (doc lines 1030-1048)
# ===========================================================================

def _extract_answer(candidate: Candidate, qd, query: str) -> str:
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
            # Plan #15: "how long ago" → compute delta
            if "how long ago" in q_lower:
                try:
                    dt = datetime.fromisoformat(date[:10])
                    delta = datetime.now() - dt
                    years = delta.days // 365
                    months = (delta.days % 365) // 30
                    if years > 0:
                        return f"{years} year{'s' if years != 1 else ''} ago"
                    elif months > 0:
                        return f"{months} month{'s' if months != 1 else ''} ago"
                    else:
                        return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
                except Exception:
                    pass
            # "what year" → year only
            if "what year" in q_lower:
                return date[:4]
            # "when" with no qualifier → year
            if "when" in q_lower and "what date" not in q_lower and "what time" not in q_lower:
                if len(date) >= 10 and "-" in date:
                    return date[:4]
            return date
        return candidate.object

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
                if entities:
                    return ", ".join(str(e) for e in entities)
            except Exception:
                pass
        return candidate.subject or candidate.object

    # Default: episodic
    # Object field may contain pronouns ("them"), determiners ("this"),
    # or empty fragments from grammar engine extraction. When the object
    # isn't a contentful noun phrase, fall back to source_text.
    # Use spaCy POS tagging to detect — no word lists.
    obj = candidate.object or ""
    if obj.strip():
        if _is_contentful_object(obj):
            return obj

    if candidate.episodic_fact:
        return candidate.episodic_fact

    return candidate.source_text or ""


# ===========================================================================
# AGGREGATION QUERIES — COUNT / LIST (doc lines 1283-1305)
# ===========================================================================

def _handle_count_query(conn: sqlite3.Connection, user_id: int,
                        qd) -> Optional[ReconstructionResult]:
    """Handle 'how many' queries with SQL COUNT."""
    conditions = [_BASE_WHERE, "is_current = 1"]
    params: list = [user_id]

    if qd.match_predicate:
        conditions.append("predicate LIKE ?")
        params.append(f"%{qd.match_predicate}%")
    if qd.match_object:
        obj = _clean_article(qd.match_object)
        if obj:
            conditions.append("object LIKE ?")
            params.append(f"%{obj}%")
    if qd.match_schema:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)

    if len(conditions) <= 2:
        return None

    sql = f"""
        SELECT COUNT(DISTINCT subject) as cnt
        FROM relationships
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
    """Handle 'name all' / 'list everyone' queries with SQL DISTINCT."""
    conditions = [_BASE_WHERE, "is_current = 1"]
    params: list = [user_id]

    if qd.match_predicate:
        conditions.append("predicate LIKE ?")
        params.append(f"%{qd.match_predicate}%")
    if qd.match_object:
        obj = _clean_article(qd.match_object)
        if obj:
            conditions.append("object LIKE ?")
            params.append(f"%{obj}%")
    if qd.match_schema:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)

    if len(conditions) <= 2:
        return None

    sql = f"""
        SELECT DISTINCT subject
        FROM relationships
        WHERE {' AND '.join(conditions)}
    """
    rows = conn.execute(sql, params).fetchall()
    if rows:
        names = [r["subject"] for r in rows if r["subject"]]
        if names:
            return ReconstructionResult(
                answer=", ".join(names),
                return_field="relational",
            )
    return None


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
        SELECT {_CANDIDATE_COLS} FROM relationships
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
            """SELECT DISTINCT subject FROM relationships
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
        f"""SELECT subject, object FROM relationships
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
            FROM relationships
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

    JOIN relationships ON superseded_by to find old→new value pairs.
    """
    rows = conn.execute(
        f"""SELECT r_new.subject, r_new.predicate, r_new.object AS new_val,
                   r_old.object AS old_val, r_new.superseded_at
            FROM relationships r_new
            JOIN relationships r_old ON r_new.id = r_old.superseded_by
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
                FROM relationships WHERE {_BASE_WHERE}""",
            (user_id,),
        ).fetchone()
        if row:
            return ReconstructionResult(
                answer=str(row["cnt"]),
                return_field="episodic",
            )

    # Get distinct session timestamps
    ts_rows = conn.execute(
        f"""SELECT DISTINCT source_timestamp FROM relationships
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
                    f"""SELECT {_CANDIDATE_COLS} FROM relationships
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
            f"""SELECT {_CANDIDATE_COLS} FROM relationships
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
        f"""SELECT {_CANDIDATE_COLS} FROM relationships
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
        f"""SELECT {_CANDIDATE_COLS} FROM relationships
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
                f"""SELECT subject FROM relationships
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
    """Write verified query as a new predicted query for this edge."""
    try:
        query_emb = embed_text(query)
        conn.execute(
            """INSERT OR IGNORE INTO predicted_queries
               (relationship_id, user_id, predicted_question, answer_text,
                question_embedding, confidence)
               VALUES (?, ?, ?, ?, ?, 0.95)""",
            (edge_id, user_id, query, answer, query_emb.tobytes()),
        )
        conn.commit()
    except Exception as e:
        log.debug("PQ write-back failed: %s", e)


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
    try:
        from app.engines.grammar_engine import _get_nlp
        nlp = _get_nlp()
        doc = nlp(text.strip())
        content_pos = {"NOUN", "PROPN", "ADJ", "NUM"}
        for tok in doc:
            if tok.pos_ in content_pos:
                return True
        return False
    except Exception:
        # If spaCy unavailable, fall back to length heuristic
        return len(text.strip()) > 4


def _clean_article(text: str) -> str:
    """Strip leading articles/determiners from match_object."""
    obj = text.strip()
    for prefix in ("a ", "an ", "the ", "some "):
        if obj.lower().startswith(prefix):
            obj = obj[len(prefix):]
    return obj


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
    return ReconstructionResult(refusal=True, refusal_reason=reason)


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
    from app.engines.grammar_engine import classify_query

    # ---- Step 1: Query decomposition ----
    qd = classify_query(query)
    log.debug("classify_query(%r) -> subj=%s pred=%s obj=%s schema=%s rf=%s wh=%s",
              query, qd.match_subject, qd.match_predicate, qd.match_object,
              qd.match_schema, qd.return_field, qd.wh_word)

    with get_db_context() as conn:

        # ---- Step 2: Pronoun resolution (plan #5) ----
        _resolve_pronoun(conn, user_id, qd)

        # ---- Step 3: Question-type routing ----

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

        # Count queries
        if _is_count_query(query):
            result = _handle_count_query(conn, user_id, qd)
            if result:
                return result

        # List queries
        if _is_list_query(query):
            result = _handle_list_query(conn, user_id, qd)
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
        if _is_conditional_query(query):
            result = _handle_causal_query(conn, user_id, qd, query)
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

        # ---- Step 4: Tiered retrieval ----

        # Tier 0: Facts table
        fact_value = _tier0_facts(conn, user_id, qd)
        if fact_value and qd.return_field == "episodic":
            reranker = _get_reranker()
            if reranker:
                entity = qd.match_entity or qd.match_subject or ""
                synthetic = f"{entity} {qd.match_predicate or ''} {fact_value}"
                score = _relevance_gate(query, Candidate(
                    edge_id=0, source_text=synthetic,
                ))
                if score >= RELEVANCE_GATE_THRESHOLD:
                    return ReconstructionResult(
                        answer=fact_value,
                        return_field="episodic",
                        grounding=[f"facts:{entity}"],
                    )
            else:
                return ReconstructionResult(
                    answer=fact_value,
                    return_field="episodic",
                    grounding=[f"facts:{qd.match_entity or qd.match_subject}"],
                )

        # Trace-scoped retrieval (plan #2) — when no S/P/O available
        candidates: List[Candidate] = []
        if not _has_spo(qd):
            candidates = _trace_scoped_retrieval(conn, user_id, qd, query)

        # Tier 1: Structural SQL (with speaker attr, negation, mood filters)
        if not candidates:
            candidates = _tier1_structural(conn, user_id, qd, query, temporal_filter)

        # Tier 2: Predicted queries (if Tier 1 insufficient)
        # High-confidence PQ match short-circuits the entire pipeline —
        # the PQ was written at ingest time specifically for this query pattern.
        if len(candidates) < 3:
            t2, pq_hit = _tier2_predicted_queries(conn, user_id, query)
            if pq_hit and pq_hit[2] >= PQ_HIGH_CONFIDENCE:
                pq_answer, pq_edge_id, pq_cos = pq_hit
                # Entity check: verify the PQ edge belongs to the queried entity.
                # Cat 5 adversarial swaps speakers — "What did Melanie research?"
                # when the edge is Caroline/research/adoption. Must reject.
                query_entity = qd.match_entity or qd.match_subject or ""
                entity_ok = True
                if query_entity and query_entity.lower() not in ("user", ""):
                    edge_row = conn.execute(
                        "SELECT subject, relational_entities FROM relationships WHERE id = ?",
                        (pq_edge_id,),
                    ).fetchone()
                    if edge_row:
                        subj = (edge_row["subject"] or "").lower()
                        rel = (edge_row["relational_entities"] or "").lower()
                        qe = query_entity.lower()
                        if qe not in subj and subj not in qe and qe not in rel:
                            entity_ok = False
                if entity_ok:
                    # Don't return raw PQ answer — route through return_field
                    # extraction so temporal/emotional/relational answers are correct.
                    pq_edge_row = conn.execute(
                        f"SELECT {_CANDIDATE_COLS} FROM relationships WHERE id = ?",
                        (pq_edge_id,),
                    ).fetchone()
                    if pq_edge_row:
                        pq_candidate = _row_to_candidate(pq_edge_row, "pq_hit")
                        answer = _extract_answer(pq_candidate, qd, query)
                        log.debug("PQ short-circuit: cos=%.3f edge=%d answer=%s",
                                  pq_cos, pq_edge_id, answer[:50] if answer else "")
                        return ReconstructionResult(
                            answer=answer,
                            return_field=qd.return_field,
                            edge_ids=[pq_edge_id],
                            grounding=[pq_candidate.source_text],
                        )
            candidates = _merge_candidates(candidates, t2)

        # Tier 3: RRF hybrid (if still insufficient)
        if len(candidates) < 3:
            t3 = _tier3_rrf(conn, user_id, query)
            candidates = _merge_candidates(candidates, t3)

        if not candidates:
            # No candidates — check CWA or arc expansion
            # Arc-based reconstruction (plan #13)
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

        # ---- Step 4b: Predicate cosine pre-scoring on ALL candidates ----
        # Doc lines 1951-1965: runs cheaply on ALL candidates BEFORE cross-encoder
        # narrows to top-20. Catches predicate synonymy ("visit" vs "went_to").
        if qd.match_predicate:
            try:
                _qpred_emb = embed_text(qd.match_predicate.lower())
                for c in candidates:
                    if c.predicate_embedding:
                        try:
                            _pemb = np.frombuffer(c.predicate_embedding, dtype=np.float32)
                            if _pemb.shape == _qpred_emb.shape:
                                c.score += float(np.dot(_pemb, _qpred_emb)) * 0.3
                        except Exception:
                            pass
                candidates.sort(key=lambda c: c.score, reverse=True)
            except Exception:
                pass

        # ---- Step 5: Cross-encoder reranking ----
        candidates = _rerank(query, candidates)

        # ---- Step 6: Ranking signal stack (plan #8) ----
        candidates = _apply_ranking_signals(candidates, query, qd)

        # ---- Step 7: Verification loop ----
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

        # ---- Step 8: Cross-encoder relevance gate ----
        relevance = _relevance_gate(query, best)
        if relevance < RELEVANCE_GATE_THRESHOLD:
            return _refuse("relevance_below_threshold")

        # ---- Step 9: Negation check (for yes/no queries) ----
        if _is_yesno_query(query, qd.wh_word):
            if best.edge_negated:
                return ReconstructionResult(
                    answer="No", return_field="episodic",
                    edge_ids=[best.edge_id], grounding=[best.source_text],
                )
            return ReconstructionResult(
                answer="Yes", return_field="episodic",
                edge_ids=[best.edge_id], grounding=[best.source_text],
            )

        # ---- Step 10: Mood check (factual queries skip conditional edges) ----
        if not _is_conditional_query(query) and best.edge_mood == "conditional":
            for c in verified[1:]:
                if c.edge_mood != "conditional":
                    best = c
                    break

        # ---- Step 11: Tier 4 cluster expansion for context (plan #13) ----
        # (Expands context but doesn't change the answer — for grounding)
        context_edges = _expand_cluster(conn, user_id, best)
        extra_grounding = [c.source_text for c in context_edges[:3] if c.source_text]

        # ---- Step 12: Answer extraction via return_field routing ----
        answer = _extract_answer(best, qd, query)

        # ---- Step 13: PQ write-back ----
        if answer:
            _write_back_pq(conn, user_id, best.edge_id, query, answer)

        grounding = [best.source_text] + extra_grounding if extra_grounding else [best.source_text]

        return ReconstructionResult(
            answer=answer,
            return_field=qd.return_field,
            edge_ids=[best.edge_id],
            grounding=grounding,
        )
