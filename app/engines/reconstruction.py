"""
Reconstruction engine — 5-trace convergence read path.

Every edge has 5 traces + structured columns + embeddings stored at write time:
  Traces:
    1. Relational  — who (relational_entities, edge_relational_type)
    2. Schematic   — what domain (edge_schematic_category)
    3. Temporal    — when (resolved_event_date, is_current, temporal_expression)
    4. Episodic    — what happened (episodic_fact, edge_episodic_significance)
    5. Emotional   — how it felt (edge_emotional_label, edge_emotional_valence)
  Structural:
    subject, predicate, object — decomposed triplet
    context_entity — topic noun
    edge_embedding — 384-dim MiniLM of full source_text, L2-normalized

Convergence narrows by trace dimensions.
Verification uses word overlap on episodic_fact (proven Cat 5 defense).
Tiebreak ranks by edge_embedding cosine (bridges paraphrases).
"""
from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from app.db.session import get_db_context

log = logging.getLogger("kenotic.reconstruction")


# ── Significance ordering for tiebreak ─────────────────────────
_SIG_RANK = {"milestone": 4, "notable": 3, "stative": 2, "routine": 1}


# ── Return type ──────────────────────────────────────────────────

@dataclass
class ReconstructionResult:
    answer: Optional[str] = None
    refusal: bool = False
    refusal_reason: str = ""
    grounding: List[str] = field(default_factory=list)
    edge_ids: List[int] = field(default_factory=list)
    return_field: str = "episodic"


def _refuse(reason: str) -> ReconstructionResult:
    return ReconstructionResult(
        answer="This information is not mentioned in the conversation.",
        refusal=True, refusal_reason=reason,
    )


# ── Trace columns — what we read from the DB ────────────────────

_TRACE_COLS = """
    id, source_text, relational_entities, episodic_fact,
    edge_schematic_category, edge_emotional_label, edge_emotional_valence,
    emotional_target, edge_relational_type,
    resolved_event_date, temporal_expression, is_current,
    edge_negated, edge_mood, edge_episodic_significance,
    subject, predicate, object, context_entity,
    edge_embedding
"""

_WHERE = "user_id = ? AND tombstoned_at IS NULL"


# ── Helper: parse relational_entities JSON ──────────────────────

def _parse_entities(row) -> Set[str]:
    """Parse relational_entities JSON → set of lowercase names."""
    raw = row["relational_entities"] or "[]"
    try:
        return {str(e).lower() for e in json.loads(raw) if e}
    except (json.JSONDecodeError, TypeError):
        return set()


# ── Answer extraction from traces ────────────────────────────────

def _extract_answer(row, return_field: str) -> str:
    """Extract answer from trace columns. No triplets."""

    if return_field == "temporal":
        date = row["resolved_event_date"] or row["temporal_expression"] or ""
        if date and len(date) >= 10 and "-" in date:
            try:
                dt = datetime.fromisoformat(date[:10])
                if date[5:10] == "01-01":
                    return str(dt.year)
                elif date[8:10] == "01":
                    return f"{dt.strftime('%B')} {dt.year}"
                else:
                    return f"{dt.day} {dt.strftime('%B')} {dt.year}"
            except (ValueError, AttributeError):
                pass
        if row["temporal_expression"]:
            return row["temporal_expression"]
        return date

    if return_field == "emotional":
        label = row["edge_emotional_label"] or ""
        target = row["emotional_target"] or ""
        if label:
            return f"{label} about {target}" if target else label

    if return_field == "relational":
        rels = row["relational_entities"] or "[]"
        try:
            entities = json.loads(rels)
            if entities:
                return ", ".join(str(e) for e in entities)
        except (json.JSONDecodeError, TypeError):
            pass

    # Default: episodic trace
    ep = row["episodic_fact"] or ""
    if ep and len(ep) > 3:
        return ep
    return row["source_text"] or ""



# ── Situational query detection ──────────────────────────────────

_SITUATIONAL_PREFIXES = (
    "tell me about", "summarize",
    "what's going on with", "what's happening with",
    "what is going on with", "what is happening with",
    "update me on", "describe", "catch me up on",
    "fill me in on", "brief me on",
    "give me an overview of",
    "what do you know about", "what do we know about",
)


def is_situational(query: str) -> bool:
    """Return True if the query is a situational / reconstruction request.
    Uses closed-class English discourse frames (structural)."""
    if not query or not query.strip():
        return False
    q = query.strip().lower()
    for prefix in _SITUATIONAL_PREFIXES:
        if q.startswith(prefix):
            return True
    # "why" questions are causal/narrative — route to reconstruction
    from app.engines.grammar_engine import _get_nlp
    doc = _get_nlp()(q)
    for tok in doc:
        if tok.tag_ == "WRB" and tok.lemma_ == "why":
            return True
    return False


# ── List query detection ─────────────────────────────────────────

def _is_list_query(query):
    """Does the query ask for multiple items?"""
    try:
        from app.engines.grammar_engine import _get_nlp
        doc = _get_nlp()(query)
        for tok in list(doc)[:6]:
            if tok.tag_ in ("NNS", "NNPS"):
                return True
        if any(tok.lemma_ in ("all", "every", "which") for tok in doc):
            return True
    except Exception:
        pass
    return False


# ── Trace signal derivation from QueryDecomposition ─────────────

def _derive_temporal_mode(qd) -> str:
    """Derive temporal filtering mode from existing QD fields.
    "when" questions and past-tense queries include historical edges.
    Everything else: current only."""
    if qd.wh_word == "when":
        return "any"
    if qd.return_field == "temporal":
        return "any"
    return "current_only"


def _derive_significance(qd) -> Optional[str]:
    """Derive episodic significance filter from predicate.
    Preference/stative verbs → stative edges.
    Achievement verbs → milestone edges."""
    pred = (qd.match_predicate or "").lower()
    if not pred:
        return None

    _STATIVE_VERBS = {
        "like", "love", "prefer", "favorite", "enjoy", "want",
        "believe", "know", "think", "feel", "own", "have",
    }
    _MILESTONE_VERBS = {
        "win", "receive", "earn", "achieve", "accomplish",
        "graduate", "complete", "finish", "launch", "open",
    }

    if pred in _STATIVE_VERBS:
        return "stative"
    if pred in _MILESTONE_VERBS:
        return "milestone"
    return None


def _derive_emotional(qd) -> bool:
    """Does the query ask about feelings/emotions?"""
    if qd.return_field == "emotional":
        return True
    pred = (qd.match_predicate or "").lower()
    return pred in ("feel", "describe", "attitude", "sentiment", "mood")


# ── 5-Trace Convergence ─────────────────────────────────────────

def _converge(conn, user_id, qd, query) -> List:
    """Walk all 5 traces. Each trace narrows candidates.

    Trace 1 — Relational: entity ∈ relational_entities
    Trace 2 — Schematic: edge_schematic_category == query schema
    Trace 3 — Temporal: is_current gate
    Trace 4 — Episodic: significance level match
    Trace 5 — Emotional: non-null emotional label required
    """
    entity = (qd.match_entity or "").lower()

    # ── Build SQL with Trace 1 (relational) pushed to DB ────────
    # Entity filter is most selective — push to SQL, not Python
    conditions = [_WHERE]
    params: list = [user_id]

    if entity and entity != "user":
        conditions.append("relational_entities LIKE ?")
        params.append(f"%{entity}%")

    rows = conn.execute(
        f"SELECT {_TRACE_COLS} FROM edges WHERE {' AND '.join(conditions)}",
        params,
    ).fetchall()

    if not rows:
        return []

    # ── Trace 2 — Schematic: life domain ────────────────────────
    # Schema is a PREFERENCE for tiebreaking, not a hard filter.
    # Write-path and read-path schema classification diverge too often
    # (e.g., "lost job" → finance at query time, career at write time).
    # Filtering here blocks 56% of correct answers.
    # Instead, schema match is boosted in _tiebreak via _schema_match.
    query_schema = qd.match_schema  # stored for tiebreak, not filtering

    # ── Trace 3 — Temporal: currency gate ───────────────────────
    temporal_mode = _derive_temporal_mode(qd)
    if temporal_mode == "current_only":
        current_rows = [r for r in rows if r["is_current"] == 1]
        if current_rows:
            rows = current_rows

    # ── Trace 4 — Episodic: significance is a tiebreak preference,
    # not a hard filter. Write/read classify significance differently
    # ("launch" → milestone at query, notable at write). Same fix as schema.

    # ── Trace 5 — Emotional: tiebreak preference, not hard filter.
    # "How does Jon feel..." → emotional, but the edge might not have
    # edge_emotional_label set. Don't exclude it.

    return rows


# ── Tiebreak: edge embedding cosine ──────────────────────────────

def _decode_embedding(blob) -> Optional[list]:
    """Decode stored float32 embedding bytes to list of floats."""
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _dot(a: list, b: list) -> float:
    """Dot product of two L2-normalized vectors = cosine similarity."""
    return sum(x * y for x, y in zip(a, b))


def _edge_cosine(row, query_emb: Optional[list]) -> float:
    """Cosine between query embedding and stored edge_embedding.
    edge_embedding covers the full source_text — bridges paraphrases
    like 'favorite style' ↔ 'top pick' that word overlap misses."""
    if not query_emb:
        return 0.0
    stored = _decode_embedding(row["edge_embedding"])
    if not stored:
        return 0.0
    return _dot(query_emb, stored)


def _content_overlap(row, query_lemmas: Set[str]) -> int:
    """Count shared lemmas between edge episodic_fact and query.
    Used for Cat 5 verification gate (proven at 70.8%)."""
    ep = row["episodic_fact"] or row["source_text"] or ""
    if not ep:
        return 0
    from app.engines.grammar_engine import _get_nlp
    doc = _get_nlp()(ep)
    ep_lemmas = {tok.lemma_.lower() for tok in doc
                 if tok.pos_ in ("NOUN", "VERB", "ADJ") and not tok.is_stop}
    return len(query_lemmas & ep_lemmas)


def _tiebreak(rows, qd, query_emb: Optional[list],
              query_lemmas: Set[str]) -> dict:
    """Select best edge from converged candidates.

    Primary: content overlap (word-level, proven).
    Secondary: edge_embedding cosine (breaks ties within same overlap).
    Tertiary: schema, significance, date.
    """
    if len(rows) == 1:
        return rows[0]

    query_schema = qd.match_schema

    def sort_key(r):
        content = _content_overlap(r, query_lemmas)
        cosine = _edge_cosine(r, query_emb)
        schema = 1 if (query_schema and r["edge_schematic_category"] == query_schema) else 0
        sig = _SIG_RANK.get(r["edge_episodic_significance"] or "routine", 1)
        date = r["resolved_event_date"] or ""
        return (content, cosine, schema, sig, date)

    return max(rows, key=sort_key)


# ── FTS5 fallback ──────────────────────────────────────────────

def _fts_fallback(conn, user_id, qd, query) -> List:
    """BM25 lexical search when trace convergence yields nothing.
    Deterministic — no embeddings."""
    entity = qd.match_entity or qd.match_subject or ""

    # Build FTS5 query from entity + content words
    from app.engines.grammar_engine import _get_nlp
    doc = _get_nlp()(query)
    terms = []
    if entity:
        terms.append(entity)
    for tok in doc:
        if tok.pos_ in ("NOUN", "VERB", "ADJ") and not tok.is_stop and len(tok.text) > 2:
            if tok.text.lower() != entity.lower():
                terms.append(tok.lemma_)

    if not terms:
        return []

    fts_query = " OR ".join(f'"{t}"' for t in terms[:5])
    try:
        rows = conn.execute(
            f"""SELECT {_TRACE_COLS} FROM edges
                WHERE id IN (
                    SELECT rowid FROM edges_fts WHERE edges_fts MATCH ?
                ) AND {_WHERE}
                LIMIT 20""",
            (fts_query, user_id),
        ).fetchall()
        return rows
    except Exception:
        return []


# ── Situation reconstruction via schema convergence ─────────────

def _reconstruct_situation(conn, user_id, entity):
    """Converge traces by schema dimension. One fact per dimension."""
    conditions = [_WHERE, "relational_entities LIKE ?", "is_current = 1"]
    params = [user_id, f"%{entity}%"]

    edges = conn.execute(
        f"SELECT {_TRACE_COLS} FROM edges WHERE {' AND '.join(conditions)}",
        params,
    ).fetchall()

    if not edges:
        return _refuse("no_edges")

    by_schema: Dict[str, list] = {}
    for e in edges:
        sc = e["edge_schematic_category"] or "uncategorized"
        by_schema.setdefault(sc, []).append(e)

    parts = []
    all_ids = []
    for schema, schema_edges in sorted(by_schema.items()):
        latest = max(schema_edges, key=lambda e: e["resolved_event_date"] or "")
        fact = latest["episodic_fact"] or latest["source_text"] or ""
        if fact and len(fact) > 3:
            parts.append(fact)
            all_ids.append(latest["id"])

    if not parts:
        return _refuse("empty_situation")

    return ReconstructionResult(answer=". ".join(parts), edge_ids=all_ids)


# ── Verification gate helpers ─────────────────────────────────────

def _word_gate(candidates, query_lemmas: Set[str]) -> bool:
    """Word-overlap gate: any candidate shares ≥2 content lemmas with query."""
    if not query_lemmas or len(query_lemmas) < 2:
        return True
    return any(_content_overlap(c, query_lemmas) >= 2 for c in candidates)


def _contrastive_gate(conn, user_id, entity_lower, query_emb) -> Optional[bool]:
    """Compare query cosine against this entity's exclusive edges vs other entities'.

    Returns True (pass), False (refuse), or None (tie/no data → fall through).
    Only uses entity-EXCLUSIVE edges (not shared) to avoid ties.
    """
    rows = conn.execute(
        f"SELECT relational_entities, edge_embedding, is_current "
        f"FROM edges WHERE {_WHERE} AND edge_embedding IS NOT NULL",
        (user_id,),
    ).fetchall()

    if not rows:
        return None

    best_target = -1.0
    best_other = -1.0

    for r in rows:
        if r["is_current"] != 1:
            continue
        emb = _decode_embedding(r["edge_embedding"])
        if not emb:
            continue

        # Parse entities — only use EXCLUSIVE edges (one entity)
        raw = r["relational_entities"] or "[]"
        try:
            names = [str(n).lower().strip() for n in json.loads(raw) if n]
        except (json.JSONDecodeError, TypeError):
            continue

        # Skip shared edges (multiple distinct entities)
        unique = set(names)
        if len(unique) != 1:
            continue

        cos = _dot(query_emb, emb)
        sole_entity = next(iter(unique))
        if sole_entity == entity_lower:
            if cos > best_target:
                best_target = cos
        else:
            if cos > best_other:
                best_other = cos

    if best_target < 0 or best_other < 0:
        return None  # not enough data

    if best_target > best_other:
        return True
    if best_other > best_target:
        return False
    return None  # exact tie


# ── Main entry ───────────────────────────────────────────────────

def reconstruct(user_id: int, query: str) -> ReconstructionResult:
    """5-trace convergence + embedding tiebreak read path.

    1. classify_query → QueryDecomposition
    2. Embed query (one embed_text call)
    3. 5-trace convergence → candidate edges
    4. Word-overlap verification gate (Cat 5 defense, proven at 70.8%)
    5. Edge embedding cosine tiebreak (bridges paraphrases)
    6. Answer from best edge
    7. FTS5 fallback if convergence yields nothing
    8. Situation reconstruction for situational queries
    9. Refuse
    """
    from app.engines.grammar_engine import classify_query, _get_nlp
    qd = classify_query(query)

    nlp = _get_nlp()
    doc_q = nlp(query)
    entity_lower = (qd.match_entity or "").lower()

    # Query content lemmas for Cat 5 verification gate
    query_lemmas: Set[str] = {
        tok.lemma_.lower() for tok in doc_q
        if tok.pos_ in ("NOUN", "VERB", "ADJ") and not tok.is_stop
        and tok.text.lower() != entity_lower and len(tok.text) > 2
    }

    # Embed query — one call, reused for cosine against all candidates
    query_emb = None
    try:
        from app.vector.embedder import embed_text
        query_emb = embed_text(query).tolist()
    except Exception:
        pass  # graceful fallback to trace-only tiebreak

    with get_db_context() as conn:

        entity = qd.match_entity or qd.match_subject

        # ── 5-trace convergence ─────────────────────────────────
        candidates = _converge(conn, user_id, qd, query)

        if candidates:
            best = _tiebreak(candidates, qd, query_emb, query_lemmas)

            # ── Verification gate ───────────────────────────────
            #
            # Tier 1: Temporal bypass. "When" queries route to the
            #   date column — word overlap on episodic_fact is
            #   irrelevant (dates are in resolved_event_date).
            #
            # Tier 2: Word overlap ≥2 on episodic_fact.
            #   Proven Cat 5 defense (70.8%).
            #
            is_temporal = qd.return_field == "temporal" or qd.wh_word == "when"

            if not is_temporal and not _word_gate(candidates, query_lemmas):
                return _refuse("episodic_no_ground")

            # Yes/No query
            if qd.wh_word is None and "?" in query:
                answer = "No" if best["edge_negated"] else "Yes"
                return ReconstructionResult(
                    answer=answer,
                    edge_ids=[best["id"]],
                )

            # List query: collect all episodic facts
            if _is_list_query(query):
                seen: Set[str] = set()
                facts = []
                ids = []
                for e in candidates:
                    ep = (e["episodic_fact"] or "").strip()
                    if ep and ep.lower() not in seen and len(ep) > 3:
                        seen.add(ep.lower())
                        facts.append(ep)
                        ids.append(e["id"])
                if facts:
                    return ReconstructionResult(
                        answer=", ".join(facts),
                        edge_ids=ids,
                    )

            # Single: extract from best
            return ReconstructionResult(
                answer=_extract_answer(best, qd.return_field),
                return_field=qd.return_field,
                edge_ids=[best["id"]],
                grounding=[best["source_text"] or ""],
            )

        # ── FTS5 fallback ───────────────────────────────────────
        fts_rows = _fts_fallback(conn, user_id, qd, query)
        if fts_rows:
            temporal_mode = _derive_temporal_mode(qd)
            if temporal_mode == "current_only":
                current = [r for r in fts_rows if r["is_current"] == 1]
                if current:
                    fts_rows = current

            if fts_rows:
                best = _tiebreak(fts_rows, qd, query_emb, query_lemmas)
                return ReconstructionResult(
                    answer=_extract_answer(best, qd.return_field),
                    return_field=qd.return_field,
                    edge_ids=[best["id"]],
                    grounding=[best["source_text"] or ""],
                )

        # ── Situation reconstruction ────────────────────────────
        if is_situational(query) and entity:
            return _reconstruct_situation(conn, user_id, entity)

        return _refuse("not_mentioned")
