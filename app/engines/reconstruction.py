"""
Reconstruction engine — trace + PQ + verification.

No scores. No thresholds. No word overlap gates.

How it works:
  1. Embed the query
  2. Pull all edges for this entity
  3. Find the edge whose PQ embedding is closest to the query embedding
  4. Verify: does the entity match?
  5. Return the trace field that answers the question

The PQ embedding IS the matching signal. It was computed at write time
from the episodic trace. If it matches the query semantically, the edge
answers the question. If no PQ matches, refuse — the information isn't there.
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


# ── Embedding helpers ────────────────────────────────────────────

def _decode_embedding(blob) -> Optional[list]:
    """Decode stored float32 embedding bytes to list of floats."""
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _dot(a: list, b: list) -> float:
    """Dot product of two L2-normalized vectors = cosine similarity."""
    return sum(x * y for x, y in zip(a, b))


# ── Entity verification ─────────────────────────────────────────

def _predicate_coherent(row, query_verb: str) -> bool:
    """Does this edge's predicate relate to what the query asks?
    No query verb → pass (can't check). Lemma or WordNet synonym match."""
    if not query_verb:
        return True

    edge_pred = (row["predicate"] or "").replace("_", " ").lower()
    if not edge_pred:
        return True

    # Lemma match
    if query_verb in edge_pred or edge_pred in query_verb:
        return True

    # WordNet synonym match
    from nltk.corpus import wordnet as wn
    q_synsets = set(wn.synsets(query_verb, pos=wn.VERB))
    q_synsets |= set(wn.synsets(query_verb, pos=wn.NOUN))
    if not q_synsets:
        return False

    q_lemma_names = set()
    for ss in q_synsets:
        for lemma in ss.lemmas():
            q_lemma_names.add(lemma.name().replace("_", " ").lower())

    edge_parts = set(edge_pred.split())
    if q_lemma_names & edge_parts:
        return True

    # Reverse: edge predicate synsets contain query verb
    for ep in edge_parts:
        e_synsets = set(wn.synsets(ep, pos=wn.VERB))
        e_synsets |= set(wn.synsets(ep, pos=wn.NOUN))
        for ss in e_synsets:
            for lemma in ss.lemmas():
                if lemma.name().replace("_", " ").lower() == query_verb:
                    return True

    # Content fallback: query verb in episodic_fact text
    ep = (row["episodic_fact"] or "").lower()
    src = (row["source_text"] or "").lower()
    combined = f"{ep} {src}"
    combined_words = set(combined.split())
    combined_lemmas = set(combined_words)
    for w in combined_words:
        vl = wn.morphy(w, wn.VERB)
        if vl:
            combined_lemmas.add(vl)
    if query_verb in combined_lemmas:
        return True

    return False


def _content_matches(edge, query_content: set, entity_lower: str, nlp) -> bool:
    """Does PQ or episodic_fact share ≥2 content lemmas with query?
    Uses spaCy lemmatization only — no synonym expansion."""
    # Check PQs
    for col in ("pq_1", "pq_2", "pq_3", "pq_4"):
        pq = edge[col] if col in edge.keys() else None
        if not pq or len(pq) < 4:
            continue
        pq_words = {
            tok.lemma_.lower() for tok in nlp(pq)
            if tok.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not tok.is_stop
            and len(tok.text) > 2 and tok.text.lower() != entity_lower
        }
        if len(query_content & pq_words) >= 2:
            return True

    # Check episodic_fact
    ep = edge["episodic_fact"] or edge["source_text"] or ""
    if ep:
        ep_words = {
            tok.lemma_.lower() for tok in nlp(ep)
            if tok.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not tok.is_stop
            and len(tok.text) > 2 and tok.text.lower() != entity_lower
        }
        if len(query_content & ep_words) >= 2:
            return True

    return False


def _entity_matches(row, query_entity: str) -> bool:
    """Does this edge involve the query entity?"""
    if not query_entity:
        return True
    qe = query_entity.lower()
    subject = (row["subject"] or "").lower()
    if qe in subject or subject in qe:
        return True
    rel = (row["relational_entities"] or "").lower()
    if qe in rel:
        return True
    return False


# ── Answer extraction from traces ────────────────────────────────

def _extract_answer(row, return_field: str) -> str:
    """Extract answer from the appropriate trace column."""

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
    if not query or not query.strip():
        return False
    q = query.strip().lower()
    for prefix in _SITUATIONAL_PREFIXES:
        if q.startswith(prefix):
            return True
    return False


# ── Situation reconstruction ─────────────────────────────────────

_TRACE_COLS = """
    id, source_text, relational_entities, episodic_fact,
    edge_schematic_category, edge_emotional_label, edge_emotional_valence,
    emotional_target, edge_relational_type,
    resolved_event_date, temporal_expression, is_current,
    edge_negated, edge_mood, edge_episodic_significance,
    subject, predicate, object, context_entity,
    edge_embedding, pq_1_embedding,
    pq_1, pq_2, pq_3, pq_4
"""

_WHERE = "user_id = ? AND tombstoned_at IS NULL"


def _reconstruct_situation(conn, user_id, entity):
    """One fact per life domain for this entity."""
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


# ── Main entry ───────────────────────────────────────────────────

def reconstruct(user_id: int, query: str) -> ReconstructionResult:
    """Trace + PQ + verification. No scores.

    1. Classify query (entity, return_field)
    2. Embed query
    3. Pull all edges for entity
    4. Rank by PQ embedding cosine (semantic match)
    5. First edge where entity matches → answer
    6. No match → refuse
    """
    from app.engines.grammar_engine import classify_query, _get_nlp
    qd = classify_query(query)

    entity = qd.match_entity or qd.match_subject
    is_temporal = qd.return_field == "temporal" or qd.wh_word == "when"

    # Embed query
    query_emb = None
    try:
        from app.vector.embedder import embed_text
        query_emb = embed_text(query).tolist()
    except Exception:
        pass

    if not query_emb:
        return _refuse("no_embedding")

    with get_db_context() as conn:

        # ── Situational: schema-diverse reconstruction ─────────
        if is_situational(query) and entity:
            return _reconstruct_situation(conn, user_id, entity)

        # ── Pull all edges for this entity ─────────────────────
        conditions = [_WHERE]
        params: list = [user_id]

        entity_lower = (entity or "").lower()
        if entity_lower and entity_lower != "user":
            conditions.append("relational_entities LIKE ?")
            params.append(f"%{entity_lower}%")

        # Temporal: include historical. Others: prefer current.
        if not is_temporal:
            conditions.append("is_current = 1")

        rows = conn.execute(
            f"SELECT {_TRACE_COLS} FROM edges WHERE {' AND '.join(conditions)}",
            params,
        ).fetchall()

        # If current-only returned nothing, try all
        if not rows and not is_temporal:
            conditions_all = [_WHERE]
            params_all: list = [user_id]
            if entity_lower and entity_lower != "user":
                conditions_all.append("relational_entities LIKE ?")
                params_all.append(f"%{entity_lower}%")
            rows = conn.execute(
                f"SELECT {_TRACE_COLS} FROM edges WHERE {' AND '.join(conditions_all)}",
                params_all,
            ).fetchall()

        if not rows:
            return _refuse("no_edges")

        # ── Filter: for emotional queries, only edges with emotional data ──
        is_emotional = qd.return_field == "emotional"
        if is_emotional:
            emo_rows = [r for r in rows if r["edge_emotional_label"]]
            if emo_rows:
                rows = emo_rows

        # ── Rank by PQ embedding cosine ────────────────────────
        scored = []
        for row in rows:
            pq_emb = _decode_embedding(
                row["pq_1_embedding"] if "pq_1_embedding" in row.keys() else None
            )
            edge_emb = _decode_embedding(row["edge_embedding"])

            # Best of PQ cosine and edge cosine
            pq_cos = _dot(query_emb, pq_emb) if pq_emb else 0.0
            edge_cos = _dot(query_emb, edge_emb) if edge_emb else 0.0
            best_cos = max(pq_cos, edge_cos)

            scored.append((best_cos, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        # ── Verification: entity + PQ match ────────────────────
        # Walk ranked list. First edge where:
        #   1. Entity matches
        #   2. PQ shares ≥ 2 content words with query (binary: matches or doesn't)
        # All fail → refuse.
        query_verb = (qd.match_predicate or "").lower() or None

        nlp = _get_nlp()
        query_content = {
            tok.lemma_.lower() for tok in nlp(query)
            if tok.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not tok.is_stop
            and len(tok.text) > 2 and tok.text.lower() != entity_lower
        }

        for cos, edge in scored:
            if not _entity_matches(edge, entity or ""):
                continue
            # Skip predicate check for yes/no — just checking existence
            is_yesno = qd.wh_word is None and "?" in query
            if not is_yesno:
                if not _predicate_coherent(edge, query_verb or ""):
                    continue

            # Content verification (refusal gate) — skip for temporal
            if not is_temporal:
                content_ok = _content_matches(
                    edge, query_content, entity_lower, nlp,
                )
                if not content_ok:
                    continue

            # Yes/No query
            if qd.wh_word is None and "?" in query:
                answer = "No" if edge["edge_negated"] else "Yes"
                return ReconstructionResult(
                    answer=answer,
                    edge_ids=[edge["id"]],
                )

            return ReconstructionResult(
                answer=_extract_answer(edge, qd.return_field),
                return_field=qd.return_field,
                edge_ids=[edge["id"]],
                grounding=[edge["source_text"] or ""],
            )

        return _refuse("not_mentioned")
