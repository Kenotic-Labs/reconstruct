"""
Recon Lab — trace-only reconstruction engine.

Uses ONLY what the DB stores:
  - Relational trace (relational_entities) → find edges by entity
  - Episodic trace (episodic_fact, source_text) → content matching
  - Temporal trace (resolved_event_date, is_current) → time filtering
  - Schematic trace (edge_schematic_category) → domain matching
  - Emotional trace (edge_emotional_label) → emotional queries
  - PQs (pq_1..pq_4) → question-to-question matching
  - Embedding (edge_embedding) → cosine similarity
  - Verification → embedding gate (last check)

NO subject/predicate/object columns. NO SPO matching.

Usage:
    python recon_lab.py --db path/to/db --user-id 1 --query "What does Jon love?"
"""

from __future__ import annotations

import json
import struct
import logging
import argparse
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

log = logging.getLogger("recon_lab")


# ── Result type ─────────────────────────────────────────────────

@dataclass
class ReconResult:
    answer: Optional[str] = None
    refusal: bool = False
    refusal_reason: str = ""
    edge_ids: List[int] = field(default_factory=list)
    return_field: str = "episodic"
    trace_score: int = 0


def _refuse(reason: str) -> ReconResult:
    return ReconResult(
        answer="This information is not mentioned in the conversation.",
        refusal=True, refusal_reason=reason,
    )


# ── Trace columns — ONLY traces, NO SPO ─────────────────────────

_TRACE_COLS = """
    id, source_text, relational_entities, episodic_fact,
    edge_schematic_category, edge_emotional_label, edge_emotional_valence,
    emotional_target, edge_relational_type,
    resolved_event_date, temporal_expression, is_current,
    edge_negated, edge_mood, edge_episodic_significance,
    context_entity, edge_embedding,
    pq_1, pq_2, pq_3, pq_4,
    pq_1_embedding, pq_2_embedding, pq_3_embedding, pq_4_embedding
"""

_WHERE = "user_id = ? AND tombstoned_at IS NULL"


# ── Helpers ──────────────────────────────────────────────────────

def _parse_entities(row) -> Set[str]:
    raw = row["relational_entities"] or "[]"
    try:
        return {str(e).lower() for e in json.loads(raw) if e}
    except (json.JSONDecodeError, TypeError):
        return set()


def _decode_embedding(blob) -> Optional[list]:
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _dot(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── Answer extraction from traces ──────────────────────────────

def _extract_answer(row, return_field: str) -> str:
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


# ── Query analysis ──────────────────────────────────────────────

def _get_nlp():
    from app.engines.grammar_engine import _get_nlp
    return _get_nlp()


def _analyze_query(query: str):
    """Extract everything we need from the query using spaCy.
    Returns dict with: entity, lemmas, nouns, wh_word, return_field, query_emb, doc."""
    from app.engines.grammar_engine import classify_query
    qd = classify_query(query)

    nlp = _get_nlp()
    doc = nlp(query)
    raw_entity = (qd.match_entity or qd.match_subject or "").lower()

    # Copula fix: classify_query may return full NP ("jon favorite style of dance")
    # for "What is X's Y?" questions. Extract just the PROPN entity.
    entity = raw_entity
    if entity and " " in entity:
        entity_doc = nlp(entity)
        propns = [tok.text.lower() for tok in entity_doc if tok.pos_ == "PROPN"]
        if propns:
            entity = propns[0]

    lemmas = {
        tok.lemma_.lower() for tok in doc
        if tok.pos_ in ("NOUN", "VERB", "ADJ") and not tok.is_stop
        and tok.text.lower() != entity and len(tok.text) > 2
    }

    nouns = {
        tok.lemma_.lower() for tok in doc
        if tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop
        and tok.text.lower() != entity and len(tok.text) > 2
    }

    query_emb = None
    try:
        from app.vector.embedder import embed_text
        query_emb = embed_text(query).tolist()
    except Exception:
        pass

    return {
        "entity": entity,
        "lemmas": lemmas,
        "nouns": nouns,
        "wh_word": qd.wh_word,
        "return_field": qd.return_field,
        "schema": qd.match_schema or "",
        "query_emb": query_emb,
        "doc": doc,
        "qd": qd,
    }


# ── Situational / list detection ────────────────────────────────

_SITUATIONAL_PREFIXES = (
    "tell me about", "summarize", "what's going on with",
    "what's happening with", "what is going on with",
    "what is happening with", "update me on", "describe",
    "catch me up on", "fill me in on", "brief me on",
    "give me an overview of", "what do you know about",
)


def _is_situational(query: str) -> bool:
    q = query.strip().lower()
    for prefix in _SITUATIONAL_PREFIXES:
        if q.startswith(prefix):
            return True
    doc = _get_nlp()(q)
    for tok in doc:
        if tok.tag_ == "WRB" and tok.lemma_ == "why":
            return True
    return False


def _is_list_query(query: str) -> bool:
    doc = _get_nlp()(query)
    for tok in list(doc)[:6]:
        if tok.tag_ in ("NNS", "NNPS"):
            return True
    if any(tok.lemma_ in ("all", "every", "which") for tok in doc):
        return True
    return False


# ── Temporal mode ───────────────────────────────────────────────

def _temporal_mode(qa) -> str:
    if qa["wh_word"] == "when":
        return "any"
    if qa["return_field"] == "temporal":
        return "any"
    return "current_only"


# ── Step 1: Find edges by relational trace ──────────────────────

def _find_by_entity(conn, user_id: int, entity: str) -> List:
    """Relational trace: entity ∈ relational_entities."""
    if not entity or entity == "user":
        rows = conn.execute(
            f"SELECT {_TRACE_COLS} FROM edges WHERE {_WHERE}",
            (user_id,),
        ).fetchall()
        return rows

    rows = conn.execute(
        f"SELECT {_TRACE_COLS} FROM edges WHERE {_WHERE} AND relational_entities LIKE ?",
        (user_id, f"%{entity}%"),
    ).fetchall()
    return rows


# ── Step 2: Filter by temporal trace ────────────────────────────

def _filter_temporal(rows: List, mode: str) -> List:
    if mode == "any":
        return rows
    current = [r for r in rows if r["is_current"] == 1]
    return current if current else rows


# ── Step 3: Score by ALL traces ─────────────────────────────────

def _cosine(row, query_emb) -> float:
    if not query_emb:
        return 0.0
    stored = _decode_embedding(row["edge_embedding"])
    if not stored:
        return 0.0
    return _dot(query_emb, stored)


def _content_overlap(row, query_lemmas: Set[str]) -> int:
    """Episodic trace: lemma overlap between query and episodic_fact."""
    ep = row["episodic_fact"] or row["source_text"] or ""
    if not ep:
        return 0
    nlp = _get_nlp()
    doc = nlp(ep)
    ep_lemmas = {tok.lemma_.lower() for tok in doc
                 if tok.pos_ in ("NOUN", "VERB", "ADJ") and not tok.is_stop}
    return len(query_lemmas & ep_lemmas)


def _pq_cosine(row, query_emb: Optional[list]) -> float:
    """PQ trace: best cosine between query embedding and stored PQ embeddings.
    PQ embeddings are pre-computed at write time — same as edge_embedding.
    Returns the highest cosine across all 4 PQs."""
    if not query_emb:
        return 0.0
    best = 0.0
    for col in ("pq_1_embedding", "pq_2_embedding", "pq_3_embedding", "pq_4_embedding"):
        blob = row[col] if col in row.keys() else None
        stored = _decode_embedding(blob)
        if not stored:
            continue
        cosine = _dot(query_emb, stored)
        if cosine > best:
            best = cosine
    return best


def _score_edge(row, qa: dict) -> int:
    """Score an edge using ALL traces. No gates — traces add up.

    PQ cosine > 0.85 → +4 (question semantically matches stored PQ — strongest)
    PQ cosine > 0.7  → +3
    PQ cosine > 0.5  → +1
    Content ≥ 2      → +2 (query words in episodic_fact)
    Content = 1      → +1
    Schema match      → +1
    Context entity    → +1
    Edge cosine > 0.7 → +1
    Notable/milestone → +1
    """
    score = 0

    # PQ trace — embedding cosine (not word overlap)
    pq_cos = _pq_cosine(row, qa["query_emb"])
    if pq_cos > 0.85:
        score += 4
    elif pq_cos > 0.7:
        score += 3
    elif pq_cos > 0.5:
        score += 1

    # Episodic trace
    content = _content_overlap(row, qa["lemmas"])
    if content >= 2:
        score += 2
    elif content == 1:
        score += 1

    # Schematic trace
    if qa["schema"] and row["edge_schematic_category"] == qa["schema"]:
        score += 1

    # Context entity
    ctx = (row["context_entity"] or "").lower()
    if ctx and ctx in qa["nouns"]:
        score += 1

    # Embedding cosine
    if _cosine(row, qa["query_emb"]) > 0.7:
        score += 1

    # Significance
    sig = row["edge_episodic_significance"] or "routine"
    if sig in ("notable", "milestone"):
        score += 1

    return score


# ── Step 4: Verification gate — embedding cosine ────────────────

def _verify_edge(row, query_emb: Optional[list], query_lemmas: Set[str],
                  entity: str) -> bool:
    """Last gate: does the picked edge actually answer THIS question?

    Two checks — both must pass:
      1. PQ cosine > 0.65 — the query semantically matches a stored PQ.
         This catches topic-level mismatches.
      2. PQ word overlap ≥ 1 content word — at least one non-entity content
         word from the query appears in a PQ. "ballet" must be in a PQ
         for a ballet question to pass. This catches Cat 5 entity-swaps
         where the topic matches but the specific claim doesn't.

    Temporal bypass: date queries don't need this check.
    """
    # Check 1: PQ cosine
    best_cosine = _pq_cosine(row, query_emb)
    if best_cosine < 0.65:
        return False

    # Check 2: PQ content word overlap (Cat 5 defense)
    nlp = _get_nlp()
    entity_words = set(entity.lower().split())

    for col in ("pq_1", "pq_2", "pq_3", "pq_4"):
        pq = row[col] if col in row.keys() else None
        if not pq or len(pq) < 4:
            continue
        pq_lemmas = {tok.lemma_.lower() for tok in nlp(pq)
                     if tok.pos_ in ("NOUN", "VERB", "ADJ")
                     and not tok.is_stop and len(tok.text) > 2
                     and tok.text.lower() not in entity_words}
        overlap = query_lemmas & pq_lemmas
        if len(overlap) >= 1:
            return True

    return False


# ── Step 5: Situation reconstruction ────────────────────────────

def _reconstruct_situation(conn, user_id: int, entity: str) -> ReconResult:
    rows = conn.execute(
        f"SELECT {_TRACE_COLS} FROM edges WHERE {_WHERE} AND relational_entities LIKE ? AND is_current = 1",
        (user_id, f"%{entity}%"),
    ).fetchall()
    if not rows:
        return _refuse("no_edges")

    by_schema: Dict[str, list] = {}
    for e in rows:
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
    return ReconResult(answer=". ".join(parts), edge_ids=all_ids)


# ── Step 6: FTS5 fallback ──────────────────────────────────────

def _fts_fallback(conn, user_id: int, qa: dict, query: str) -> List:
    entity = qa["entity"]
    doc = qa["doc"]
    terms = []
    if entity:
        terms.append(entity)
    for tok in doc:
        if tok.pos_ in ("NOUN", "VERB", "ADJ") and not tok.is_stop and len(tok.text) > 2:
            if tok.text.lower() != entity:
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


# ── Main: reconstruct ──────────────────────────────────────────

def reconstruct(conn, user_id: int, query: str) -> ReconResult:
    """Trace-only reconstruction. No SPO.

    1. Analyze query → entity, lemmas, wh_word, return_field, embedding
    2. Find edges by relational trace (entity)
    3. Filter by temporal trace (is_current)
    4. Score ALL edges by all traces (PQ + episodic + schema + cosine + context)
    5. Best edge → extract answer from trace column
    6. Verify via embedding cosine (Cat 5 defense)
    7. FTS5 fallback if no trace candidates
    8. Situation reconstruction for situational queries
    9. Refuse
    """
    qa = _analyze_query(query)
    entity = qa["entity"]

    result = None

    # ── Step 1: relational trace → candidate edges ──────────
    candidates = _find_by_entity(conn, user_id, entity)

    if candidates:
        # ── Step 2: temporal filter ──────────────────────────
        mode = _temporal_mode(qa)
        candidates = _filter_temporal(candidates, mode)

        # ── Step 3: score and rank ──────────────────────────
        ranked = sorted(candidates, key=lambda r: (
            _score_edge(r, qa),
            _pq_cosine(r, qa["query_emb"]),
            _cosine(r, qa["query_emb"]),
        ), reverse=True)

        # Take best
        best = ranked[0] if ranked else None

        if best and _score_edge(best, qa) > 0:
            # Yes/No query
            if qa["wh_word"] is None and "?" in query:
                answer = "No" if best["edge_negated"] else "Yes"
                result = ReconResult(
                    answer=answer,
                    edge_ids=[best["id"]],
                    trace_score=_score_edge(best, qa),
                )

            # List query
            elif _is_list_query(query):
                seen: Set[str] = set()
                facts = []
                ids = []
                for e in ranked:
                    if _score_edge(e, qa) == 0:
                        break
                    ep = (e["episodic_fact"] or "").strip()
                    if ep and ep.lower() not in seen and len(ep) > 3:
                        seen.add(ep.lower())
                        facts.append(ep)
                        ids.append(e["id"])
                if facts:
                    result = ReconResult(
                        answer=", ".join(facts),
                        edge_ids=ids,
                    )

            # Single answer
            if result is None:
                result = ReconResult(
                    answer=_extract_answer(best, qa["return_field"]),
                    return_field=qa["return_field"],
                    edge_ids=[best["id"]],
                    trace_score=_score_edge(best, qa),
                )

    # ── FTS5 fallback ──────────────────────────────────────
    if result is None:
        fts_rows = _fts_fallback(conn, user_id, qa, query)
        if fts_rows:
            fts_rows = _filter_temporal(fts_rows, _temporal_mode(qa))
            if fts_rows:
                fts_ranked = sorted(fts_rows, key=lambda r: (
                    _score_edge(r, qa),
                    _pq_cosine(r, qa["query_emb"]),
                    _cosine(r, qa["query_emb"]),
                ), reverse=True)
                best = fts_ranked[0]
                if _score_edge(best, qa) > 0:
                    result = ReconResult(
                        answer=_extract_answer(best, qa["return_field"]),
                        return_field=qa["return_field"],
                        edge_ids=[best["id"]],
                        trace_score=_score_edge(best, qa),
                    )

    # ── Situation reconstruction ────────────────────────────
    if result is None and _is_situational(query) and entity:
        result = _reconstruct_situation(conn, user_id, entity)

    # ── LAST GATE: embedding verification ───────────────────
    # Temporal bypass: dates verified by temporal trace, not embedding.
    is_temporal = qa["return_field"] == "temporal" or qa["wh_word"] == "when"

    if result and not result.refusal and result.answer:
        if is_temporal:
            return result
        if result.edge_ids:
            picked = next(
                (c for c in (candidates or []) if c["id"] == result.edge_ids[0]),
                None,
            )
            if picked and not _verify_edge(picked, qa["query_emb"],
                                                qa["lemmas"], qa["entity"]):
                return _refuse("embedding_no_match")
        return result

    return _refuse("not_mentioned")


# ── CLI entry ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    result = reconstruct(conn, args.user_id, args.query)
    print(f"Answer:  {result.answer}")
    print(f"Refusal: {result.refusal} ({result.refusal_reason})")
    print(f"Edges:   {result.edge_ids}")
    print(f"Score:   {result.trace_score}")
    print(f"Field:   {result.return_field}")


if __name__ == "__main__":
    main()
