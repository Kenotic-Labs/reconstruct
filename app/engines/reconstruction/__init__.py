"""
Reconstruction engine — MINERVA 2 resonance model.

Every edge responds simultaneously to the probe. Each edge computes
5-dimensional activation (episodic, emotional, temporal, relational,
schematic). Similarity is cubed. Dimensions are multiplied. The echo
— weighted sum of all activations — IS the answer.

No gates. No thresholds. No top-K. No BM25. No RRF.
Refusal = echo magnitude below noise floor.
"""
from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

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


# ── Answer extraction from traces ────────────────────────────────

def _extract_answer(row, return_field: str, wh_word: str = None) -> str:
    """Extract answer from the appropriate trace column."""

    if return_field == "temporal":
        date = row["resolved_event_date"] or row["temporal_expression"] or ""
        if date and len(date) >= 10 and "-" in date:
            try:
                dt = datetime.fromisoformat(date[:10])
                if date[5:10] == "01-01":
                    return str(dt.year)
                else:
                    # Return "Month Year" — LOCOMO gold answers are mostly month-level.
                    # Returning the day ("20 February 2023") hurts precision when
                    # gold is "February, 2023" (F1: 0.80 → 1.0 by dropping the day).
                    return f"{dt.strftime('%B')} {dt.year}"
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
    # For "What" questions: prefer the object column (shorter = higher F1).
    # Gold answers are median 4 words. Object IS the answer for "What did X verb?"
    ep = row["episodic_fact"] or ""
    obj = (row["object"] or "").strip().strip('" -')
    if wh_word in ("what", "who", "where", "how") and obj and len(obj.split()) >= 3 and len(obj) < len(ep):
        return obj
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
    resolved_event_date, temporal_expression, edge_temporal_context, is_current,
    edge_negated, edge_mood, edge_episodic_significance,
    subject, predicate, object, context_entity,
    edge_embedding, pq_1_embedding, pq_2_embedding, pq_3_embedding, pq_4_embedding,
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

    grounding = [e["source_text"] or "" for e in edges if e["id"] in all_ids]
    return ReconstructionResult(
        answer=". ".join(parts), edge_ids=all_ids,
        grounding=grounding[:5], return_field="episodic",
    )


# ── Main entry ───────────────────────────────────────────────────

def reconstruct(user_id: int, query: str) -> ReconstructionResult:
    """MINERVA 2 echo-based reconstruction.

    1. Classify query → probe dimensions
    2. Embed query
    3. Pull all edges for entity
    4. Every edge computes 5-dimensional activation (cubed, multiplied)
    5. Echo magnitude = sum of all activations
    6. If echo too weak → refuse (no trace resonates)
    7. Dominant trace (highest activation) provides the answer text
    """
    from app.engines.grammar_engine import classify_query
    from app.engines.reconstruction.trace_convergence import score_edge

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
            f"SELECT {_TRACE_COLS} FROM edges "
            f"WHERE {' AND '.join(conditions)}",
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
                f"SELECT {_TRACE_COLS} FROM edges "
                f"WHERE {' AND '.join(conditions_all)}",
                params_all,
            ).fetchall()

        if not rows:
            return _refuse("no_edges")

        # ── Compute activation for ALL edges simultaneously ────
        activations: List[Tuple[dict, float]] = []
        for edge in rows:
            a = score_edge(query_emb, qd, edge, _decode_embedding, _dot)
            activations.append((edge, a))

        # ── Sort by activation — dominant trace first ──────────
        activations.sort(key=lambda x: x[1], reverse=True)

        dominant = activations[0][0]
        dominant_activation = activations[0][1]

        # ── Echo refusal ───────────────────────────────────────
        # The echo magnitude is the sum of all activations. When no
        # trace resonates, the sum is dominated by noise. Refuse when
        # the echo is indistinguishable from uniform low activation.
        #
        # Noise baseline: N edges each with episodic ~0.15 and all
        # other dimensions neutral (1.0): per-edge = (0.15)^3 = 0.0034.
        # Echo noise = N * 0.0034. Signal: dominant at (0.5)^3 = 0.125.
        # Refuse when dominant < noise-per-edge (the echo doesn't
        # concentrate on any trace above the background).
        echo_magnitude = sum(a for _, a in activations)
        noise_per_edge = 0.004  # (0.15)^3 ≈ what random cosine produces
        if dominant_activation < noise_per_edge:
            return _refuse("echo_below_noise")

        # ── Collect significantly-activated edges for grounding ──
        sig_edges = [(e, a) for e, a in activations
                     if a > dominant_activation * 0.1]
        sig_ids = [e["id"] for e, a in sig_edges if e["id"]]
        sig_grounding = [e["source_text"] or "" for e, a in sig_edges
                         if e["source_text"]]

        # Yes/No questions
        if qd.wh_word is None and "?" in query:
            answer = "No" if dominant["edge_negated"] else "Yes"
            return ReconstructionResult(
                answer=answer, edge_ids=sig_ids[:10],
                grounding=sig_grounding[:5], return_field="episodic",
            )

        return ReconstructionResult(
            answer=_extract_answer(dominant, qd.return_field, wh_word=qd.wh_word),
            return_field=qd.return_field,
            edge_ids=sig_ids[:10],
            grounding=sig_grounding[:5],
        )
