"""
MINERVA 2 resonance model — 5-dimensional trace activation.

Every edge responds simultaneously to the probe. Each edge computes
its resonance across 5 trace dimensions. Similarity is cubed per
MINERVA 2 (Hintzman 1986): strong matches amplified, weak suppressed.
Dimensions are MULTIPLIED — multi-dimensional overlap is exponentially
stronger. No gates. No thresholds. No top-K.

The 5 dimensions:
  1. Episodic   — semantic content match (embedding cosine)
  2. Emotional  — affective resonance (label + valence)
  3. Temporal   — time alignment (date proximity + context)
  4. Relational — entity identity (subject + relational_entities)
  5. Schematic  — life domain relevance (category match)

Dimensions irrelevant to the query score 1.0 (neutral).
1.0^3 = 1.0. No penalty. No help. Only probed dimensions contribute.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Optional

log = logging.getLogger("kenotic.trace_convergence")


# ── Schema relatedness map ──────────────────────────────────────
# Related categories score 0.7 instead of 0.3
_SCHEMA_RELATED = {
    "career": {"finance", "education"},
    "finance": {"career"},
    "education": {"career"},
    "health": {"hobby"},
    "hobby": {"health", "experience"},
    "family": {"social"},
    "social": {"family"},
    "experience": {"hobby"},
    "emotional": {"social", "family"},
    "identity": set(),
}


def _episodic(query_emb: list, edge, decode_embedding, dot) -> float:
    """Dimension 1: Semantic content match.
    Max cosine across PQ embeddings + edge embedding."""
    cosines = []
    for col in ("pq_1_embedding", "pq_2_embedding",
                "pq_3_embedding", "pq_4_embedding"):
        emb = decode_embedding(edge[col] if col in edge.keys() else None)
        if emb:
            cosines.append(dot(query_emb, emb))
    edge_emb = decode_embedding(edge["edge_embedding"])
    if edge_emb:
        cosines.append(dot(query_emb, edge_emb))
    if not cosines:
        return 0.0
    return max(max(c, 0.0) for c in cosines)


def _emotional(qd, edge) -> float:
    """Dimension 2: Affective resonance.
    If query doesn't probe emotion → 1.0 (neutral).
    If query probes emotion → match label and valence."""
    is_emotional = qd.return_field == "emotional" or qd.wh_word == "how"
    if not is_emotional:
        return 1.0  # neutral — not probed

    label = (edge["edge_emotional_label"] or "").strip().lower()
    if not label:
        return 0.3  # query asks for emotion, edge has none

    # Edge has an emotional label — it carries affective data.
    # For now: 0.8 (present and plausibly relevant).
    # Future: compare probe emotion word to label via WordNet synonyms.
    return 0.8


def _temporal(qd, edge) -> float:
    """Dimension 3: Time alignment.
    If query doesn't probe time → 1.0 (neutral).
    If query probes time → score by data presence and context match."""
    is_temporal = qd.return_field == "temporal" or qd.wh_word == "when"
    if not is_temporal:
        return 1.0  # neutral — not probed

    date = (edge["resolved_event_date"] or "").strip()
    expr = (edge["temporal_expression"] or "").strip()

    if not date and not expr:
        return 0.1  # query asks for time, edge has no temporal data

    # Edge has temporal data. Score by context alignment.
    edge_ctx = (edge["edge_temporal_context"] or "present").lower()

    # "When" questions typically seek past events or specific dates.
    # Edges with resolved dates are more valuable.
    if date and len(date) >= 10:
        return 1.0  # has a resolved date — strong temporal signal
    if expr:
        return 0.7  # has expression but no resolved date

    return 0.5


def _relational(qd, edge) -> float:
    """Dimension 4: Entity identity.
    The WHO dimension. Wrong person → 0.0 → activation collapses."""
    entity = (qd.match_entity or qd.match_subject or "").lower()
    if not entity or entity == "user":
        return 1.0  # no specific entity probed — neutral

    subject = (edge["subject"] or "").lower()

    # Exact subject match
    if entity in subject or subject in entity:
        return 1.0

    # Entity in relational_entities (mentioned but not subject)
    rel = (edge["relational_entities"] or "").lower()
    if entity in rel:
        return 0.8

    # No match → 0.0. Cubed = 0. Activation collapses. Correct.
    return 0.0


def _schematic(qd, edge) -> float:
    """Dimension 5: Life domain relevance.
    If query domain unclear → 1.0 (neutral).
    If query domain known → score by category match."""
    query_schema = (getattr(qd, "match_schema", None) or "").lower()
    if not query_schema:
        return 1.0  # can't determine domain — neutral

    edge_schema = (edge["edge_schematic_category"] or "").lower()
    if not edge_schema or edge_schema == "uncategorized":
        return 0.5  # edge has no category — neutral-ish

    # Exact match
    if query_schema == edge_schema:
        return 1.0

    # Related categories
    related = _SCHEMA_RELATED.get(query_schema, set())
    if edge_schema in related:
        return 0.7

    # Unrelated
    return 0.3


def score_edge(
    query_emb: list,
    qd,
    edge,
    decode_embedding,
    dot,
) -> float:
    """MINERVA 2 activation for a single edge.

    activation = episodic³ × emotional³ × temporal³ × relational³ × schematic³

    Each dimension returns [0.0, 1.0]. Cubed. Multiplied.
    Multi-dimensional match = exponentially strong.
    Any dimension at 0 = total collapse (wrong entity).
    Irrelevant dimensions = 1.0 (neutral, no effect).
    """
    e = _episodic(query_emb, edge, decode_embedding, dot)
    m = _emotional(qd, edge)
    t = _temporal(qd, edge)
    r = _relational(qd, edge)
    s = _schematic(qd, edge)

    return (e ** 3) * (m ** 3) * (t ** 3) * (r ** 3) * (s ** 3)
