"""
MINERVA 2 resonance model — 5-dimensional trace activation.

Every edge responds simultaneously to the probe. Each edge computes
its resonance across 5 trace dimensions. Similarity is cubed per
MINERVA 2 (Hintzman 1986): strong matches amplified, weak suppressed.
Dimensions are MULTIPLIED — multi-dimensional overlap is exponentially
stronger. No gates. No thresholds. No top-K.

The 5 dimensions:
  1. Episodic   — semantic content match (embedding cosine)
  2. Emotional  — affective resonance (WordNet synonym match on labels)
  3. Temporal   — time alignment (context direction + date presence)
  4. Relational — entity identity (subject + relational_entities)
  5. Schematic  — life domain relevance (category match)

Dimensions irrelevant to the query score 1.0 (neutral).
1.0^3 = 1.0. No penalty. No help. Only probed dimensions contribute.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

log = logging.getLogger("kenotic.trace_convergence")


# ── Schema relatedness map ──────────────────────────────────────
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


# ── Dimension 1: Episodic ──────────────────────────────────────

def _episodic(query_emb: list, edge, decode_embedding, dot) -> float:
    """Semantic content match. Max cosine across PQ + edge embeddings."""
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


# ── Dimension 2: Emotional ─────────────────────────────────────

def _emotional(qd, edge, query_text: str = "") -> float:
    """Affective resonance via WordNet synonym matching.
    Not probed → 1.0. Probed → match query emotion words against edge label."""
    is_emotional = qd.return_field == "emotional" or qd.wh_word == "how"
    if not is_emotional:
        return 1.0

    label = (edge["edge_emotional_label"] or "").strip().lower()
    if not label:
        return 0.3  # query asks for emotion, edge carries none

    if not query_text:
        return 0.8  # has emotion, can't compare (no query text)

    # Extract emotion words from query, compare against edge label
    try:
        from app.engines.grammar_engine import _is_emotion_word, _get_nlp
        nlp = _get_nlp()
        query_emotions = {
            tok.lemma_.lower() for tok in nlp(query_text)
            if tok.pos_ in ("ADJ", "NOUN", "VERB")
            and _is_emotion_word(tok.lemma_.lower(), tok.pos_)
        }
        if not query_emotions:
            return 0.8  # query probes emotion but no specific emotion word

        # Direct match
        if label in query_emotions:
            return 1.0

        # WordNet synonym match
        from nltk.corpus import wordnet as wn
        label_synsets = set()
        for pos in (wn.ADJ, wn.NOUN, wn.VERB):
            label_synsets.update(wn.synsets(label, pos=pos)[:2])
        for qw in query_emotions:
            for pos in (wn.ADJ, wn.NOUN, wn.VERB):
                if set(wn.synsets(qw, pos=pos)[:2]) & label_synsets:
                    return 0.9  # synonym match

        return 0.5  # has emotion but doesn't match query's specific emotion
    except Exception:
        return 0.8  # fallback: has emotion, assume plausible


# ── Dimension 3: Temporal ──────────────────────────────────────

def _temporal(qd, edge) -> float:
    """Time alignment via context direction + date presence.
    Not probed → 1.0. Probed → continuous scoring."""
    is_temporal = qd.return_field == "temporal" or qd.wh_word == "when"
    if not is_temporal:
        return 1.0

    date = (edge["resolved_event_date"] or "").strip()
    expr = (edge["temporal_expression"] or "").strip()
    edge_ctx = (edge["edge_temporal_context"] or "present").lower()

    if not date and not expr:
        return 0.1  # no temporal data at all

    # Base score: has some temporal data
    score = 0.5

    # Resolved date is strongest temporal signal
    if date and len(date) >= 10:
        score = 0.8

    # Context alignment: "when" questions typically seek past events.
    # Past-context edges are more likely to be the answer.
    if edge_ctx == "past":
        score = min(score + 0.2, 1.0)
    elif edge_ctx == "future":
        # Future events less likely for "when did" but valid for "when will"
        score = max(score - 0.1, 0.3)

    return score


# ── Dimension 4: Relational ────────────────────────────────────

def _relational(qd, edge) -> float:
    """Entity identity. Wrong person → 0.0 → activation collapses."""
    entity = (qd.match_entity or qd.match_subject or "").lower()
    if not entity or entity == "user":
        return 1.0

    subject = (edge["subject"] or "").lower()
    if entity in subject or subject in entity:
        return 1.0

    rel = (edge["relational_entities"] or "").lower()
    if entity in rel:
        return 0.8

    return 0.0  # cubed = 0, activation collapses


# ── Dimension 5: Schematic ─────────────────────────────────────

def _schematic(qd, edge) -> float:
    """Life domain relevance. Unknown query domain → 1.0 (neutral)."""
    query_schema = (getattr(qd, "match_schema", None) or "").lower()
    if not query_schema:
        return 1.0

    edge_schema = (edge["edge_schematic_category"] or "").lower()
    if not edge_schema or edge_schema == "uncategorized":
        return 0.5

    if query_schema == edge_schema:
        return 1.0

    related = _SCHEMA_RELATED.get(query_schema, set())
    if edge_schema in related:
        return 0.7

    return 0.3


# ── MINERVA 2 activation ──────────────────────────────────────

def score_edge(
    query_emb: list,
    qd,
    edge,
    decode_embedding,
    dot,
    query_text: str = "",
) -> float:
    """MINERVA 2 activation for a single edge.

    activation = episodic³ × emotional³ × temporal³ × relational³ × schematic³
    """
    e = _episodic(query_emb, edge, decode_embedding, dot)
    m = _emotional(qd, edge, query_text=query_text)
    t = _temporal(qd, edge)
    r = _relational(qd, edge)
    s = _schematic(qd, edge)

    return (e ** 3) * (m ** 3) * (t ** 3) * (r ** 3) * (s ** 3)
