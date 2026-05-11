#!/usr/bin/env python3
"""Generate app/engines/retrieval.py from scratch."""

CONTENT = r'''# -*- coding: cp1252 -*-
"""
RetrievalEngine -- moat pipeline read path.

Lookup pipeline:
    1. Entry Cosine: max(pq_cosine, edge_cosine, predicate_cosine) per edge.
    2. Expand: entity union from graph.
    3. Group: cluster + arc anchor filter.
    4. Relate: BFS proximity filter.
    5. Exit Cosine: rank survivors.
    6. Verification loop (every candidate, no exceptions):
       CHECK 1 -- ENTITY: edge mentions the query entity.
       CHECK 2 -- EXISTENCE: is_current=1, not tombstoned.
       CHECK 3 -- FACT CONSISTENCY: if facts table has a canonical value
       for this edge's schema+entity, the edge's value must match.
       First candidate to pass all checks = answer.
       All rejected = StructuralRefusal.

reconstruct() shares stages 1-5, then groups by cluster_id and fuses
each cluster through the deterministic grammar engine.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from app.db.session import get_db_context
from app.vector import embedder as _embedder_module
from app.vector.embedder import embed_text
from app.engines.retrieval_types import Candidate

log = logging.getLogger(__name__)


@dataclass
class Answer:
    text: Optional[str]
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    confidence: float = 0.0
    source: str = "pq_cosine"
    survivors: int = 0
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    convergence_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralRefusal:
    reason: str
    text: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    confidence: float = 0.0
    source: str = "structural_refusal"
    survivors: int = 0
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    convergence_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Cluster:
    key_type: str
    key_value: str
    edges: List[Dict[str, Any]] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    dominant_mood: Optional[str] = None
    mean_valence: Optional[float] = None
    pivotal_edge_ids: List[int] = field(default_factory=list)
    timeline_edge_ids: List[int] = field(default_factory=list)


@dataclass
class Situation:
    narrative: str
    clusters: List[Cluster] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    dominant_mood: Optional[str] = None
    pivotal_events: List[int] = field(default_factory=list)
    timeline: List[int] = field(default_factory=list)
    grounding_map: Dict[int, List[int]] = field(default_factory=dict)
    source: str = "reconstruct"
    survivors: int = 0
    convergence_details: Dict[str, Any] = field(default_factory=dict)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    try:
        da = float(np.linalg.norm(a))
        db = float(np.linalg.norm(b))
        if da == 0.0 or db == 0.0:
            return 0.0
        return float(np.dot(a, b) / (da * db))
    except Exception:
        return 0.0


def _cosine_from_blob(q_emb: np.ndarray, blob):
    if blob is None:
        return 0.0
    v = np.frombuffer(blob, dtype=np.float32)
    if v.size != q_emb.size:
        return 0.0
    denom = float(np.linalg.norm(q_emb) * np.linalg.norm(v))
    if denom == 0.0:
        return 0.0
    return float(np.dot(q_emb, v) / denom)


def _deserialize_emb(blob) -> Optional[np.ndarray]:
    if blob is None:
        return None
    try:
        return np.frombuffer(blob, dtype=np.float32).copy()
    except Exception:
        return None


def _split_words(text: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    for ch in text or "":
        if ch.isalnum():
            buf.append(ch.lower())
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def _normalize_token(token: str) -> str:
    t = (token or "").strip().lower()
    if not t:
        return ""
    irregular = {
        "am": "be", "is": "be", "are": "be",
        "was": "be", "were": "be", "been": "be", "being": "be",
        "has": "have", "had": "have",
        "does": "do", "did": "do",
    }
    if t in irregular:
        return irregular[t]
    try:
        from nltk.corpus import wordnet as _wn
        lemma = _wn.morphy(t, _wn.VERB)
        if lemma:
            return lemma
        lemma = _wn.morphy(t, _wn.NOUN)
        if lemma:
            return lemma
    except Exception:
        pass
    if len(t) > 4 and t.endswith("ied"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("ed"):
        return t[:-2]
    if len(t) > 4 and t.endswith("es"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _normalized_words(text: str) -> Set[str]:
    words: Set[str] = set()
    for tok in _split_words(text):
        norm = _normalize_token(tok)
        if norm:
            words.add(norm)
    return words


def _edge_mentions_entity(edge_subject: str, edge_object: str, query_entity: Optional[str]) -> bool:
    """Binary: does this edge mention the query entity in either position?"""
    if not query_entity:
        return True
    subj = " ".join(_split_words(edge_subject))
    obj = " ".join(_split_words(edge_object))
    ent = " ".join(_split_words(query_entity))
    if not ent:
        return True
    return subj == ent or subj == "user" or obj == ent or obj == "user"


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces to single space without regex."""
    return " ".join(text.split())


'''

with open("app/engines/retrieval.py", "w", encoding="cp1252") as f:
    f.write(CONTENT)
print("Part 1 written:", len(CONTENT), "chars")
