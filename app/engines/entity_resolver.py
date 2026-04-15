"""Query-time entity resolver.

Input: user_id, query_text.
Output: list of {id, name, entity_type, score} for user entities whose
embedding cosine-matches a noun-phrase from the query above tau_entity.

No curated word lists. NP extraction uses spaCy en_core_web_sm; stopwords
are the closed WH/auxiliary grammatical class already used by
predicted_queries.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from app.db.session import get_db_context
from app.vector import embedder as _embedder_module


TAU_ENTITY = 0.55

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
    min_score: float = TAU_ENTITY,
) -> List[Dict[str, Any]]:
    candidates = _extract_candidates(query_text)
    if not candidates:
        return []

    phrase_embs = [(c, _embedder_module.embed_text(c)) for c in candidates]

    sql = ("SELECT id, name, entity_type, embedding FROM entities "
           "WHERE user_id = ?")
    with get_db_context() as conn:
        rows = conn.execute(sql, (user_id,)).fetchall()

    best: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        if not r["embedding"]:
            continue
        ev = np.frombuffer(r["embedding"], dtype=np.float32)
        top = 0.0
        for _, pe in phrase_embs:
            c = _cosine(ev, pe)
            if c > top:
                top = c
        if top >= min_score:
            best[r["id"]] = {
                "id": r["id"], "name": r["name"],
                "entity_type": r["entity_type"], "score": top,
            }

    ranked = sorted(best.values(), key=lambda x: -x["score"])
    return ranked[:top_k]
