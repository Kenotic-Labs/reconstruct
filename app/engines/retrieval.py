"""
Retrieval adapter — bridges the MINERVA 2 reconstruction engine
to the API surface that callers (ATANT, SDK, tests, demos) expect.

Exports:
  - RetrievalEngine / get_retrieval_engine — class adapter
  - Answer — successful reconstruction result
  - StructuralRefusal — refusal result (echo below noise)
  - Situation — situational reconstruction result
  - RetrievalAnswer — legacy dataclass (run_atant)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.engines.reconstruction import (
    reconstruct, ReconstructionResult, is_situational,
)
from app.db.session import get_db_context


# ── Compatibility types ─────────────────────────────────────────
# The old retrieval engine returned Answer, StructuralRefusal, or
# Situation objects. 16 files import these. Rather than updating
# all 16, we provide thin wrappers over ReconstructionResult.

@dataclass
class Answer:
    """A successful reconstruction — the echo converged on an answer."""
    text: str = ""
    return_field: str = "episodic"
    edge_ids: List[int] = field(default_factory=list)
    grounding: List[str] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StructuralRefusal:
    """The echo was too weak — no trace resonated. Correct refusal."""
    text: str = "This information is not mentioned in the conversation."
    reason: str = ""


@dataclass
class Situation:
    """Situational reconstruction — schema-grouped facts."""
    text: str = ""
    edge_ids: List[int] = field(default_factory=list)
    grounding: List[str] = field(default_factory=list)


@dataclass
class RetrievalAnswer:
    """Legacy format for run_atant_cumulative.py."""
    text: str = ""
    candidates: List[Dict[str, Any]] = field(default_factory=list)


# ── Main adapter ────────────────────────────────────────────────

class RetrievalEngine:

    def __init__(self, memory=None, temporal=None):
        pass  # resonance model needs no external engines

    def retrieve(self, user_id: int, query: str) -> RetrievalAnswer:
        """Legacy retrieve — returns RetrievalAnswer for ATANT/tests."""
        result = reconstruct(user_id, query)
        candidates = _build_candidates(result.edge_ids)
        return RetrievalAnswer(
            text=result.answer or "",
            candidates=candidates,
        )

    def answer(self, user_id: int, query: str):
        """Rich retrieve — returns Answer, StructuralRefusal, or Situation."""
        result = reconstruct(user_id, query)

        if result.refusal:
            return StructuralRefusal(
                text=result.answer or "",
                reason=result.refusal_reason,
            )

        if is_situational(query):
            return Situation(
                text=result.answer or "",
                edge_ids=result.edge_ids,
                grounding=result.grounding,
            )

        return Answer(
            text=result.answer or "",
            return_field=result.return_field,
            edge_ids=result.edge_ids,
            grounding=result.grounding,
            candidates=_build_candidates(result.edge_ids),
        )


def get_retrieval_engine(memory=None, temporal=None) -> RetrievalEngine:
    return RetrievalEngine()


# ── Helpers ─────────────────────────────────────────────────────

def _cosine_from_blob(query_vec, blob) -> float:
    """Compatibility: compute cosine between a numpy array and a stored blob."""
    from app.engines.reconstruction import _decode_embedding, _dot
    emb = _decode_embedding(blob)
    if emb is None:
        return 0.0
    q = list(query_vec) if not isinstance(query_vec, list) else query_vec
    return _dot(q, emb)


def _build_candidates(edge_ids: List[int]) -> List[Dict[str, Any]]:
    """Fetch edge triples from DB for candidate display."""
    if not edge_ids:
        return []
    try:
        with get_db_context() as conn:
            placeholders = ",".join("?" for _ in edge_ids)
            rows = conn.execute(
                f"SELECT id, subject, predicate, object, source_text "
                f"FROM edges WHERE id IN ({placeholders})",
                edge_ids,
            ).fetchall()
            return [
                {
                    "subject": row["subject"] or "",
                    "predicate": row["predicate"] or "",
                    "object": row["object"] or "",
                    "source_text": row["source_text"] or "",
                }
                for row in rows
            ]
    except Exception:
        return []
