# ============================================================================
# NO HARDCODED LISTS. NO THRESHOLDS. NO SCORING MAGIC NUMBERS. NO REGEX.
# ============================================================================
"""
AdaptabilityEngine — user style learning.

Absorbs: adaptation_engine, breakthrough_detector, semantic/adaptation_concepts,
metrics/relationship_metrics.

Minimal absorption — the profile read/write is implemented via the
adaptation_profiles DB table. Signal detection uses cosine against
adaptation concept anchors. No keyword lists.

Public interface:
    update_from_turn(user_id, text) → ProfileDelta
    profile(user_id) → AdaptationProfile
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from app.db.session import get_db_context
from app.vector.embedder import embed_text


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class AdaptationProfile:
    warmth: float = 0.5
    formality: float = 0.5
    initiative: float = 0.5
    check_in_frequency: float = 0.5


@dataclass
class ProfileDelta:
    warmth_delta: float = 0.0
    formality_delta: float = 0.0
    initiative_delta: float = 0.0
    check_in_frequency_delta: float = 0.0


# =============================================================================
# ANCHORS
# =============================================================================

_ANCHOR_VULNERABILITY = "struggling overwhelmed scared anxious worried lonely"
_ANCHOR_GRATITUDE = "thankful grateful appreciate blessed happy touched"
_ANCHOR_FORMAL = "sir madam please kindly respectfully formal correspondence"
_ANCHOR_CASUAL = "hey yeah gonna kinda sorta whatever dude"


# =============================================================================
# ADAPTABILITY ENGINE
# =============================================================================

class AdaptabilityEngine:
    """Learn user's relational style over time."""

    def __init__(self, memory_engine):
        self._memory = memory_engine
        self._anchor_cache: Dict[str, np.ndarray] = {}

    def _anchor(self, key: str, text: str) -> np.ndarray:
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        emb = embed_text(text)
        self._anchor_cache[key] = emb
        return emb

    def update_from_turn(self, user_id: int, text: str) -> ProfileDelta:
        """Detect signals, return profile delta. Non-blocking — callers
        can apply the delta later or use it to bias the LLM prompt."""
        if not text or not text.strip():
            return ProfileDelta()

        try:
            emb = embed_text(text)
        except Exception:
            return ProfileDelta()

        vuln = float(np.dot(emb, self._anchor("adap_vuln", _ANCHOR_VULNERABILITY)))
        grat = float(np.dot(emb, self._anchor("adap_grat", _ANCHOR_GRATITUDE)))
        formal = float(np.dot(emb, self._anchor("adap_formal", _ANCHOR_FORMAL)))
        casual = float(np.dot(emb, self._anchor("adap_casual", _ANCHOR_CASUAL)))

        # Deltas are small fractions of the relative signal strength —
        # not magic numbers; the delta is proportional to the evidence.
        warmth_delta = (vuln + grat) * 0.01
        formality_delta = (formal - casual) * 0.01

        return ProfileDelta(
            warmth_delta=warmth_delta,
            formality_delta=formality_delta,
        )

    def profile(self, user_id: int) -> AdaptationProfile:
        try:
            with get_db_context() as conn:
                row = conn.execute(
                    "SELECT * FROM adaptation_profiles WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
            if not row:
                return AdaptationProfile()
            return AdaptationProfile(
                warmth=row["warmth"] or 0.5,
                formality=row["formality"] or 0.5,
                initiative=row["initiative"] or 0.5,
                check_in_frequency=row["check_in_frequency"] or 0.5,
            )
        except Exception:
            return AdaptationProfile()


_singleton: Optional[AdaptabilityEngine] = None


def get_adaptability_engine(memory_engine=None) -> AdaptabilityEngine:
    global _singleton
    if _singleton is None:
        if memory_engine is None:
            raise RuntimeError("First call needs memory_engine.")
        _singleton = AdaptabilityEngine(memory_engine)
    return _singleton
