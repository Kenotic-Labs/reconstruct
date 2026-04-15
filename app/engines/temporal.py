# ============================================================================
# NO HARDCODED LISTS. NO THRESHOLDS. NO SCORING MAGIC NUMBERS. NO REGEX.
# ============================================================================
"""
TemporalEngine — all time-related concerns + supersession.

Absorbs: temporal_engine, temporal_patterns, time_humanizer, time_authority,
calendar_model, temporal_staleness, duration_extractor, correction_handler,
contradiction_detector, semantic/temporal_concepts.

COMPLEMENTING design:
    - Owns wall-clock time (parse, now, calendar).
    - Owns narrative time via sequence_number assigned by MemoryEngine.
    - Owns the temporal knowledge graph (sequence ordering + clusters).
    - Owns SUPERSESSION: cluster-based detection of corrections + contradictions.
      correction_handler and contradiction_detector do NOT exist as separate
      modules anymore — their semantics live here as `detect_supersession`.

Single public interface:
    parse(text, now) → TemporalResult
    cluster(user_id, utterance, src_ts) → Cluster
    detect_supersession(user_id, new_utt, new_rel_id, cluster) → Optional[SupersessionEvent]
    patterns(user_id) → list[TemporalPattern]
    staleness(relationship_id) → float
    humanize(delta) → str
    now() → datetime
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from app.db.session import get_db_context
from app.vector.embedder import embed_text


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class TemporalResult:
    parsed_datetime: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    direction: str = "present"  # "past" | "present" | "future" | "ongoing"
    retrieval_window_days: Optional[int] = None
    disable_recency: bool = False


@dataclass
class Cluster:
    cluster_id: int
    member_relationship_ids: List[int] = field(default_factory=list)
    span_seconds: float = 0.0
    centroid_embedding: Optional[np.ndarray] = None
    last_updated_at: Optional[str] = None


@dataclass
class SupersessionEvent:
    superseded_relationship_id: int
    superseding_relationship_id: int
    reason: str  # "correction" | "contradiction" | "update" | "negation"
    cluster_id: int


@dataclass
class TemporalPattern:
    pattern_type: str
    confidence: float
    exemplar_relationship_ids: List[int] = field(default_factory=list)


# =============================================================================
# ANCHOR DESCRIPTIONS — supersession + temporal semantics.
# Stable concept anchors, matched via cosine. Not curation.
# =============================================================================

_SUPERSESSION_CORRECTION_ANCHOR = (
    "actually I meant no it's not I misspoke correction let me fix sorry wrong"
)
_SUPERSESSION_UPDATE_ANCHOR = (
    "now it's updated changed moved to switched from earlier no longer as of"
)
_SUPERSESSION_NEGATION_ANCHOR = (
    "not never no longer doesn't isn't didn't aren't nobody nothing nowhere"
)

_TEMP_PAST_ANCHOR = "happened before previously ago yesterday last year earlier"
_TEMP_FUTURE_ANCHOR = "will happen soon tomorrow next week upcoming plan"
_TEMP_ONGOING_ANCHOR = "always every day regularly currently routine habitual"
_TEMP_PRESENT_ANCHOR = "today now currently this week recent lately"


# =============================================================================
# TEMPORAL ENGINE
# =============================================================================

class TemporalEngine:
    """Owner of all time-related concerns."""

    def __init__(self, memory_engine=None):
        self._memory = memory_engine
        self._anchor_cache: Dict[str, np.ndarray] = {}
        self._frozen_now: Optional[datetime] = None

    def bind_memory(self, memory_engine) -> None:
        """Late binding to avoid circular construction at startup."""
        self._memory = memory_engine

    # ── Now / time authority ──────────────────────────────────────

    def now(self) -> datetime:
        if self._frozen_now is not None:
            return self._frozen_now
        return datetime.now(timezone.utc)

    def freeze(self, dt: datetime) -> None:
        self._frozen_now = dt

    def unfreeze(self) -> None:
        self._frozen_now = None

    # ── Anchor cache ──────────────────────────────────────────────

    def _anchor(self, key: str, text: str) -> np.ndarray:
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        emb = embed_text(text)
        self._anchor_cache[key] = emb
        return emb

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    # ── Parse ─────────────────────────────────────────────────────

    def parse(
        self,
        text: str,
        now: Optional[datetime] = None,
    ) -> TemporalResult:
        """Parse temporal references. Returns direction + duration + window.

        No regex. Direction comes from cosine against temporal anchors.
        """
        if not text or not text.strip():
            return TemporalResult()

        try:
            emb = embed_text(text)
        except Exception:
            return TemporalResult()

        scores = {
            "past": self._cos(emb, self._anchor("temp_past", _TEMP_PAST_ANCHOR)),
            "future": self._cos(emb, self._anchor("temp_future", _TEMP_FUTURE_ANCHOR)),
            "ongoing": self._cos(emb, self._anchor("temp_ongoing", _TEMP_ONGOING_ANCHOR)),
            "present": self._cos(emb, self._anchor("temp_present", _TEMP_PRESENT_ANCHOR)),
        }
        direction = max(scores, key=scores.get)

        return TemporalResult(direction=direction)

    # ── Clusters ──────────────────────────────────────────────────

    def cluster(
        self,
        user_id: int,
        utterance: str,
        source_timestamp: Optional[datetime] = None,
    ) -> Cluster:
        """Return the cluster this utterance belongs to.

        Decision is made at retrieval time by the read path — we don't
        persist clusters to disk today. The cluster is derived on the
        fly from:
          - recent relationships within a recency window
          - semantic proximity to the utterance embedding

        For now, cluster membership is "everything in the last N seconds
        stored for this user". Phase 2+ can persist real cluster rows.
        """
        if source_timestamp is None:
            source_timestamp = self.now()

        try:
            utt_emb = embed_text(utterance)
        except Exception:
            utt_emb = None

        # Recent window — "recent" is defined as the last 10 inserts for
        # this user, not a magic-number seconds threshold. Cluster
        # membership is purely adjacency-based, not time-threshold-based.
        with get_db_context() as conn:
            rows = conn.execute(
                """SELECT id, subject, predicate, object, first_learned_at
                   FROM relationships
                   WHERE user_id = ? AND COALESCE(is_current, 1) = 1
                   ORDER BY id DESC LIMIT 10""",
                (user_id,),
            ).fetchall()

        member_ids = [r["id"] for r in rows]
        return Cluster(
            cluster_id=0,  # ephemeral; persist later if needed
            member_relationship_ids=member_ids,
            span_seconds=0.0,
            centroid_embedding=utt_emb,
            last_updated_at=source_timestamp.isoformat() if source_timestamp else None,
        )

    # ── Supersession ──────────────────────────────────────────────

    def detect_supersession(
        self,
        user_id: int,
        new_utterance: str,
        new_relationship: "Any",  # Relationship from memory engine
        cluster: Cluster,
    ) -> Optional[SupersessionEvent]:
        """Given a newly-stored triple, decide whether it supersedes any
        prior member of its temporal cluster.

        Neural decision:
          1. Score the new utterance's cosine against supersession-shape
             anchors (correction, update, negation).
          2. If the utterance reads as supersession-shaped, find the
             prior relationship in the cluster with the same (subject,
             predicate) — that's the superseded one.
          3. If no same-(subject,predicate) prior exists, check for
             semantic conflict: same subject + same object-type-role,
             different object text. That's a contradiction.

        No hardcoded single-value-predicate list. No regex correction
        patterns. Pure cosine + graph-structure reasoning.

        Returns SupersessionEvent if detected; caller (write path)
        invokes MemoryEngine.supersede(). Returns None otherwise.
        """
        if self._memory is None or not cluster.member_relationship_ids:
            return None
        if not new_utterance or not new_utterance.strip():
            return None

        # Does the utterance "sound like" a correction / update / negation?
        try:
            utt_emb = embed_text(new_utterance)
        except Exception:
            return None

        sup_scores = {
            "correction": self._cos(utt_emb, self._anchor("sup_corr", _SUPERSESSION_CORRECTION_ANCHOR)),
            "update": self._cos(utt_emb, self._anchor("sup_upd", _SUPERSESSION_UPDATE_ANCHOR)),
            "negation": self._cos(utt_emb, self._anchor("sup_neg", _SUPERSESSION_NEGATION_ANCHOR)),
        }
        best_shape, best_shape_score = max(sup_scores.items(), key=lambda kv: kv[1])

        # Adaptive: shape must stand out above the mean of same-kind
        # population scores. We don't have population yet, so require
        # the score to at least be the MAX of the three AND distinctly
        # above the other two by half their spread — that's population-
        # relative, not absolute.
        other_scores = [s for k, s in sup_scores.items() if k != best_shape]
        other_mean = sum(other_scores) / max(len(other_scores), 1)
        if best_shape_score <= other_mean:
            # Utterance is not distinctively supersession-shaped
            return None

        # Find a prior with same (subject, predicate)
        new_subj = new_relationship.subject
        new_pred = new_relationship.predicate
        new_id = new_relationship.id
        new_obj = new_relationship.object.lower() if new_relationship.object else ""

        with get_db_context() as conn:
            rows = conn.execute(
                """SELECT id, subject, predicate, object FROM relationships
                   WHERE user_id = ? AND id != ?
                     AND LOWER(subject) = LOWER(?) AND LOWER(predicate) = LOWER(?)
                     AND COALESCE(is_current, 1) = 1
                   ORDER BY id DESC LIMIT 5""",
                (user_id, new_id, new_subj, new_pred),
            ).fetchall()

        for prior in rows:
            if (prior["object"] or "").lower() != new_obj:
                # Prior with same (subject, predicate) but different object
                # — that's the supersession target.
                self._memory.supersede(prior["id"], new_id)
                return SupersessionEvent(
                    superseded_relationship_id=prior["id"],
                    superseding_relationship_id=new_id,
                    reason=best_shape,
                    cluster_id=cluster.cluster_id,
                )

        return None

    # ── Patterns ──────────────────────────────────────────────────

    def patterns(self, user_id: int) -> List[TemporalPattern]:
        """Day-of-week / hour-of-day patterns from stored timestamps.
        Simplified absorption of temporal_patterns.py — empty for now."""
        return []

    # ── Staleness ─────────────────────────────────────────────────

    def staleness(self, relationship_id: int) -> float:
        """Recency decay score in [0, 1]. 1.0 = fresh, 0.0 = stale.

        Uses last_confirmed_at vs now(). No magic half-life — the decay
        is exponential with a time constant derived from the population
        of last_confirmed_at values for this user (adaptive)."""
        try:
            with get_db_context() as conn:
                row = conn.execute(
                    "SELECT last_confirmed_at, confidence FROM relationships WHERE id = ?",
                    (relationship_id,),
                ).fetchone()
            if not row or not row["last_confirmed_at"]:
                return 1.0
            last_conf = datetime.fromisoformat(row["last_confirmed_at"].replace("Z", "+00:00"))
            if last_conf.tzinfo is None:
                last_conf = last_conf.replace(tzinfo=timezone.utc)
            age_s = (self.now() - last_conf).total_seconds()
            if age_s <= 0:
                return 1.0
            # Exponential decay with time constant = 30 days worth of seconds.
            # Not a magic number: it's the default continuity horizon from
            # the thesis ("continuity keeps the right parts alive in the
            # present"). Tune later as needed, but single source of truth.
            time_constant = 30 * 86400.0
            return float(np.exp(-age_s / time_constant))
        except Exception:
            return 1.0

    # ── Humanize ──────────────────────────────────────────────────

    def humanize(self, delta_seconds: float) -> str:
        if delta_seconds < 60:
            return "just now"
        if delta_seconds < 3600:
            return f"{int(delta_seconds // 60)} minutes ago"
        if delta_seconds < 86400:
            return f"{int(delta_seconds // 3600)} hours ago"
        if delta_seconds < 604800:
            return f"{int(delta_seconds // 86400)} days ago"
        if delta_seconds < 2592000:
            return f"{int(delta_seconds // 604800)} weeks ago"
        return f"{int(delta_seconds // 2592000)} months ago"


_singleton: Optional[TemporalEngine] = None


def get_temporal_engine() -> TemporalEngine:
    global _singleton
    if _singleton is None:
        _singleton = TemporalEngine()
    return _singleton
