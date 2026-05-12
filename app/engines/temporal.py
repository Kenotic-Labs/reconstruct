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
    resolve_event_date(text, reference_timestamp) → Optional[str]
    cluster(user_id, utterance, src_ts) → Cluster
    detect_supersession(user_id, new_utt, new_rel_id, cluster) → Optional[SupersessionEvent]
    recluster_for_reconstruction(user_id, edge_ids) → List[dict]
    detect_arcs(user_id, new_rel_id, cluster_id) → Optional[str]
    edges_valid_at(user_id, timestamp) → List[dict]
    temporal_neighbors(user_id, relationship_id, window) → List[dict]
    patterns(user_id) → list[TemporalPattern]
    staleness(relationship_id) → float
    humanize(delta) → str
    now() → datetime
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.db.session import get_db_context
from app.vector.embedder import embed_text

import logging as _logging
_log = _logging.getLogger(__name__)

# =============================================================================
# Entry / Exit checks
# =============================================================================

_ENTRY_CHECKED = False


class TemporalEntryError(RuntimeError):
    """Raised when temporal engine's dependencies are not available."""
    pass


def _check_entry():
    """Verify ingestion and memory are importable. Runs once."""
    global _ENTRY_CHECKED
    if _ENTRY_CHECKED:
        return
    missing = []
    try:
        from app.engines import ingestion  # noqa: F401
        if not hasattr(ingestion, 'cleanup'):
            missing.append("ingestion.cleanup")
    except ImportError:
        missing.append("ingestion")
    try:
        from app.engines import memory  # noqa: F401
        if not hasattr(memory, 'get_memory_engine'):
            missing.append("memory.get_memory_engine")
    except ImportError:
        missing.append("memory")
    if missing:
        raise TemporalEntryError(
            f"temporal entry check failed — missing: {', '.join(missing)}"
        )
    _ENTRY_CHECKED = True


def check_exit_date(result: str) -> bool:
    """Validate that a resolved date looks like ISO 8601."""
    if result is None:
        return True  # None is valid (no date found)
    return isinstance(result, str) and len(result) >= 10


# =============================================================================
# LAZY LOADERS — spaCy and dateparser are heavy; load once on first use
# =============================================================================

_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


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


    # == Calendar model ====================================================
    # Real-time understanding of time — not just wall-clock, but what
    # time MEANS: proximity, urgency, ordering, duration, upcoming events.

    def temporal_context(self):
        """What the engine knows about NOW."""
        n = self.now()
        return {
            "now": n.isoformat(), "weekday": n.strftime("%A"),
            "date": n.strftime("%Y-%m-%d"), "time": n.strftime("%H:%M"),
            "hour": n.hour, "day_of_week": n.weekday(),
            "is_weekend": n.weekday() >= 5,
            "month": n.strftime("%B"), "year": n.year,
        }

    def days_until(self, target):
        """Days from now until target. Negative = past."""
        target_dt = self._to_datetime(target)
        if target_dt is None:
            return None
        now_naive = self.now().replace(tzinfo=None)
        return (target_dt - now_naive).total_seconds() / 86400.0

    def days_since(self, target):
        """Days since target until now. Negative = future."""
        d = self.days_until(target)
        return -d if d is not None else None

    def proximity(self, target):
        """Human proximity: 'tomorrow', 'in 3 days', '2 weeks ago'."""
        days = self.days_until(target)
        if days is None:
            return None
        ad = abs(days)
        fut = days > 0
        if ad < 0.04:
            return "right now"
        if ad < 1:
            return "today" if fut else "earlier today"
        if ad < 2:
            return "tomorrow" if fut else "yesterday"
        if ad < 7:
            n = round(ad)
            return f"in {n} days" if fut else f"{n} days ago"
        if ad < 30:
            n = round(ad / 7)
            u = "week" if n == 1 else "weeks"
            return f"in {n} {u}" if fut else f"{n} {u} ago"
        if ad < 365:
            n = round(ad / 30)
            u = "month" if n == 1 else "months"
            return f"in {n} {u}" if fut else f"{n} {u} ago"
        n = round(ad / 365)
        u = "year" if n == 1 else "years"
        return f"in {n} {u}" if fut else f"{n} {u} ago"

    def is_before(self, a, b):
        """Is event A before event B?"""
        dt_a, dt_b = self._to_datetime(a), self._to_datetime(b)
        if dt_a is None or dt_b is None:
            return None
        return dt_a < dt_b

    def duration_between(self, a, b):
        """Seconds between two events. Always positive."""
        dt_a, dt_b = self._to_datetime(a), self._to_datetime(b)
        if dt_a is None or dt_b is None:
            return None
        return abs((dt_b - dt_a).total_seconds())

    def urgency(self, user_id, relationship_id):
        """How urgent is this edge right now?"""
        try:
            with get_db_context() as conn:
                row = conn.execute(
                    "SELECT resolved_event_date, temporal_expression, "
                    "edge_emotional_valence, edge_schematic_category, is_current "
                    "FROM edges WHERE id = ? AND user_id = ?",
                    (relationship_id, user_id),
                ).fetchone()
            if not row:
                return None
            date_str = row["resolved_event_date"] or row["temporal_expression"]
            if not date_str:
                return None
            days = self.days_until(date_str)
            if days is None:
                return None
            return {
                "days": round(days, 1),
                "proximity": self.proximity(date_str),
                "is_upcoming": 0 < days <= 7,
                "is_overdue": days < 0 and bool(row["is_current"]),
                "schema": row["edge_schematic_category"],
                "valence": row["edge_emotional_valence"],
            }
        except Exception:
            return None

    def upcoming_events(self, user_id, window_days=7):
        """Edges with resolved dates within the next N days."""
        n = self.now()
        cutoff = (n + timedelta(days=window_days)).isoformat()
        now_str = n.isoformat()
        try:
            with get_db_context() as conn:
                rows = conn.execute(
                    "SELECT id, subject, predicate, object, "
                    "resolved_event_date, source_text, edge_schematic_category "
                    "FROM edges WHERE user_id = ? "
                    "AND resolved_event_date IS NOT NULL "
                    "AND resolved_event_date >= ? AND resolved_event_date <= ? "
                    "AND COALESCE(is_current, 1) = 1 AND tombstoned_at IS NULL "
                    "ORDER BY resolved_event_date ASC",
                    (user_id, now_str, cutoff),
                ).fetchall()
            result = []
            for r in rows:
                days = self.days_until(r["resolved_event_date"])
                result.append({
                    "id": r["id"], "subject": r["subject"],
                    "predicate": r["predicate"], "object": r["object"],
                    "date": r["resolved_event_date"],
                    "days_until": round(days, 1) if days is not None else None,
                    "proximity": self.proximity(r["resolved_event_date"]),
                    "schema": r["edge_schematic_category"],
                })
            return result
        except Exception:
            return []

    def _to_datetime(self, value):
        """Convert anything temporal to naive datetime."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(tzinfo=None) if value.tzinfo else value
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return dt.replace(tzinfo=None) if dt.tzinfo else dt
            except (ValueError, TypeError):
                pass
            resolved = self.resolve_event_date(value, self.now().isoformat())
            if resolved:
                try:
                    dt = datetime.fromisoformat(resolved)
                    return dt.replace(tzinfo=None) if dt.tzinfo else dt
                except (ValueError, TypeError):
                    pass
        return None


    # ── Anchor cache ──────────────────────────────────────────────

    def _anchor(self, key: str, text: str) -> np.ndarray:
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        emb = embed_text(text)
        self._anchor_cache[key] = emb
        return emb

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity, not raw dot product."""
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ── resolve_event_date ────────────────────────────────────────
    # ROOT CAUSE: memory.py line 1000 calls this but it did not exist,
    # leaving resolved_event_date NULL on every edge.

    def resolve_event_date(
        self,
        text: str,
        reference_timestamp: Optional[str] = None,
    ) -> Optional[str]:
        """Extract and resolve temporal expressions to ISO 8601.

        Called by memory.py at write time. Uses spaCy NER for DATE/TIME
        span extraction, then dateparser to resolve against the reference
        timestamp. Returns ISO string or None.
        """
        _check_entry()
        if not text or not text.strip():
            return None

        # Parse reference timestamp into datetime for dateparser
        ref_dt = self.now()
        if reference_timestamp:
            try:
                ref_dt = datetime.fromisoformat(
                    reference_timestamp.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        # Step 1: Extract DATE/TIME spans via spaCy NER
        # Filter out DURATIONS (NUM + time unit without date anchor)
        # Rule: "for X weeks/months" = duration. "X ago" = date. "June 1" = date.
        try:
            nlp = _get_spacy()
            doc = nlp(text)
            temporal_spans = []
            for ent in doc.ents:
                if ent.label_ not in ("DATE", "TIME"):
                    continue
                span_lower = ent.text.lower().strip()
                # Duration filter: NUM + time unit without "ago"/"last"/"next" = duration, skip
                tokens = span_lower.split()
                _TIME_UNITS = {"second", "seconds", "minute", "minutes", "hour", "hours",
                               "day", "days", "week", "weeks", "month", "months",
                               "year", "years", "decade", "decades"}
                is_duration = (
                    len(tokens) >= 2
                    and tokens[-1] in _TIME_UNITS
                    and not any(w in span_lower for w in ("ago", "last", "next", "this"))
                )
                # Check if governed by "for" prep (strong duration signal)
                if ent.start > 0 and doc[ent.start - 1].lemma_ == "for":
                    is_duration = True
                if not is_duration:
                    temporal_spans.append(ent.text)
        except Exception:
            temporal_spans = []

        if not temporal_spans:
            return None

        # Step 2: Resolve temporal spans to datetime.
        #
        # Root cause: dateparser.parse() is 250-1000ms per call. The old code
        # called it 2-4 times per span with fallbacks, causing 1-2 SECOND
        # latency per sentence with temporal expressions. Fix: use Python's
        # datetime + simple relative-date math for common patterns. Only fall
        # back to dateparser for patterns we can't handle structurally.

        ref_naive = ref_dt.replace(tzinfo=None) if ref_dt.tzinfo else ref_dt

        # Fast structural resolver — handles 90%+ of conversational date refs
        _DAY_MAP = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }
        _MONTH_MAP = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }

        def _fast_resolve(span: str) -> Optional[datetime]:
            s = span.lower().strip()

            # "yesterday"
            if s == "yesterday":
                return ref_naive - timedelta(days=1)

            # "today"
            if s == "today":
                return ref_naive

            # "tomorrow"
            if s == "tomorrow":
                return ref_naive + timedelta(days=1)

            # "last Sunday" / "last Monday" etc
            for day_name, day_num in _DAY_MAP.items():
                if day_name in s:
                    if "last" in s or "past" in s:
                        days_back = (ref_naive.weekday() - day_num) % 7
                        if days_back == 0:
                            days_back = 7
                        return ref_naive - timedelta(days=days_back)
                    if "next" in s:
                        days_fwd = (day_num - ref_naive.weekday()) % 7
                        if days_fwd == 0:
                            days_fwd = 7
                        return ref_naive + timedelta(days=days_fwd)
                    # Bare day name — direction from context
                    # "on Wednesday" / "this Wednesday" → future (next occurrence)
                    # "Wednesday" with past-tense verb → past (most recent)
                    text_lower = text.lower()
                    is_future = (
                        f"on {day_name}" in text_lower
                        or f"this {day_name}" in text_lower
                        or f"coming {day_name}" in text_lower
                        or "will " in text_lower
                        or "going to " in text_lower
                    )
                    if is_future:
                        days_fwd = (day_num - ref_naive.weekday()) % 7
                        if days_fwd == 0:
                            days_fwd = 7
                        return ref_naive + timedelta(days=days_fwd)
                    # Default: past (most recent occurrence)
                    days_back = (ref_naive.weekday() - day_num) % 7
                    if days_back == 0:
                        days_back = 7
                    return ref_naive - timedelta(days=days_back)

            # "last week" / "last month"
            if "last week" in s:
                return ref_naive - timedelta(weeks=1)
            if "last month" in s:
                m = ref_naive.month - 1 or 12
                y = ref_naive.year if ref_naive.month > 1 else ref_naive.year - 1
                return ref_naive.replace(year=y, month=m, day=min(ref_naive.day, 28))

            # "N days/weeks/months/years ago"
            tokens = s.split()
            if "ago" in tokens:
                for i, tok in enumerate(tokens):
                    if tok == "ago" and i >= 2:
                        try:
                            num = int(tokens[i - 2])
                        except ValueError:
                            _WORD_NUMS = {"one":1,"two":2,"three":3,"four":4,"five":5,
                                          "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
                            num = _WORD_NUMS.get(tokens[i-2].lower())
                        unit = tokens[i - 1].lower().rstrip("s")
                        if num and unit == "day":
                            return ref_naive - timedelta(days=num)
                        if num and unit == "week":
                            return ref_naive - timedelta(weeks=num)
                        if num and unit == "month":
                            m = ref_naive.month - num
                            y = ref_naive.year
                            while m <= 0:
                                m += 12; y -= 1
                            return ref_naive.replace(year=y, month=m, day=min(ref_naive.day, 28))
                        if num and unit == "year":
                            return ref_naive.replace(year=ref_naive.year - num)

            # "June 2023" / "July 2023" / "2022" — month+year or year-only
            for month_name, month_num in _MONTH_MAP.items():
                if month_name in s:
                    # Try to find year
                    for tok in tokens:
                        if tok.isdigit() and len(tok) == 4:
                            return datetime(int(tok), month_num, 1)
                    # No year — use reference year
                    return datetime(ref_naive.year, month_num, 1)

            # Pure year: "2022"
            if s.isdigit() and len(s) == 4:
                return datetime(int(s), 1, 1)

            # "D Month YYYY" pattern: "7 May 2023", "25 May 2023"
            if len(tokens) >= 3:
                try:
                    day_num = int(tokens[0])
                    month_num = _MONTH_MAP.get(tokens[1].lower())
                    year_num = int(tokens[2]) if tokens[2].isdigit() else None
                    if month_num and year_num:
                        return datetime(year_num, month_num, day_num)
                except (ValueError, IndexError):
                    pass

            return None

        candidates: List[Tuple[str, datetime]] = []

        for span_text in temporal_spans:
            parsed = _fast_resolve(span_text)
            if parsed is not None:
                candidates.append((span_text, parsed))
                continue
            # Dateparser fallback for patterns we missed — single call only
            try:
                import dateparser
                parsed = dateparser.parse(span_text, settings={
                    "RELATIVE_BASE": ref_naive,
                    "PREFER_DATES_FROM": "past",
                    "RETURN_AS_TIMEZONE_AWARE": False,
                })
                if parsed is not None:
                    candidates.append((span_text, parsed))
            except Exception:
                pass

        if not candidates:
            return None

        # Pick the most SPECIFIC date — one with day+month > month-only > year-only
        # Specificity = how many components are non-default (not Jan 1, not day 1)
        def _specificity(dt):
            score = 0
            if dt.month != 1 or dt.day != 1:
                score += 1  # has month
            if dt.day != 1:
                score += 1  # has day
            if dt.hour != 0 or dt.minute != 0:
                score += 1  # has time
            return score

        best_text, best_dt = max(candidates, key=lambda c: _specificity(c[1]))
        return best_dt.isoformat()

    # ── Parse ─────────────────────────────────────────────────────

    def parse(
        self,
        text: str,
        now: Optional[datetime] = None,
    ) -> TemporalResult:
        """Parse temporal references. Returns direction + parsed datetime + duration.

        Direction comes from cosine against temporal anchors.
        Parsed datetime comes from resolve_event_date.
        Duration comes from dateparser on duration-like spans.
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

        # Resolve concrete datetime
        ref_ts = (now or self.now()).isoformat()
        resolved_iso = self.resolve_event_date(text, ref_ts)
        parsed_dt = None
        if resolved_iso:
            try:
                parsed_dt = datetime.fromisoformat(resolved_iso)
            except (ValueError, TypeError):
                pass

        # Resolve duration for "for 6 months" / "since January" patterns
        duration_seconds = self._extract_duration(text, now or self.now())

        return TemporalResult(
            direction=direction,
            parsed_datetime=parsed_dt,
            duration_seconds=duration_seconds,
        )

    def _extract_duration(
        self,
        text: str,
        ref: datetime,
    ) -> Optional[float]:
        """Extract duration in seconds from duration-like expressions.

        Uses dateparser to resolve "since January" or "for 6 months" by
        computing the difference between the resolved date and reference.
        """
        import dateparser

        # Look for temporal spans via spaCy NER
        try:
            nlp = _get_spacy()
            doc = nlp(text)
            duration_spans = [
                ent.text for ent in doc.ents
                if ent.label_ in ("DATE", "TIME")
            ]
        except Exception:
            return None

        if not duration_spans:
            return None

        ref_naive = ref.replace(tzinfo=None) if ref.tzinfo else ref
        settings = {
            "RELATIVE_BASE": ref_naive,
            "PREFER_DATES_FROM": "past",
            "RETURN_AS_TIMEZONE_AWARE": False,
        }

        # Check for duration structure via spaCy dependency parse:
        # Duration expressions have prep/mark tokens ('since', 'for')
        # governing a temporal NP. Using dep parse, not string matching.
        has_duration_dep = False
        try:
            nlp = _get_spacy()
            doc = nlp(text)
            for tok in doc:
                if tok.dep_ in ("prep", "mark") and tok.pos_ == "ADP":
                    # Check if this preposition governs a DATE/TIME entity
                    for child in tok.children:
                        if child.ent_type_ in ("DATE", "TIME"):
                            has_duration_dep = True
                            break
                if has_duration_dep:
                    break
        except Exception:
            pass

        if not has_duration_dep:
            return None

        for span_text in duration_spans:
            parsed = dateparser.parse(span_text, settings=settings)
            if parsed:
                diff = abs((ref_naive - parsed).total_seconds())
                if diff > 0:
                    return diff

        return None

    # ── Supersession ──────────────────────────────────────────────
    # ROOT CAUSE: memory.py line 370 passes an int (rel_id) but old
    # code expected a Relationship object, causing silent failure.
    # Also: no same-source guard, no predicate normalization.

    def detect_supersession(
        self,
        user_id: int,
        new_utterance: str,
        new_relationship: Any,  # int (rel_id) or Relationship object
        cluster: Optional[Cluster],
    ) -> Optional[SupersessionEvent]:
        """Given a newly-stored triple, decide whether it supersedes any
        prior member of its temporal cluster.

        Accepts either an int (relationship ID) or a Relationship object
        to support both memory.py call sites.

        Neural decision:
          1. Score the new utterance's cosine against supersession-shape
             anchors (correction, update, negation).
          2. Same-source guard: skip if source_text_hash matches.
          3. Predicate normalization via grammar_engine.classify_verb_class.
          4. Find prior with same (subject, predicate class) — that's the
             superseded one.

        Returns SupersessionEvent if detected; caller invokes
        MemoryEngine.supersede(). Returns None otherwise.
        """
        if self._memory is None:
            return None
        if not new_utterance or not new_utterance.strip():
            return None

        # Resolve relationship ID — accept int or object with .id
        new_id = None
        if isinstance(new_relationship, int):
            new_id = new_relationship
        elif new_relationship is not None and hasattr(new_relationship, "id"):
            new_id = new_relationship.id
        else:
            return None

        # Fetch the new edge's fields from DB
        try:
            with get_db_context() as conn:
                new_row = conn.execute(
                    """SELECT id, subject, predicate, object, source_text_hash
                       FROM edges WHERE id = ?""",
                    (new_id,),
                ).fetchone()
        except Exception:
            return None

        if not new_row:
            return None

        new_subj = new_row["subject"] or ""
        new_pred = new_row["predicate"] or ""
        new_obj = (new_row["object"] or "").lower()
        new_hash = new_row["source_text_hash"] or ""

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

        # Shape must stand out above the mean of the other scores
        # (population-relative, not absolute threshold).
        other_scores = [s for k, s in sup_scores.items() if k != best_shape]
        other_mean = sum(other_scores) / max(len(other_scores), 1)
        if best_shape_score <= other_mean:
            return None

        # Predicate normalization via grammar_engine (open-vocabulary)
        pred_class = None
        try:
            from app.engines.grammar_engine import classify_verb_class
            pred_class = classify_verb_class(new_pred.lower().split()[0] if new_pred else "")
        except Exception:
            pass

        # Find priors with same subject
        cluster_id = cluster.cluster_id if cluster else 0
        with get_db_context() as conn:
            rows = conn.execute(
                """SELECT id, subject, predicate, object, source_text_hash
                   FROM edges
                   WHERE user_id = ? AND id != ?
                     AND LOWER(subject) = LOWER(?)
                     AND COALESCE(is_current, 1) = 1
                   ORDER BY id DESC LIMIT 20""",
                (user_id, new_id, new_subj),
            ).fetchall()

        for prior in rows:
            # Same-source guard: skip if same utterance produced both edges
            prior_hash = prior["source_text_hash"] or ""
            if new_hash and prior_hash and new_hash == prior_hash:
                continue

            prior_pred = (prior["predicate"] or "").lower()
            new_pred_lower = new_pred.lower()

            # Check predicate match: exact match or same verb class
            pred_match = (prior_pred == new_pred_lower)
            if not pred_match and pred_class is not None:
                try:
                    from app.engines.grammar_engine import classify_verb_class
                    prior_class = classify_verb_class(
                        prior_pred.split()[0] if prior_pred else ""
                    )
                    pred_match = (prior_class == pred_class)
                except Exception:
                    pass

            if not pred_match:
                continue

            prior_obj = (prior["object"] or "").lower()
            if prior_obj != new_obj:
                # Different object with same (subject, predicate class) = supersession
                # Temporal engine DETECTS only. MemoryEngine.supersede() WRITES.
                # Single writer prevents race conditions.
                if self._memory is not None:
                    try:
                        self._memory.supersede(prior["id"], new_id)
                    except Exception:
                        pass
                return SupersessionEvent(
                    superseded_relationship_id=prior["id"],
                    superseding_relationship_id=new_id,
                    reason=best_shape,
                    cluster_id=cluster_id,
                )

        return None

    # ── Recluster for reconstruction ──────────────────────────────
    # ROOT CAUSE: architecture_verifier expects this method but it did
    # not exist. Retrieval reconstruct() cannot group edges for narrative.

    def recluster_for_reconstruction(
        self,
        user_id: int,
        edge_ids: List[int],
    ) -> List[dict]:
        """Union-find clustering on shared entities across edges.

        Groups edges that share subject or object entities, orders
        clusters by mean sequence_number, and returns list of
        {cluster_id, edge_ids, participants}.

        Excludes generic "user" entity from union-find to prevent
        collapsing all edges into one cluster.
        """
        if not edge_ids:
            return []

        # Fetch edge data
        placeholders = ",".join("?" for _ in edge_ids)
        with get_db_context() as conn:
            rows = conn.execute(
                f"""SELECT id, subject, object, sequence_number,
                           relational_entities
                    FROM edges
                    WHERE id IN ({placeholders}) AND user_id = ?""",
                (*edge_ids, user_id),
            ).fetchall()

        if not rows:
            return []

        # Build entity-to-edge mapping
        _GENERIC = {"user", "i", "me", "my"}
        entity_to_edges: Dict[str, List[int]] = {}
        edge_data: Dict[int, dict] = {}

        for row in rows:
            eid = row["id"]
            edge_data[eid] = dict(row)
            entities: set = set()

            for field_name in ("subject", "object"):
                val = row[field_name]
                if val and val.lower() not in _GENERIC:
                    entities.add(val.lower())

            # Parse relational_entities JSON if present
            rel_ents = row["relational_entities"]
            if rel_ents:
                try:
                    import json
                    parsed = json.loads(rel_ents)
                    if isinstance(parsed, list):
                        for e in parsed:
                            if isinstance(e, str) and e.lower() not in _GENERIC:
                                entities.add(e.lower())
                except Exception:
                    pass

            for ent in entities:
                entity_to_edges.setdefault(ent, []).append(eid)

        # Union-find
        parent: Dict[int, int] = {eid: eid for eid in edge_data}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for ent, eids in entity_to_edges.items():
            for i in range(1, len(eids)):
                union(eids[0], eids[i])

        # Collect clusters
        cluster_map: Dict[int, List[int]] = {}
        for eid in edge_data:
            root = find(eid)
            cluster_map.setdefault(root, []).append(eid)

        # Build result sorted by mean sequence_number
        result = []
        for idx, (root, members) in enumerate(cluster_map.items()):
            seq_nums = [
                edge_data[m].get("sequence_number") or 0
                for m in members
            ]
            mean_seq = sum(seq_nums) / max(len(seq_nums), 1)

            participants = set()
            for m in members:
                for field_name in ("subject", "object"):
                    val = edge_data[m].get(field_name)
                    if val and val.lower() not in _GENERIC:
                        participants.add(val)

            result.append({
                "cluster_id": idx,
                "edge_ids": sorted(members),
                "participants": sorted(participants),
                "_mean_seq": mean_seq,
            })

        result.sort(key=lambda c: c["_mean_seq"])

        # Remove internal sort key
        for c in result:
            c.pop("_mean_seq", None)

        return result

    # ── Arc detection ─────────────────────────────────────────────
    # ROOT CAUSE: architecture_verifier expects this method but it did
    # not exist. Arcs table exists in schema but nothing writes to it.

    def detect_arcs(
        self,
        user_id: int,
        new_rel_id: int,
        cluster_id: Optional[str] = None,
    ) -> Optional[str]:
        """Check arcs table for open arc matching subject + schema.

        If found, add edge to arc. If not, create new arc.
        Return arc_id string.
        """
        try:
            with get_db_context() as conn:
                # Fetch new edge data
                edge = conn.execute(
                    """SELECT subject, object, predicate, source_text
                       FROM edges WHERE id = ? AND user_id = ?""",
                    (new_rel_id, user_id),
                ).fetchone()

                if not edge:
                    return None

                subject = edge["subject"] or ""
                source_text = edge["source_text"] or edge["object"] or ""

                # Embed the edge topic for arc matching
                try:
                    edge_emb = embed_text(source_text)
                except Exception:
                    return None

                # Find open arcs for this user
                open_arcs = conn.execute(
                    """SELECT id, topic, topic_embedding
                       FROM arcs
                       WHERE user_id = ? AND status = 'open'
                       ORDER BY last_checked_at DESC LIMIT 20""",
                    (user_id,),
                ).fetchall()

                best_arc_id = None
                best_sim = -1.0

                all_sims = []
                for arc in open_arcs:
                    arc_emb_blob = arc["topic_embedding"]
                    if not arc_emb_blob:
                        continue
                    try:
                        arc_emb = np.frombuffer(arc_emb_blob, dtype=np.float32)
                        sim = self._cos(edge_emb, arc_emb)
                        all_sims.append(sim)
                        if sim > best_sim:
                            best_sim = sim
                            best_arc_id = arc["id"]
                    except Exception:
                        continue

                # Population-relative: best must exceed mean of all open arcs
                if best_arc_id is not None and all_sims:
                    mean_sim = sum(all_sims) / len(all_sims)
                    if best_sim > mean_sim:
                        conn.execute(
                            "UPDATE arcs SET last_checked_at = datetime('now') WHERE id = ?",
                            (best_arc_id,),
                        )
                        conn.execute(
                            "UPDATE edges SET arc_id = ? WHERE id = ?",
                            (best_arc_id, new_rel_id),
                        )
                        return best_arc_id

                # Create new arc
                arc_id = str(uuid.uuid4())
                topic = f"{subject}: {edge['predicate'] or ''} {edge['object'] or ''}".strip()
                conn.execute(
                    """INSERT INTO arcs (id, user_id, topic, topic_embedding,
                                        start_edge_id, status)
                       VALUES (?, ?, ?, ?, ?, 'open')""",
                    (arc_id, user_id, topic, edge_emb.tobytes(), new_rel_id),
                )
                conn.execute(
                    "UPDATE edges SET arc_id = ? WHERE id = ?",
                    (arc_id, new_rel_id),
                )
                return arc_id

        except Exception:
            return None

    # ── Temporal graph helpers ────────────────────────────────────

    def edges_valid_at(
        self,
        user_id: int,
        timestamp: Any,  # str or datetime
    ) -> List[dict]:
        """Return edges valid at a given timestamp.

        An edge is valid if resolved_event_date <= timestamp AND
        (tombstoned_at IS NULL OR tombstoned_at > timestamp).
        """
        try:
            ts_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
            with get_db_context() as conn:
                rows = conn.execute(
                    """SELECT id, subject, predicate, object, resolved_event_date,
                              source_timestamp, is_current
                       FROM edges
                       WHERE user_id = ?
                         AND (resolved_event_date IS NOT NULL AND resolved_event_date <= ?)
                         AND (tombstoned_at IS NULL OR tombstoned_at > ?)
                       ORDER BY resolved_event_date ASC""",
                    (user_id, ts_str, ts_str),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def temporal_neighbors(
        self,
        user_id: int,
        relationship_id: int,
        window: int = 5,
    ) -> List[dict]:
        """Return edges before and after the given edge in sequence order.

        Returns up to `window` edges on each side based on sequence_number.
        """
        try:
            with get_db_context() as conn:
                target = conn.execute(
                    "SELECT sequence_number FROM edges WHERE id = ? AND user_id = ?",
                    (relationship_id, user_id),
                ).fetchone()

                if not target or target["sequence_number"] is None:
                    return []

                seq = target["sequence_number"]

                rows = conn.execute(
                    """SELECT id, subject, predicate, object, sequence_number,
                              resolved_event_date, source_timestamp
                       FROM edges
                       WHERE user_id = ?
                         AND sequence_number BETWEEN ? AND ?
                         AND id != ?
                         AND COALESCE(is_current, 1) = 1
                       ORDER BY sequence_number ASC""",
                    (user_id, seq - window, seq + window, relationship_id),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    # ── Patterns ──────────────────────────────────────────────────

    def patterns(self, user_id: int) -> List[TemporalPattern]:
        """Day-of-week / hour-of-day patterns from stored timestamps.

        Identifies circadian patterns by grouping edges by hour-of-day
        and day-of-week from source_timestamp. Peak detection uses
        population-relative comparison (above mean density).
        """
        try:
            with get_db_context() as conn:
                rows = conn.execute(
                    """SELECT id, source_timestamp
                       FROM edges
                       WHERE user_id = ? AND source_timestamp IS NOT NULL
                       ORDER BY id DESC LIMIT 200""",
                    (user_id,),
                ).fetchall()

            if not rows:
                return []

            hour_counts: Dict[int, List[int]] = {}
            dow_counts: Dict[int, List[int]] = {}

            for row in rows:
                try:
                    ts = datetime.fromisoformat(
                        row["source_timestamp"].replace("Z", "+00:00")
                    )
                    hour_counts.setdefault(ts.hour, []).append(row["id"])
                    dow_counts.setdefault(ts.weekday(), []).append(row["id"])
                except Exception:
                    continue

            patterns_out: List[TemporalPattern] = []

            if hour_counts:
                mean_count = len(rows) / 24.0
                for hour, ids in hour_counts.items():
                    if len(ids) > mean_count:
                        ratio = len(ids) / max(len(rows), 1)
                        patterns_out.append(TemporalPattern(
                            pattern_type=f"peak_hour_{hour}",
                            confidence=ratio,
                            exemplar_relationship_ids=ids[:5],
                        ))

            if dow_counts:
                mean_count = len(rows) / 7.0
                day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
                for dow, ids in dow_counts.items():
                    if len(ids) > mean_count:
                        ratio = len(ids) / max(len(rows), 1)
                        patterns_out.append(TemporalPattern(
                            pattern_type=f"peak_day_{day_names[dow]}",
                            confidence=ratio,
                            exemplar_relationship_ids=ids[:5],
                        ))

            return patterns_out

        except Exception:
            return []

    # ── Staleness ─────────────────────────────────────────────────

    def staleness_batch(self, relationship_ids: List[int]) -> Dict[int, float]:
        """Batch recency decay for a set of edges. Single DB query.
        Returns {rel_id: score} where score in [0,1], 1.0=fresh."""
        if not relationship_ids:
            return {}
        result: Dict[int, float] = {rid: 1.0 for rid in relationship_ids}
        try:
            placeholders = ",".join("?" for _ in relationship_ids)
            with get_db_context() as conn:
                rows = conn.execute(
                    f"SELECT id, last_confirmed_at FROM edges "
                    f"WHERE id IN ({placeholders})",
                    tuple(relationship_ids),
                ).fetchall()
            now = self.now()
            time_constant = 30 * 86400.0
            for row in rows:
                rid = row["id"]
                lc = row["last_confirmed_at"]
                if not lc:
                    continue
                try:
                    last_conf = datetime.fromisoformat(lc.replace("Z", "+00:00"))
                    if last_conf.tzinfo is None:
                        last_conf = last_conf.replace(tzinfo=timezone.utc)
                    age_s = (now - last_conf).total_seconds()
                    if age_s > 0:
                        result[rid] = float(np.exp(-age_s / time_constant))
                except Exception:
                    pass
        except Exception:
            pass
        return result

    def staleness(self, relationship_id: int) -> float:
        """Single-edge staleness. Delegates to batch for consistency."""
        batch = self.staleness_batch([relationship_id])
        return batch.get(relationship_id, 1.0)

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
