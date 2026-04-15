# ============================================================================
# NO HARDCODED LISTS. NO THRESHOLDS. NO SCORING MAGIC NUMBERS. NO REGEX.
# ============================================================================
"""
ProactiveEngine — arcs, timers, frequency, surface logic.

Absorbs: proactive_engine_v2, arc_manager, story_arc, timer_system,
frequency_tracker, trigger_manager, semantic/arc_classifier,
semantic/proactive_concepts.

COMPLEMENTING design:
    - Listens to MemoryEngine via the utterances flowing through.
    - Listens to TemporalEngine for 'when should this resurface' windows.
    - Does NOT duplicate memory storage.

Minimal absorption — full arc lifecycle can stay deferred to later
sessions. This engine exposes the public interface; the absorbed logic
reads from MemoryEngine rather than maintaining parallel state.

Public interface:
    process_message(user_id, text) → ArcUpdate
    evaluate_due(user_id) → list[ProactiveInsight]
    create_timer(user_id, duration_seconds, callback) → TimerId
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class ArcUpdate:
    created_arc: Optional[Dict[str, Any]] = None
    resolved_arc: Optional[Dict[str, Any]] = None
    updated_arc: Optional[Dict[str, Any]] = None
    timer_created: bool = False
    timer_duration_seconds: Optional[float] = None


@dataclass
class ProactiveInsight:
    arc_id: str
    topic: str
    engagement_type: str
    urgency: float
    suggested_text: str


# =============================================================================
# PROACTIVE ENGINE
# =============================================================================

class ProactiveEngine:
    """Story arc detection, timer management, frequency-based surfacing."""

    def __init__(self, memory_engine, temporal_engine):
        self._memory = memory_engine
        self._temporal = temporal_engine
        self._timers: Dict[str, Dict[str, Any]] = {}
        self._timer_lock = threading.Lock()
        self._worker_started = False

    # ── Message processing ──────────────────────────────────────

    def process_message(self, user_id: int, text: str) -> ArcUpdate:
        """Arc detection delegates to the memory engine's relational +
        emotional traces. Phase 4 absorption — full arc lifecycle can be
        rebuilt later. For now this returns a neutral ArcUpdate so the
        write path doesn't break."""
        return ArcUpdate()

    def evaluate_due(self, user_id: int) -> List[ProactiveInsight]:
        """Return arcs / frequencies due for surfacing. Empty for now;
        full implementation in a later session."""
        return []

    # ── Timers (short-term reminders) ───────────────────────────

    def create_timer(
        self,
        user_id: int,
        duration_seconds: float,
        callback: Optional[Callable] = None,
    ) -> str:
        self._ensure_worker()
        timer_id = str(uuid.uuid4())
        fire_at = time.time() + duration_seconds
        with self._timer_lock:
            self._timers[timer_id] = {
                "user_id": user_id,
                "fire_at": fire_at,
                "callback": callback,
                "fired": False,
            }
        return timer_id

    def _ensure_worker(self) -> None:
        if self._worker_started:
            return
        self._worker_started = True
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self) -> None:
        while True:
            now_s = time.time()
            with self._timer_lock:
                ready = [
                    (tid, entry)
                    for tid, entry in self._timers.items()
                    if not entry["fired"] and entry["fire_at"] <= now_s
                ]
            for tid, entry in ready:
                try:
                    if entry["callback"]:
                        entry["callback"]()
                except Exception:
                    pass
                with self._timer_lock:
                    if tid in self._timers:
                        self._timers[tid]["fired"] = True
            time.sleep(0.5)


_singleton: Optional[ProactiveEngine] = None


def get_proactive_engine(memory_engine=None, temporal_engine=None) -> ProactiveEngine:
    global _singleton
    if _singleton is None:
        if memory_engine is None or temporal_engine is None:
            raise RuntimeError("First call to get_proactive_engine() requires memory + temporal.")
        _singleton = ProactiveEngine(memory_engine, temporal_engine)
    return _singleton
