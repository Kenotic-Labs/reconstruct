"""
Nura continuity layer — the five engines.

Every domain concern lives in exactly ONE engine. No parallel flows between
engines. Each engine is a class with a narrow public interface; internal
state lives behind that interface.

    MemoryEngine        — relationships + entities + derived traces (the store)
    RetrievalEngine     — single read path (DTCM convergence + coherence check)
    TemporalEngine      — parse + knowledge graph + clusters + supersession
    ProactiveEngine     — arcs + timers + frequency
    AdaptabilityEngine  — user style learning

Infrastructure (not engines): vector/embedder, db/session, services/*,
voice/*, guards/*, core/*, integration/backbone, preprocessing/speech_cleaner.

Rule: if it's a domain concern, it belongs in an engine. No free-floating
utility modules implementing domain logic.
"""
from app.engines.memory import MemoryEngine
from app.engines.retrieval import RetrievalEngine
from app.engines.temporal import TemporalEngine
from app.engines.proactive import ProactiveEngine
from app.engines.adaptability import AdaptabilityEngine

__all__ = [
    "MemoryEngine",
    "RetrievalEngine",
    "TemporalEngine",
    "ProactiveEngine",
    "AdaptabilityEngine",
]
