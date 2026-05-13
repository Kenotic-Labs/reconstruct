"""Typed return shapes exposed by the SDK.

Standalone definitions — no dependency on any engine module.
Root cause: sdk/types.py imported Situation/Cluster/Answer from
app.engines.retrieval which was deleted (v3 regression revert).
These types are now self-contained so the SDK compiles regardless
of which engine sits behind it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Answer:
    """Single factual answer."""
    text: str = ""
    grounding: List[str] = field(default_factory=list)
    edge_ids: List[int] = field(default_factory=list)
    return_field: str = "episodic"
    refusal: bool = False
    refusal_reason: str = ""


@dataclass
class Cluster:
    """Group of related edges in a situation."""
    cluster_id: str = ""
    edge_ids: List[int] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)


@dataclass
class Situation:
    """Reconstructed living state — narrative summary."""
    narrative: str = ""
    clusters: List[Cluster] = field(default_factory=list)
    grounding: List[str] = field(default_factory=list)
    edge_ids: List[int] = field(default_factory=list)


# Ambiguity lives in architecture (app/types.py). Re-export for SDK users.
from app.types import Ambiguity


@dataclass
class ProcessResult:
    """Result of k.process() — wraps whatever the architecture produced."""
    action: str          # "stored", "answered", "reconstructed", "forgot", "skipped", "proactive"
    result: Any = None   # int (store count), Answer, Situation, list, None
    proactive: List = field(default_factory=list)
    triples_stored: int = 0
    ambiguities: List = field(default_factory=list)  # List[Ambiguity]


__all__ = ["Situation", "Cluster", "Answer", "ProcessResult"]
