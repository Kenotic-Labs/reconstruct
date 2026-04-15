"""Candidate dataclass shared across RetrievalEngine stages.

Each stage of the Moat pipeline annotates the Candidate with its own
contribution. No stage computes a weighted combined score — structural
stages filter, cosine stages order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Set


@dataclass
class Candidate:
    relationship_id: int
    edge: Dict[str, Any]
    entry_cosine: float = 0.0
    entity_overlap: int = 0
    cluster_members: int = 0
    hops_to_entity: int = -1
    exit_cosine: float = 0.0
    source_stages: Set[str] = field(default_factory=set)
