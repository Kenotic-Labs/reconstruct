"""Compatibility types for legacy test imports."""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Candidate:
    """A retrieval candidate edge."""
    edge_id: int = 0
    subject: str = ""
    predicate: str = ""
    object: str = ""
    source_text: str = ""
    score: float = 0.0
