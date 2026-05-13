"""Architecture-level types used by engines."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Ambiguity:
    """A detected conflict between new and existing data."""
    entity: str          # what's being talked about ("bus route")
    old_value: str       # what was stored before ("42")
    new_value: str       # what was just said ("44")
    old_edge_id: int     # the existing edge
    new_edge_id: int     # the new edge
    question: str        # what the LLM should ask ("Is route 44 replacing route 42?")
