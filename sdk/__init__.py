"""
Reconstruct from Kenotic — the continuity layer as a library.

Thin wrapper over app.engines.* with an opinionated contract:
  - Raw user text in, structured continuity out
  - Validation gate ALWAYS ON (SDK assumes messy input)
  - Grammar engine ALWAYS ON (SDK renders to English for no-LLM callers)
  - No runtime levers, no env flags

Public API:
    Reconstruct             — one-function entry point
    Kenotic                 — class client (fine-grained control)
    Situation, Cluster, Answer — typed return shapes

Usage:
    from sdk import Reconstruct
    Reconstruct("Maya started at Vantage Systems in 2023.")
    Reconstruct("Where does Maya work?")
    Reconstruct("What's going on in Maya's life?")
"""
from sdk.client import Kenotic, Reconstruct
from sdk.types import Situation, Cluster, Answer

__version__ = "0.1.0"
__all__ = ["Kenotic", "Reconstruct", "Situation", "Cluster", "Answer", "__version__"]
