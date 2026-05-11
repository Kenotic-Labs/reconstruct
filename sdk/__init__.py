"""
kenotic-sdk — the continuity layer as a library.

Thin wrapper over app.engines.* with an opinionated contract:
  - Raw user text in, structured continuity out
  - Validation gate ALWAYS ON (SDK assumes messy input)
  - Grammar engine ALWAYS ON (SDK renders to English for no-LLM callers)
  - No runtime levers, no env flags

Public API:
    Kenotic                 — main client
    Situation, Cluster, Answer — typed return shapes

Usage:
    from sdk import Kenotic
    k = Kenotic(user_id=1, db_path="./my_memory.db")
    k.ingest("Maya started at Vantage Systems in 2023.")
    a = k.retrieve("Where does Maya work?")
    s = k.reconstruct("Summarize Maya's situation.")
"""
from sdk.client import Kenotic, KenoticV1
from sdk.types import Situation, Cluster, Answer

__version__ = "0.1.0"
__all__ = ["Kenotic", "KenoticV1", "Situation", "Cluster", "Answer", "__version__"]
