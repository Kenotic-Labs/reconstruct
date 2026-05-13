"""
Backbone — orchestrates all engines.

Single entry point for the architecture. Handles all routing:
  1. Question vs statement (grammar engine classification)
  2. Retrieval vs conversational (memory engine context window)
  3. Situational vs factual (wh_type classification)

Product wrappers (SDK, MCP, Raya) call this. Never the other way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ProcessResult:
    """What the architecture returns. Product wrappers read this."""
    action: str          # "stored", "answered", "reconstructed", "forgot", "skipped"
    answer: Optional[str] = None
    edge_ids: List[int] = field(default_factory=list)
    grounding: List[str] = field(default_factory=list)
    return_field: str = "episodic"
    refusal: bool = False
    refusal_reason: str = ""
    edges_stored: int = 0
    proactive: List = field(default_factory=list)
    ambiguities: List = field(default_factory=list)


def process(
    text: str,
    *,
    user_id: int = 0,
    speaker: str = "user",
    listener: Optional[str] = None,
    speaker_is_user: bool = True,
    source_timestamp: Optional[str] = None,
) -> ProcessResult:
    """Architecture entry point. Text in, result out.

    Routing (all decided by engines, not by product code):
      1. Grammar engine classifies: question / command / backchannel / statement
      2. Memory engine context window: retrieval mode vs conversational mode
      3. WH-type engine: situational vs factual

    Always writes first (every turn may contain facts).
    Then answers if in retrieval mode and it's a question.
    """
    if not text or not text.strip():
        return ProcessResult(action="skipped")

    # ── Step 1: Classify (grammar engine) ───────────────────────
    from app.engines.grammar_engine import _get_nlp, classify_utterance

    doc = _get_nlp()(text)
    utt = classify_utterance(doc)

    if utt.is_backchannel:
        return ProcessResult(action="skipped")

    if utt.is_command:
        # Route to forget
        from app.engines.memory import get_memory_engine
        mem = get_memory_engine()
        # Extract entity to forget from text
        for tok in doc:
            if tok.dep_ in ("dobj", "pobj") and tok.pos_ in ("PROPN", "NOUN"):
                count = mem.forget(user_id, by="entity", scope=tok.text)
                return ProcessResult(action="forgot", edges_stored=count)
        return ProcessResult(action="skipped")

    # ── Step 2: Check retrieval mode BEFORE writing ─────────────
    # (writing adds to context buffer, which would pollute the check)
    from app.engines.memory import get_memory_engine
    mem = get_memory_engine()

    is_question = utt.is_question
    retrieval = is_question and mem.is_retrieval_mode(user_id, speaker)

    # ── Step 3: Always write ────────────────────────────────────
    # Every turn may contain facts — even questions have imposed facts.
    # Memory engine handles per-decomposition mood filtering.
    count = mem.ingest_text(
        user_id=user_id,
        text=text,
        speaker=speaker,
        listener=listener,
        speaker_is_user=speaker_is_user,
        source_timestamp=source_timestamp,
    )

    ambiguities = list(mem.pending_ambiguities)
    mem.pending_ambiguities.clear()

    # ── Step 4: Answer if retrieval mode ────────────────────────
    if retrieval:
        # Step 4a: Situational vs factual (wh_type engine)
        try:
            from app.engines.reconstruction import is_situational
            if is_situational(text):
                return _reconstruct_situation(user_id, text, count)
        except Exception:
            pass

        # Step 4b: Factual retrieval (reconstruction engine)
        from app.engines.reconstruction import reconstruct
        rr = reconstruct(user_id, text)

        return ProcessResult(
            action="answered",
            answer=rr.answer,
            edge_ids=rr.edge_ids,
            grounding=rr.grounding,
            return_field=rr.return_field,
            refusal=rr.refusal,
            refusal_reason=rr.refusal_reason,
            edges_stored=count,
            ambiguities=ambiguities,
        )

    # ── Step 5: Statement or conversational question — stored ───
    return ProcessResult(
        action="stored",
        edges_stored=count,
        ambiguities=ambiguities,
    )


def _reconstruct_situation(user_id: int, text: str, edges_stored: int) -> ProcessResult:
    """Situational reconstruction — converge all traces by schema."""
    from app.engines.reconstruction import reconstruct
    rr = reconstruct(user_id, text)
    return ProcessResult(
        action="reconstructed",
        answer=rr.answer,
        edge_ids=rr.edge_ids,
        grounding=rr.grounding,
        refusal=rr.refusal,
        refusal_reason=rr.refusal_reason,
        edges_stored=edges_stored,
    )
