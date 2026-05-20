"""
Kenotic — the main SDK client.

One class, one entry point: process().

  k = Kenotic(user_id=1, db_path="test.db")
  result = k.process("I got a job at Google starting Tuesday.")
  result = k.process("Where do I work?")
  result = k.process("What's going on in my life?")
  result = k.process("Forget about Google.")

process() classifies intent via the grammar engine and routes internally
to the appropriate capability (ingest, retrieve, reconstruct, forget,
check_proactive). Returns a ProcessResult with action discriminator.

The 8 internal methods (ingest, retrieve, reconstruct, forget, show,
trace, check_proactive, profile) remain available for direct use by
runners and tests that need fine-grained control.

No runtime levers. Validation, grammar, and grounding are all always
on in the SDK variant.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from sdk.types import Situation, Answer, ProcessResult

_log = logging.getLogger("kenotic.sdk")


class Kenotic:
    """Continuity-layer client. One per user/database."""

    def __init__(
        self,
        user_id: int = 0,
        db_path: Union[str, Path] = "~/.kenotic/memory.db",
        *,
        embed_device: str = "cuda",
    ):
        """Initialize an isolated continuity layer bound to a SQLite file.

        Args:
          user_id: integer partition key. Defaults to 0 (single-user mode
                   used by the MCP Continuity Bridge). Edges are stored
                   under this ID and never cross to other user_ids at
                   retrieval time.
          db_path: path to the SQLite file. Created if missing.
          embed_device: 'cuda' or 'cpu' for MiniLM embedding inference.
        """
        self.user_id = int(user_id)
        self.db_path = str(db_path)

        # Wire env used by the engines before import-time side effects
        os.environ.setdefault("RAYA_EMBED_DEVICE", embed_device)

        # Point the engines at the caller's DB path
        from config.settings import settings
        settings.sqlite_path = self.db_path

        # Ensure schema exists
        self._init_schema()

        # Lazy engine construction — deferred to first use to keep
        # __init__ cheap
        self._memory = None
        self._temporal = None

        # kenoticArchitectureV1 runtime verifier — runs ONCE per instance.
        # Pure observability: delegates to architecture_verifier, logs summary.
        # Never blocks init, never changes control flow.
        self._architecture_status: List[Dict[str, str]] = self._run_verifier()

    # -- Architecture verifier --------------------------------------

    def _run_verifier(self) -> List[Dict[str, str]]:
        """Run kenoticArchitectureV1 and log a one-line summary.

        Returns the full results list for caching. If the verifier itself
        fails (import error, DB issue), returns an empty list and logs the
        exception — never crashes init.
        """
        try:
            from scripts.architecture_verifier import (
                verify_kenotic_architecture_v1,
            )
            results = verify_kenotic_architecture_v1(db_path=self.db_path)
        except Exception:
            _log.warning(
                "kenoticArchitectureV1: verifier could not run",
                exc_info=True,
            )
            return []

        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = sum(1 for r in results if r["status"] == "FAIL")
        total = len(results)

        _log.info(
            "kenoticArchitectureV1: %d/%d PASS, %d FAIL",
            passed, total, failed,
        )
        for r in results:
            if r["status"] == "FAIL":
                _log.warning(
                    "  FAIL: %s -- %s", r["check"], r["detail"],
                )
        return results

    def architecture_status(self) -> List[Dict[str, str]]:
        """Return cached kenoticArchitectureV1 verifier results.

        Each entry is a dict with keys: check, status, detail.
        Ran once at init; this method returns the cached snapshot.
        """
        return self._architecture_status

    # -- Private engine loader --------------------------------------

    def _engines(self):
        # Root cause: old code imported get_retrieval_engine from
        # app.engines.retrieval which was deleted after v3 regression.
        # Guard changed from _retrieval to _memory. Retrieval replaced
        # by reconstruction engine (app.engines.reconstruction).
        if self._memory is None:
            from app.engines.memory import get_memory_engine
            from app.engines.temporal import get_temporal_engine
            self._memory = get_memory_engine()
            self._temporal = get_temporal_engine()
            self._temporal.bind_memory(self._memory)
        return self._memory, self._temporal

    def _init_schema(self):
        import sqlite3
        from app.db.models import MIGRATIONS, run_schema_upgrades
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript(MIGRATIONS)
        run_schema_upgrades(conn)
        conn.commit()
        conn.close()

    # -- Public API -------------------------------------------------

    def ingest(
        self,
        text: str,
        *,
        source_timestamp: Optional[str] = None,
        speaker: Optional[str] = None,
        listener: Optional[str] = None,
        speaker_is_user: bool = True,
        confidence: float = 0.9,
        model_response: Optional[str] = None,
        llm_id: Optional[str] = None,
    ) -> int:
        """Extract triples from raw text and store them.

        The singular write path runs:
        ingestion cleanup -> grammar engine extraction -> temporal engine
        -> memory store.

        Source identity:
          - User input:    source_tag="user", speaker=user's name
          - LLM response:  source_tag="llm:{llm_id}", speaker=llm_id
          "I" always resolves to speaker — grammar engine handles it.

        Args:
          text: the raw user utterance or conversation turn.
          source_timestamp: ISO datetime of the utterance.
          speaker: who said it. "I" resolves to this name.
          confidence: 0.0—1.0 confidence for every resulting triple.
          model_response: the AI's response text.
          llm_id: which LLM generated model_response (e.g. "claude",
                  "gpt", "cursor", "grok"). Stored in source_tag as
                  "llm:{llm_id}" for provenance tracking.

        Returns:
          Combined number of triples stored from both passes.
        """
        memory, _ = self._engines()

        # User input — source_tag = "user"
        count = memory.ingest_text(
            user_id=self.user_id,
            text=text,
            source_timestamp=source_timestamp,
            speaker=speaker,
            listener=listener,
            speaker_is_user=speaker_is_user,
            confidence=confidence,
            source_tag="user",
        )

        # LLM response — source_tag = "llm:{llm_id}"
        # "I" in LLM output resolves to llm_id.
        # "you" in LLM output resolves to the user (speaker).
        if model_response and model_response.strip():
            _llm_speaker = llm_id or "assistant"
            _llm_tag = f"llm:{llm_id}" if llm_id else "llm:unknown"
            count += memory.ingest_text(
                user_id=self.user_id,
                text=model_response,
                source_timestamp=source_timestamp,
                speaker=_llm_speaker,
                listener=speaker,  # LLM's "you" = the user
                speaker_is_user=False,
                confidence=confidence,
                source_tag=_llm_tag,
            )
        return count

    def retrieve(self, query: str) -> Union[Answer, Situation]:
        """Query the continuity layer.

        Situational queries ('summarize X', 'why is Y', 'what's
        happening with Z') route to reconstruction and return a
        Situation. Lookup queries ('who is Maya's manager?') route to
        Filter->Complete and return an Answer.

        Routing is determined by WH-grammar + discourse verbs in the
        query — not by a flag.
        """
        # Route situational queries to the reconstruction path.
        # When explicit_reconstruct_only is True, only fire reconstruct()
        # if the query contains an explicit trigger word ("reconstruct",
        # "what's going on", "summarize", "tell me about").
        # When False, is_situational() routes automatically via WH-grammar.
        from config.settings import settings
        try:
            if settings.explicit_reconstruct_only:
                _lower = query.lower()
                _triggers = ("reconstruct", "what's going on", "whats going on",
                             "summarize", "tell me about", "what is going on",
                             "what's happening", "whats happening",
                             "how is everything", "catch me up",
                             "give me a summary", "overview")
                if any(t in _lower for t in _triggers):
                    return self.reconstruct(query)
            else:
                from app.engines.wh_type import is_situational
                if is_situational(query):
                    return self.reconstruct(query)
        except Exception:
            pass  # fail-open: fall through to lookup

        # Root cause: old code called retrieval.retrieve() from deleted
        # app.engines.retrieval. Now routes through reconstruction engine.
        from app.engines.reconstruction import reconstruct as _reconstruct
        rr = _reconstruct(self.user_id, query)
        if rr.refusal:
            # Use reconstruction engine's answer if it provides one (e.g. low-coverage CWA),
            # otherwise fall back to the standard refusal text.
            refusal_text = rr.answer or "This information is not mentioned in the conversation."
            return Answer(
                text=refusal_text,
                refusal=True, refusal_reason=rr.refusal_reason,
            )
        return Answer(
            text=rr.answer or "",
            grounding=rr.grounding,
            edge_ids=rr.edge_ids,
            return_field=rr.return_field,
        )

    def forget(
        self,
        by: str,            # 'entity' | 'time_range' | 'source' | 'triple_id'
        scope,              # str | int | (str, str) tuple / list
    ) -> int:
        """Tombstone memory by entity, time range, source tag, or triple id.

        Returns the count of tombstones emitted. Idempotent — calling
        again with the same scope returns 0.
        """
        memory, _ = self._engines()
        if by == 'entity':
            return memory.forget_by_entity(self.user_id, scope)
        if by == 'time_range':
            start, end = scope  # tuple/list unpack
            return memory.forget_by_time_range(self.user_id, start, end)
        if by == 'source':
            return memory.forget_by_source(self.user_id, scope)
        if by == 'triple_id':
            return memory.forget_by_triple_id(self.user_id, int(scope))
        raise ValueError(f"Unknown forget scope: {by}")

    def show(
        self,
        facet: str,         # 'time' | 'entity' | 'source' | 'trace'
        value: Optional[str] = None,
        limit: int = 100,
        export_raw_text: bool = False,
    ) -> list:
        """Return a browsable view of stored memory.

        Never returns trace internals (edge_*, embeddings, convergence
        state, salience). If export_raw_text=True, includes the original
        source_text field. Otherwise, returns only fact summaries.
        """
        memory, _ = self._engines()
        rows = memory.list_by_facet(
            self.user_id, facet, value=value, limit=limit
        )
        if not export_raw_text:
            for r in rows:
                r.pop('source_text', None)
        return rows

    def trace(self, subject: str, predicate: str) -> list:
        """Return the full supersession history for a subject+predicate pair.

        Surfaces every object value ever stored for this (subject, predicate)
        combination — both active and superseded — ordered oldest to newest.
        Each entry is a dict with keys: id, object, is_active,
        first_learned_at, source_timestamp, source_tag, superseded_by,
        superseded_at, sequence_number.

        Pure delegation to MemoryEngine.trace(). No logic added here.
        """
        memory, _ = self._engines()
        return memory.trace(self.user_id, subject, predicate)

    def check_proactive(self) -> list:
        """Return proactive insights (arcs due for surfacing).

        Queries the arcs table for open arcs. Returns a list of
        ProactiveInsight dataclasses. Returns [] if the arcs table
        doesn't exist or no arcs are due.
        """
        from app.engines.proactive import get_proactive_engine
        memory, temporal = self._engines()
        proactive = get_proactive_engine(memory, temporal)
        return proactive.evaluate_due(self.user_id)

    def profile(self):
        """Return the current adaptation profile for this user.

        Returns an AdaptationProfile dataclass with warmth, formality,
        initiative, and check_in_frequency dimensions (all 0.0—1.0).
        """
        from app.engines.adaptability import get_adaptability_engine
        memory, _ = self._engines()
        adapt = get_adaptability_engine(memory)
        return adapt.profile(self.user_id)


    def process(
        self,
        text: str,
        *,
        speaker: str = "user",
        listener: Optional[str] = None,
        speaker_is_user: bool = True,
        source_timestamp: Optional[str] = None,
        model_response: Optional[str] = None,
        check_proactive: bool = False,
        llm_id: Optional[str] = None,
    ) -> ProcessResult:
        """Thin wrapper — all routing lives in backbone.process().

        This method only handles SDK-specific concerns (proactive checks,
        model_response storage, llm_id tagging). The architectural
        decisions (question vs statement, retrieval vs conversational,
        situational vs factual) are all in app.engines.backbone.
        """
        # Proactive-only mode: no text, just check for due arcs.
        if check_proactive and not text.strip():
            insights = self.check_proactive()
            return ProcessResult(
                action="proactive",
                result=insights,
                proactive=insights,
            )

        # Architecture handles everything
        from app.engines.backbone import process as arch_process
        result = arch_process(
            text,
            user_id=self.user_id,
            speaker=speaker,
            listener=listener,
            speaker_is_user=speaker_is_user,
            source_timestamp=source_timestamp,
        )

        # SDK wrapping: convert backbone.ProcessResult → sdk.ProcessResult
        sdk_result = ProcessResult(
            action=result.action,
            result=result.answer if result.action == "answered" else result.edges_stored,
            triples_stored=result.edges_stored,
            ambiguities=result.ambiguities,
        )

        # SDK-specific: proactive check after store
        if check_proactive and result.action == "stored":
            insights = self.check_proactive()
            sdk_result.proactive = insights

        return sdk_result

    @staticmethod
    def _parse_forget_target(text: str) -> Optional[str]:
        """Extract the entity target from a forget/delete/remove command.

        Strips the imperative verb and common prepositions to isolate the
        entity name. Returns None if nothing remains.
        """
        stripped = text.strip().rstrip(".").rstrip("!")
        tokens = stripped.split()
        if not tokens:
            return None
        # Drop the imperative verb
        rest = tokens[1:]
        # Drop leading prepositions ("about", "all about", "everything about")
        skip = {"about", "all", "everything"}
        while rest and rest[0].lower() in skip:
            rest = rest[1:]
        target = " ".join(rest).strip()
        return target if target else None

    def reconstruct(self, query: str) -> Situation:
        """Force the reconstruction path.

        Root cause: old code called retrieval.reconstruct() from deleted
        app.engines.retrieval. Now routes through reconstruction engine.
        """
        from app.engines.reconstruction import reconstruct as _reconstruct
        rr = _reconstruct(self.user_id, query)
        return Situation(
            narrative=rr.answer or "",
            grounding=rr.grounding,
            edge_ids=rr.edge_ids,
        )


# -- Module-level singleton -----------------------------------------------

_singleton: Optional[Kenotic] = None


def Reconstruct(
    text: str,
    *,
    speaker: str = "user",
    listener: Optional[str] = None,
    speaker_is_user: bool = True,
    source_timestamp: Optional[str] = None,
    model_response: Optional[str] = None,
    llm_id: Optional[str] = None,
    check_proactive: bool = False,
    user_id: int = 0,
    db_path: Union[str, Path] = "~/.kenotic/memory.db",
    embed_device: str = "cuda",
    locomo_mode: bool = False,
):
    """The entire Kenotic continuity architecture in ONE function call.

    Pipeline:
        Text in
          → Ingestion (messy → clean)
          → Grammar Engine (clean → 5 traces + SPO + PQs + types)
          → Temporal Engine (event dates, supersession, arcs)
          → Memory Engine (writes to edges + edge_extraction)
          → Reconstruction Engine (reads from edges, answers questions)

    Routing:
        - Statements  → ingestion → grammar → temporal → memory (write)
        - Questions   → reconstruction (read)
        - Commands    → forget (soft tombstone)
        - Backchannels → skip

    Source identity:
        User input:   speaker="Sam", source_tag="user"
        LLM response: llm_id="claude" → speaker="claude", source_tag="llm:claude"
        "I" resolves to speaker. Grammar engine handles it.

    Args:
        text: any English text -- statement, question, command, anything.
        speaker: who said it (for pronoun resolution). "I" → speaker name.
        source_timestamp: ISO datetime of the utterance.
        model_response: the AI's response text.
        llm_id: which LLM generated model_response ("claude", "gpt",
                "cursor", "grok"). Stored as source_tag="llm:{llm_id}".
        check_proactive: if True with empty text, returns arcs due for surfacing.
        user_id: partition key (default 0 for single-user).
        db_path: SQLite file path.
        embed_device: 'cuda' or 'cpu'.
        locomo_mode: if True, forces short factual answers.

    Returns:
        ProcessResult -- contains action, result, proactive insights, triples_stored.
    """
    global _singleton
    if _singleton is None or _singleton.db_path != str(db_path) or _singleton.user_id != int(user_id):
        _singleton = Kenotic(user_id=user_id, db_path=db_path, embed_device=embed_device)

    # LOCOMO mode: force explicit_reconstruct_only so all questions go
    # through factual retrieval path, not situational reconstruction.
    # This produces short answers that match LOCOMO gold format.
    if locomo_mode:
        from config.settings import settings
        settings.explicit_reconstruct_only = True

    return _singleton.process(
        text,
        speaker=speaker,
        listener=listener,
        speaker_is_user=speaker_is_user,
        source_timestamp=source_timestamp,
        model_response=model_response,
        llm_id=llm_id,
        check_proactive=check_proactive,
    )
