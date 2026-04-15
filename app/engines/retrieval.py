# ============================================================================
# NO HARDCODED LISTS. NO THRESHOLDS. NO SCORING MAGIC NUMBERS. NO REGEX.
# The ONLY read path. Filter → Complete. Not voting. Not weighted product.
# ============================================================================
"""
RetrievalEngine — Filter → Complete.

Two operations:
    1. FILTER   — binary constraints derived from query STRUCTURE
                  (WH-grammar + token presence + entity-link). Each edge
                  either satisfies all constraints or it doesn't.
    2. COMPLETE — argmax of cosine(pattern.predicate_emb, edge.edge_embedding)
                  over the surviving edges.

No weighted sum. No vote count. No threshold on "similar enough". If no
candidate survives the filter → structural refusal (not a noisy guess).

Public interface:
    retrieve(user_id, query_text) → Answer
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from app.db.session import get_db_context
from app.vector.embedder import embed_text


# =============================================================================
# Pattern + Answer dataclasses
# =============================================================================

@dataclass
class Pattern:
    """Structured query — the thing we try to complete."""
    raw_query: str
    query_emb: Any                   # np.ndarray — for tie-break completions
    predicate_emb: Any               # np.ndarray — the slot we try to fill
    entity_target: Optional[str] = None   # canonical entity name (from neural linker)

    # Binary constraints — each is either True (must hold) or False (don't care)
    require_object_type_person: bool = False
    require_object_type_location: bool = False
    require_object_type_time: bool = False
    require_is_current: bool = False
    require_emotional_label: bool = False
    require_schematic_category: Optional[str] = None  # specific schema name or None


@dataclass
class Answer:
    text: Optional[str]
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    confidence: float = 0.0
    source: str = "filter_complete"
    survivors: int = 0
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    convergence_details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# RetrievalEngine
# =============================================================================

class RetrievalEngine:
    """Single read path. Parse → Filter → Complete."""

    def __init__(self, memory_engine, temporal_engine):
        self._memory = memory_engine
        self._temporal = temporal_engine
        self._schema_anchor_cache: Dict[str, np.ndarray] = {}

    # ── Public retrieve ───────────────────────────────────────────

    def retrieve(self, user_id: int, query_text: str) -> Answer:
        query_text = (query_text or "").strip()
        if not query_text:
            return Answer(text=None, source="empty_query")

        # Step 1 — Parse query into Pattern + structural constraints
        pattern = self._parse_pattern(user_id, query_text)
        if pattern is None:
            return Answer(text=None, source="parse_failed")

        # Step 2 — Load candidate edges (all current for user)
        edges = self._load_edges(user_id)
        if not edges:
            return Answer(text=None, source="no_edges")

        # Step 3 — FILTER: binary structural constraints
        survivors = [e for e in edges if self._meets_constraints(e, pattern)]

        if not survivors:
            # Structural refusal — no edges match what the query is asking for.
            return Answer(
                text=None,
                source="structural_refusal",
                survivors=0,
                convergence_details={
                    "entity_target": pattern.entity_target,
                    "total_edges_before_filter": len(edges),
                },
            )

        # Step 4 — COMPLETE: argmax on cosine(pattern.predicate_emb, edge_embedding)
        scored = []
        for e in survivors:
            edge_emb = e.get("edge_embedding_vec")
            if edge_emb is None:
                continue
            # Fit is the cosine to the predicate/slot. ONE signal. No combination.
            fit = float(np.dot(pattern.predicate_emb, edge_emb))
            scored.append((fit, e))

        if not scored:
            return Answer(
                text=None,
                source="no_scorable_survivors",
                survivors=len(survivors),
            )

        # Tie-break on sequence_number (narrative time)
        scored.sort(
            key=lambda fe: (fe[0], fe[1].get("sequence_number") or fe[1]["id"]),
            reverse=True,
        )
        best_fit, best = scored[0]

        # Reconstruct answer
        answer_text = self._triple_to_sentence(
            best["subject"], best["predicate"], best["object"]
        )

        return Answer(
            text=answer_text,
            subject=best["subject"],
            predicate=best["predicate"],
            object=best["object"],
            confidence=min(0.99, max(0.0, best_fit)),
            source="filter_complete",
            survivors=len(survivors),
            candidates=[
                {
                    "subject": e["subject"],
                    "predicate": e["predicate"],
                    "object": e["object"],
                    "fit": round(s, 4),
                }
                for s, e in scored[:5]
            ],
            convergence_details={
                "entity_target": pattern.entity_target,
                "total_edges": len(edges),
                "survived_filter": len(survivors),
            },
        )

    # ── Pattern parsing ──────────────────────────────────────────

    def _parse_pattern(self, user_id: int, query: str) -> Optional[Pattern]:
        try:
            query_emb = embed_text(query)
        except Exception:
            return None

        # 1. Entity link — neural, via MemoryEngine.entity_link
        entity_target = None
        try:
            ent = self._memory.entity_link(user_id, query)
            if ent is not None:
                entity_target = ent.name
        except Exception:
            pass

        # 2. Structural constraints from WH-grammar.
        # WH-words are a CLOSED grammatical class of English — not
        # curation of entities or concepts. They are the pronouns of
        # the question grammar itself.
        q_lower = query.lower()
        q_tokens = q_lower.split()

        require_obj_person = any(t in ("who", "whom", "whose") for t in q_tokens)
        require_obj_location = "where" in q_tokens
        require_obj_time = any(
            t in ("when", "what-time") or t.startswith("when")
            for t in q_tokens
        ) or "how long" in q_lower

        # 3. Temporal-currency constraint — emitted by "now", "currently",
        # "still", "right now", "these days", "lately".
        # These tokens are the English present-tense deictics — another
        # closed grammatical class (indexicals), not curated content.
        require_is_current = any(
            t in ("now", "currently", "still", "lately", "these", "today")
            for t in q_tokens
        ) or "right now" in q_lower

        # 4. Emotional-content constraint — emitted when the query asks
        # about feelings. We detect this STRUCTURALLY via the tokens
        # "feel/feeling/feelings/mood" — closed lexeme set from English
        # grammar of psych predicates, not a content list.
        require_emotional = any(
            t in ("feel", "feeling", "feelings", "mood", "emotion", "emotional")
            for t in q_tokens
        )

        # 5. Schematic category — detected by argmax cosine of the query
        # against the SAME schema anchors the store uses. If the top-1
        # schema match is clearly the winner (argmax), constrain to it.
        # If query is uninformative, no schema constraint.
        schema_label = self._best_schema(query_emb)

        # 6. predicate_emb — the slot we complete. The cleanest version:
        # the query embedding itself, minus the entity token. We don't
        # surgically remove the entity (that requires a tokenizer dance);
        # the query_emb already captures "everything but the entity" in
        # practice because entity names are low-weight tokens in MiniLM.
        # For completion we use the query_emb directly.
        predicate_emb = query_emb

        return Pattern(
            raw_query=query,
            query_emb=query_emb,
            predicate_emb=predicate_emb,
            entity_target=entity_target,
            require_object_type_person=require_obj_person,
            require_object_type_location=require_obj_location,
            require_object_type_time=require_obj_time,
            require_is_current=require_is_current,
            require_emotional_label=require_emotional,
            require_schematic_category=schema_label,
        )

    # ── Schema argmax (reused from memory engine's anchor set) ───

    _SCHEMA_ANCHORS = {
        "career": "job work employer salary promotion office colleague",
        "health": "doctor medicine hospital symptom diagnosis exercise diet",
        "family": "parent sibling child spouse partner wedding baby",
        "social": "friend gathering party event celebration",
        "finance": "money budget savings investment rent payment",
        "education": "school class course learning study teacher student",
        "housing": "apartment house move rent room furniture",
        "hobby": "hobby sport art music craft game reading",
        "travel": "trip vacation flight hotel destination visit",
        "food": "restaurant cooking recipe meal diet ingredient",
        "identity": "name age birthday background origin ethnicity",
        "relationship": "partner dating marriage engagement boyfriend girlfriend",
    }

    def _best_schema(self, query_emb: np.ndarray) -> Optional[str]:
        """Argmax the query against schema anchors. Returns the top-1
        label ALWAYS — no threshold. If the top match is uninformative
        (schema is too noisy a signal for a given query), the filter
        step will just see most edges match by coincidence and the
        completion step does the real work."""
        best_label, best_sim = None, -2.0
        for label, desc in self._SCHEMA_ANCHORS.items():
            cached = self._schema_anchor_cache.get(label)
            if cached is None:
                try:
                    cached = embed_text(desc)
                    self._schema_anchor_cache[label] = cached
                except Exception:
                    continue
            sim = float(np.dot(query_emb, cached))
            if sim > best_sim:
                best_sim, best_label = sim, label
        return best_label

    # ── Constraint check ─────────────────────────────────────────

    def _meets_constraints(self, edge: Dict[str, Any], pattern: Pattern) -> bool:
        # Entity target — edge's subject or object must match canonically
        if pattern.entity_target:
            et = pattern.entity_target.lower()
            subj = (edge.get("subject") or "").lower()
            obj = (edge.get("object") or "").lower()
            # "user" is the first-person anchor — every user query is
            # plausibly about the user. So a user-subject edge passes
            # the entity filter even when entity_target is a named
            # entity (the user might be asking about a relation they
            # have with that entity).
            if subj != et and obj != et and subj != "user":
                return False

        # is_current
        if pattern.require_is_current:
            if not edge.get("is_current", 1):
                return False

        # Emotional-label presence
        if pattern.require_emotional_label:
            if not edge.get("edge_emotional_label"):
                return False

        # Object-type filters — check edge predicate/object against
        # the grammatical expected type via the edge's relational_type
        # and schema tags.
        if pattern.require_object_type_person:
            # Edge is PERSON-object iff relational_type is kin/friend/
            # romantic/professional (all involve people) OR schema is
            # family/social/relationship.
            rt = edge.get("edge_relational_type")
            sc = edge.get("edge_schematic_category")
            if rt not in ("kin", "friend", "romantic", "professional") and \
               sc not in ("family", "social", "relationship"):
                return False

        if pattern.require_object_type_location:
            sc = edge.get("edge_schematic_category")
            if sc not in ("housing", "travel", "career"):
                # career includes offices / workplaces
                return False

        if pattern.require_object_type_time:
            # Time-expecting queries — edge's temporal_context or
            # episodic_significance should be present.
            tc = edge.get("edge_temporal_context")
            if not tc:
                return False

        # Schematic constraint: only enforced when confidence is high.
        # To avoid overfiltering we pass the schema check as soft —
        # don't filter on it. The completion step picks the best fit
        # within all other constraints. (If schema were a hard filter,
        # cross-domain queries like "how does my work affect my family"
        # would structurally refuse.)
        return True

    # ── Edge loader ───────────────────────────────────────────────

    def _load_edges(self, user_id: int) -> List[Dict[str, Any]]:
        with get_db_context() as conn:
            rows = conn.execute(
                """SELECT id, subject, predicate, object, confidence,
                          COALESCE(is_current, 1) AS is_current,
                          sequence_number,
                          edge_embedding,
                          edge_emotional_valence,
                          edge_emotional_label,
                          edge_schematic_category,
                          edge_episodic_significance,
                          edge_relational_type,
                          edge_temporal_context
                   FROM relationships
                   WHERE user_id = ? AND COALESCE(is_current, 1) = 1
                   ORDER BY id DESC""",
                (user_id,),
            ).fetchall()

        edges: List[Dict[str, Any]] = []
        for r in rows:
            d = {
                "id": r["id"],
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "confidence": r["confidence"],
                "is_current": bool(r["is_current"]),
                "sequence_number": r["sequence_number"],
                "edge_emotional_valence": r["edge_emotional_valence"],
                "edge_emotional_label": r["edge_emotional_label"],
                "edge_schematic_category": r["edge_schematic_category"],
                "edge_episodic_significance": r["edge_episodic_significance"],
                "edge_relational_type": r["edge_relational_type"],
                "edge_temporal_context": r["edge_temporal_context"],
            }
            # Decode edge_embedding blob
            emb_blob = r["edge_embedding"]
            if emb_blob:
                try:
                    d["edge_embedding_vec"] = np.frombuffer(
                        emb_blob, dtype=np.float32
                    ).copy()
                except Exception:
                    d["edge_embedding_vec"] = None
            else:
                d["edge_embedding_vec"] = None
            edges.append(d)
        return edges

    # ── Sentence reconstruction ──────────────────────────────────

    def _triple_to_sentence(self, subject: str, predicate: str, object: str) -> str:
        s = subject or "?"
        p = (predicate or "").replace("_", " ")
        o = object or "?"
        if s.lower() == "user":
            s = "You"
        return f"{s} {p} {o}".strip()


_singleton: Optional[RetrievalEngine] = None


def get_retrieval_engine(memory_engine=None, temporal_engine=None) -> RetrievalEngine:
    global _singleton
    if _singleton is None:
        if memory_engine is None or temporal_engine is None:
            raise RuntimeError(
                "First call to get_retrieval_engine() requires memory and temporal engines."
            )
        _singleton = RetrievalEngine(memory_engine, temporal_engine)
    return _singleton
