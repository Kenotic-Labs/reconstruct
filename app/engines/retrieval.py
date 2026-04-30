# -*- coding: cp1252 -*-
# ============================================================================
# NO HARDCODED LISTS. NO THRESHOLDS. NO SCORING MAGIC NUMBERS. NO REGEX.
# Question-to-question cosine + binary type-coherence gate +
# sequence_number tie-break. Empty set -> StructuralRefusal.
# ============================================================================
"""
RetrievalEngine -- question-to-question indexing read path.

Pipeline:

    1. Embed the query.
    2. Cosine over predicted_queries.question_embedding for this user
       (all rows on live relationships).
    3. Dedupe by relationship_id (an edge can have several PQs).
    4. Binary coherence gate: WH-type -> relationship.subject_type or
       object_type must match. No thresholds.
    5. Tie-break by sequence_number DESC.
    6. On empty result set -> StructuralRefusal.

reconstruct() shares steps 1..3, then groups by cluster_id and fuses
each cluster through the existing deterministic grammar engine.

Cosine lives ONLY at:
  (a) question-to-question retrieval (this file), and
  (b) entity_link at write time (MemoryEngine).

Everywhere else: structural rules only.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from app.db.session import get_db_context
from app.vector import embedder as _embedder_module
from app.vector.embedder import embed_text
from app.engines.wh_type import parse_expected_answer_type
from app.engines.retrieval_types import Candidate

log = logging.getLogger(__name__)

# How many deduped edges we keep before the coherence gate / grouping.
# Structural cutoff derived from narrative density (not a tuned score):
# one "narrative day" is ~10 edges; we take two days' worth of nearest
# question-neighbors and let the binary filters decide.
_COSINE_POOL_SIZE = 20


# =============================================================================
# Answer / Cluster / Situation / StructuralRefusal dataclasses
# =============================================================================


@dataclass
class Answer:
    text: Optional[str]
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    confidence: float = 0.0
    source: str = "pq_cosine"
    survivors: int = 0
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    convergence_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralRefusal:
    """First-class output when the pipeline yields no coherent answer.

    reason codes:
        empty_query          -- input text was blank
        no_edges             -- user has no live relationships
        no_coherent_answer   -- top-k exists but no candidate matches the
                                WH-type coherence gate
    """
    reason: str
    # Match Answer's public surface so downstream doesn't break on
    # logging / serialization.
    text: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    confidence: float = 0.0
    source: str = "structural_refusal"
    survivors: int = 0
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    convergence_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Cluster:
    key_type: str                                 # 'cluster_id' | 'unclustered'
    key_value: str
    edges: List[Dict[str, Any]] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    dominant_mood: Optional[str] = None
    mean_valence: Optional[float] = None
    pivotal_edge_ids: List[int] = field(default_factory=list)
    timeline_edge_ids: List[int] = field(default_factory=list)


@dataclass
class Situation:
    narrative: str
    clusters: List[Cluster] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    dominant_mood: Optional[str] = None
    pivotal_events: List[int] = field(default_factory=list)
    timeline: List[int] = field(default_factory=list)
    grounding_map: Dict[int, List[int]] = field(default_factory=dict)
    source: str = "reconstruct"
    survivors: int = 0
    convergence_details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# RetrievalEngine
# =============================================================================


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    try:
        da = float(np.linalg.norm(a))
        db = float(np.linalg.norm(b))
        if da == 0.0 or db == 0.0:
            return 0.0
        return float(np.dot(a, b) / (da * db))
    except Exception:
        return 0.0


def _cosine_from_blob(q_emb: np.ndarray, blob):
    """Cosine between a query vector and a float32 embedding BLOB.
    Returns 0.0 on null blob or size mismatch — never raises."""
    if blob is None:
        return 0.0
    v = np.frombuffer(blob, dtype=np.float32)
    if v.size != q_emb.size:
        return 0.0
    denom = float(np.linalg.norm(q_emb) * np.linalg.norm(v))
    if denom == 0.0:
        return 0.0
    return float(np.dot(q_emb, v) / denom)


def _deserialize_emb(blob) -> Optional[np.ndarray]:
    if blob is None:
        return None
    try:
        return np.frombuffer(blob, dtype=np.float32).copy()
    except Exception:
        return None


class RetrievalEngine:
    """Single read path. Cosine(PQ) -> coherence -> tie-break."""

    _ENTRY_POOL_SIZE = 80
    _ENTITY_EDGE_LIMIT = 10

    def __init__(self, memory_engine=None, temporal_engine=None):
        self._memory = memory_engine
        self._temporal = temporal_engine

    # ── Stage 1: Entry Cosine ────────────────────────────────────

    def _stage1_entry(self, user_id: int, q_emb: np.ndarray) -> List[Candidate]:
        """Entry Cosine: max(pq_cosine, edge_cosine, predicate_cosine) per
        relationship_id, top-K by entry_cosine. All three cosine
        dimensions always run — no gating. Stores each separately so
        Exit can rank by the three-signal max while Entry casts the wide
        net.

        predicate_cosine isolates the *relational* dimension of the
        triple: cosine(query, embed(predicate)).  This discriminates
        between same-subject edges whose entities produce similar PQ
        and edge embeddings but whose predicates differ (the core
        wrong_ranking pattern)."""
        pq_rows = self._fetch_pq_rows(user_id)
        edge_rows = self._fetch_edge_rows(user_id)

        by_rid: Dict[int, Candidate] = {}

        # PQ pool first — multiple PQs per edge possible; keep the max.
        for row in pq_rows:
            rid = row["relationship_id"]
            cos = _cosine_from_blob(q_emb, row.get("question_embedding"))
            existing = by_rid.get(rid)
            if existing is None:
                cand = Candidate(
                    relationship_id=rid, edge=dict(row),
                    pq_cosine=cos,
                )
                cand.source_stages.add("entry")
                by_rid[rid] = cand
            elif cos > existing.pq_cosine:
                existing.pq_cosine = cos

        # Edge pool.
        for row in edge_rows:
            rid = row["relationship_id"]
            cos = _cosine_from_blob(q_emb, row.get("edge_embedding"))
            existing = by_rid.get(rid)
            if existing is None:
                cand = Candidate(
                    relationship_id=rid, edge=dict(row),
                    edge_cosine=cos,
                )
                cand.source_stages.add("entry")
                by_rid[rid] = cand
            elif cos > existing.edge_cosine:
                existing.edge_cosine = cos

        # Predicate pool — embed the predicate text for each unique edge,
        # giving the query a relational-alignment signal that is
        # independent of which entities fill the subject/object slots.
        _pred_emb_cache: Dict[str, np.ndarray] = {}
        for c in by_rid.values():
            pred_text = (c.edge.get("predicate") or "").replace("_", " ")
            if not pred_text:
                continue
            if pred_text not in _pred_emb_cache:
                _pred_emb_cache[pred_text] = embed_text(pred_text)
            c.predicate_cosine = _cosine(q_emb, _pred_emb_cache[pred_text])

        # entry_cosine = max(pq, edge, predicate) — recall key.
        for c in by_rid.values():
            c.entry_cosine = max(c.pq_cosine, c.edge_cosine,
                                 c.predicate_cosine)

        out = sorted(by_rid.values(), key=lambda c: -c.entry_cosine)
        return out[: self._ENTRY_POOL_SIZE]

    # ── Stage 2: Expand (entity union) ───────────────────────────

    def _count_entity_overlap(
        self, names: Set[str], subject: str, object_: str,
    ) -> Tuple[int, int]:
        """Return (total_overlap, non_self_overlap).
        Exact case-sensitive match of entity name to edge
        subject/object fields — same semantics as the original
        entity_overlap computation."""
        total = 0
        non_self = 0
        for n in names:
            if n == subject or n == object_:
                total += 1
                if n != "user":
                    non_self += 1
        return total, non_self

    def _stage2_expand(
        self, user_id: int, query_text: str, candidates: List[Candidate]
    ) -> List[Candidate]:
        from app.engines import entity_resolver
        entities = entity_resolver.resolve_query_entities(user_id, query_text)

        # Self-reference invariant: every query is from the user's own
        # perspective. Kenotic canonicalizes first-person mentions to
        # subject='user', so the query entity set always includes 'user'
        # even when no proper-noun entity is resolvable. This is
        # structural, not a heuristic — it's the data-model contract.
        names = {e["name"] for e in entities}
        names.add("user")
        if not names:  # unreachable, but defensive
            return candidates
        by_rid: Dict[int, Candidate] = {
            c.relationship_id: c for c in candidates
        }

        placeholders = ",".join("?" * len(names))
        sql = f"""
            SELECT r.* FROM relationships r
             WHERE r.user_id = ?
               AND COALESCE(r.is_current, 1) = 1
               AND r.tombstoned_at IS NULL
               AND (r.subject IN ({placeholders})
                    OR r.object IN ({placeholders}))
        """
        params = [user_id] + list(names) + list(names)
        with get_db_context() as conn:
            rows = conn.execute(sql, params).fetchall()

        for row in rows:
            rid = row["id"]
            overlap, non_self = self._count_entity_overlap(
                names, row["subject"] or "", row["object"] or "",
            )
            if rid in by_rid:
                cand = by_rid[rid]
                if overlap > cand.entity_overlap:
                    cand.entity_overlap = overlap
                if non_self > cand.non_self_entity_overlap:
                    cand.non_self_entity_overlap = non_self
                cand.source_stages.add("expand")
            else:
                cand = Candidate(relationship_id=rid, edge=dict(row),
                                 entity_overlap=overlap,
                                 non_self_entity_overlap=non_self)
                cand.source_stages.add("expand")
                by_rid[rid] = cand

        # Also score existing candidates that happen to touch entities.
        for rid, cand in by_rid.items():
            if cand.entity_overlap == 0:
                e = cand.edge
                overlap, non_self = self._count_entity_overlap(
                    names, e.get("subject") or "", e.get("object") or "",
                )
                if overlap:
                    cand.entity_overlap = overlap
                    cand.non_self_entity_overlap = non_self
                    cand.source_stages.add("expand")

        # Schematic co-expansion: pull in edges that share BOTH an
        # entity AND an edge_schematic_category with current anchors.
        # This expands within category boundaries (career stays with
        # career, health with health) rather than pulling in unrelated
        # edges that happen to mention the same person.
        anchor_categories = {
            c.edge.get("edge_schematic_category")
            for c in by_rid.values()
            if c.entity_overlap > 0 and c.edge.get("edge_schematic_category")
            and c.edge.get("edge_schematic_category") != "uncategorized"
        }
        if anchor_categories and names:
            cat_placeholders = ",".join("?" * len(anchor_categories))
            name_placeholders = ",".join("?" * len(names))
            schema_sql = f"""
                SELECT r.* FROM relationships r
                 WHERE r.user_id = ?
                   AND COALESCE(r.is_current, 1) = 1
                   AND r.tombstoned_at IS NULL
                   AND r.edge_schematic_category IN ({cat_placeholders})
                   AND (r.subject IN ({name_placeholders})
                        OR r.object IN ({name_placeholders}))
            """
            schema_params = (
                [user_id]
                + list(anchor_categories)
                + list(names)
                + list(names)
            )
            with get_db_context() as conn:
                schema_rows = conn.execute(schema_sql, schema_params).fetchall()
            for row in schema_rows:
                rid = row["id"]
                if rid in by_rid:
                    continue
                overlap, non_self = self._count_entity_overlap(
                    names, row["subject"] or "", row["object"] or "",
                )
                cand = Candidate(
                    relationship_id=rid, edge=dict(row),
                    entity_overlap=overlap,
                    non_self_entity_overlap=non_self,
                )
                cand.source_stages.add("expand")
                by_rid[rid] = cand

        return list(by_rid.values())

    # ── Stage 3: Group (cluster ∪ arc anchor) ────────────────────

    def _stage3_group(self, candidates: List[Candidate]) -> List[Candidate]:
        anchor_clusters = {
            c.edge.get("cluster_id") for c in candidates
            if c.entity_overlap > 0 and c.edge.get("cluster_id")
        }
        anchor_arcs = {
            c.edge.get("arc_id") for c in candidates
            if c.entity_overlap > 0 and c.edge.get("arc_id")
        }
        if not anchor_clusters and not anchor_arcs:
            return candidates  # inapplicable — pass through

        survivors = [
            c for c in candidates
            if (c.edge.get("cluster_id") in anchor_clusters
                or c.edge.get("arc_id") in anchor_arcs)
        ]

        cluster_counts = Counter(
            c.edge.get("cluster_id") for c in survivors
            if c.edge.get("cluster_id")
        )
        for c in survivors:
            cid = c.edge.get("cluster_id")
            if cid:
                c.cluster_members = cluster_counts.get(cid, 0)
            c.source_stages.add("group")

        return survivors

    # ── Stage 4: Relate (BFS filter) ─────────────────────────────

    def _stage4_relate(
        self, user_id: int, query_text: str, candidates: List[Candidate]
    ) -> List[Candidate]:
        from app.engines import entity_resolver
        from app.engines.memgraph import (
            build_adjacency, min_hops_to_entities,
        )

        entities = entity_resolver.resolve_query_entities(user_id, query_text)

        # Self-reference invariant — see _stage2_expand. When the
        # candidate pool actually contains 'user' edges (Kenotic's
        # first-person canonical subject), add 'user' to the query
        # entity set so BFS starts from the user's subgraph too.
        # Otherwise the data model doesn't have a user anchor and we
        # fall back to purely named entities.
        has_user_edges = any(
            c.edge.get("subject") == "user" or c.edge.get("object") == "user"
            for c in candidates
        )
        query_names = {e["name"] for e in entities}
        if has_user_edges:
            query_names.add("user")
        if not query_names:
            return candidates  # inapplicable — pass through

        sql = """
            SELECT subject, object FROM relationships
             WHERE user_id = ?
               AND COALESCE(is_current, 1) = 1
               AND tombstoned_at IS NULL
        """
        with get_db_context() as conn:
            all_edges = [dict(r) for r in conn.execute(sql, (user_id,))]
        adj = build_adjacency(all_edges)

        survivors: List[Candidate] = []
        for c in candidates:
            h = min_hops_to_entities(
                subject=c.edge.get("subject") or "",
                object_=c.edge.get("object") or "",
                query_entities=query_names, adj=adj,
            )
            if h >= 0:
                c.hops_to_entity = h
                c.source_stages.add("relate")
                survivors.append(c)
        return survivors

    # ── Stage 5: Exit Cosine ─────────────────────────────────────

    def _stage5_exit(
        self, q_emb: np.ndarray, candidates: List[Candidate]
    ) -> List[Candidate]:
        """Exit Cosine: max(pq_cosine, edge_cosine, predicate_cosine).
        Three cosine dimensions, each measuring a different aspect of
        relevance:

        - pq_cosine: answerability (does a predicted question match?)
        - edge_cosine: surface similarity (does the full triple match?)
        - predicate_cosine: relational alignment (does the *relation
          type* match what the query is asking about?)

        Additive — never discards a signal, just picks the strongest
        dimension per candidate. The three-signal max widens precision
        coverage: when two same-subject edges tie on PQ and edge
        cosine, predicate_cosine can break the tie structurally
        by favouring the edge whose predicate aligns with the query's
        relational intent."""
        for c in candidates:
            c.exit_cosine = max(c.pq_cosine, c.edge_cosine,
                                c.predicate_cosine)
            c.source_stages.add("exit")
        return sorted(candidates, key=lambda c: -c.exit_cosine)

    # ── Stage 6: Validate (warn-only) ────────────────────────────

    def _stage6_validate(
        self, query_text: str, top: Candidate
    ) -> Optional[str]:
        """Warn-only. Returns warning string or None. Never drops."""
        expected = parse_expected_answer_type(query_text)
        if expected is None:
            return None
        if (top.edge.get("object_type") == expected
                or top.edge.get("subject_type") == expected):
            return None
        return "type_mismatch"

    # ── Public API ───────────────────────────────────────────────

    def _try_facts_lookup(
        self, user_id: int, query_text: str,
    ) -> Optional[Answer]:
        """Direct fact lookup. Returns Answer if a fact matches, None
        otherwise. Falls through to the moat pipeline on any miss.

        Uses classify_query to get match_subject + match_schema, then
        builds the fact key schema::VerbClass::subject. The facts table
        has exactly ONE current value per key — no ranking needed."""
        try:
            from app.engines.grammar_engine import (
                classify_query, classify_verb_class,
            )
            qd = classify_query(query_text)
            # Skip facts for temporal queries — facts store values not dates.
            if qd.return_field == "temporal":
                return None
            subject = qd.match_subject or qd.match_entity
            schema = qd.match_schema
            if not subject or not schema:
                return None

            verb_class = "UNKNOWN"
            if qd.match_predicate:
                vc = classify_verb_class(qd.match_predicate)
                verb_class = vc.name

            with get_db_context() as conn:
                # Try exact key
                key = f"{schema}::{verb_class}::{subject}"
                row = conn.execute(
                    "SELECT value, confidence FROM facts "
                    "WHERE user_id = ? AND key = ?",
                    (user_id, key),
                ).fetchone()
                if row and row["value"]:
                    return Answer(
                        text=row["value"],
                        confidence=row["confidence"] or 1.0,
                        source="fact_fast_path",
                        convergence_details={
                            "query": query_text, "fact_key": key,
                        },
                    )

                # No wildcard fallback — exact key only. The wildcard
                # schema::%::subject is too loose (career::%::Sam matches
                # any career fact regardless of what the query asks).
        except Exception:
            pass
        return None

    def retrieve(self, user_id: int, query_text: str):
        """Moat pipeline — Lookup mode.
        Step 0: facts fast-path (O(1) for stative facts).
        Then: Entry → Expand → Group → Relate → Exit Cosine → Validate → top-1."""
        query_text = (query_text or "").strip()
        if not query_text:
            return StructuralRefusal(reason="empty_query")

        # Step 0: Facts fast-path
        fact_answer = self._try_facts_lookup(user_id, query_text)
        if fact_answer is not None:
            return fact_answer

        try:
            q_emb = _embedder_module.embed_text(query_text)
        except Exception as e:
            return StructuralRefusal(
                reason="embed_failed",
                convergence_details={"error": str(e)[:200]},
            )

        candidates = self._stage1_entry(user_id, q_emb)
        candidates = self._stage2_expand(user_id, query_text, candidates)
        if not candidates:
            return StructuralRefusal(
                reason="no_candidates",
                convergence_details={"query": query_text},
            )

        candidates = self._stage3_group(candidates)
        candidates = self._stage4_relate(user_id, query_text, candidates)
        if not candidates:
            return StructuralRefusal(
                reason="no_structural_match",
                convergence_details={"query": query_text},
            )

        candidates = self._stage5_exit(q_emb, candidates)

        # Final ranking: three-signal exit_cosine with
        # sequence_number DESC tie-break. Structural signals
        # (entity_overlap, cluster_members, hops_to_entity) already
        # gated the pool at Stages 2-4 — they decided who survives,
        # not who wins at the top. The three-signal max
        # (pq + edge + predicate) is the final discriminator —
        # pq measures answerability, edge measures surface similarity,
        # and predicate measures relational alignment.
        # Infer schema for tiebreaking — same function as write path.
        _inferred_schema = None
        try:
            from app.engines.grammar_engine import (
                _get_nlp as _ge_nlp, _get_root as _ge_root,
                classify_verb_class, _reclassify_location_by_object,
                _extract_schematic, _VERB_CLASS_TO_SCHEMA,
                _noun_to_schema_via_wordnet,
            )
            _qdoc = _ge_nlp()(query_text)
            _qroot = _ge_root(_qdoc)
            # Strategy 1: content noun IS a schema name.
            # Pull actual schemas from DB + verb class map (no hardcoded set).
            _known = set(_VERB_CLASS_TO_SCHEMA.values())
            try:
                with get_db_context() as _sc:
                    for _sr in _sc.execute(
                        "SELECT DISTINCT edge_schematic_category FROM relationships WHERE edge_schematic_category IS NOT NULL"
                    ).fetchall():
                        if _sr[0]:
                            _known.add(_sr[0].lower())
            except Exception:
                pass
            for tok in _qdoc:
                if tok.pos_ == "NOUN" and not tok.is_stop and tok.lemma_.lower() in _known:
                    _inferred_schema = tok.lemma_.lower()
                    break
            # Strategy 1b: noun-to-schema via WordNet
            if not _inferred_schema:
                for tok in _qdoc:
                    if tok.pos_ == "NOUN" and not tok.is_stop:
                        _ns = _noun_to_schema_via_wordnet(tok.lemma_)
                        if _ns and _ns != "uncategorized":
                            _inferred_schema = _ns
                            break
            # Strategy 2: verb class
            if not _inferred_schema and _qroot is not None:
                _v = _qroot
                if _qroot.pos_ == "AUX":
                    for ch in _qroot.children:
                        if ch.dep_ in ("xcomp", "ccomp") and ch.pos_ == "VERB":
                            _v = ch
                            break
                _vc = classify_verb_class(_v.lemma_)
                _vc = _reclassify_location_by_object(_qdoc, _v, _vc)
                _s = _extract_schematic(_qdoc, _qroot, _vc)
                if _s and _s != "uncategorized":
                    _inferred_schema = _s
        except Exception:
            pass

        # Detect if query is present-tense via spaCy morph (no word list).
        # Past tense on ROOT or AUX → query asks about the past.
        _query_is_present = True
        try:
            _adv_doc = _adv_nlp(query_text)
            for _tok in _adv_doc:
                if _tok.dep_ in ("ROOT", "aux") and "Past" in _tok.morph.get("Tense", []):
                    _query_is_present = False
                    break
                # "used to" construction: VBD lemma="use" + xcomp
                if (_tok.lemma_ == "use" and _tok.tag_ == "VBD"
                        and any(c.dep_ == "xcomp" for c in _tok.children)):
                    _query_is_present = False
                    break
        except Exception:
            pass

        # exit_cosine first — PQ cosine=1.0 (exact match) must win.
        # Schema and is_current are tiebreakers only.
        _schema_l = (_inferred_schema or "").lower()
        candidates.sort(
            key=lambda c: (
                -c.exit_cosine,
                -(1 if _schema_l and (c.edge.get("edge_schematic_category") or "").lower() == _schema_l else 0),
                -(1 if _query_is_present and not c.edge.get("is_historical") else 0),
                -(c.edge.get("sequence_number") or 0),
            )
        )

        top = candidates[0]

        # ── Adversarial speaker filter ─────────────────────────────
        # If the query mentions a specific entity (NER PERSON/ORG/GPE),
        # verify the winning edge is ABOUT that entity.
        _query_entities: list = []
        try:
            from app.engines.grammar_engine import _get_nlp as _ge_get_nlp
            _adv_nlp = _ge_get_nlp()  # cached, not reloaded
            _qdoc = _adv_nlp(query_text)
            _query_entities = [
                ent.text.lower() for ent in _qdoc.ents
                if ent.label_ in ("PERSON", "ORG", "GPE")
            ]
            if not _query_entities:
                _query_entities = [
                    tok.text.lower() for tok in _qdoc
                    if tok.pos_ == "PROPN"
                ]
        except Exception:
            pass

        if _query_entities:
            def _edge_matches_entity(edge_dict, qe_list):
                """Check if any query entity appears in the edge's text.
                Also maps 'user' ↔ query entity (edges store subject='user'
                for first-person, but queries use the speaker's name)."""
                _text = " ".join([
                    (edge_dict.get("subject") or ""),
                    (edge_dict.get("object") or ""),
                    (edge_dict.get("relational_entities") or ""),
                    (edge_dict.get("source_text") or ""),
                ]).lower()
                # Direct match
                if any(qe in _text for qe in qe_list):
                    return True
                # "user" ↔ speaker mapping: if subject is "user", the
                # edge is first-person. Check relational_entities for
                # the speaker's real name (always appended by write path).
                if (edge_dict.get("subject") or "").lower() == "user":
                    rel = (edge_dict.get("relational_entities") or "").lower()
                    if any(qe in rel for qe in qe_list):
                        return True
                return False

            if not _edge_matches_entity(top.edge, _query_entities):
                _found_match = False
                for alt in candidates[1:]:
                    if _edge_matches_entity(alt.edge, _query_entities):
                        top = alt
                        _found_match = True
                        break
                if not _found_match:
                    return StructuralRefusal(
                        reason="no_entity_match",
                        convergence_details={
                            "query": query_text,
                            "query_entities": _query_entities,
                        },
                    )

        warning = self._stage6_validate(query_text, top)

        subj = top.edge.get("subject") or ""
        pred = top.edge.get("predicate") or ""
        obj = top.edge.get("object") or ""
        source_text = (top.edge.get("source_text") or "").strip()

        # ── Answer extraction by wh_type ───────────────────────────
        # Return the SHORT answer from the edge's structural fields,
        # not a full reconstruction. LoCoMo F1 scores against short
        # gold answers — verbose text kills precision.
        expected_type = parse_expected_answer_type(query_text)
        if expected_type == "TIME":
            # Temporal: prefer resolved_event_date or temporal_expression
            answer_text = (
                top.edge.get("temporal_expression")
                or top.edge.get("resolved_event_date")
                or obj
            )
        elif expected_type == "PERSON":
            # Relational: the entity that ISN'T the query entity.
            # "user" is canonical first-person, never the answer.
            _subj_l = subj.lower()
            if _subj_l == "user":
                answer_text = obj
            elif _query_entities and any(qe in _subj_l for qe in _query_entities):
                answer_text = obj
            else:
                answer_text = subj
        elif expected_type == "LOCATION":
            answer_text = obj or subj
        else:
            # Episodic default: object is the short noun phrase
            answer_text = obj

        # If answer is empty or very short, fall back to source_text
        if not answer_text or len(answer_text.strip()) <= 1:
            answer_text = source_text

        return Answer(
            text=answer_text,
            subject=subj, predicate=pred, object=obj,
            confidence=1.0, source="moat_pipeline_lookup",
            survivors=len(candidates),
            convergence_details={
                "query": query_text,
                "entry_cosine": top.entry_cosine,
                "entity_overlap": top.entity_overlap,
                "non_self_entity_overlap": top.non_self_entity_overlap,
                "cluster_members": top.cluster_members,
                "hops_to_entity": top.hops_to_entity,
                "exit_cosine": top.exit_cosine,
                "predicate_cosine": top.predicate_cosine,
                "sequence_number": top.edge.get("sequence_number"),
                "source_stages": sorted(top.source_stages),
                "validate_warning": warning,
                "source_text": source_text,
            },
        )

    _RECONSTRUCT_TOP_N = 20

    def reconstruct(self, user_id: int, query_text: str) -> Situation:
        """Moat pipeline — Reconstruction mode. Same stages 1-5 as
        retrieve(). Output: top-N grouped by cluster_id, fused via the
        existing grammar engine."""
        query_text = (query_text or "").strip()
        if not query_text:
            return Situation(
                narrative="", source="structural_refusal",
                convergence_details={"reason": "empty_query"},
            )
        try:
            q_emb = _embedder_module.embed_text(query_text)
        except Exception as e:
            return Situation(
                narrative="", source="structural_refusal",
                convergence_details={"reason": "embed_failed",
                                     "error": str(e)[:200]},
            )

        candidates = self._stage1_entry(user_id, q_emb)
        candidates = self._stage2_expand(user_id, query_text, candidates)
        if not candidates:
            return Situation(
                narrative="", source="structural_refusal", survivors=0,
                convergence_details={"reason": "no_candidates"},
            )
        candidates = self._stage3_group(candidates)
        candidates = self._stage4_relate(user_id, query_text, candidates)
        candidates = self._stage5_exit(q_emb, candidates)

        # Same Moat lexicographic ranking as Lookup — pick top-N for fuse.
        candidates.sort(
            key=lambda c: (
                -c.exit_cosine,
                -(c.edge.get("sequence_number") or 0),
            )
        )
        survivors = candidates[: self._RECONSTRUCT_TOP_N]
        if not survivors:
            return Situation(
                narrative="", source="structural_refusal", survivors=0,
                convergence_details={"reason": "no_structural_match"},
            )

        # Read-time reclustering via TemporalEngine — entity-graph
        # connected components instead of fragmented write-time cluster_id.
        survivor_edges = [c.edge for c in survivors]
        if (self._temporal is not None
                and hasattr(self._temporal, 'recluster_for_reconstruction')):
            groups = self._temporal.recluster_for_reconstruction(survivor_edges)
        else:
            # Fallback: write-time cluster_id grouping
            groups = defaultdict(list)
            for e in survivor_edges:
                key = e.get("cluster_id") or "__unclustered__"
                groups[key].append(e)

        cluster_tuples: List[Tuple[str, str, List[Dict[str, Any]]]] = []
        for key, edges in groups.items():
            kt = "unclustered" if key == "__unclustered__" else "cluster_id"
            cluster_tuples.append((kt, key, edges))

        fused = [self._fuse_cluster(ct) for ct in cluster_tuples]
        narrative, grounding_map = self._render_situation_deterministic(fused)

        all_participants: Set[str] = set()
        all_pivotal: List[int] = []
        all_timeline: List[int] = []
        mood_votes: Counter = Counter()
        for c in fused:
            all_participants.update(c.participants)
            all_pivotal.extend(c.pivotal_edge_ids)
            all_timeline.extend(c.timeline_edge_ids)
            if c.dominant_mood:
                mood_votes[c.dominant_mood] += len(c.edges)
        dominant_mood = mood_votes.most_common(1)[0][0] if mood_votes else None

        only_unclustered = (
            set(groups.keys()) == {"__unclustered__"}
        )

        # Arc detection: identify temporal arcs (subject continuity
        # across sequence_numbers) in the survivor edges. Surfaced
        # in convergence_details for downstream consumers.
        arc_info: Dict[str, Any] = {}
        if (self._temporal is not None
                and hasattr(self._temporal, 'detect_arcs')):
            arcs = self._temporal.detect_arcs(survivor_edges)
            if arcs:
                arc_info = {
                    "arc_count": len(arcs),
                    "arc_keys": list(arcs.keys()),
                    "arc_edge_counts": {
                        k: len(v) for k, v in arcs.items()
                    },
                }

        return Situation(
            narrative=narrative, clusters=fused,
            participants=sorted(all_participants),
            dominant_mood=dominant_mood,
            pivotal_events=all_pivotal, timeline=all_timeline,
            grounding_map=grounding_map,
            source="unclustered" if only_unclustered else "reconstruct",
            survivors=sum(len(c.edges) for c in fused),
            convergence_details=arc_info,
        )

    # ── Fetchers (used by Stage 1 Entry) ──────────────────────────

    def _fetch_pq_rows(self, user_id: int) -> List[Dict[str, Any]]:
        sql = """
            SELECT pq.id              AS pq_id,
                   pq.relationship_id AS relationship_id,
                   pq.question_embedding AS question_embedding,
                   r.id               AS id,
                   r.subject          AS subject,
                   r.predicate        AS predicate,
                   r.object           AS object,
                   r.confidence       AS confidence,
                   r.id               AS sequence_number,
                   r.cluster_id       AS cluster_id,
                   r.subject_type     AS subject_type,
                   r.object_type      AS object_type,
                   r.edge_emotional_valence  AS edge_emotional_valence,
                   r.edge_emotional_label    AS edge_emotional_label,
                   r.edge_episodic_significance AS edge_episodic_significance,
                   r.edge_temporal_context   AS edge_temporal_context,
                   r.edge_relational_type    AS edge_relational_type,
                   r.edge_embedding          AS edge_embedding,
                   r.arc_id                  AS arc_id,
                   r.source_text             AS source_text,
                   r.relational_entities     AS relational_entities,
                   r.temporal_expression     AS temporal_expression,
                   r.resolved_event_date     AS resolved_event_date,
                   r.emotional_target        AS emotional_target,
                   r.is_historical           AS is_historical,
                   r.edge_schematic_category AS edge_schematic_category,
                   r.predicate_embedding     AS predicate_embedding
              FROM predicted_queries pq
              JOIN relationships r ON r.id = pq.relationship_id
             WHERE pq.user_id = ?
               AND r.tombstoned_at IS NULL
        """
        out: List[Dict[str, Any]] = []
        try:
            with get_db_context() as conn:
                rs = conn.execute(sql, (user_id,)).fetchall()
        except Exception as e:
            log.warning("pq_query_failed: %s", e)
            return []
        for r in rs:
            out.append(dict(r))
        return out

    def _fetch_edge_rows(self, user_id: int) -> List[Dict[str, Any]]:
        sql = """
            SELECT r.id              AS id,
                   r.id              AS relationship_id,
                   r.subject         AS subject,
                   r.predicate       AS predicate,
                   r.object          AS object,
                   r.confidence      AS confidence,
                   r.id              AS sequence_number,
                   r.cluster_id      AS cluster_id,
                   r.subject_type    AS subject_type,
                   r.object_type     AS object_type,
                   r.edge_embedding  AS edge_embedding,
                   r.edge_emotional_valence  AS edge_emotional_valence,
                   r.edge_emotional_label    AS edge_emotional_label,
                   r.edge_episodic_significance AS edge_episodic_significance,
                   r.edge_temporal_context   AS edge_temporal_context,
                   r.edge_relational_type    AS edge_relational_type,
                   r.arc_id                  AS arc_id,
                   r.source_text             AS source_text,
                   r.relational_entities     AS relational_entities,
                   r.temporal_expression     AS temporal_expression,
                   r.resolved_event_date     AS resolved_event_date,
                   r.emotional_target        AS emotional_target,
                   r.is_historical           AS is_historical,
                   r.edge_schematic_category AS edge_schematic_category,
                   r.predicate_embedding     AS predicate_embedding
              FROM relationships r
             WHERE r.user_id = ?
               AND r.tombstoned_at IS NULL
        """
        out: List[Dict[str, Any]] = []
        try:
            with get_db_context() as conn:
                rs = conn.execute(sql, (user_id,)).fetchall()
        except Exception as e:
            log.warning("edge_query_failed: %s", e)
            return []
        for r in rs:
            out.append(dict(r))
        return out

    # ── Cluster fusion (structural grammar engine) ───────────────

    def _fuse_cluster(
        self,
        cluster_tuple: Tuple[str, str, List[Dict[str, Any]]],
    ) -> Cluster:
        key_type, key_value, edges = cluster_tuple

        # Only PERSON/ORG/LOCATION entities appear as participants.
        # "user" is the canonical self-reference and always qualifies.
        # subject_type / object_type are set at write time by
        # MemoryEngine._label_entity_types(). When NULL, the value
        # is excluded — fail-closed, not fail-open.
        _PARTICIPANT_TYPES = {"PERSON", "ORG", "LOCATION"}
        participants: Set[str] = set()
        for e in edges:
            subj = e.get("subject") or ""
            if subj:
                if subj.lower() == "user":
                    participants.add(subj)
                elif (e.get("subject_type") or "").upper() in _PARTICIPANT_TYPES:
                    participants.add(subj)
            obj = e.get("object") or ""
            if obj:
                if (e.get("object_type") or "").upper() in _PARTICIPANT_TYPES:
                    # Structural filter: multi-word objects containing
                    # prepositions, determiners, or verbs are descriptive
                    # phrases ("Riya on Wednesday", "at her company"),
                    # not entity names. POS-based, not a word list.
                    _is_phrase = False
                    if " " in obj:
                        try:
                            import nltk as _nltk_part
                            _obj_tags = _nltk_part.pos_tag(obj.split())
                            _phrase_pos = {"IN", "TO", "DT", "VB",
                                           "VBD", "VBG", "VBN", "VBZ",
                                           "VBP", "PRP$"}
                            if any(t in _phrase_pos for _, t in _obj_tags):
                                _is_phrase = True
                        except Exception:
                            pass
                    if not _is_phrase:
                        participants.add(obj)

        # Dominant mood from stored labels — structural passthrough of
        # write-time emotion classification. No thresholds on valence.
        mood_labels = [
            e["edge_emotional_label"]
            for e in edges
            if e.get("edge_emotional_label")
        ]
        if mood_labels:
            from collections import Counter as _Counter
            dominant_mood = _Counter(mood_labels).most_common(1)[0][0]
        else:
            dominant_mood = None

        pivotal_ids = [
            e["id"] for e in edges
            if e.get("edge_episodic_significance")
            and str(e["edge_episodic_significance"]).lower() != "routine"
        ]

        timeline = sorted(
            edges,
            key=lambda e: (
                e.get("sequence_number") if e.get("sequence_number") is not None
                else e.get("id", 0)
            ),
        )
        timeline_ids = [e["id"] for e in timeline]

        return Cluster(
            key_type=key_type,
            key_value=str(key_value),
            edges=timeline,
            participants=sorted(participants),
            dominant_mood=dominant_mood,
            mean_valence=None,
            pivotal_edge_ids=pivotal_ids,
            timeline_edge_ids=timeline_ids,
        )

    def _render_situation_deterministic(
        self,
        clusters: List[Cluster],
    ) -> Tuple[str, Dict[int, List[int]]]:
        sentences: List[str] = []
        grounding: Dict[int, List[int]] = {}

        for ci, cluster in enumerate(clusters):
            if not cluster.edges:
                continue
            intro = self._cluster_intro(cluster, ci)
            if intro:
                sentences.append(intro)
                grounding[len(sentences) - 1] = list(cluster.timeline_edge_ids)

            prev_subject: Optional[str] = None
            prev_predicate: Optional[str] = None
            for ei, e in enumerate(cluster.edges):
                sent = self._edge_to_sentence(
                    e, prev_subject,
                    prev_predicate=prev_predicate,
                    position_in_cluster=ei,
                    cluster_size=len(cluster.edges),
                )
                if not sent:
                    continue
                sentences.append(sent)
                grounding[len(sentences) - 1] = [e["id"]]
                prev_subject = (e.get("subject") or "").lower() or None
                prev_predicate = (e.get("predicate") or "").lower() or None

        narrative = " ".join(sentences)
        return narrative, grounding

    # ── Grammar helpers (structural, no curated lexicons) ──

    def _cluster_intro(
        self, cluster: Cluster, cluster_index: int = 0,
    ) -> Optional[str]:
        """Generate a contextual cluster introduction from structural
        metadata: participants, mood, and cluster type. No curated
        templates -- the intro is assembled from stored data."""
        kt = cluster.key_type

        # Build a participant phrase from non-user participants.
        named = [p for p in cluster.participants if p.lower() != "user"]
        has_user = any(p.lower() == "user" for p in cluster.participants)

        if kt == "unclustered" and cluster_index > 0:
            return "Separately:"

        # Skip intro for small clusters — the edges speak for themselves.
        # Structural gate: intro only adds value when the cluster has
        # enough edges to form a sub-narrative worth framing.
        if len(cluster.edges) < 3:
            return None

        # If we have named participants, use them in the intro.
        if named:
            if has_user and len(named) == 1:
                people_phrase = f"you and {named[0]}"
            elif has_user and len(named) > 1:
                people_phrase = "you, " + ", ".join(named[:-1]) + f" and {named[-1]}"
            elif len(named) == 1:
                people_phrase = named[0]
            else:
                people_phrase = ", ".join(named[:-1]) + f" and {named[-1]}"

            mood = cluster.dominant_mood
            if mood:
                return self._finalize_sentence(
                    f"Here is what happened with {people_phrase} \u2014 "
                    f"the overall feeling was {mood}"
                )
            return self._finalize_sentence(
                f"Here is what happened with {people_phrase}"
            )

        # No named participants -- generic but still not robotic.
        return None

    def _edge_to_sentence(
        self,
        edge: Dict[str, Any],
        prev_subject: Optional[str],
        *,
        prev_predicate: Optional[str] = None,
        position_in_cluster: int = 0,
        cluster_size: int = 1,
    ) -> str:
        """Render ONE edge into ONE sentence via structural dispatch.

        Three branches gated on stored columns, not predicate content:
          1. Emotional edge: predicate is 'has_emotion'
          2. Emotional edge (legacy): edge_emotional_label == object
          3. Default SPO via predicate_shape parser.

        Discourse connectives are derived from episodic significance
        and temporal context — never from curated templates.
        """
        s_raw = (edge.get("subject") or "").strip()
        p_raw = (edge.get("predicate") or "").strip()
        o_raw = (edge.get("object") or "").strip()
        s_lower = s_raw.lower()
        is_user_subject = s_lower == "user"
        tense = (edge.get("edge_temporal_context") or "").lower().strip()

        subject_surface = "You" if is_user_subject else s_raw

        # ── Discourse connective selection ──
        # Structural signals: same subject continuity, episodic significance,
        # temporal context. No curated phrase lists — the connective is
        # selected by the intersection of two categorical dimensions:
        #   (same_subject x episodic_significance)
        connective = ""
        same_subject = prev_subject is not None and prev_subject == s_lower
        sig = (edge.get("edge_episodic_significance") or "").lower()

        if same_subject:
            if sig == "pivotal":
                connective = "Then "
            elif sig == "milestone":
                connective = "At that point, "
            elif sig == "resolution":
                connective = "In the end, "
            else:
                connective = ""  # routine continuation — no filler word
            if is_user_subject:
                subject_surface = "you"
        elif prev_subject is not None and position_in_cluster > 0:
            # Subject switch within a cluster — signal the shift.
            if sig == "pivotal":
                connective = "Meanwhile, "
                if is_user_subject:
                    subject_surface = "you"
            # else: no connective — the subject change itself signals shift

        # ── Branch 1: explicit emotion predicate (has_emotion) ──
        if p_raw.lower() == "has_emotion" and o_raw:
            verb = "felt" if tense == "past" else "feel" if is_user_subject else "feels"
            body = f"{subject_surface} {verb} {o_raw}"
            return self._finalize_sentence(connective + body)

        # ── Structural SPO via predicate-shape parser ──
        from app.engines.predicate_shape import parse_predicate, inflect_verb

        tense = (edge.get("edge_temporal_context") or "").lower().strip()
        person = "2s" if (is_user_subject or (subject_surface and subject_surface.lower() == "you")) else "3s"
        parsed = parse_predicate(p_raw)

        # ── Branch 2: emotional edge (label equals object, legacy) ──
        # Only fires when the predicate is NOT a parseable verb phrase.
        # This prevents false matches where the cosine classifier
        # assigned the object text as the emotional label (e.g.,
        # predicate="has_event", object="team presentation",
        # emotional_label="team presentation" — a coincidence, not
        # a real emotional edge).
        if not parsed.ok:
            emo_label = (edge.get("edge_emotional_label") or "").strip()
            if emo_label and emo_label.lower() == o_raw.lower() and o_raw:
                verb = "felt" if tense == "past" else "feel" if is_user_subject else "feels"
                body = f"{subject_surface} {verb} {o_raw}"
                return self._finalize_sentence(connective + body)

        if not parsed.ok:
            try:
                log.debug(
                    "predicate_shape_fallback: predicate=%r reason=%r edge_id=%s",
                    p_raw, parsed.failure_reason, edge.get("id"),
                )
            except Exception:
                pass
            pred_natural = p_raw.replace("_", " ")
            # Noun-compound predicates (birth_date, phone_number) use
            # copular form "Your X is Y". Verb-headed compounds that
            # failed parse (e.g. head POS-mistagged) render as verb
            # phrases: "{Subject} {pred} {obj}". POS on the head token
            # is the structural discriminator — no word lists.
            noun_compound = (
                "noun_compound" in (parsed.failure_reason or "")
                or "head_not_verb" in (parsed.failure_reason or "")
            )
            from app.engines.predicate_shape import is_verb_token as _is_verb
            from app.engines.predicate_shape import _pos_tag as _pt_fallback
            from app.engines.predicate_shape import _wn_morphy as _wn_morphy_raw
            def _wn_morphy_v(tok):
                return _wn_morphy_raw(tok, "v")
            segs = [s for s in p_raw.split("_") if s]
            head_token = segs[0] if segs else p_raw
            head_pos = _pt_fallback(head_token)
            # Gerund/participial-noun compound detection: if head is
            # VBG or VBN tagged and a subsequent segment is NN-tagged,
            # it's a modifier acting as noun prefix ("scheduling_note",
            # "recommended_action"), not a verb phrase. Treat as noun
            # compound for rendering.
            if head_pos in ("VBG", "VBN") and not noun_compound:
                for seg in segs[1:]:
                    if _pt_fallback(seg).startswith("NN"):
                        noun_compound = True
                        break
            verb_headed = _is_verb(head_token) and not noun_compound
            # Adjective detection: POS JJ* is primary. For NN-tagged
            # words, check WordNet: if it has adj synsets but NO noun
            # synsets, the tagger mistagged an adjective as a noun
            # (e.g., "allergic" → NN in isolation, but WN says adj).
            adj_headed = head_pos in ("JJ", "JJR", "JJS")
            if not adj_headed and head_pos.startswith("NN") and not noun_compound:
                try:
                    from nltk.corpus import wordnet as _wn_adj
                    has_adj = bool(_wn_adj.synsets(head_token.lower(), pos="a")
                                   or _wn_adj.synsets(head_token.lower(), pos="s"))
                    has_noun = bool(_wn_adj.synsets(head_token.lower(), pos="n"))
                    if has_adj and not has_noun:
                        adj_headed = True
                except Exception:
                    pass
            prep_headed = head_pos in ("IN", "TO", "RB")
            # Adverb-verb pattern: head is RB/adverb and next segment
            # is a verb (e.g., "now_feels", "now_scheduled_for",
            # "originally_set_for", "previously_requested"). Structural
            # tell: RB head + verb-shaped second segment.
            adverb_verb = (
                head_pos == "RB"
                and len(segs) >= 2
                and _is_verb(segs[1])
            )
            # Modal-verb pattern: head is MD ("should", "would", "could",
            # "might") followed by any word. Modals are structural
            # verb-selectors — the complement is grammatically always a
            # verb regardless of its WordNet synset distribution.
            modal_verb = (
                head_pos == "MD"
                and len(segs) >= 2
            )
            if verb_headed:
                # Verb-headed compound: render as verb phrase
                _o_art = self._maybe_article(o_raw) if o_raw else ""
                if subject_surface:
                    body = f"{subject_surface} {pred_natural} {_o_art}".strip()
                else:
                    body = f"{pred_natural} {_o_art}".strip()
            elif adverb_verb:
                # Adverb + verb: detect tense from verb morphology,
                # then render as "{subject} {adv} {verb_inflected} {rest} {obj}"
                adv = segs[0]
                verb_seg = segs[1]
                verb_lemma = _wn_morphy_v(verb_seg) or verb_seg
                # Tense from morphology: if verb_seg != lemma, the verb
                # carries its own tense; detect via POS.
                if verb_seg.lower() != verb_lemma.lower():
                    vpos = _pt_fallback(verb_seg)
                    vm = _wn_morphy_raw(verb_seg, "v")
                    is_mistag = vpos.startswith("NN") and vm and vm == verb_lemma
                    if vpos in ("VBD", "VBN"):
                        v_inflected = verb_seg  # past morphology
                    elif vpos == "VBZ" or is_mistag:
                        v_inflected = inflect_verb(verb_lemma, "present", person)
                    elif vpos == "VBG":
                        copula = "are" if person == "2s" else "is"
                        rest_parts = [p for p in segs[2:] if p]
                        parts = [subject_surface, adv, copula, verb_seg] + rest_parts
                        if o_raw:
                            parts.append(self._maybe_article(o_raw))
                        body = " ".join(p for p in parts if p).strip()
                        return self._finalize_sentence(connective + body)
                    else:
                        v_inflected = inflect_verb(verb_lemma, "present", person)
                else:
                    # Base form: surface == lemma (ambiguous tense, e.g.,
                    # "set", "put", "cut"). Use cosine-detected tense as
                    # tiebreaker — it's the only structural signal left.
                    v_inflected = inflect_verb(verb_lemma, tense, person)
                parts = [subject_surface, adv, v_inflected] + [s for s in segs[2:] if s]
                if o_raw:
                    parts.append(self._maybe_article(o_raw))
                body = " ".join(p for p in parts if p).strip()
            elif modal_verb:
                # Modal + verb: render as "{subject} {modal} {verb_base} {rest} {obj}"
                modal = segs[0]
                verb_seg = segs[1]
                verb_lemma = _wn_morphy_v(verb_seg) or verb_seg
                parts = [subject_surface, modal, verb_lemma] + [s for s in segs[2:] if s]
                if o_raw:
                    parts.append(self._maybe_article(o_raw))
                body = " ".join(p for p in parts if p).strip()
            elif adj_headed:
                # Adjective-headed predicate (allergic_to, anxious_because):
                # copular with "is/are" + adjective phrase.
                # "You are allergic to shellfish"
                copula = "are" if is_user_subject else "is"
                if subject_surface:
                    body = f"{subject_surface} {copula} {pred_natural} {o_raw}".strip()
                else:
                    body = f"{copula} {pred_natural} {o_raw}".strip()
            elif prep_headed:
                # Prepositional-headed predicate (in_relationship_with,
                # on_hiring_panel, together_for): copular with "is/are"
                # + predicate phrase. "You are in relationship with Mika"
                copula = "are" if is_user_subject else "is"
                if subject_surface:
                    body = f"{subject_surface} {copula} {pred_natural} {o_raw}".strip()
                else:
                    body = f"{copula} {pred_natural} {o_raw}".strip()
            elif is_user_subject and noun_compound:
                # Lemmatize the head for copular: "shows pattern" → "show pattern"
                _nc_segs = [s for s in p_raw.split("_") if s]
                _nc_head = _nc_segs[0] if _nc_segs else ""
                _nc_lemma = _wn_morphy_raw(_nc_head, "n") or _nc_head
                _nc_natural = " ".join([_nc_lemma] + _nc_segs[1:])
                body = f"Your {_nc_natural} is {o_raw}".strip()
            elif is_user_subject:
                body = f"Your {pred_natural} is {o_raw}".strip()
            elif subject_surface:
                body = f"{subject_surface} {pred_natural} {o_raw}".strip()
            else:
                body = f"{pred_natural} {o_raw}".strip()
            return self._finalize_sentence(connective + body)

        if parsed.verb_lemma == "be":
            if tense == "past":
                verb_surface = "were" if person == "2s" else "was"
            else:
                verb_surface = "are" if person == "2s" else "is"
        else:
            # Participial adjective detection: if the stored verb surface
            # is VBN/JJ-tagged AND the predicate structure is
            # verb + preposition WITHOUT an embedded noun between them,
            # the construction is adjectival copular:
            # "Mika is excited about X", not "Mika excites about X".
            # Structural tell: VBN/JJ tag + preposition + no embedded noun.
            # If there IS an embedded noun (asked_user_to), the verb is
            # active past tense with a direct object, not adjectival.
            from app.engines.predicate_shape import _pos_tag as _pt
            surface_tag = _pt(parsed.verb_surface)
            if (surface_tag in ("VBN", "JJ")
                    and parsed.preposition
                    and parsed.embedded_noun is None):
                # Copular construction: "{subject} is/was {surface} {prep} {obj}"
                copula = "were" if person == "2s" and tense == "past" else \
                         "was" if tense == "past" else \
                         "are" if person == "2s" else "is"
                parts = []
                if subject_surface:
                    parts.append(subject_surface)
                parts.append(copula)
                parts.append(parsed.verb_surface)
                parts.extend(parsed.middle)
                if o_raw:
                    parts.append(self._maybe_article(o_raw))
                body = " ".join([p for p in parts if p]).strip()
                return self._finalize_sentence(connective + body)
            # ── Structural tense rule ──
            # The verb's OWN morphology (baked into the stored predicate)
            # is the ground truth for tense. The cosine-detected
            # edge_temporal_context is a noisy secondary signal.
            #
            # When verb_surface != verb_lemma, the predicate carries
            # inflected morphology ("has", "feels", "received", "booked").
            # We derive the rendering tense FROM that morphology via POS
            # tag, ignoring the cosine label. This prevents "will has",
            # "will received", etc.
            #
            # When verb_surface == verb_lemma (base form), the predicate
            # did not carry tense — use the cosine-detected tense to
            # inflect.
            morphology_carries_tense = (
                parsed.verb_surface.lower() != parsed.verb_lemma.lower()
            )

            if morphology_carries_tense:
                # POS-tag the stored surface to determine its tense.
                # Caveat: NLTK POS-tags verbs like "feels", "needs",
                # "prefers" as NNS in isolation. The structural tell for
                # a mistagged verb: morphy(surface, 'v') returns a
                # DIFFERENT lemma. In that case, treat as present-tense
                # verb and re-inflect for person agreement.
                from app.engines.predicate_shape import _pos_tag as _pt_morph
                from app.engines.predicate_shape import _wn_morphy as _morph_check
                surface_pos = _pt_morph(parsed.verb_surface)

                # Detect mistagged verbs: NNS/NN-tagged but morphy('v')
                # resolves to the known lemma — structurally a verb.
                morphy_v = _morph_check(parsed.verb_surface, "v")
                is_mistagged_verb = (
                    surface_pos.startswith("NN")
                    and morphy_v is not None
                    and morphy_v == parsed.verb_lemma
                )

                if surface_pos in ("VBD", "VBN"):
                    # Past-tense morphology — use stored surface, no "will"
                    verb_surface = parsed.verb_surface
                    tense = "past"
                elif surface_pos == "VBG":
                    # Present participle ("helping", "working", "studying")
                    # Render as progressive: "is/are {-ing}"
                    copula = "are" if person == "2s" else "is"
                    parts = []
                    if subject_surface:
                        parts.append(subject_surface)
                    parts.append(copula)
                    parts.append(parsed.verb_surface)
                    parts.extend(parsed.middle)
                    if o_raw:
                        parts.append(self._maybe_article(o_raw))
                    body = " ".join([p for p in parts if p]).strip()
                    return self._finalize_sentence(connective + body)
                elif is_mistagged_verb:
                    # NN/NNS-tagged but structurally a verb (morphy
                    # resolves to the known lemma). Sub-classify the
                    # actual tense from suffix morphology — the POS
                    # tagger failed but the morphology is unambiguous:
                    #   -ing suffix  → progressive (VBG)
                    #   verb.exc map → irregular past (VBD)
                    #   otherwise    → present 3s (VBZ, e.g. "feels")
                    _sfc = parsed.verb_surface.lower()
                    if _sfc.endswith("ing"):
                        # Progressive: "planning" → "are planning"
                        copula = "are" if person == "2s" else "is"
                        parts = []
                        if subject_surface:
                            parts.append(subject_surface)
                        parts.append(copula)
                        parts.append(parsed.verb_surface)
                        parts.extend(parsed.middle)
                        if o_raw:
                            parts.append(self._maybe_article(o_raw))
                        body = " ".join([p for p in parts if p]).strip()
                        return self._finalize_sentence(connective + body)
                    else:
                        # Check WordNet verb exceptions: if the surface
                        # form appears as an irregular past of the lemma,
                        # it's VBD ("met" → past of "meet"), not VBZ.
                        _is_irreg_past = False
                        try:
                            from nltk.corpus import wordnet as _wn_tense
                            _wn_tense.morphy("be", "v")  # trigger load
                            _vexc = _wn_tense._exception_map.get("v", {})
                            _exc_base = _vexc.get(_sfc)
                            if _exc_base:
                                _bases = _exc_base if isinstance(_exc_base, (list, tuple)) else [_exc_base]
                                if parsed.verb_lemma in _bases:
                                    _is_irreg_past = True
                        except Exception:
                            pass
                        if _is_irreg_past:
                            # Irregular past: use stored surface as-is
                            verb_surface = parsed.verb_surface
                            tense = "past"
                        else:
                            # Present 3s mistagged as NNS ("feels", "needs")
                            verb_surface = inflect_verb(parsed.verb_lemma, "present", person)
                            tense = "present"
                elif surface_pos == "VBZ":
                    # 3rd-person singular present ("has", "feels")
                    # Re-inflect for correct person agreement.
                    verb_surface = inflect_verb(parsed.verb_lemma, "present", person)
                    tense = "present"
                else:
                    # Other inflected forms — re-inflect for person, present
                    verb_surface = inflect_verb(parsed.verb_lemma, "present", person)
                    tense = "present"
            else:
                # Base form — use cosine-detected tense to inflect
                verb_surface = inflect_verb(parsed.verb_lemma, tense, person)

        passive = parsed.preposition == "by" and tense == "past"
        if passive:
            from app.engines.predicate_shape import _regular_past
            pp = None
            try:
                from nltk.corpus import wordnet as wn
                wn.morphy("be", "v")
                vexc = wn._exception_map.get("v", {})
                import nltk
                cands = [k for k, v in vexc.items() if (isinstance(v, (list, tuple)) and parsed.verb_lemma in v)]
                vbn_cands = [c for c in cands if nltk.pos_tag([c])[0][1] == "VBN"]
                if vbn_cands:
                    pp = sorted(vbn_cands, key=len)[0]
            except Exception:
                pass
            if not pp:
                pp = _regular_past(parsed.verb_lemma)
            aux = "were" if person == "2s" else "was"
            parts = []
            if subject_surface:
                parts.append(subject_surface)
            parts.append(aux)
            parts.append(pp)
            parts.extend(parsed.middle)
            parts.append(self._maybe_article(o_raw))
            body = " ".join([p for p in parts if p]).strip()
            return self._finalize_sentence(connective + body)

        parts = []
        if subject_surface:
            parts.append(subject_surface)
        # Only prepend "will" when tense is future AND the morphology
        # did NOT carry its own tense (base-form verbs only). If the
        # stored surface was already inflected, the tense variable was
        # overridden above to match the morphology.
        if tense == "future" and not morphology_carries_tense and verb_surface.lower() != "will":
            parts.append("will")
        parts.append(verb_surface)
        # Word-order fix: when the predicate carries a preposition +
        # embedded noun (entity) and the object starts with an adjective,
        # the object is a descriptor that should precede the prepositional
        # phrase: "feel jealous but happy about Riya" not
        # "feel about Riya jealous but happy". Structural tell: the
        # object's first token is JJ-tagged (adjective).
        from app.engines.predicate_shape import _pos_tag as _pt_mid
        _obj_first_tag = ""
        if o_raw:
            _obj_first_word = o_raw.split()[0] if o_raw.split() else ""
            _obj_first_tag = _pt_mid(_obj_first_word) if _obj_first_word else ""
        _reorder_obj_before_prep = (
            parsed.preposition is not None
            and parsed.embedded_noun is not None
            and _obj_first_tag.startswith("JJ")
        )
        if _reorder_obj_before_prep:
            # Place object before the prep+entity tail
            if o_raw:
                parts.append(self._maybe_article(o_raw))
            for seg in parsed.middle:
                # Title-case embedded noun (entity reference in predicate)
                if seg == parsed.embedded_noun and seg[0].islower():
                    parts.append(seg.title())
                else:
                    parts.append(seg)
        else:
            # Normal order: middle segments then object
            _has_embedded_article = False
            for seg in parsed.middle:
                if _pt_mid(seg) == "NN" and parsed.embedded_noun == seg:
                    parts.append(self._maybe_article(seg))
                    _has_embedded_article = True
                else:
                    parts.append(seg)
            if o_raw:
                # Skip article on object when the predicate already
                # carries an embedded noun with its own article — the
                # object modifies that noun, not a new noun phrase.
                # Insert a linking preposition "for" when the embedded
                # noun has no preposition after it and the object is a
                # bare noun phrase (not starting with a preposition).
                if _has_embedded_article:
                    # Insert linking "for" when the embedded noun has
                    # no trailing preposition and the object doesn't
                    # start with one. POS-tagged: IN/TO = preposition.
                    _obj_lead = o_raw.split()[0] if o_raw.split() else ""
                    _obj_lead_tag = _pt_mid(_obj_lead) if _obj_lead else ""
                    _needs_link = (
                        parsed.preposition is None
                        and _obj_lead_tag not in ("IN", "TO")
                    )
                    if _needs_link:
                        parts.append("for")
                    parts.append(o_raw)
                else:
                    # Skip article after infinitive "to" — the object
                    # starts a verb phrase, not a noun phrase.
                    _last_mid = parsed.middle[-1] if parsed.middle else ""
                    if _last_mid.lower() == "to":
                        parts.append(o_raw)
                    else:
                        parts.append(self._maybe_article(o_raw))
        body = " ".join([p for p in parts if p]).strip()
        return self._finalize_sentence(connective + body)

    # ── Article insertion ──

    @staticmethod
    def _maybe_article(obj: str) -> str:
        """Prepend 'a'/'an' to an object phrase when the first word is a
        bare singular count noun (POS=NN).  POS-based, not a word list.

        Skips when:
          - object is empty or starts with a determiner/possessive/number
          - first word is a proper noun (NNP/NNPS)
          - first word is plural (NNS)
          - first word is an adjective (JJ) followed by a proper noun
            (the adjective modifies a name, not a count noun)
        """
        if not obj:
            return obj
        tokens = obj.split()
        if not tokens:
            return obj
        first = tokens[0]
        # Already has a determiner, possessive, or digit lead
        fl = first.lower()
        if fl in ("a", "an", "the", "this", "that", "these", "those",
                  "my", "your", "his", "her", "its", "our", "their",
                  "some", "any", "every", "each", "no"):
            return obj
        if first[0].isdigit():
            return obj
        # Proper noun heuristic: if the first word starts with an
        # uppercase letter, it's likely a proper noun (entity name,
        # day of week, place). Objects are raw stored text, never
        # sentence-initial, so uppercase = proper noun.
        if first[0].isupper():
            return obj
        try:
            from app.engines.predicate_shape import _pos_tag as _pt_art
            tag = _pt_art(first)
            if tag == "NN":
                # Singular count noun — needs article
                article = "an" if fl[0] in "aeiou" else "a"
                return article + " " + obj
        except Exception:
            pass
        return obj

    # ── Morphological polish for _finalize_sentence ──

    def _substitute_user_pronoun(self, text: str) -> str:
        if not text:
            return text
        out_tokens = []
        for tok in text.split(" "):
            stripped = tok
            trailing = ""
            while stripped and stripped[-1] in ".,;:!?)":
                trailing = stripped[-1] + trailing
                stripped = stripped[:-1]
            leading = ""
            while stripped and stripped[0] in "(":
                leading += stripped[0]
                stripped = stripped[1:]

            lower = stripped.lower()
            if lower == "user":
                replacement = "You" if stripped and stripped[0].isupper() else "you"
                out_tokens.append(leading + replacement + trailing)
            elif lower == "user's":
                replacement = "Your" if stripped and stripped[0].isupper() else "your"
                out_tokens.append(leading + replacement + trailing)
            else:
                out_tokens.append(tok)
        return " ".join(out_tokens)

    def _finalize_sentence(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        # Collapse multiple spaces (from empty join segments)
        import re as _re_finalize
        text = _re_finalize.sub(r"  +", " ", text)
        text = self._substitute_user_pronoun(text)
        if not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text += "."
        try:
            from app.engines.sentence_model import polish as _polish
            polished = _polish(text)
            if polished and polished.strip():
                text = polished
        except Exception:
            pass
        return text

    # ── Lookup entity aggregation ────────────────────────────────

    def _fetch_entity_edges(
        self, user_id: int, entity: str, exclude_ids: Set[int]
    ) -> List[Dict[str, Any]]:
        """Fetch current edges where the entity appears as subject or object.

        Structural: direct graph lookup by entity name. No cosine,
        no scoring — pure entity identity. Returns edges ordered by
        sequence_number DESC (most recent first), limited to 5.
        """
        try:
            with get_db_context() as conn:
                # Fetch subject-matches first (most relevant), then
                # object-matches. Structural priority: edges where the
                # entity IS the subject are more informative about the
                # entity than edges where it merely appears in the object.
                subj_rows = conn.execute(
                    """SELECT id, subject, predicate, object,
                              edge_temporal_context, edge_emotional_label,
                              edge_episodic_significance
                       FROM relationships
                       WHERE user_id = ?
                         AND LOWER(subject) LIKE ('%' || LOWER(?) || '%')
                         AND COALESCE(is_current, 1) = 1
                         AND tombstoned_at IS NULL
                       ORDER BY id DESC
                       LIMIT ?""",
                    (user_id, entity, self._ENTITY_EDGE_LIMIT),
                ).fetchall()
                obj_rows = conn.execute(
                    """SELECT id, subject, predicate, object,
                              edge_temporal_context, edge_emotional_label,
                              edge_episodic_significance
                       FROM relationships
                       WHERE user_id = ?
                         AND LOWER(subject) NOT LIKE ('%' || LOWER(?) || '%')
                         AND LOWER(object) LIKE ('%' || LOWER(?) || '%')
                         AND COALESCE(is_current, 1) = 1
                         AND tombstoned_at IS NULL
                       ORDER BY id DESC
                       LIMIT ?""",
                    (user_id, entity, entity, self._ENTITY_EDGE_LIMIT),
                ).fetchall()
                rows = list(subj_rows) + list(obj_rows)
            seen_ids = set()
            result = []
            for r in rows:
                if r["id"] not in exclude_ids and r["id"] not in seen_ids:
                    result.append(dict(r))
                    seen_ids.add(r["id"])
                if len(result) >= 10:
                    break
            return result
        except Exception:
            return []

    def _lookup_primary_entity_from_query(
        self, user_id: int, query_text: str, top: Candidate
    ) -> Optional[str]:
        """Extract the primary non-self entity for lookup aggregation.

        Structural extraction: find the first capitalized token in the
        query that is not a common English function word. This captures
        proper nouns (Mika, Rohan, Biscuit, Derek, Meridian Labs) without
        NER or word lists — capitalization is the structural signal for
        proper-noun-hood in English.

        Fallback: non-self entity from the winning edge.

        Returns the entity lowercased, or None if no non-self entity.
        """
        if query_text:
            tokens = query_text.split()
            _skip = {
                "who", "what", "where", "when", "why", "how",
                "is", "are", "do", "does", "did", "was", "were",
                "the", "my", "me", "i", "a", "an",
            }
            # Collect consecutive capitalized tokens as a single entity
            # (e.g. "Meridian Labs" -> "meridian labs")
            # Structural rule: sentence-initial capitalization (index 0)
            # is English orthography, not a proper-noun signal. Only
            # mid-sentence capitalization indicates a proper noun.
            entity_parts: List[str] = []
            for i, tok in enumerate(tokens):
                clean = tok.strip("?.,!;:'\"()")
                if not clean:
                    if entity_parts:
                        return " ".join(entity_parts).lower()
                    continue
                if (clean[0].isupper() and clean.lower() not in _skip
                        and i > 0):
                    entity_parts.append(clean)
                else:
                    if entity_parts:
                        return " ".join(entity_parts).lower()
            if entity_parts:
                return " ".join(entity_parts).lower()
        # Fallback to winning edge's non-self entity
        subj = (top.edge.get("subject") or "").strip().lower()
        obj = (top.edge.get("object") or "").strip().lower()
        if subj and subj != "user":
            return subj
        if obj and obj != "user":
            return obj
        return None

    # ── Sentence reconstruction (retrieve single-fact path) ───────

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
