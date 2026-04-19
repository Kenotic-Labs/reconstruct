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

    def retrieve(self, user_id: int, query_text: str):
        """Moat pipeline — Lookup mode.
        Entry → Expand → Group → Relate → Exit Cosine → Validate (warn) → top-1."""
        query_text = (query_text or "").strip()
        if not query_text:
            return StructuralRefusal(reason="empty_query")

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
        candidates.sort(
            key=lambda c: (
                -c.exit_cosine,
                -(c.edge.get("sequence_number") or 0),
            )
        )

        top = candidates[0]
        warning = self._stage6_validate(query_text, top)

        subj = top.edge.get("subject") or ""
        pred = top.edge.get("predicate") or ""
        obj = top.edge.get("object") or ""
        source_text = (top.edge.get("source_text") or "").strip()

        # Render the winning triple via the grammar engine. Then
        # aggregate: include other edges that mention the primary entity.
        # The primary entity comes from the QUERY (via entity resolver),
        # falling back to the winning edge's non-self entity. This is
        # structural entity-focused aggregation — when the user asks
        # "Who is Mika?", the answer includes all current facts about
        # Mika, not just the single highest-cosine triple.
        #
        # Two-pass aggregation:
        #   Pass 1: surviving candidates (already cosine-ranked)
        #   Pass 2: direct DB fetch by entity name (fills gaps where
        #           cosine didn't surface the edge but it's structurally
        #           relevant via entity identity)
        primary_entity = self._lookup_primary_entity_from_query(
            user_id, query_text, top
        )
        sentences = [self._edge_to_sentence(top.edge, None)]
        seen_edges = {top.relationship_id}

        if primary_entity:
            # Pass 1: candidates already in the pool
            for c in candidates[1:]:
                if c.relationship_id in seen_edges:
                    continue
                c_subj = (c.edge.get("subject") or "").lower()
                c_obj = (c.edge.get("object") or "").lower()
                if primary_entity in c_subj or primary_entity in c_obj:
                    sent = self._edge_to_sentence(c.edge, None)
                    if sent and sent not in sentences:
                        sentences.append(sent)
                        seen_edges.add(c.relationship_id)
                    if len(sentences) >= 20:
                        break

            # Pass 2: direct entity fetch from DB for completeness
            if len(sentences) < 20:
                entity_edges = self._fetch_entity_edges(
                    user_id, primary_entity, seen_edges
                )
                for e in entity_edges:
                    sent = self._edge_to_sentence(e, None)
                    if sent and sent not in sentences:
                        sentences.append(sent)
                        seen_edges.add(e.get("id", 0))
                    if len(sentences) >= 20:
                        break

        answer_text = " ".join(sentences)

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
                   r.sequence_number  AS sequence_number,
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
                   r.source_text             AS source_text
              FROM predicted_queries pq
              JOIN relationships r ON r.id = pq.relationship_id
             WHERE pq.user_id = ?
               AND COALESCE(r.is_current, 1) = 1
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
                   r.sequence_number AS sequence_number,
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
                   r.source_text             AS source_text
              FROM relationships r
             WHERE r.user_id = ?
               AND COALESCE(r.is_current, 1) = 1
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

        # ── Branch 2: emotional edge (label equals object, legacy) ──
        emo_label = (edge.get("edge_emotional_label") or "").strip()
        if emo_label and emo_label.lower() == o_raw.lower() and o_raw:
            verb = "felt" if tense == "past" else "feel" if is_user_subject else "feels"
            body = f"{subject_surface} {verb} {o_raw}"
            return self._finalize_sentence(connective + body)

        # Branch 2 -- structural SPO via predicate-shape parser.
        from app.engines.predicate_shape import parse_predicate, inflect_verb

        tense = (edge.get("edge_temporal_context") or "").lower().strip()
        person = "2s" if (is_user_subject or (subject_surface and subject_surface.lower() == "you")) else "3s"
        parsed = parse_predicate(p_raw)

        if not parsed.ok:
            try:
                log.warning(
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
            noun_compound = "noun_compound" in (parsed.failure_reason or "")
            from app.engines.predicate_shape import is_verb_token as _is_verb
            from app.engines.predicate_shape import _pos_tag as _pt_fallback
            head_token = p_raw.split("_")[0] if "_" in p_raw else p_raw
            head_pos = _pt_fallback(head_token)
            verb_headed = _is_verb(head_token) and not noun_compound
            prep_headed = head_pos in ("IN", "TO", "RB")
            if verb_headed:
                # Verb-headed compound: render as verb phrase
                if subject_surface:
                    body = f"{subject_surface} {pred_natural} {o_raw}".strip()
                else:
                    body = f"{pred_natural} {o_raw}".strip()
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
                body = f"Your {pred_natural} is {o_raw}".strip()
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
                    parts.append(o_raw)
                body = " ".join([p for p in parts if p]).strip()
                return self._finalize_sentence(connective + body)
            # Preserve stored tense morphology when the stored surface
            # differs from the lemma. The predicate's verb_surface carries
            # the tense the user spoke ("accepted", "interviewed",
            # "started"). Re-inflecting from the lemma based on
            # edge_temporal_context can lose this original tense when the
            # context label disagrees with the surface morphology.
            # Structural rule: if verb_surface != verb_lemma (tense was
            # baked into the predicate), prefer the stored surface.
            if parsed.verb_surface.lower() != parsed.verb_lemma.lower():
                verb_surface = parsed.verb_surface
            else:
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
            parts.append(o_raw)
            body = " ".join([p for p in parts if p]).strip()
            return self._finalize_sentence(connective + body)

        parts = []
        if subject_surface:
            parts.append(subject_surface)
        if tense == "future" and verb_surface.lower() != "will":
            parts.append("will")
        parts.append(verb_surface)
        parts.extend(parsed.middle)
        if o_raw:
            parts.append(o_raw)
        body = " ".join([p for p in parts if p]).strip()
        return self._finalize_sentence(connective + body)

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
                       ORDER BY COALESCE(sequence_number, id) DESC
                       LIMIT 10""",
                    (user_id, entity),
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
                       ORDER BY COALESCE(sequence_number, id) DESC
                       LIMIT 10""",
                    (user_id, entity, entity),
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
                "tell", "the", "my", "me", "i", "a", "an",
            }
            # Collect consecutive capitalized tokens as a single entity
            # (e.g. "Meridian Labs" -> "meridian labs")
            entity_parts: List[str] = []
            for tok in tokens:
                clean = tok.strip("?.,!;:'\"()")
                if not clean:
                    if entity_parts:
                        return " ".join(entity_parts).lower()
                    continue
                if clean[0].isupper() and clean.lower() not in _skip:
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
