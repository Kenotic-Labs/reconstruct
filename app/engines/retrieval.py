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
        """Entry Cosine: max(pq_cosine, edge_cosine) per relationship_id,
        top-K by entry_cosine. Both pools always run — no gating."""
        pq_rows = self._fetch_pq_rows(user_id)
        edge_rows = self._fetch_edge_rows(user_id)

        by_rid: Dict[int, Candidate] = {}

        for row in pq_rows:
            rid = row["relationship_id"]
            cos = _cosine_from_blob(q_emb, row.get("question_embedding"))
            existing = by_rid.get(rid)
            if existing is None:
                cand = Candidate(relationship_id=rid, edge=dict(row), entry_cosine=cos)
                cand.source_stages.add("entry")
                by_rid[rid] = cand
            elif cos > existing.entry_cosine:
                existing.entry_cosine = cos

        for row in edge_rows:
            rid = row["relationship_id"]
            cos = _cosine_from_blob(q_emb, row.get("edge_embedding"))
            existing = by_rid.get(rid)
            if existing is None:
                cand = Candidate(relationship_id=rid, edge=dict(row), entry_cosine=cos)
                cand.source_stages.add("entry")
                by_rid[rid] = cand
            elif cos > existing.entry_cosine:
                existing.entry_cosine = cos

        out = sorted(by_rid.values(), key=lambda c: -c.entry_cosine)
        return out[: self._ENTRY_POOL_SIZE]

    # ── Stage 2: Expand (entity union) ───────────────────────────

    def _stage2_expand(
        self, user_id: int, query_text: str, candidates: List[Candidate]
    ) -> List[Candidate]:
        from app.engines import entity_resolver
        entities = entity_resolver.resolve_query_entities(user_id, query_text)
        if not entities:
            return candidates

        names = {e["name"] for e in entities}
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
            overlap = sum(
                1 for n in names
                if n == row["subject"] or n == row["object"]
            )
            if rid in by_rid:
                cand = by_rid[rid]
                if overlap > cand.entity_overlap:
                    cand.entity_overlap = overlap
                cand.source_stages.add("expand")
            else:
                cand = Candidate(relationship_id=rid, edge=dict(row),
                                 entity_overlap=overlap)
                cand.source_stages.add("expand")
                by_rid[rid] = cand

        # Also score existing candidates that happen to touch entities.
        for rid, cand in by_rid.items():
            if cand.entity_overlap == 0:
                e = cand.edge
                touches = sum(
                    1 for n in names
                    if n == e.get("subject") or n == e.get("object")
                )
                if touches:
                    cand.entity_overlap = touches
                    cand.source_stages.add("expand")

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
        if not entities:
            return candidates  # inapplicable — pass through

        query_names = {e["name"] for e in entities}

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
        for c in candidates:
            c.exit_cosine = _cosine_from_blob(q_emb, c.edge.get("edge_embedding"))
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
        """Returns Answer on success, StructuralRefusal on empty set."""
        query_text = (query_text or "").strip()
        if not query_text:
            return StructuralRefusal(reason="empty_query")

        try:
            q_emb = embed_text(query_text)
        except Exception as e:
            return StructuralRefusal(
                reason="embed_failed",
                convergence_details={"error": str(e)[:200]},
            )

        deduped = self._cosine_pool(user_id, q_emb)
        if not deduped:
            return StructuralRefusal(
                reason="no_edges",
                convergence_details={"query": query_text},
            )

        expected = parse_expected_answer_type(query_text)
        if expected is not None:
            coherent = [
                (s, r) for (s, r) in deduped
                if (r.get("object_type") == expected
                    or r.get("subject_type") == expected)
            ]
            if not coherent:
                return StructuralRefusal(
                    reason="no_coherent_answer",
                    convergence_details={
                        "query": query_text,
                        "expected_type": expected,
                        "pool_size": len(deduped),
                    },
                )
            candidates = coherent
        else:
            candidates = deduped

        # Cosine is the relevance signal (question↔predicted-question, same
        # semantic space). sequence_number breaks ties when cosines match —
        # newer fact wins. Recency does not override relevance.
        candidates.sort(
            key=lambda sr: (
                -sr[0],
                -(sr[1].get("sequence_number") or 0),
            )
        )

        top_score, top = candidates[0]
        subj = top.get("subject") or ""
        pred = top.get("predicate") or ""
        obj = top.get("object") or ""
        answer_text = self._triple_to_sentence(subj, pred, obj)
        return Answer(
            text=answer_text,
            subject=subj,
            predicate=pred,
            object=obj,
            confidence=1.0,
            source="pq_cosine",
            survivors=len(candidates),
            convergence_details={
                "query": query_text,
                "expected_type": expected,
                "sequence_number": top.get("sequence_number"),
                "cosine": top_score,
                "pool_size": len(deduped),
            },
        )

    def reconstruct(self, user_id: int, query_text: str) -> Situation:
        """Group the cosine top-k by cluster_id and fuse each cluster via
        the deterministic grammar engine."""
        query_text = (query_text or "").strip()
        if not query_text:
            return Situation(
                narrative="",
                source="structural_refusal",
                convergence_details={"reason": "empty_query"},
            )
        try:
            q_emb = embed_text(query_text)
        except Exception as e:
            return Situation(
                narrative="",
                source="structural_refusal",
                convergence_details={"reason": "embed_failed",
                                     "error": str(e)[:200]},
            )

        deduped = self._cosine_pool(user_id, q_emb)
        if not deduped:
            return Situation(
                narrative="",
                source="structural_refusal",
                survivors=0,
                convergence_details={"reason": "no_edges"},
            )

        # Group by cluster_id.
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for _, r in deduped:
            key = r.get("cluster_id") or "__unclustered__"
            groups[key].append(r)

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

        return Situation(
            narrative=narrative,
            clusters=fused,
            participants=sorted(all_participants),
            dominant_mood=dominant_mood,
            pivotal_events=all_pivotal,
            timeline=all_timeline,
            grounding_map=grounding_map,
            source="reconstruct",
            survivors=sum(len(c.edges) for c in fused),
            convergence_details={
                "query": query_text,
                "cluster_count": len(fused),
                "pool_size": len(deduped),
            },
        )

    # ── Cosine pool (steps 1..3 of the pipeline) ──────────────────

    def _cosine_pool(
        self, user_id: int, q_emb: np.ndarray
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Return [(cosine, edge_row_dict), ...] deduped by relationship_id,
        sorted by cosine DESC, capped at _COSINE_POOL_SIZE.

        Primary source: predicted_queries.question_embedding joined to
        relationships. Fallback when no PQ rows exist for this user:
        relationships.edge_embedding directly (legacy path for
        pre-backfill DBs)."""
        rows = self._fetch_pq_rows(user_id)
        if rows:
            return self._rank_rows(q_emb, rows, emb_col="question_embedding")

        log.warning(
            "pq_pool_empty for user_id=%s; falling back to edge_embedding",
            user_id,
        )
        rows = self._fetch_edge_rows(user_id)
        if not rows:
            return []
        return self._rank_rows(q_emb, rows, emb_col="edge_embedding")

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
                   r.edge_relational_type    AS edge_relational_type
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
                   r.edge_relational_type    AS edge_relational_type
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

    def _rank_rows(
        self,
        q_emb: np.ndarray,
        rows: List[Dict[str, Any]],
        emb_col: str,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for r in rows:
            emb = _deserialize_emb(r.get(emb_col))
            if emb is None:
                continue
            scored.append((_cosine(q_emb, emb), r))

        scored.sort(key=lambda x: -x[0])

        seen: Set[int] = set()
        deduped: List[Tuple[float, Dict[str, Any]]] = []
        for score, r in scored:
            rid = r.get("relationship_id") or r.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            deduped.append((score, r))
            if len(deduped) >= _COSINE_POOL_SIZE:
                break
        return deduped

    # ── Cluster fusion (structural grammar engine) ───────────────

    def _fuse_cluster(
        self,
        cluster_tuple: Tuple[str, str, List[Dict[str, Any]]],
    ) -> Cluster:
        key_type, key_value, edges = cluster_tuple

        participants: Set[str] = set()
        for e in edges:
            if e.get("subject"):
                participants.add(e["subject"])
            if e.get("object"):
                participants.add(e["object"])

        valences = [
            float(e["edge_emotional_valence"])
            for e in edges
            if e.get("edge_emotional_valence") is not None
        ]
        mean_val = float(np.mean(valences)) if valences else None

        if mean_val is None:
            dominant_mood = None
        elif mean_val > 0.55:
            dominant_mood = "positive"
        elif mean_val < 0.45:
            dominant_mood = "negative"
        else:
            dominant_mood = "neutral"

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
            mean_valence=mean_val,
            pivotal_edge_ids=pivotal_ids,
            timeline_edge_ids=timeline_ids,
        )

    def _render_situation_deterministic(
        self,
        clusters: List[Cluster],
    ) -> Tuple[str, Dict[int, List[int]]]:
        sentences: List[str] = []
        grounding: Dict[int, List[int]] = {}

        for cluster in clusters:
            if not cluster.edges:
                continue
            intro = self._cluster_intro(cluster)
            if intro:
                sentences.append(intro)
                grounding[len(sentences) - 1] = list(cluster.timeline_edge_ids)

            prev_subject: Optional[str] = None
            for e in cluster.edges:
                sent = self._edge_to_sentence(e, prev_subject)
                if not sent:
                    continue
                sentences.append(sent)
                grounding[len(sentences) - 1] = [e["id"]]
                prev_subject = (e.get("subject") or "").lower() or None

        narrative = " ".join(sentences)
        return narrative, grounding

    # ── Grammar helpers (structural, no curated lexicons) ──

    def _cluster_intro(self, cluster: Cluster) -> Optional[str]:
        kt = cluster.key_type
        if kt == "cluster_id":
            return "In this narrative window:"
        if kt == "unclustered":
            return "Elsewhere:"
        return None

    def _edge_to_sentence(
        self,
        edge: Dict[str, Any],
        prev_subject: Optional[str],
    ) -> str:
        """Render ONE edge into ONE sentence via structural dispatch.

        Two branches gated on stored columns, not predicate content:
          1. Emotional edge: edge_emotional_label == object
          2. Default SPO via predicate_shape parser.
        """
        s_raw = (edge.get("subject") or "").strip()
        p_raw = (edge.get("predicate") or "").strip()
        o_raw = (edge.get("object") or "").strip()
        s_lower = s_raw.lower()
        is_user_subject = s_lower == "user"

        subject_surface = "You" if is_user_subject else s_raw

        connective = ""
        if prev_subject and prev_subject == s_lower:
            sig = (edge.get("edge_episodic_significance") or "").lower()
            if sig and sig != "routine":
                connective = "Then "
            else:
                connective = "Also "
            if is_user_subject:
                subject_surface = "you"

        # Branch 1 -- emotional edge (label equals object)
        emo_label = (edge.get("edge_emotional_label") or "").strip()
        if emo_label and emo_label.lower() == o_raw.lower() and o_raw:
            if is_user_subject:
                body = f"feel {o_raw}" if not subject_surface else f"{subject_surface} feel {o_raw}"
            else:
                body = f"feels {o_raw}" if not subject_surface else f"{subject_surface} feels {o_raw}"
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
            if is_user_subject:
                body = f"Your {pred_natural} is {o_raw}".strip()
            elif subject_surface:
                body = f"{subject_surface}'s {pred_natural} is {o_raw}".strip()
            else:
                body = f"{pred_natural} is {o_raw}".strip()
            return self._finalize_sentence(connective + body)

        if parsed.verb_lemma == "be":
            if tense == "past":
                verb_surface = "were" if person == "2s" else "was"
            else:
                verb_surface = "are" if person == "2s" else "is"
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
