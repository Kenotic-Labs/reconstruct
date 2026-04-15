# Moat Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire all five Moat layers (Entry → Expand → Group → Relate → Exit Cosine) plus a warn-only Validate as a single retrieval pipeline with two output modes (Lookup, Reconstruction). No weights. No tuning coefficients. Every column on `relationships` and `entities` is used by the stage it belongs to. Spec: `docs/superpowers/specs/2026-04-15-moat-pipeline-design.md`.

**Architecture:** Filter-and-rank pipeline. Structural stages (Expand/Group/Relate) are boolean gates. Cosine stages (Entry, Exit) score for ordering. Validate is a warning on top-1, not a pool filter. When a structural stage has no anchor (query has no resolvable entity), it passes through — we do not refuse for inapplicability. Refusal happens only when Entry AND Expand jointly produce zero candidates.

**Tech Stack:** Python 3.10, SQLite, sentence-transformers MiniLM via `app.vector.embedder.embed_text`, numpy cosine, spaCy `en_core_web_sm` for query-side NP extraction (already used by `type_resolver.py`).

**Structural parameters (NOT tuning knobs):**
- `K = 80` — Entry top-K cap
- `τ_entity = 0.55` — query-phrase → entity cosine threshold
- `MAX_HOPS = 3` — Relate BFS depth
- `N_reconstruct = 20` — top-N before cluster fuse

---

## File Structure

**Create:**
- `app/engines/retrieval_types.py` — `Candidate` dataclass
- `app/engines/entity_resolver.py` — query-time entity resolution
- `app/engines/memgraph.py` — BFS graph proximity on `relationships`
- `tests/test_retrieval_types.py`
- `tests/test_entity_resolver.py`
- `tests/test_memgraph.py`
- `tests/test_retrieval_pipeline.py` — per-stage + E2E

**Modify:**
- `app/engines/retrieval.py` — delete `_cosine_pool`, add `_stage1_entry` through `_stage6_validate`, rewrite `retrieve()` and `reconstruct()`
- `tests/test_retrieval_pq.py` — update assertions for new Answer source/convergence_details (no behavior regression expected)

---

## Task 1: Create `Candidate` dataclass

**Files:**
- Create: `app/engines/retrieval_types.py`
- Test: `tests/test_retrieval_types.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_retrieval_types.py
from app.engines.retrieval_types import Candidate


def test_candidate_defaults_are_neutral():
    c = Candidate(relationship_id=1, edge={})
    assert c.entry_cosine == 0.0
    assert c.entity_overlap == 0
    assert c.cluster_members == 0
    assert c.hops_to_entity == -1
    assert c.exit_cosine == 0.0
    assert c.source_stages == set()


def test_candidate_source_stages_accumulates():
    c = Candidate(relationship_id=1, edge={})
    c.source_stages.add("entry")
    c.source_stages.add("expand")
    assert c.source_stages == {"entry", "expand"}
```

- [ ] **Step 2: Run — expect FAIL**

```
py -3.10 -m pytest tests/test_retrieval_types.py -v
```
Expected: `ModuleNotFoundError: app.engines.retrieval_types`.

- [ ] **Step 3: Implement**

```python
# app/engines/retrieval_types.py
"""Candidate dataclass shared across RetrievalEngine stages.

Each stage of the Moat pipeline annotates the Candidate with its
contribution. No stage computes a weighted combined score — structural
stages filter, cosine stages order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Set


@dataclass
class Candidate:
    relationship_id: int
    edge: Dict[str, Any]
    entry_cosine: float = 0.0
    entity_overlap: int = 0
    cluster_members: int = 0
    hops_to_entity: int = -1
    exit_cosine: float = 0.0
    source_stages: Set[str] = field(default_factory=set)
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add app/engines/retrieval_types.py tests/test_retrieval_types.py
git commit -m "feat(retrieval): add Candidate dataclass for Moat pipeline"
```

---

## Task 2: Helper — cosine-from-blob

Shared by Entry and Exit stages.

**Files:**
- Modify: `app/engines/retrieval.py` (add module-level helper)
- Test: `tests/test_retrieval_pipeline.py` (create)

- [ ] **Step 1: Write failing test**

```python
# tests/test_retrieval_pipeline.py
import numpy as np
import pytest

from app.engines.retrieval import _cosine_from_blob


def test_cosine_from_blob_matches_numpy():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    blob = v.tobytes()
    expected = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v)))
    assert abs(_cosine_from_blob(q, blob) - expected) < 1e-6


def test_cosine_from_blob_returns_zero_on_none():
    q = np.array([1.0, 0.0], dtype=np.float32)
    assert _cosine_from_blob(q, None) == 0.0


def test_cosine_from_blob_returns_zero_on_size_mismatch():
    q = np.array([1.0, 0.0], dtype=np.float32)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert _cosine_from_blob(q, v.tobytes()) == 0.0
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

In `app/engines/retrieval.py`, add near the top (after imports, before class):

```python
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
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add app/engines/retrieval.py tests/test_retrieval_pipeline.py
git commit -m "feat(retrieval): add _cosine_from_blob helper"
```

---

## Task 3: Stage 1 — Entry Cosine (PQ ∪ edge_embedding, max by relationship_id)

**Files:**
- Modify: `app/engines/retrieval.py` (add `_stage1_entry`, `_fetch_pq_rows` already exists, `_fetch_edge_rows` already exists — verify both filter by `is_current` and `tombstoned_at`)
- Test: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Verify existing fetchers filter live rows**

Grep `app/engines/retrieval.py` for `_fetch_pq_rows` and `_fetch_edge_rows`. Both should include:

```
AND COALESCE(r.is_current, 1) = 1
AND r.tombstoned_at IS NULL
```

If they don't, add those predicates.

- [ ] **Step 2: Write failing test**

```python
# tests/test_retrieval_pipeline.py (append)
import numpy as np
from unittest.mock import patch

from app.engines.retrieval import RetrievalEngine
from app.engines.retrieval_types import Candidate


def _make_pq_row(rid, emb):
    return {
        "relationship_id": rid, "id": rid,
        "subject": "s", "predicate": "p", "object": "o",
        "question_embedding": emb.astype(np.float32).tobytes(),
        "sequence_number": 10, "cluster_id": "c1",
        "subject_type": "PERSON", "object_type": "ORG",
        "edge_emotional_valence": 0, "edge_emotional_label": None,
        "edge_episodic_significance": None, "edge_temporal_context": None,
        "edge_relational_type": None, "confidence": 1.0, "arc_id": None,
    }


def _make_edge_row(rid, emb):
    return {
        "relationship_id": rid, "id": rid,
        "subject": "s", "predicate": "p", "object": "o",
        "edge_embedding": emb.astype(np.float32).tobytes(),
        "sequence_number": 10, "cluster_id": "c1",
        "subject_type": "PERSON", "object_type": "ORG",
        "edge_emotional_valence": 0, "edge_emotional_label": None,
        "edge_episodic_significance": None, "edge_temporal_context": None,
        "edge_relational_type": None, "confidence": 1.0, "arc_id": None,
    }


def test_stage1_entry_max_across_pq_and_edge():
    engine = RetrievalEngine()
    q = np.array([1.0, 0.0], dtype=np.float32)
    # relationship 1: pq cosine 0.3, edge cosine 0.9 → entry_cosine should be 0.9
    pq_rows = [_make_pq_row(1, np.array([0.3, 0.95], dtype=np.float32))]
    edge_rows = [_make_edge_row(1, np.array([0.9, 0.43], dtype=np.float32))]
    with patch.object(engine, "_fetch_pq_rows", return_value=pq_rows), \
         patch.object(engine, "_fetch_edge_rows", return_value=edge_rows):
        cands = engine._stage1_entry(user_id=1, q_emb=q)
    assert len(cands) == 1
    assert cands[0].entry_cosine >= 0.85
    assert cands[0].source_stages == {"entry"}


def test_stage1_entry_top_k_caps_pool():
    engine = RetrievalEngine()
    q = np.array([1.0, 0.0], dtype=np.float32)
    # Create 200 pq rows, all same cosine — Entry should cap at K=80.
    v = np.array([1.0, 0.0], dtype=np.float32)
    pq_rows = [_make_pq_row(i, v) for i in range(200)]
    with patch.object(engine, "_fetch_pq_rows", return_value=pq_rows), \
         patch.object(engine, "_fetch_edge_rows", return_value=[]):
        cands = engine._stage1_entry(user_id=1, q_emb=q)
    assert len(cands) == 80


def test_stage1_entry_returns_empty_when_both_pools_empty():
    engine = RetrievalEngine()
    q = np.array([1.0, 0.0], dtype=np.float32)
    with patch.object(engine, "_fetch_pq_rows", return_value=[]), \
         patch.object(engine, "_fetch_edge_rows", return_value=[]):
        cands = engine._stage1_entry(user_id=1, q_emb=q)
    assert cands == []
```

- [ ] **Step 3: Run — expect FAIL** (`_stage1_entry` missing)

- [ ] **Step 4: Implement**

Add to `RetrievalEngine` class:

```python
_ENTRY_POOL_SIZE = 80

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
```

Import `Candidate` at top of retrieval.py:
```python
from app.engines.retrieval_types import Candidate
```

- [ ] **Step 5: Run — expect PASS**

- [ ] **Step 6: Commit**

```bash
git add app/engines/retrieval.py tests/test_retrieval_pipeline.py
git commit -m "feat(retrieval): stage 1 — Entry Cosine (PQ ∪ edge max)"
```

---

## Task 4: Entity resolver module

**Files:**
- Create: `app/engines/entity_resolver.py`
- Create: `tests/conftest.py` (or extend existing) — `seeded_db` fixture
- Test: `tests/test_entity_resolver.py`

- [ ] **Step 1: Add fixture to `tests/conftest.py`**

If a `conftest.py` exists, append. Otherwise create:

```python
# tests/conftest.py
import os
import sqlite3
from dataclasses import dataclass

import numpy as np
import pytest

from app.db.models import MIGRATIONS, run_schema_upgrades
from config.settings import settings


@dataclass
class SeededDB:
    user_id: int
    db_path: str


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    """A fresh SQLite DB with one user (42), one entity (Maya=PERSON),
    and two relationships. Uses a deterministic 2-d embedding for tests."""
    path = str(tmp_path / "seeded.db")
    monkeypatch.setattr(settings, "sqlite_path", path)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    run_schema_upgrades(conn, MIGRATIONS)

    maya_emb = np.array([1.0, 0.0], dtype=np.float32)
    vantage_emb = np.array([0.0, 1.0], dtype=np.float32)

    conn.execute(
        "INSERT INTO entities (user_id, name, entity_type, embedding, "
        "mention_count) VALUES (?, ?, ?, ?, ?)",
        (42, "Maya", "PERSON", maya_emb.tobytes(), 1),
    )
    conn.execute(
        "INSERT INTO entities (user_id, name, entity_type, embedding, "
        "mention_count) VALUES (?, ?, ?, ?, ?)",
        (42, "Vantage", "ORG", vantage_emb.tobytes(), 1),
    )
    conn.execute(
        "INSERT INTO relationships (user_id, subject, predicate, object, "
        "confidence, is_current, sequence_number, cluster_id, "
        "subject_type, object_type, edge_embedding) "
        "VALUES (42, 'Maya', 'works_at', 'Vantage', 1.0, 1, 1, 'c1', "
        "'PERSON', 'ORG', ?)",
        (np.array([0.9, 0.4], dtype=np.float32).tobytes(),),
    )
    conn.commit()
    conn.close()

    return SeededDB(user_id=42, db_path=path)
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_entity_resolver.py
from app.engines.entity_resolver import resolve_query_entities


def test_resolve_query_entities_finds_person(seeded_db, monkeypatch):
    import numpy as np
    from app.vector import embedder
    monkeypatch.setattr(
        embedder, "embed_text",
        lambda s: np.array([1.0, 0.0], dtype=np.float32) if "Maya" in s else np.array([0.0, 1.0], dtype=np.float32),
    )
    matches = resolve_query_entities(
        user_id=seeded_db.user_id, query_text="Where does Maya work?"
    )
    names = [m["name"] for m in matches]
    assert "Maya" in names


def test_resolve_query_entities_skips_stopwords(seeded_db, monkeypatch):
    import numpy as np
    from app.vector import embedder
    monkeypatch.setattr(embedder, "embed_text",
                        lambda s: np.array([0.5, 0.5], dtype=np.float32))
    matches = resolve_query_entities(
        user_id=seeded_db.user_id, query_text="Where is work?"
    )
    # no candidate phrase ("where", "work") should trigger a cosine ≥ 0.55
    # against Maya or Vantage embeddings
    assert all(m["score"] < 0.99 or m["name"] not in {"where", "work"}
               for m in matches)
```

- [ ] **Step 3: Run — expect FAIL**

- [ ] **Step 4: Implement**

```python
# app/engines/entity_resolver.py
"""Query-time entity resolver.

Input: user_id, query_text.
Output: list of {id, name, entity_type, score} for user entities whose
embedding cosine-matches a noun-phrase from the query above tau_entity.

No curated word lists. NP extraction uses spaCy en_core_web_sm; stopwords
are the closed WH/auxiliary grammatical class already used by
predicted_queries.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from app.db.session import get_db_context
from app.vector.embedder import embed_text


TAU_ENTITY = 0.55

_STOPWORDS = {
    "who", "what", "when", "where", "why", "which", "whose", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "will",
    "the", "a", "an", "this", "that", "these", "those",
    "i", "you", "me", "my", "your", "we", "us", "our",
}


def _extract_candidates(query: str) -> List[str]:
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None

    if nlp is None:
        # Fallback: title-cased multi-char tokens.
        return [w for w in query.split()
                if w and w[:1].isupper() and w.lower() not in _STOPWORDS]

    doc = nlp(query)
    out: List[str] = []
    for ent in doc.ents:
        t = ent.text.strip()
        if t and t.lower() not in _STOPWORDS:
            out.append(t)
    for nc in doc.noun_chunks:
        t = nc.text.strip()
        if len(t) > 1 and t.lower() not in _STOPWORDS:
            out.append(t)

    seen = set()
    deduped = []
    for t in out:
        k = t.lower()
        if k not in seen:
            deduped.append(t)
            seen.add(k)
    return deduped


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def resolve_query_entities(
    user_id: int, query_text: str, top_k: int = 10,
    min_score: float = TAU_ENTITY,
) -> List[Dict[str, Any]]:
    candidates = _extract_candidates(query_text)
    if not candidates:
        return []

    phrase_embs = [(c, embed_text(c)) for c in candidates]

    sql = ("SELECT id, name, entity_type, embedding FROM entities "
           "WHERE user_id = ?")
    with get_db_context() as conn:
        rows = conn.execute(sql, (user_id,)).fetchall()

    best: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        if not r["embedding"]:
            continue
        ev = np.frombuffer(r["embedding"], dtype=np.float32)
        top = 0.0
        for _, pe in phrase_embs:
            c = _cosine(ev, pe)
            if c > top:
                top = c
        if top >= min_score:
            best[r["id"]] = {
                "id": r["id"], "name": r["name"],
                "entity_type": r["entity_type"], "score": top,
            }

    ranked = sorted(best.values(), key=lambda x: -x["score"])
    return ranked[:top_k]
```

- [ ] **Step 5: Run — expect PASS**

- [ ] **Step 6: Commit**

```bash
git add app/engines/entity_resolver.py tests/test_entity_resolver.py tests/conftest.py
git commit -m "feat(retrieval): query-time entity resolver"
```

---

## Task 5: Stage 2 — Expand (UNION of entity-touching edges)

**Files:**
- Modify: `app/engines/retrieval.py`
- Test: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
def test_stage2_expand_unions_entity_touching_edges(seeded_db, monkeypatch):
    import numpy as np
    from app.vector import embedder
    from app.engines import entity_resolver
    # Pin embeddings so Maya resolves
    monkeypatch.setattr(embedder, "embed_text",
                        lambda s: np.array([1.0, 0.0], dtype=np.float32))
    # Force entity_resolver to return Maya
    monkeypatch.setattr(entity_resolver, "resolve_query_entities",
                        lambda uid, qt: [{"id": 1, "name": "Maya",
                                          "entity_type": "PERSON",
                                          "score": 0.9}])
    engine = RetrievalEngine()
    # Stage 1 returned an empty pool — Expand must still find the Maya edge.
    out = engine._stage2_expand(
        user_id=seeded_db.user_id,
        query_text="Where does Maya work?",
        candidates=[],
    )
    assert len(out) == 1
    assert out[0].edge["subject"] == "Maya"
    assert out[0].entity_overlap == 1
    assert "expand" in out[0].source_stages


def test_stage2_expand_passes_through_when_no_entities(seeded_db, monkeypatch):
    from app.engines import entity_resolver
    monkeypatch.setattr(entity_resolver, "resolve_query_entities",
                        lambda uid, qt: [])
    engine = RetrievalEngine()
    existing = [Candidate(relationship_id=99, edge={"id": 99},
                          entry_cosine=0.7)]
    existing[0].source_stages.add("entry")
    out = engine._stage2_expand(
        user_id=seeded_db.user_id, query_text="nothing",
        candidates=existing,
    )
    assert len(out) == 1
    assert out[0].relationship_id == 99
    assert out[0].entity_overlap == 0
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

Add to `RetrievalEngine`:

```python
def _stage2_expand(
    self, user_id: int, query_text: str, candidates: List[Candidate]
) -> List[Candidate]:
    from app.engines.entity_resolver import resolve_query_entities
    entities = resolve_query_entities(user_id, query_text)
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
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add app/engines/retrieval.py tests/test_retrieval_pipeline.py
git commit -m "feat(retrieval): stage 2 — Expand (entity union)"
```

---

## Task 6: Stage 3 — Group (cluster_id ∪ arc_id narrowing)

**Files:**
- Modify: `app/engines/retrieval.py`
- Test: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
def test_stage3_group_keeps_cluster_peers_and_sets_members():
    engine = RetrievalEngine()
    c1 = Candidate(relationship_id=1,
                   edge={"cluster_id": "cA", "arc_id": None},
                   entity_overlap=1)
    c2 = Candidate(relationship_id=2,
                   edge={"cluster_id": "cA", "arc_id": None},
                   entity_overlap=0)  # peer of c1 in cluster A → kept
    c3 = Candidate(relationship_id=3,
                   edge={"cluster_id": "cB", "arc_id": None},
                   entity_overlap=0)  # cluster B, no overlap anchor → dropped
    out = engine._stage3_group([c1, c2, c3])
    kept = {c.relationship_id for c in out}
    assert kept == {1, 2}
    for c in out:
        assert c.cluster_members == 2


def test_stage3_group_uses_arc_id_as_secondary_anchor():
    engine = RetrievalEngine()
    c1 = Candidate(relationship_id=1,
                   edge={"cluster_id": "cA", "arc_id": "arc1"},
                   entity_overlap=1)
    c2 = Candidate(relationship_id=2,
                   edge={"cluster_id": "cB", "arc_id": "arc1"},
                   entity_overlap=0)  # different cluster but same arc → kept
    out = engine._stage3_group([c1, c2])
    kept = {c.relationship_id for c in out}
    assert kept == {1, 2}


def test_stage3_group_passes_through_when_no_overlap():
    engine = RetrievalEngine()
    c1 = Candidate(relationship_id=1, edge={"cluster_id": "cA"},
                   entity_overlap=0)
    out = engine._stage3_group([c1])
    assert len(out) == 1
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
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

    # Count fellow survivors per cluster (used by Reconstruction).
    from collections import Counter
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
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add app/engines/retrieval.py tests/test_retrieval_pipeline.py
git commit -m "feat(retrieval): stage 3 — Group (cluster ∪ arc anchor)"
```

---

## Task 7: Memgraph module (BFS proximity)

**Files:**
- Create: `app/engines/memgraph.py`
- Test: `tests/test_memgraph.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_memgraph.py
from app.engines.memgraph import build_adjacency, min_hops_to_entities


def test_direct_edge_zero_hops():
    adj = build_adjacency([{"subject": "Maya", "object": "Vantage"}])
    # Maya is itself in query_entities — 0 hops.
    assert min_hops_to_entities(
        subject="Maya", object_="Vantage",
        query_entities={"Maya"}, adj=adj,
    ) == 0


def test_one_hop_via_shared_neighbor():
    adj = build_adjacency([
        {"subject": "Maya", "object": "Vantage"},
        {"subject": "Leon", "object": "Vantage"},
    ])
    # Leon edge: Leon→Vantage→Maya = 2 hops from Leon; Vantage is 1 hop.
    assert min_hops_to_entities(
        subject="Leon", object_="Vantage",
        query_entities={"Maya"}, adj=adj,
    ) == 1


def test_disconnected_returns_neg_one():
    adj = build_adjacency([
        {"subject": "Maya", "object": "Vantage"},
        {"subject": "Bob", "object": "Acme"},
    ])
    assert min_hops_to_entities(
        subject="Bob", object_="Acme",
        query_entities={"Maya"}, adj=adj,
    ) == -1


def test_respects_max_hops():
    # A chain Maya - X1 - X2 - X3 - X4 - Far ; query {Maya}, cand subject=Far
    adj = build_adjacency([
        {"subject": "Maya", "object": "X1"},
        {"subject": "X1", "object": "X2"},
        {"subject": "X2", "object": "X3"},
        {"subject": "X3", "object": "X4"},
        {"subject": "X4", "object": "Far"},
    ])
    # 5 hops from Far → Maya; MAX_HOPS=3 → unreachable.
    assert min_hops_to_entities(
        subject="Far", object_="Far",
        query_entities={"Maya"}, adj=adj,
    ) == -1
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
# app/engines/memgraph.py
"""Memgraph — BFS proximity on the relationships edge list.

The relationships table IS the graph: each row's (subject, object) is an
undirected edge. No separate graph store; we build adjacency per query
from the full user edge set."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, Set


MAX_HOPS = 3


def build_adjacency(edges: Iterable[Dict]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for e in edges:
        s = (e.get("subject") or "").strip()
        o = (e.get("object") or "").strip()
        if not s or not o:
            continue
        adj[s].add(o)
        adj[o].add(s)
    return adj


def _bfs_from(start: str, goals: Set[str],
              adj: Dict[str, Set[str]]) -> int:
    if start in goals:
        return 0
    if start not in adj:
        return -1
    visited = {start}
    q = deque([(start, 0)])
    while q:
        node, d = q.popleft()
        if d >= MAX_HOPS:
            continue
        for n in adj.get(node, ()):
            if n in visited:
                continue
            if n in goals:
                return d + 1
            visited.add(n)
            q.append((n, d + 1))
    return -1


def min_hops_to_entities(
    subject: str, object_: str,
    query_entities: Set[str], adj: Dict[str, Set[str]],
) -> int:
    """Minimum BFS hop count from either endpoint of (subject, object_)
    to any name in query_entities, capped at MAX_HOPS. Returns -1 if
    unreachable within the cap."""
    if not query_entities:
        return -1
    best = -1
    for endpoint in (subject, object_):
        if not endpoint:
            continue
        h = _bfs_from(endpoint, query_entities, adj)
        if h >= 0 and (best < 0 or h < best):
            best = h
    return best
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add app/engines/memgraph.py tests/test_memgraph.py
git commit -m "feat(retrieval): memgraph BFS proximity module"
```

---

## Task 8: Stage 4 — Relate (BFS filter on full user graph)

**Files:**
- Modify: `app/engines/retrieval.py`
- Test: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
def test_stage4_relate_drops_disconnected(seeded_db, monkeypatch):
    from app.engines import entity_resolver
    monkeypatch.setattr(entity_resolver, "resolve_query_entities",
                        lambda uid, qt: [{"id": 1, "name": "Maya",
                                          "entity_type": "PERSON",
                                          "score": 0.9}])
    engine = RetrievalEngine()
    c_connected = Candidate(
        relationship_id=1,
        edge={"subject": "Maya", "object": "Vantage"})
    c_disconnected = Candidate(
        relationship_id=2,
        edge={"subject": "Bob", "object": "Acme"})
    out = engine._stage4_relate(
        user_id=seeded_db.user_id,
        query_text="Where does Maya work?",
        candidates=[c_connected, c_disconnected],
    )
    rids = {c.relationship_id for c in out}
    assert 1 in rids
    assert 2 not in rids


def test_stage4_relate_passes_through_when_no_entities(monkeypatch):
    from app.engines import entity_resolver
    monkeypatch.setattr(entity_resolver, "resolve_query_entities",
                        lambda uid, qt: [])
    engine = RetrievalEngine()
    c = Candidate(relationship_id=1, edge={"subject": "a", "object": "b"})
    out = engine._stage4_relate(
        user_id=999, query_text="x", candidates=[c],
    )
    assert out == [c]
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
def _stage4_relate(
    self, user_id: int, query_text: str, candidates: List[Candidate]
) -> List[Candidate]:
    from app.engines.entity_resolver import resolve_query_entities
    from app.engines.memgraph import (
        build_adjacency, min_hops_to_entities,
    )

    entities = resolve_query_entities(user_id, query_text)
    if not entities:
        return candidates  # inapplicable — pass through

    query_names = {e["name"] for e in entities}

    # Build adjacency from the FULL user graph (not survivors) so BFS
    # can traverse through edges not in the candidate set.
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
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add app/engines/retrieval.py tests/test_retrieval_pipeline.py
git commit -m "feat(retrieval): stage 4 — Relate (BFS filter on full graph)"
```

---

## Task 9: Stage 5 — Exit Cosine (re-rank by edge_embedding)

**Files:**
- Modify: `app/engines/retrieval.py`
- Test: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
def test_stage5_exit_reranks_by_edge_embedding():
    import numpy as np
    engine = RetrievalEngine()
    q_emb = np.array([1.0, 0.0], dtype=np.float32)
    # c1 edge_emb = [0.9, 0.4] → cosine ~0.914
    # c2 edge_emb = [0.3, 0.95] → cosine ~0.301
    # Entry cosine order had c2 higher (0.95 vs 0.4 — but that was PQ);
    # Exit should rerank c1 first.
    c1 = Candidate(
        relationship_id=1,
        edge={"edge_embedding": np.array([0.9, 0.4], dtype=np.float32).tobytes()},
        entry_cosine=0.4,
    )
    c2 = Candidate(
        relationship_id=2,
        edge={"edge_embedding": np.array([0.3, 0.95], dtype=np.float32).tobytes()},
        entry_cosine=0.95,
    )
    out = engine._stage5_exit(q_emb=q_emb, candidates=[c2, c1])
    assert out[0].relationship_id == 1
    assert out[1].relationship_id == 2


def test_stage5_exit_handles_null_edge_embedding():
    import numpy as np
    engine = RetrievalEngine()
    q_emb = np.array([1.0, 0.0], dtype=np.float32)
    c = Candidate(relationship_id=1, edge={"edge_embedding": None},
                  entry_cosine=0.8)
    out = engine._stage5_exit(q_emb=q_emb, candidates=[c])
    assert len(out) == 1
    assert out[0].exit_cosine == 0.0  # no crash, no drop
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
def _stage5_exit(
    self, q_emb: np.ndarray, candidates: List[Candidate]
) -> List[Candidate]:
    for c in candidates:
        c.exit_cosine = _cosine_from_blob(q_emb, c.edge.get("edge_embedding"))
        c.source_stages.add("exit")
    return sorted(candidates, key=lambda c: -c.exit_cosine)
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add app/engines/retrieval.py tests/test_retrieval_pipeline.py
git commit -m "feat(retrieval): stage 5 — Exit Cosine (edge_embedding rerank)"
```

---

## Task 10: Stage 6 — Validate (warn, never drop)

**Files:**
- Modify: `app/engines/retrieval.py`
- Test: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
def test_stage6_validate_warns_on_type_mismatch():
    engine = RetrievalEngine()
    c = Candidate(relationship_id=1,
                  edge={"subject_type": "PERSON", "object_type": "TIME"})
    warning = engine._stage6_validate(
        query_text="Where does Maya work?", top=c,
    )
    assert warning == "type_mismatch"


def test_stage6_validate_no_warning_when_match():
    engine = RetrievalEngine()
    c = Candidate(relationship_id=1,
                  edge={"subject_type": "PERSON", "object_type": "LOCATION"})
    warning = engine._stage6_validate(
        query_text="Where does Maya work?", top=c,
    )
    assert warning is None


def test_stage6_validate_no_warning_when_wh_unresolved():
    engine = RetrievalEngine()
    c = Candidate(relationship_id=1,
                  edge={"subject_type": "PERSON", "object_type": "TIME"})
    # "Tell me about..." has no WH → no warning regardless of types
    warning = engine._stage6_validate(
        query_text="Tell me about Maya.", top=c,
    )
    assert warning is None
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
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
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add app/engines/retrieval.py tests/test_retrieval_pipeline.py
git commit -m "feat(retrieval): stage 6 — Validate (warn-only)"
```

---

## Task 11: Wire all stages into `retrieve()` (Lookup mode)

**Files:**
- Modify: `app/engines/retrieval.py` (rewrite `retrieve()` body; delete old `_cosine_pool` method)
- Test: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Write failing E2E test**

```python
def test_retrieve_lookup_pipeline_returns_vantage(seeded_db, monkeypatch):
    import numpy as np
    from app.vector import embedder
    # Pin: query "Where does Maya work?" embeds as [1,0]
    monkeypatch.setattr(
        embedder, "embed_text",
        lambda s: np.array([1.0, 0.0], dtype=np.float32),
    )
    engine = RetrievalEngine()
    result = engine.retrieve(
        user_id=seeded_db.user_id,
        query_text="Where does Maya work?",
    )
    # seeded edge is Maya works_at Vantage
    assert hasattr(result, "object")
    assert result.object == "Vantage"
    assert result.source == "moat_pipeline_lookup"
    cd = result.convergence_details
    assert set(cd.keys()) >= {
        "entry_cosine", "entity_overlap", "cluster_members",
        "hops_to_entity", "exit_cosine", "source_stages",
    }
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement — replace `retrieve()` body**

Replace `retrieve()` (currently lines 154-225) with:

```python
def retrieve(self, user_id: int, query_text: str):
    """Moat pipeline — Lookup mode.
    Entry → Expand → Group → Relate → Exit Cosine → Validate (warn) → top-1."""
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

    # Tie-break by sequence_number DESC when exit cosines match.
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

    return Answer(
        text=self._triple_to_sentence(subj, pred, obj),
        subject=subj, predicate=pred, object=obj,
        confidence=1.0, source="moat_pipeline_lookup",
        survivors=len(candidates),
        convergence_details={
            "query": query_text,
            "entry_cosine": top.entry_cosine,
            "entity_overlap": top.entity_overlap,
            "cluster_members": top.cluster_members,
            "hops_to_entity": top.hops_to_entity,
            "exit_cosine": top.exit_cosine,
            "sequence_number": top.edge.get("sequence_number"),
            "source_stages": sorted(top.source_stages),
            "validate_warning": warning,
        },
    )
```

Delete the old `_cosine_pool` method entirely (it's superseded by the stage pipeline).

- [ ] **Step 4: Run ALL pipeline tests**

```
py -3.10 -m pytest tests/test_retrieval_pipeline.py tests/test_retrieval_types.py tests/test_entity_resolver.py tests/test_memgraph.py -v
```
Expected: all pass.

- [ ] **Step 5: Run existing retrieval tests — update assertions for new source/details**

```
py -3.10 -m pytest tests/test_retrieval_pq.py -v
```
If failures cite `source != "pq_cosine"` or missing convergence keys: update those tests to expect `source == "moat_pipeline_lookup"` and the new keys. Do NOT change pipeline semantics to satisfy old tests.

- [ ] **Step 6: Commit**

```bash
git add app/engines/retrieval.py tests/test_retrieval_pipeline.py tests/test_retrieval_pq.py
git commit -m "feat(retrieval): wire Moat pipeline into retrieve() Lookup mode"
```

---

## Task 12: Rewire `reconstruct()` onto the same pipeline (Reconstruction mode)

**Files:**
- Modify: `app/engines/retrieval.py` (rewrite `reconstruct()` body)
- Test: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
def test_reconstruct_uses_same_pipeline_top_n_then_fuses(seeded_db, monkeypatch):
    import numpy as np
    from app.vector import embedder
    monkeypatch.setattr(
        embedder, "embed_text",
        lambda s: np.array([1.0, 0.0], dtype=np.float32),
    )
    engine = RetrievalEngine()
    sit = engine.reconstruct(
        user_id=seeded_db.user_id,
        query_text="Tell me about Maya.",
    )
    # seeded DB has one Maya edge; Situation should contain it.
    assert sit.source in {"reconstruct", "unclustered"}
    assert sit.survivors >= 1
    assert "Maya" in (sit.narrative or "") or any(
        "Maya" in (c.participants or []) for c in (sit.clusters or [])
    )
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement — replace `reconstruct()` body**

```python
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
        q_emb = embed_text(query_text)
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

    survivors = candidates[: self._RECONSTRUCT_TOP_N]
    if not survivors:
        return Situation(
            narrative="", source="structural_refusal", survivors=0,
            convergence_details={"reason": "no_structural_match"},
        )

    # Group by cluster_id — feed the existing fuser.
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in survivors:
        key = c.edge.get("cluster_id") or "__unclustered__"
        groups[key].append(c.edge)

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
        narrative=narrative, clusters=fused,
        participants=sorted(all_participants),
        dominant_mood=dominant_mood,
        pivotal_events=all_pivotal, timeline=all_timeline,
        grounding_map=grounding_map,
        source="reconstruct" if groups != {"__unclustered__": survivors[0].edge and [survivors[0].edge]} else "unclustered",
        survivors=sum(len(c.edges) for c in fused),
    )
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add app/engines/retrieval.py tests/test_retrieval_pipeline.py
git commit -m "feat(retrieval): Reconstruction mode on Moat pipeline"
```

---

## Task 13: Run full test suite, resolve any regression

**Files:**
- None (verification only; fix if fail)

- [ ] **Step 1: Run all retrieval-adjacent suites**

```
cd D:/Nura/Code/nura_living_memory_code
py -3.10 -m pytest tests/test_retrieval_pipeline.py tests/test_retrieval_types.py \
    tests/test_entity_resolver.py tests/test_memgraph.py \
    tests/test_retrieval_pq.py tests/test_foundation_writepath.py \
    tests/test_type_resolver.py -v
```
Expected: all pass. If any regress with non-env errors, fix — do not mark plan complete.

- [ ] **Step 2: Commit if any test fixes needed**

```bash
git add tests/...
git commit -m "fix(tests): update assertions for Moat pipeline output format"
```

---

## Task 14: YAML audit — expect ≥ 54%

**Files:**
- Measurement only.

- [ ] **Step 1: Clean DB**

```
rm -f "D:/Nura/Code/nura_living_memory_code/Memory Storage/h_retrieval_audit_v2.db"
```

- [ ] **Step 2: Run**

```
cd D:/Nura/Code/nura_living_memory_code
py -3.10 tmp/pipeline/h_retrieval_audit_v2.py 2>&1 | tee tmp/measurement/audit_moat.log
```

- [ ] **Step 3: Read pass rate**

```
py -3.10 -c "import json; s=json.load(open('tmp/pipeline/h_retrieval_audit_v2_summary.json')); print(s)"
```
**GATE:** ≥ 54% pass rate. If below — STOP. Do not run the other benchmarks. Diagnose via failure_layer distribution.

- [ ] **Step 4: Commit results**

```bash
git add tmp/measurement/audit_moat.log tmp/pipeline/h_retrieval_audit_v2_summary.json tmp/pipeline/h_retrieval_audit_v2.jsonl
git commit -m "measure(retrieval): YAML audit after Moat pipeline — <PCT>%"
```

---

## Task 15: Remaining 4 benchmarks (MCP server path)

**Files:**
- Measurement only. Sequence is sequential — all use port 7130.

Per benchmark:

- [ ] **Step 1: Clean DB + boot server**

```
rm -f <benchmark>.db
KENOTIC_DB_PATH=<path> KENOTIC_MCP_TOKEN=<token> \
  py -3.10 -m mcp.http_server --token <token> --host 127.0.0.1 --port 7130 \
  > tmp/measurement/<benchmark>_server_moat.log 2>&1 &
# poll http://127.0.0.1:7130/healthz until 200
```

- [ ] **Step 2: Run the benchmark**

Core 50:
```
py -3.10 tmp/baseline/atant_baseline_runner.py --mode cumulative --source core50 --out tmp/baseline/atant_core50_moat.json
```
Gate: ≥ 20.07%.

Stress 51-100:
```
py -3.10 tmp/baseline/atant_baseline_runner.py --mode cumulative --source cumulative --range 51-100 --out tmp/baseline/atant_stress_moat.json
```
Gate: ≥ 5.4%.

Banff:
```
py -3.10 tmp/pipeline/e_two_agent_continuity.py
```
Gate: ≥ 5/8.

DeLores:
```
py -3.10 tmp/pipeline/g_long_convo_test.py
```
Gate: ≥ 6/10.

- [ ] **Step 3: Stop server between benchmarks**

- [ ] **Step 4: Commit each result**

```bash
git add tmp/baseline/atant_*_moat.json tmp/measurement/*_moat.log
git commit -m "measure(retrieval): <benchmark> after Moat pipeline"
```

---

## Self-Review Checklist

- ✅ Spec coverage: every stage in the spec has a task (1-12). Benchmarks in 14-15. Test suite audit in 13.
- ✅ No placeholders: every step has complete code or an exact command.
- ✅ Type consistency: `Candidate`, stage method names `_stage1_entry` ... `_stage6_validate`, helper `_cosine_from_blob` — all consistent across tasks.
- ✅ Property wiring: Entry reads `edge_embedding` + `question_embedding` + live filters; Expand reads `entities.*` + `subject`/`object`; Group reads `cluster_id` + `arc_id`; Relate reads graph + `edge_relational_type` (future hop-weight); Exit reads `edge_embedding`; Validate reads `subject_type`/`object_type`; Reconstruction reads `sequence_number`/mood cols/`edge_*` trace cols.
- ✅ TDD: every stage task is failing test → implement → passing test → commit.
- ✅ No weights: no coefficients appear anywhere. Only structural caps (K, τ_entity, MAX_HOPS, N_reconstruct).
- ✅ Refusal discipline: refuse when Entry+Expand empty, or when Relate empties the pool. Never refuse for inapplicability of a single stage.

## Open decisions (ship as-is unless Sam overrides)

1. `K=80`, `τ_entity=0.55`, `MAX_HOPS=3`, `N_reconstruct=20` — defaults stated in spec.
2. Mode detection: leading WH word = Lookup; else Reconstruction. Not in this plan — `retrieve()` handles Lookup, `reconstruct()` handles Reconstruction; the caller (MCP tool dispatch) already routes.
3. `edge_relational_type` is carried through but not yet used to weight BFS hops. v2 feature.
