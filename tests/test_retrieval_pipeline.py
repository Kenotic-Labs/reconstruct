import numpy as np
import pytest
from unittest.mock import patch

from app.engines.retrieval import _cosine_from_blob, RetrievalEngine
from app.engines.retrieval_types import Candidate


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


def test_stage2_expand_unions_entity_touching_edges(seeded_db, monkeypatch):
    import numpy as np
    from app.vector import embedder
    from app.engines import entity_resolver
    monkeypatch.setattr(embedder, "embed_text",
                        lambda s: np.array([1.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(entity_resolver, "resolve_query_entities",
                        lambda uid, qt: [{"id": 1, "name": "Maya",
                                          "entity_type": "PERSON",
                                          "score": 0.9}])
    engine = RetrievalEngine()
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


def test_stage3_group_keeps_cluster_peers_and_sets_members():
    engine = RetrievalEngine()
    c1 = Candidate(relationship_id=1,
                   edge={"cluster_id": "cA", "arc_id": None},
                   entity_overlap=1)
    c2 = Candidate(relationship_id=2,
                   edge={"cluster_id": "cA", "arc_id": None},
                   entity_overlap=0)
    c3 = Candidate(relationship_id=3,
                   edge={"cluster_id": "cB", "arc_id": None},
                   entity_overlap=0)
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
                   entity_overlap=0)
    out = engine._stage3_group([c1, c2])
    kept = {c.relationship_id for c in out}
    assert kept == {1, 2}


def test_stage3_group_passes_through_when_no_overlap():
    engine = RetrievalEngine()
    c1 = Candidate(relationship_id=1, edge={"cluster_id": "cA"},
                   entity_overlap=0)
    out = engine._stage3_group([c1])
    assert len(out) == 1


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


def test_stage5_exit_reranks_by_edge_embedding():
    engine = RetrievalEngine()
    q_emb = np.array([1.0, 0.0], dtype=np.float32)
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
    engine = RetrievalEngine()
    q_emb = np.array([1.0, 0.0], dtype=np.float32)
    c = Candidate(relationship_id=1, edge={"edge_embedding": None},
                  entry_cosine=0.8)
    out = engine._stage5_exit(q_emb=q_emb, candidates=[c])
    assert len(out) == 1
    assert out[0].exit_cosine == 0.0


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
    warning = engine._stage6_validate(
        query_text="Tell me about Maya.", top=c,
    )
    assert warning is None
