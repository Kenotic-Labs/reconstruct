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
