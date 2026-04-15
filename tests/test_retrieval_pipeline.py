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
