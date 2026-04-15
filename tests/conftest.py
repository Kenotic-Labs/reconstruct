import os
import sqlite3
from dataclasses import dataclass

import numpy as np
import pytest

from app.db.session import init_db
from config.settings import settings


@dataclass
class SeededDB:
    user_id: int
    db_path: str


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    """A fresh SQLite DB with one user (42), two entities (Maya=PERSON,
    Vantage=ORG), and one relationship (Maya works_at Vantage). Uses a
    deterministic 2-d embedding for tests."""
    path = str(tmp_path / "seeded.db")
    monkeypatch.setattr(settings, "sqlite_path", path)

    init_db(path)

    # sequence_number is added lazily by the write path; add it here.
    conn = sqlite3.connect(path)
    try:
        conn.execute("ALTER TABLE relationships ADD COLUMN sequence_number INTEGER")
        conn.commit()
    except Exception:
        pass
    conn.close()

    maya_emb = np.array([1.0, 0.0], dtype=np.float32)
    vantage_emb = np.array([0.0, 1.0], dtype=np.float32)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
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
