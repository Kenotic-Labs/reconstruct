# -*- coding: cp1252 -*-
"""
test_foundation_writepath.py -- write-path foundation layer tests
(2026-04-14).

Covers:
  - Entity type resolution on a 10-case corpus (accepts plausible
    alternatives where NER / WordNet honest-miss).
  - Predicted query generation produces >= 2 parseable questions on 5
    known triples.
  - MemoryEngine.store() now writes subject_type/object_type on
    relationships, entity_type on entities, and predicted_queries rows.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Temp DB fixture ───────────────────────────────────────────────

@pytest.fixture
def tmp_db(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="kenotic_foundation_")
    db_path = os.path.join(tmpdir, "memory.db")

    from config.settings import settings
    monkeypatch.setattr(settings, "sqlite_path", db_path)

    from app.db.models import MIGRATIONS, run_schema_upgrades
    conn = sqlite3.connect(db_path)
    conn.executescript(MIGRATIONS)
    run_schema_upgrades(conn)
    try:
        conn.execute("ALTER TABLE relationships ADD COLUMN sequence_number INTEGER")
    except Exception:
        pass
    conn.commit()
    conn.close()
    yield db_path


@pytest.fixture
def engine(tmp_db, monkeypatch):
    """MemoryEngine with the expensive embedding skipped, but the
    foundation helpers (type + predicted queries) kept real.

    Embeddings in _prepare_row fail-open (try/except) so no method
    stub needed -- just stub the embedder module to avoid MiniLM.
    """
    from app.engines import memory as memmod

    try:
        from app.vector import embedder as _embedder_mod
        monkeypatch.setattr(_embedder_mod, "embed_text", lambda text: __import__('numpy').zeros(384, dtype=__import__('numpy').float32))
    except Exception:
        pass

    eng = memmod.MemoryEngine()
    return eng


USER = 77


# ── 1. Entity-type resolver (10 cases) ────────────────────────────

def test_resolver_10_cases():
    from app.engines.type_resolver import resolve_entity_type
    cases = [
        ("Maya", {"PERSON"}),
        ("Google", {"ORG"}),
        ("Ann Arbor", {"LOCATION"}),
        ("Tuesday", {"TIME"}),
        ("Honda Civic", {"PRODUCT", "ORG"}),  # NER edge case
        ("interview", {"EVENT", "GENERIC"}),  # WordNet sense dependency
        ("five dollars", {"QUANTITY"}),
        ("World War II", {"EVENT"}),
        ("Mona Lisa", {"WORK_OF_ART", "PERSON"}),  # NER ambiguity
        ("dog", {"GENERIC"}),
    ]
    results = {}
    for name, allowed in cases:
        got = resolve_entity_type(name)
        results[name] = got
        assert got in allowed, f"{name} -> {got}, expected one of {allowed}"
    # At least 7 of 10 should land on their primary intent (sanity).
    primary = {
        "Maya": "PERSON", "Google": "ORG", "Ann Arbor": "LOCATION",
        "Tuesday": "TIME", "five dollars": "QUANTITY",
        "World War II": "EVENT", "dog": "GENERIC",
    }
    hit = sum(1 for k, v in primary.items() if results[k] == v)
    assert hit >= 6, f"primary accuracy too low: {hit}/7 ({results})"


# ── 2. Predicted query generator (5 triples, >= 2 questions each) ─

@pytest.mark.slow
def test_predicted_queries_5_triples():
    from app.engines.predicted_queries import generate_predicted_queries
    # Each triple has >= 2 WH-types applicable (by structural rule).
    triples = [
        ("Maya", "works_at", "Google", "PERSON", "ORG"),            # WHAT + WHO
        ("Sam", "lives_in", "Ann_Arbor", "PERSON", "LOCATION"),      # WHAT + WHO + WHERE
        ("meeting", "scheduled_on", "Tuesday", "EVENT", "TIME"),     # WHAT + WHEN
        ("Emily", "married_to", "Jake", "PERSON", "PERSON"),         # WHAT + WHO
        ("report", "submitted_on", "Monday", "GENERIC", "TIME"),     # WHAT + WHEN
    ]
    for s, p, o, st, ot in triples:
        pairs = generate_predicted_queries(s, p, o, st, ot)
        assert len(pairs) >= 2, f"{(s,p,o)} -> only {len(pairs)} questions"
        for q, emb in pairs:
            assert q and q.strip(), "empty question"
            assert emb is not None and emb.size > 0, "empty embedding"


# ── 3. store() integration ────────────────────────────────────────

def test_store_writes_types_and_predicted_queries(engine, tmp_db):
    rid = engine.store(
        user_id=USER,
        subject="Maya",
        predicate="works_at",
        object="Google",
        source_text="Maya works at Google in Mountain View.",
    )
    assert rid > 0

    from app.db.session import get_db_context
    with get_db_context() as conn:
        rel = conn.execute(
            "SELECT subject_type, object_type FROM relationships WHERE id = ?",
            (rid,),
        ).fetchone()
        assert rel is not None
        assert rel["subject_type"] == "PERSON"
        assert rel["object_type"] == "ORG"

        # Entities backfilled.
        ent_rows = conn.execute(
            "SELECT name, entity_type FROM entities WHERE user_id = ?",
            (USER,),
        ).fetchall()
        by_name = {r["name"].lower(): r["entity_type"] for r in ent_rows}
        assert by_name.get("maya") == "PERSON"
        assert by_name.get("google") == "ORG"

        # Predicted queries written.
        pq_rows = conn.execute(
            "SELECT predicted_question, answer_subject, answer_text FROM predicted_queries WHERE relationship_id = ?",
            (rid,),
        ).fetchall()
        assert len(pq_rows) >= 2
        for r in pq_rows:
            assert r["predicted_question"]
            assert r["answer_subject"] == "Maya"
            assert "Maya" in r["answer_text"]


def test_store_idempotent_on_replay(engine):
    # Re-storing the same triple should not duplicate predicted_queries.
    r1 = engine.store(user_id=USER, subject="Sam", predicate="lives_in", object="Michigan",
                      source_text="Sam lives in Michigan.")
    r2 = engine.store(user_id=USER, subject="Sam", predicate="lives_in", object="Michigan",
                      source_text="Sam lives in Michigan.")
    assert r1 == r2  # upsert returns same id

    from app.db.session import get_db_context
    with get_db_context() as conn:
        pq = conn.execute(
            "SELECT COUNT(*) AS n FROM predicted_queries WHERE relationship_id = ?",
            (r1,),
        ).fetchone()
        # Some positive number; no explosion.
        assert 2 <= pq["n"] <= 10
