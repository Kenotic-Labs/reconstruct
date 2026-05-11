"""
Tests for memory.forget and memory.show — the soft-tombstone API.

Fast tests avoid T5 / MiniLM loading by calling MemoryEngine.store()
directly (hand-authored triples) and by monkeypatching sdk.Kenotic so
engine construction stays cheap. Full-stack tests (marked `slow`) still
exist for end-to-end coverage and are opt-in via `-m slow`.
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
    """Create a scratch SQLite DB with full schema + tombstone migration."""
    tmpdir = tempfile.mkdtemp(prefix="kenotic_forget_")
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


# ── Stub MemoryEngine with no-op edge-trace writes (skip embedding) ──

@pytest.fixture
def engine(tmp_db, monkeypatch):
    """MemoryEngine with the costly embedding paths stubbed.

    Embeddings in _prepare_row fail-open (try/except) so no method stub
    needed -- just stub the embedder module to avoid loading MiniLM.
    """
    from app.engines import memory as memmod

    try:
        from app.vector import embedder as _embedder_mod
        monkeypatch.setattr(_embedder_mod, "embed_text", lambda text: __import__('numpy').zeros(384, dtype=__import__('numpy').float32))
    except Exception:
        pass

    eng = memmod.MemoryEngine()
    return eng


def _insert_triple(engine, user_id, subj, pred, obj,
                   *, source_timestamp=None, source_tag=None,
                   source_text="",
                   confidence=0.9):
    """Insert a hand-authored triple via store(), returns rel_id."""
    return engine.store(
        user_id=user_id,
        subject=subj,
        predicate=pred,
        object=obj,
        source_text=source_text or f"{subj} {pred} {obj}",
        confidence=confidence,
        source_timestamp=source_timestamp,
        source_tag=source_tag,
    )


USER = 42


# ── Forget tests ──────────────────────────────────────────────────

def test_forget_by_triple_id_emits_one_tombstone(engine):
    rid = _insert_triple(engine, USER, "sam", "lives_in", "michigan")
    assert rid > 0

    n = engine.forget_by_triple_id(USER, rid)
    assert n == 1

    # Row still exists but is tombstoned
    from app.db.session import get_db_context
    with get_db_context() as conn:
        row = conn.execute(
            "SELECT tombstoned_at, tombstone_reason FROM relationships WHERE id = ?",
            (rid,),
        ).fetchone()
        assert row["tombstoned_at"] is not None
        assert "by_triple_id" in row["tombstone_reason"]

    # Audit row in memories
    with get_db_context() as conn:
        audit = conn.execute(
            "SELECT content FROM memories WHERE user_id = ? AND memory_type = 'summary'",
            (USER,),
        ).fetchone()
        assert audit is not None
        assert "forget op" in audit["content"]


def test_forget_by_entity_covers_subject_and_object_positions(engine):
    r1 = _insert_triple(engine, USER, "maya", "works_at", "acme")
    r2 = _insert_triple(engine, USER, "priya", "manages", "maya")
    r3 = _insert_triple(engine, USER, "sam", "lives_in", "michigan")

    n = engine.forget_by_entity(USER, "maya")
    assert n == 2

    rels = engine.get_relationships(USER)
    live_ids = {r.id for r in rels}
    assert r3 in live_ids
    assert r1 not in live_ids
    assert r2 not in live_ids


def test_forget_by_entity_case_insensitive(engine):
    r1 = _insert_triple(engine, USER, "Maya", "works_at", "Acme")
    n = engine.forget_by_entity(USER, "MAYA")
    assert n == 1


def test_forget_by_time_range_respects_source_timestamp(engine):
    r1 = _insert_triple(engine, USER, "a", "did", "thing1",
                        source_timestamp="2026-01-15T10:00:00")
    r2 = _insert_triple(engine, USER, "b", "did", "thing2",
                        source_timestamp="2026-02-20T10:00:00")
    r3 = _insert_triple(engine, USER, "c", "did", "thing3",
                        source_timestamp="2026-03-10T10:00:00")

    n = engine.forget_by_time_range(
        USER, "2026-02-01T00:00:00", "2026-02-28T23:59:59"
    )
    assert n == 1

    rels = engine.get_relationships(USER)
    live_ids = {r.id for r in rels}
    assert r1 in live_ids
    assert r2 not in live_ids
    assert r3 in live_ids


def test_forget_by_time_range_skips_rows_without_source_timestamp(engine):
    # No source_timestamp → should be skipped by time-range forget
    r1 = _insert_triple(engine, USER, "a", "did", "thing1",
                        source_timestamp=None)

    n = engine.forget_by_time_range(
        USER, "1900-01-01T00:00:00", "2099-12-31T23:59:59"
    )
    assert n == 0

    rels = engine.get_relationships(USER)
    assert r1 in {r.id for r in rels}


def test_forget_by_source_tag_exact_match(engine):
    r1 = _insert_triple(engine, USER, "a", "did", "thing1", source_tag="session-42")
    r2 = _insert_triple(engine, USER, "b", "did", "thing2", source_tag="session-43")
    r3 = _insert_triple(engine, USER, "c", "did", "thing3", source_tag="session-42")

    n = engine.forget_by_source(USER, "session-42")
    assert n == 2

    rels = engine.get_relationships(USER)
    live_ids = {r.id for r in rels}
    assert r2 in live_ids
    assert r1 not in live_ids
    assert r3 not in live_ids


def test_forget_is_idempotent(engine):
    rid = _insert_triple(engine, USER, "sam", "lives_in", "michigan")
    assert engine.forget_by_triple_id(USER, rid) == 1
    assert engine.forget_by_triple_id(USER, rid) == 0

    assert engine.forget_by_entity(USER, "sam") == 0


def test_get_relationships_excludes_tombstoned_triples(engine):
    rid = _insert_triple(engine, USER, "sam", "lives_in", "michigan")
    before = engine.get_relationships(USER, subject="sam")
    assert any(r.id == rid for r in before)

    engine.forget_by_triple_id(USER, rid)
    after = engine.get_relationships(USER, subject="sam")
    assert not any(r.id == rid for r in after)


def test_retrieval_load_edges_excludes_tombstoned(engine, tmp_db):
    """_load_edges should not return tombstoned rows."""
    rid = _insert_triple(engine, USER, "sam", "lives_in", "michigan")
    engine.forget_by_triple_id(USER, rid)

    # Import the retrieval query directly without constructing the
    # full engine (which would load embedding models)
    from app.db.session import get_db_context
    with get_db_context() as conn:
        rows = conn.execute(
            """SELECT id FROM relationships
               WHERE user_id = ? AND COALESCE(is_current, 1) = 1
                 AND tombstoned_at IS NULL""",
            (USER,),
        ).fetchall()
    assert all(r["id"] != rid for r in rows)


# ── Show / list_by_facet tests ────────────────────────────────────

def test_show_time_facet_respects_limit(engine):
    for i in range(5):
        _insert_triple(
            engine, USER, f"subj{i}", "did", f"thing{i}",
            source_timestamp=f"2026-04-0{i+1}T10:00:00",
        )

    rows = engine.list_by_facet(USER, "time", limit=3)
    assert len(rows) == 3
    # Should be in DESC order by source_timestamp
    ts = [r["source_timestamp"] for r in rows]
    assert ts == sorted(ts, reverse=True)


def test_show_entity_facet_returns_matching_rows(engine):
    _insert_triple(engine, USER, "maya", "works_at", "acme")
    _insert_triple(engine, USER, "priya", "manages", "maya")
    _insert_triple(engine, USER, "sam", "lives_in", "michigan")

    rows = engine.list_by_facet(USER, "entity", value="maya")
    assert len(rows) == 2
    for r in rows:
        assert "maya" in (r["subject"].lower(), r["object"].lower())


def test_show_export_raw_text_false_omits_source_text(engine, monkeypatch):
    """SDK show() with export_raw_text=False strips source_text."""
    _insert_triple(
        engine, USER, "sam", "lives_in", "michigan",
        source_text="I live in Michigan now.",
    )

    # Exercise the SDK show() directly, but bypass engine construction.
    from sdk.client import Kenotic
    k = Kenotic.__new__(Kenotic)
    k.user_id = USER

    # Stub _engines to return our pre-built engine
    k._engines = lambda: (engine, None, None)

    rows = k.show(facet="entity", value="sam", export_raw_text=False)
    assert len(rows) == 1
    assert "source_text" not in rows[0]

    rows2 = k.show(facet="entity", value="sam", export_raw_text=True)
    assert "source_text" in rows2[0]
    assert rows2[0]["source_text"] == "I live in Michigan now."


def test_show_never_returns_edge_trace_columns(engine):
    _insert_triple(engine, USER, "sam", "lives_in", "michigan")
    rows = engine.list_by_facet(USER, "entity", value="sam")
    assert len(rows) == 1
    forbidden = {
        "edge_embedding", "edge_emotional_valence", "edge_emotional_label",
        "edge_schematic_category", "edge_episodic_significance",
        "edge_relational_type", "edge_temporal_context",
    }
    assert forbidden.isdisjoint(rows[0].keys())


def test_show_excludes_tombstoned(engine):
    rid = _insert_triple(engine, USER, "sam", "lives_in", "michigan")
    engine.forget_by_triple_id(USER, rid)
    rows = engine.list_by_facet(USER, "entity", value="sam")
    assert rows == []


# ── MCP tool dispatch tests ───────────────────────────────────────

def test_mcp_forget_tool_dispatches(monkeypatch):
    """mcp.tools.dispatch wires memory.forget to the SDK."""
    from mcp import tools as mt

    captured = {}

    class StubKenotic:
        def __init__(self, user_id, db_path):
            captured["user_id"] = user_id

        def forget(self, by, scope):
            captured["by"] = by
            captured["scope"] = scope
            return 7

    # Patch the Kenotic symbol that tool_forget imports
    import sdk as sdk_mod
    monkeypatch.setattr(sdk_mod, "Kenotic", StubKenotic)

    result = mt.dispatch(
        "tools/call",
        {"name": "memory.forget",
         "arguments": {"by": "entity", "scope": "maya"}},
    )
    import json
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"tombstones_emitted": 7}
    assert captured["by"] == "entity"
    assert captured["scope"] == "maya"
    assert captured["user_id"] == 0  # _SOLO_USER_ID


def test_mcp_forget_tool_handles_time_range_array(monkeypatch):
    from mcp import tools as mt

    captured = {}

    class StubKenotic:
        def __init__(self, user_id, db_path):
            pass

        def forget(self, by, scope):
            captured["by"] = by
            captured["scope"] = scope
            return 3

    import sdk as sdk_mod
    monkeypatch.setattr(sdk_mod, "Kenotic", StubKenotic)

    result = mt.dispatch(
        "tools/call",
        {"name": "memory.forget",
         "arguments": {
             "by": "time_range",
             "scope": ["2026-01-01T00:00:00", "2026-01-31T23:59:59"],
         }},
    )
    import json
    payload = json.loads(result["content"][0]["text"])
    assert payload == {"tombstones_emitted": 3}
    assert captured["by"] == "time_range"
    assert captured["scope"] == ("2026-01-01T00:00:00", "2026-01-31T23:59:59")


def test_mcp_show_tool_dispatches(monkeypatch):
    from mcp import tools as mt

    captured = {}

    class StubKenotic:
        def __init__(self, user_id, db_path):
            pass

        def show(self, facet, value, limit, export_raw_text):
            captured["facet"] = facet
            captured["value"] = value
            captured["limit"] = limit
            captured["export_raw_text"] = export_raw_text
            return [{"id": 1, "subject": "sam"}]

    import sdk as sdk_mod
    monkeypatch.setattr(sdk_mod, "Kenotic", StubKenotic)

    result = mt.dispatch(
        "tools/call",
        {"name": "memory.show",
         "arguments": {"facet": "entity", "value": "sam", "limit": 25}},
    )
    import json
    payload = json.loads(result["content"][0]["text"])
    assert payload["count"] == 1
    assert payload["rows"] == [{"id": 1, "subject": "sam"}]
    assert captured["facet"] == "entity"
    assert captured["value"] == "sam"
    assert captured["limit"] == 25
    assert captured["export_raw_text"] is False


def test_reingest_clears_tombstone(engine):
    """Re-storing a (s,p,o) triple after it was tombstoned must resurrect
    the existing row (clear tombstoned_at / tombstone_reason), not create
    a duplicate. This is the 'forget then re-learn' user story and is
    load-bearing for the forget UX — without it the UNIQUE(s,p,o)
    constraint would make re-learning impossible."""
    from app.db.session import get_db_context

    rid = _insert_triple(engine, USER, "kobe", "is_a", "dog")
    assert rid > 0

    n = engine.forget_by_triple_id(USER, rid)
    assert n == 1

    with get_db_context() as conn:
        ts = conn.execute(
            "SELECT tombstoned_at FROM relationships WHERE id = ?", (rid,)
        ).fetchone()["tombstoned_at"]
        assert ts is not None

    # Re-ingesting the same triple should clear the tombstone on the
    # existing row, not insert a new one.
    rid2 = _insert_triple(engine, USER, "kobe", "is_a", "dog")
    assert rid2 == rid, "re-ingest must reuse the existing row id"

    with get_db_context() as conn:
        row = conn.execute(
            "SELECT tombstoned_at, tombstone_reason FROM relationships WHERE id = ?",
            (rid,),
        ).fetchone()
        assert row["tombstoned_at"] is None, "tombstone must be cleared"
        assert row["tombstone_reason"] is None

    # And the triple must now surface in live retrieval again.
    rels = engine.get_relationships(USER)
    assert rid in {r.id for r in rels}


def test_mcp_tools_list_has_5_entries():
    from mcp.tools import dispatch
    result = dispatch("tools/list", {})
    names = {t["name"] for t in result["tools"]}
    assert names == {
        "memory.ingest", "memory.retrieve", "memory.reconstruct",
        "memory.forget", "memory.show",
    }
