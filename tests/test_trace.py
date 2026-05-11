"""
test_trace.py — Contract tests for MemoryEngine.trace() and Kenotic.trace().

Run: py -3.10 tests/test_trace.py

Covers:
  TC-01  Empty trace             — unknown (subject, predicate) returns []
  TC-02  Single entry            — one active fact, never superseded
  TC-03  Two-entry supersession  — X → Y, oldest-first, X inactive
  TC-04  Three-entry chain       — A → B → C, only C active
  TC-05  Predicate isolation     — salary trace excludes role entries
  TC-06  Subject isolation       — Sam trace excludes Priya entries
  TC-07  source_tag propagation  — explicit tag appears in output
  TC-08  model_comprehension tag — model_comprehension-tagged entries visible
          alongside user entries (same subject/predicate)

Architecture notes:
  - Uses MemoryEngine.store() directly (no T5/MiniLM) with _write_edge_traces
    and _upsert_entity stubbed out as no-ops so tests are fast.
  - Supersession is driven by MemoryEngine.supersede() — the write-path method
    the TemporalEngine calls — rather than re-implementing the logic here.
  - One temp SQLite file per test; cleaned up in finally blocks.
  - No pytest required; plain assertions + PASS/FAIL summary at the bottom.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_db() -> str:
    """Create a fresh temp SQLite file with the full Kenotic schema."""
    tmpdir = tempfile.mkdtemp(prefix="kenotic_trace_")
    db_path = os.path.join(tmpdir, "memory.db")

    from app.db.models import MIGRATIONS, run_schema_upgrades
    conn = sqlite3.connect(db_path)
    conn.executescript(MIGRATIONS)
    run_schema_upgrades(conn)
    # sequence_number column added lazily by write-path; add it here so
    # all tests start with a consistent schema.
    try:
        conn.execute("ALTER TABLE relationships ADD COLUMN sequence_number INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_seq "
            "ON relationships(user_id, sequence_number)"
        )
    except Exception:
        pass
    conn.commit()
    conn.close()

    return db_path


def _make_engine(db_path: str, monkeypatch_targets: bool = True):
    """
    Return a MemoryEngine pointed at db_path with costly side-effects stubbed.

    Stubs the embedder module to avoid loading MiniLM during tests.
    Embeddings in _prepare_row fail-open (try/except) so this is safe.

    The engine itself is a fresh instance (not the module singleton) so
    tests are isolated.
    """
    from config.settings import settings
    settings.sqlite_path = db_path

    from app.engines import memory as memmod

    # Stub embed_text to avoid loading MiniLM
    import numpy as _np
    _orig_embed = None
    try:
        from app.vector import embedder as _embedder_mod
        _orig_embed = _embedder_mod.embed_text
        _embedder_mod.embed_text = lambda text: _np.zeros(384, dtype=_np.float32)
    except Exception:
        pass

    engine = memmod.MemoryEngine()

    return engine, memmod, _orig_embed, None


def _restore(memmod, orig_embed, _unused):
    if orig_embed is not None:
        try:
            from app.vector import embedder as _embedder_mod
            _embedder_mod.embed_text = orig_embed
        except Exception:
            pass


def _store(engine, user_id, subj, pred, obj,
           source_tag=None, source_timestamp=None) -> int:
    """Insert one hand-authored triple; returns rel_id."""
    return engine.store(
        user_id=user_id,
        subject=subj,
        predicate=pred,
        object=obj,
        source_text=f"{subj} {pred} {obj}",
        confidence=0.9,
        source_tag=source_tag,
        source_timestamp=source_timestamp,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test runner infrastructure
# ─────────────────────────────────────────────────────────────────────────────

_results: List[Tuple[str, bool, str]] = []  # (name, passed, detail)


def run_test(name: str, fn: Callable) -> None:
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
    except AssertionError as exc:
        detail = str(exc) or "(assertion)"
        _results.append((name, False, detail))
        print(f"  FAIL  {name}: {detail}")
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        _results.append((name, False, detail))
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TC-01  Empty trace
# ─────────────────────────────────────────────────────────────────────────────

def tc_01_empty_trace():
    db_path = _make_db()
    engine, memmod, ot, oe = _make_engine(db_path)
    try:
        result = engine.trace(user_id=1, subject="Ghost", predicate="lives_in")
        assert result == [], f"Expected [], got {result!r}"
    finally:
        _restore(memmod, ot, oe)


# ─────────────────────────────────────────────────────────────────────────────
# TC-02  Single entry, never superseded
# ─────────────────────────────────────────────────────────────────────────────

def tc_02_single_entry_never_superseded():
    db_path = _make_db()
    engine, memmod, ot, oe = _make_engine(db_path)
    try:
        uid = 2
        rid = _store(engine, uid, "Sam", "lives_in", "Michigan")
        assert rid > 0, "store() returned 0 — insert failed"

        chain = engine.trace(uid, "Sam", "lives_in")
        assert len(chain) == 1, f"Expected 1 entry, got {len(chain)}: {chain}"

        entry = chain[0]
        assert entry["id"] == rid, f"id mismatch: {entry['id']} != {rid}"
        assert entry["object"] == "Michigan", f"object: {entry['object']!r}"
        assert entry["is_active"] is True, f"is_active should be True: {entry}"
        assert entry["superseded_by"] is None, (
            f"superseded_by should be None: {entry['superseded_by']!r}"
        )
    finally:
        _restore(memmod, ot, oe)


# ─────────────────────────────────────────────────────────────────────────────
# TC-03  Two-entry supersession chain
# ─────────────────────────────────────────────────────────────────────────────

def tc_03_two_entry_chain():
    db_path = _make_db()
    engine, memmod, ot, oe = _make_engine(db_path)
    try:
        uid = 3
        rid_x = _store(engine, uid, "Sam", "salary", "120k",
                       source_timestamp="2025-01-01T10:00:00")
        rid_y = _store(engine, uid, "Sam", "salary", "155k",
                       source_timestamp="2025-06-01T10:00:00")
        assert rid_x > 0 and rid_y > 0, "store() returned 0"

        # Manually supersede X with Y (as TemporalEngine would)
        engine.supersede(rid_x, rid_y)

        chain = engine.trace(uid, "Sam", "salary")
        assert len(chain) == 2, f"Expected 2 entries, got {len(chain)}: {chain}"

        # Ordering: oldest-first
        first, second = chain
        assert first["id"] == rid_x, (
            f"First entry should be rid_x={rid_x}, got {first['id']}"
        )
        assert second["id"] == rid_y, (
            f"Second entry should be rid_y={rid_y}, got {second['id']}"
        )

        # First entry (X): inactive, superseded_by Y
        assert first["is_active"] is False, (
            f"X should be inactive (superseded): {first}"
        )
        assert first["superseded_by"] == rid_y, (
            f"X.superseded_by should be {rid_y}, got {first['superseded_by']}"
        )

        # Second entry (Y): active, not superseded
        assert second["is_active"] is True, (
            f"Y should be active: {second}"
        )
        assert second["superseded_by"] is None, (
            f"Y.superseded_by should be None: {second['superseded_by']!r}"
        )
    finally:
        _restore(memmod, ot, oe)


# ─────────────────────────────────────────────────────────────────────────────
# TC-04  Three-entry chain A → B → C
# ─────────────────────────────────────────────────────────────────────────────

def tc_04_three_entry_chain():
    db_path = _make_db()
    engine, memmod, ot, oe = _make_engine(db_path)
    try:
        uid = 4
        rid_a = _store(engine, uid, "Sam", "role", "intern",
                       source_timestamp="2023-01-01T00:00:00")
        rid_b = _store(engine, uid, "Sam", "role", "engineer",
                       source_timestamp="2024-01-01T00:00:00")
        rid_c = _store(engine, uid, "Sam", "role", "founder",
                       source_timestamp="2025-01-01T00:00:00")
        assert all(r > 0 for r in (rid_a, rid_b, rid_c)), "store() returned 0"

        engine.supersede(rid_a, rid_b)
        engine.supersede(rid_b, rid_c)

        chain = engine.trace(uid, "Sam", "role")
        assert len(chain) == 3, f"Expected 3 entries, got {len(chain)}: {chain}"

        by_id = {e["id"]: e for e in chain}
        entry_a = by_id[rid_a]
        entry_b = by_id[rid_b]
        entry_c = by_id[rid_c]

        # Order: oldest first
        ids_in_order = [e["id"] for e in chain]
        assert ids_in_order == [rid_a, rid_b, rid_c], (
            f"Order wrong: {ids_in_order}"
        )

        # A: inactive, superseded_by B
        assert entry_a["is_active"] is False
        assert entry_a["superseded_by"] == rid_b, (
            f"A.superseded_by should be {rid_b}: {entry_a['superseded_by']}"
        )

        # B: inactive, superseded_by C
        assert entry_b["is_active"] is False
        assert entry_b["superseded_by"] == rid_c, (
            f"B.superseded_by should be {rid_c}: {entry_b['superseded_by']}"
        )

        # C: active, not superseded
        assert entry_c["is_active"] is True
        assert entry_c["superseded_by"] is None
    finally:
        _restore(memmod, ot, oe)


# ─────────────────────────────────────────────────────────────────────────────
# TC-05  Different predicate isolation
# ─────────────────────────────────────────────────────────────────────────────

def tc_05_predicate_isolation():
    db_path = _make_db()
    engine, memmod, ot, oe = _make_engine(db_path)
    try:
        uid = 5
        rid_salary = _store(engine, uid, "Sam", "salary", "90k")
        rid_role   = _store(engine, uid, "Sam", "role",   "engineer")
        assert rid_salary > 0 and rid_role > 0

        chain = engine.trace(uid, "Sam", "salary")
        ids = {e["id"] for e in chain}

        assert rid_salary in ids, "salary entry missing from salary trace"
        assert rid_role not in ids, (
            f"role entry leaked into salary trace: {chain}"
        )
    finally:
        _restore(memmod, ot, oe)


# ─────────────────────────────────────────────────────────────────────────────
# TC-06  Different subject isolation
# ─────────────────────────────────────────────────────────────────────────────

def tc_06_subject_isolation():
    db_path = _make_db()
    engine, memmod, ot, oe = _make_engine(db_path)
    try:
        uid = 6
        rid_sam   = _store(engine, uid, "Sam",   "works_at", "Google")
        rid_priya = _store(engine, uid, "Priya", "works_at", "Amazon")
        assert rid_sam > 0 and rid_priya > 0

        chain = engine.trace(uid, "Sam", "works_at")
        ids = {e["id"] for e in chain}

        assert rid_sam in ids, "Sam entry missing from Sam trace"
        assert rid_priya not in ids, (
            f"Priya entry leaked into Sam trace: {chain}"
        )
    finally:
        _restore(memmod, ot, oe)


# ─────────────────────────────────────────────────────────────────────────────
# TC-07  source_tag propagation
# ─────────────────────────────────────────────────────────────────────────────

def tc_07_source_tag_propagation():
    db_path = _make_db()
    engine, memmod, ot, oe = _make_engine(db_path)
    try:
        uid = 7
        tag = "session_42"
        rid = _store(engine, uid, "Sam", "city", "Ann Arbor", source_tag=tag)
        assert rid > 0

        chain = engine.trace(uid, "Sam", "city")
        assert len(chain) == 1, f"Expected 1 entry, got {len(chain)}: {chain}"

        entry = chain[0]
        assert entry["source_tag"] == tag, (
            f"source_tag should be {tag!r}, got {entry['source_tag']!r}"
        )
    finally:
        _restore(memmod, ot, oe)


# ─────────────────────────────────────────────────────────────────────────────
# TC-08  model_comprehension tag
#
# The model_response path in ingest_text() tags triples with
# source_tag="model_comprehension".  This test bypasses T5 by inserting
# a triple directly via store() with source_tag="model_comprehension"
# (same tag the pipeline writes) alongside a user triple on the same
# (subject, predicate).  Both must appear in trace().
# ─────────────────────────────────────────────────────────────────────────────

def tc_08_model_comprehension_tag():
    db_path = _make_db()
    engine, memmod, ot, oe = _make_engine(db_path)
    try:
        uid = 8
        # User asserts a fact
        rid_user = _store(engine, uid, "Sam", "project", "DTCM",
                          source_tag="session_5")
        # Model response extracted the same subject/predicate but with
        # enriched object; stored with the fixed model_comprehension tag
        rid_model = _store(engine, uid, "Sam", "project",
                           "DTCM — graph traversal system",
                           source_tag="model_comprehension")
        assert rid_user > 0 and rid_model > 0

        chain = engine.trace(uid, "Sam", "project")
        assert len(chain) == 2, (
            f"Expected 2 entries (user + model), got {len(chain)}: {chain}"
        )

        tags = {e["source_tag"] for e in chain}
        assert "session_5" in tags, (
            f"user entry (session_5) missing from trace: {tags}"
        )
        assert "model_comprehension" in tags, (
            f"model_comprehension entry missing from trace: {tags}"
        )

        # Verify the model_comprehension entry has the correct attributes
        mc_entry = next(e for e in chain if e["source_tag"] == "model_comprehension")
        assert mc_entry["id"] == rid_model
        assert "DTCM" in mc_entry["object"]
    finally:
        _restore(memmod, ot, oe)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  test_trace.py — MemoryEngine.trace() contract suite")
    print("=" * 64)

    run_test("TC-01  Empty trace",                        tc_01_empty_trace)
    run_test("TC-02  Single entry, never superseded",     tc_02_single_entry_never_superseded)
    run_test("TC-03  Two-entry supersession chain",       tc_03_two_entry_chain)
    run_test("TC-04  Three-entry chain A->B->C",           tc_04_three_entry_chain)
    run_test("TC-05  Predicate isolation",                tc_05_predicate_isolation)
    run_test("TC-06  Subject isolation",                  tc_06_subject_isolation)
    run_test("TC-07  source_tag propagation",             tc_07_source_tag_propagation)
    run_test("TC-08  model_comprehension tag",            tc_08_model_comprehension_tag)

    passed = sum(1 for _, ok, _ in _results if ok)
    total  = len(_results)
    failed = [(n, d) for n, ok, d in _results if not ok]

    print("\n" + "=" * 64)
    print(f"  RESULT: {passed}/{total} passed")
    if failed:
        print(f"  FAILURES ({len(failed)}):")
        for name, detail in failed:
            print(f"    - {name}")
            for line in detail.splitlines():
                print(f"        {line}")
    print("=" * 64 + "\n")

    sys.exit(0 if not failed else 1)
