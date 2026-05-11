# -*- coding: utf-8 -*-
"""
test_t5_subject_validation.py — Subject surface validation + deduplication suite.

Tests the two additions shipped to MemoryEngine.ingest_text() in app/engines/memory.py:
  1. Deduplication: `triples = list(dict.fromkeys(triples))` after extract()
  2. Subject surface validation: resolved subject must be grounded in source tokens
     (or exempt as "user" / "listener" / first-person resolution / partial name match)

Laws: RESEARCH FIRST | STRUCTURAL | SCALABLE | NO WHACK-A-MOLE | NO SCORES |
      NO RANGES | NO PATCHWORK | ADDITIVE

TC-01  Named subject present in source — accepted
TC-02  Hallucinated subject rejected
TC-03  First-person resolved — accepted via canonical user exemption
TC-04  Speaker name accepted via first-person exemption
TC-05  Deduplication — 13 identical triples become 1
TC-06  Regression — you-resolution suite still passes 5/5
TC-07  Partial name match — accepted

Run:
    py -3.10 -m pytest tests/test_t5_subject_validation.py -v
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tmp_db(monkeypatch):
    """Isolated SQLite DB per test. Patches settings.sqlite_path so every
    get_db_context() call in the engine uses this file."""
    tmpdir = tempfile.mkdtemp(prefix="kenotic_subj_val_")
    db_path = os.path.join(tmpdir, "memory.db")

    from config.settings import settings
    monkeypatch.setattr(settings, "sqlite_path", db_path)

    from app.db.models import MIGRATIONS, run_schema_upgrades
    conn = sqlite3.connect(db_path)
    conn.executescript(MIGRATIONS)
    run_schema_upgrades(conn)
    # sequence_number is added lazily; pre-add to avoid ALTER noise.
    try:
        conn.execute("ALTER TABLE relationships ADD COLUMN sequence_number INTEGER")
    except Exception:
        pass
    conn.commit()
    conn.close()

    yield db_path


@pytest.fixture
def engine(tmp_db, monkeypatch):
    """MemoryEngine with embedding stubbed out (expensive, not under test).
    Tests inject triples via monkeypatch on extract().

    Embeddings in _prepare_row fail-open so just stub the embedder module.
    """
    from app.engines import memory as memmod

    try:
        from app.vector import embedder as _embedder_mod
        monkeypatch.setattr(_embedder_mod, "embed_text", lambda text: __import__('numpy').zeros(384, dtype=__import__('numpy').float32))
    except Exception:
        pass

    return memmod.MemoryEngine()


# =============================================================================
# Helpers
# =============================================================================

def stored_triples(engine, user_id: int):
    """Return all live (subject, predicate, object) tuples for a user."""
    rels = engine.get_relationships(user_id)
    return [(r.subject, r.predicate, r.object) for r in rels]


# =============================================================================
# TC-01 — Named subject present in source — ACCEPTED
#
# Source: "Arjun went to Toronto". T5 extracts ("Arjun", "went_to", "Toronto").
# "arjun" is in source tokens → triple accepted.
# =============================================================================

def test_tc01_named_subject_present_accepted(engine, monkeypatch):
    """Named subject that appears verbatim in source must be stored."""
    USER_ID = 201

    monkeypatch.setattr(engine, "clean", lambda text: text)
    monkeypatch.setattr(
        engine, "extract",
        lambda text: [("Arjun", "went_to", "Toronto", False)] if "arjun" in text.lower() else [],
    )

    engine.ingest_text(
        user_id=USER_ID,
        text="Arjun went to Toronto",
        speaker=None,
    )

    triples = stored_triples(engine, USER_ID)
    assert any(s.lower() == "arjun" for s, p, o in triples), (
        f"TC-01 FAIL: expected triple with subject='Arjun' to be stored, "
        f"got triples={triples}"
    )


# =============================================================================
# TC-02 — Hallucinated subject rejected
#
# Source: "Arjun twisted his ankle". T5 hallucinates subject="cut trip short"
# (a multi-word hallucination). None of {cut, trip, short} appear in the
# source tokens → triple must be rejected.
# =============================================================================

def test_tc02_hallucinated_subject_rejected(engine, monkeypatch):
    """Subject not grounded in source tokens must be rejected."""
    USER_ID = 202

    monkeypatch.setattr(engine, "clean", lambda text: text)
    monkeypatch.setattr(
        engine, "extract",
        lambda text: [("cut trip short", "ended", "journey", False)] if "arjun" in text.lower() else [],
    )

    engine.ingest_text(
        user_id=USER_ID,
        text="Arjun twisted his ankle",
        speaker=None,
    )

    triples = stored_triples(engine, USER_ID)
    # "cut", "trip", "short" are all absent from "Arjun twisted his ankle"
    bad = [t for t in triples if "cut trip short" in t[0].lower()]
    assert not bad, (
        f"TC-02 FAIL: hallucinated subject 'cut trip short' was stored — "
        f"subject validation did not fire. Stored: {triples}"
    )
    # Also assert nothing was stored at all (the only triple was the bad one)
    assert not triples, (
        f"TC-02 FAIL: expected 0 triples stored, got {triples}"
    )


# =============================================================================
# TC-03 — First-person resolved — ACCEPTED via canonical "user" exemption
#
# Source: "I went to Banff". speaker=None. T5 extracts ("user", "went_to", "Banff")
# (canonical form after _resolve() when speaker is None). "user" is exempt →
# triple must be accepted.
# =============================================================================

def test_tc03_first_person_user_exemption_accepted(engine, monkeypatch):
    """Canonical 'user' subject (resolved first-person, speaker=None) must be stored."""
    USER_ID = 203

    monkeypatch.setattr(engine, "clean", lambda text: text)
    # Simulate T5 returning ("user", ...) — the canonical resolution of "I"
    monkeypatch.setattr(
        engine, "extract",
        lambda text: [("user", "went_to", "Banff", False)] if "banff" in text.lower() else [],
    )

    engine.ingest_text(
        user_id=USER_ID,
        text="I went to Banff",
        speaker=None,
    )

    triples = stored_triples(engine, USER_ID)
    assert any(s.lower() == "user" and "banff" in o.lower() for s, p, o in triples), (
        f"TC-03 FAIL: expected triple (user, went_to, Banff) via exemption, "
        f"got triples={triples}"
    )


# =============================================================================
# TC-04 — Speaker name accepted via first-person exemption
#
# Source: "I hiked the mountain". speaker="Sam". T5 extracts ("Sam", "hiked", "mountain")
# after _resolve() turns "I" → "Sam". Source contains "i" → first-person
# exemption fires → accepted.
# =============================================================================

def test_tc04_speaker_name_accepted_via_first_person_exemption(engine, monkeypatch):
    """Speaker name resolved from 'I' must pass subject validation."""
    USER_ID = 204

    monkeypatch.setattr(engine, "clean", lambda text: text)
    # T5 returns "i" (as T5 sees it); _resolve() will convert to "Sam".
    # We simulate: T5 returns "i" so _resolve() maps it to speaker.
    monkeypatch.setattr(
        engine, "extract",
        lambda text: [("i", "hiked", "mountain", False)] if "hiked" in text.lower() else [],
    )

    engine.ingest_text(
        user_id=USER_ID,
        text="I hiked the mountain",
        speaker="Sam",
    )

    triples = stored_triples(engine, USER_ID)
    # After _resolve("i") with speaker="Sam", stored subject is "Sam"
    assert any(s.lower() == "sam" and "mountain" in o.lower() for s, p, o in triples), (
        f"TC-04 FAIL: expected triple (Sam, hiked, mountain) after first-person "
        f"resolution, got triples={triples}"
    )


# =============================================================================
# TC-05 — Deduplication: 13 identical triples become 1
#
# Patch extract() to return [("user", "went_home", "Michigan")] * 13.
# After ingest, the DB must contain exactly 1 row for this (user_id, s, p, o).
# =============================================================================

def test_tc05_deduplication_13_to_1(engine, monkeypatch):
    """13 identical T5 triples must be deduplicated to 1 stored row."""
    USER_ID = 205

    monkeypatch.setattr(engine, "clean", lambda text: text)
    monkeypatch.setattr(
        engine, "extract",
        lambda text: [("user", "went_home", "Michigan", False)] * 13,
    )

    engine.ingest_text(
        user_id=USER_ID,
        text="I went home to Michigan",
        speaker=None,
    )

    triples = stored_triples(engine, USER_ID)
    matching = [t for t in triples if t[0].lower() == "user"
                and t[1].lower() == "went_home"
                and t[2].lower() == "michigan"]
    assert len(matching) == 1, (
        f"TC-05 FAIL: expected exactly 1 stored triple for (user, went_home, Michigan), "
        f"got {len(matching)}. All stored: {triples}"
    )


# =============================================================================
# TC-06 — Regression: you-resolution suite still passes 5/5
#
# Executes tests/test_you_resolution.py as a subprocess and asserts all 5
# tests pass. Isolation: subprocess so the running engine's T5 singleton state
# does not interfere.
# =============================================================================

def test_tc06_you_resolution_regression():
    """All 5 tests in test_you_resolution.py must still pass."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_you_resolution.py",
            "-v", "--tb=short", "-q",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    output = result.stdout + result.stderr
    print(f"\n[TC-06] you-resolution subprocess output:\n{output}")

    # Detect pass count from pytest summary line (e.g. "5 passed")
    import re
    pass_match = re.search(r"(\d+) passed", output)
    fail_match = re.search(r"(\d+) failed", output)
    error_match = re.search(r"(\d+) error", output)

    passed = int(pass_match.group(1)) if pass_match else 0
    failed = int(fail_match.group(1)) if fail_match else 0
    errors = int(error_match.group(1)) if error_match else 0

    assert result.returncode == 0 and passed == 5 and failed == 0 and errors == 0, (
        f"TC-06 FAIL: you-resolution suite did not pass 5/5. "
        f"passed={passed}, failed={failed}, errors={errors}, "
        f"returncode={result.returncode}.\n"
        f"Output:\n{output}"
    )


# =============================================================================
# TC-07 — Partial name match: accepted
#
# Source: "Priya made dinner". T5 extracts subject="Priya Patel" (multi-word).
# "priya" is a token of the subject and "priya" IS in source tokens → accepted.
# =============================================================================

def test_tc07_partial_name_match_accepted(engine, monkeypatch):
    """Multi-word subject where any token appears in source must be accepted."""
    USER_ID = 207

    monkeypatch.setattr(engine, "clean", lambda text: text)
    monkeypatch.setattr(
        engine, "extract",
        lambda text: [("Priya Patel", "made", "dinner", False)] if "priya" in text.lower() else [],
    )

    engine.ingest_text(
        user_id=USER_ID,
        text="Priya made dinner",
        speaker=None,
    )

    triples = stored_triples(engine, USER_ID)
    assert any("priya" in s.lower() and "dinner" in o.lower() for s, p, o in triples), (
        f"TC-07 FAIL: expected triple (Priya Patel, made, dinner) to be accepted "
        f"via partial name match, got triples={triples}"
    )
