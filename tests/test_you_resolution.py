# -*- coding: utf-8 -*-
"""
test_you_resolution.py — Second-person pronoun resolution regression suite.

Tests the fix in MemoryEngine._resolve() where second-person pronouns
(you, your, yourself, you're) in the model_comprehension pass are resolved
to the speaker name when speaker is known.

Laws: RESEARCH FIRST | STRUCTURAL | NO SCORES | NO RANGES | NO PATCHWORK | ADDITIVE

TC-01  User-text pass isolation          — "you" stays "you" in user pass
TC-02  Model-comprehension resolution    — "you" -> speaker in model_comprehension
TC-03  Pronoun variants                  — your / yourself also resolve
TC-04  No-speaker guard                  — no speaker -> no resolution, no crash
TC-05  End-to-end Banff integration      — 3-turn Banff scenario, 3 retrieval Qs

Run:
    py -3.10 -m pytest tests/test_you_resolution.py -v
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(monkeypatch):
    """Isolated SQLite DB per test. Patches settings.sqlite_path so every
    get_db_context() call in the engine uses this file."""
    tmpdir = tempfile.mkdtemp(prefix="kenotic_you_res_")
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
    The critical path under test is _resolve() inside ingest_text(),
    and get_relationships() for read-back.

    Embeddings in _prepare_row fail-open (try/except) so no stub needed.
    """
    from app.engines import memory as memmod

    # Stub embed_text to avoid loading MiniLM during tests
    try:
        from app.vector import embedder as _embedder_mod
        monkeypatch.setattr(_embedder_mod, "embed_text", lambda text: __import__('numpy').zeros(384, dtype=__import__('numpy').float32))
    except Exception:
        pass

    return memmod.MemoryEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SECOND_PERSON = {"you", "your", "yourself", "you're"}


def all_subjects(engine, user_id: int) -> List[str]:
    rels = engine.get_relationships(user_id)
    return [r.subject.lower() for r in rels]


def all_subjects_raw(engine, user_id: int) -> List[str]:
    """Return subjects in original case for exact comparison."""
    rels = engine.get_relationships(user_id)
    return [r.subject for r in rels]


def triples_with_object_containing(engine, user_id: int, fragment: str):
    rels = engine.get_relationships(user_id)
    return [r for r in rels if fragment.lower() in r.object.lower()]


# ─────────────────────────────────────────────────────────────────────────────
# TC-01 — User-text pass isolation
#
# Ingest text="you are great" with speaker="Sam", tag=None (default user pass).
# The user-text pass MUST NOT resolve "you" -> "Sam".
# Resolution is scoped exclusively to model_comprehension.
# ─────────────────────────────────────────────────────────────────────────────

def test_tc01_user_pass_does_not_resolve_you(engine, monkeypatch):
    """'you' in the user-text pass must stay 'you', not resolve to speaker."""
    USER_ID = 101

    # Patch extract() to return a controlled triple so the test is deterministic
    # regardless of T5 model availability.
    monkeypatch.setattr(
        engine, "extract",
        lambda text: [("you", "are", "great", False)] if "you" in text.lower() else [],
    )
    monkeypatch.setattr(engine, "clean", lambda text: text)

    engine.ingest_text(
        user_id=USER_ID,
        text="you are great",
        speaker="Sam",
        source_tag=None,        # user pass — no model_comprehension
        model_response=None,
    )

    subjects = all_subjects(engine, USER_ID)

    # "you" must be present — it was NOT resolved in user pass
    assert "you" in subjects, (
        f"TC-01 FAIL: expected 'you' to be stored as-is in user pass, "
        f"got subjects={subjects}"
    )
    # "sam" must NOT be present — no resolution happened
    assert "sam" not in subjects, (
        f"TC-01 FAIL: 'Sam' should not appear from user-text pass resolution, "
        f"got subjects={subjects}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TC-02 — Model-comprehension resolution: basic
#
# Ingest text="I went to Banff", model_response="You visited Banff last weekend"
# with speaker="Sam".
# Expect: at least one triple with subject="Sam" AND object containing "Banff".
# Expect: NO triple with subject="you".
# ─────────────────────────────────────────────────────────────────────────────

def test_tc02_model_comprehension_resolves_you(engine, monkeypatch):
    """'You' in model_response must resolve to speaker name."""
    USER_ID = 102

    def _fake_extract(text):
        t = text.lower()
        if "banff last weekend" in t or "visited banff" in t:
            return [("you", "visited", "Banff", False)]
        if "went to banff" in t or "i went" in t:
            return [("i", "went_to", "Banff", False)]
        return []

    monkeypatch.setattr(engine, "extract", _fake_extract)
    monkeypatch.setattr(engine, "clean", lambda text: text)

    engine.ingest_text(
        user_id=USER_ID,
        text="I went to Banff",
        model_response="You visited Banff last weekend",
        speaker="Sam",
    )

    rels = engine.get_relationships(USER_ID)
    subjects_lower = [r.subject.lower() for r in rels]

    # At least one triple should have subject="Sam" with object containing Banff
    banff_triples_sam = [
        r for r in rels
        if r.subject.lower() == "sam" and "banff" in r.object.lower()
    ]
    assert banff_triples_sam, (
        f"TC-02 FAIL: expected a triple (Sam, *, *Banff*) from model_response, "
        f"got triples={[(r.subject, r.predicate, r.object) for r in rels]}"
    )

    # No triple should have subject="you"
    assert "you" not in subjects_lower, (
        f"TC-02 FAIL: found unresolved 'you' subject in stored triples, "
        f"got subjects={subjects_lower}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TC-03 — Pronoun variants (your, yourself)
#
# model_response="Your salary increased. You changed yourself"
# All second-person pronouns must resolve to "Sam".
# ─────────────────────────────────────────────────────────────────────────────

def test_tc03_pronoun_variants_resolve(engine, monkeypatch):
    """'your' and 'yourself' in model_response must also resolve to speaker."""
    USER_ID = 103

    # Keyed on distinct substrings so user-text and model_response produce
    # separate triple sets (avoids the user-text pass producing second-person
    # subjects and polluting the assertion).
    def _fake_extract(text):
        t = text.lower()
        # model_response marker
        if "your salary increased" in t:
            return [("your", "increased", "salary", False), ("you", "changed", "yourself", False)]
        # user-text marker — no second-person pronouns
        if "finances improved" in t:
            return [("i", "earned_more", "this_year", False)]
        return []

    monkeypatch.setattr(engine, "extract", _fake_extract)
    monkeypatch.setattr(engine, "clean", lambda text: text)

    engine.ingest_text(
        user_id=USER_ID,
        text="finances improved",  # user-text: distinct tokens, no 2nd-person
        model_response="Your salary increased. You changed yourself",
        speaker="Sam",
    )

    rels = engine.get_relationships(USER_ID)
    subjects_lower = {r.subject.lower() for r in rels}

    # None of the second-person pronoun forms should appear as subject.
    # "yourself" in object position is fine — object is not subject-resolved.
    leaked_pronouns = SECOND_PERSON & subjects_lower
    assert not leaked_pronouns, (
        f"TC-03 FAIL: second-person pronouns leaked as subjects: "
        f"{leaked_pronouns}. All triples: "
        f"{[(r.subject, r.predicate, r.object) for r in rels]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TC-04 — No-speaker guard
#
# Ingest model_response="You went to Banff" with speaker=None.
# Must NOT crash. Must store subject="you" (not resolved, no crash).
# ─────────────────────────────────────────────────────────────────────────────

def test_tc04_no_speaker_guard(engine, monkeypatch):
    """With speaker=None, no resolution occurs; 'you' is stored as-is."""
    USER_ID = 104

    monkeypatch.setattr(
        engine, "extract",
        lambda text: [("you", "went_to", "Banff", False)] if "banff" in text.lower() else [],
    )
    monkeypatch.setattr(engine, "clean", lambda text: text)

    # Must not raise
    try:
        engine.ingest_text(
            user_id=USER_ID,
            text="placeholder",
            model_response="You went to Banff",
            speaker=None,
        )
    except Exception as exc:
        pytest.fail(f"TC-04 FAIL: ingest_text raised with speaker=None: {exc}")

    subjects_lower = all_subjects(engine, USER_ID)

    # "you" must be stored as-is (no speaker to resolve to)
    assert "you" in subjects_lower, (
        f"TC-04 FAIL: expected 'you' to be stored when speaker=None, "
        f"got subjects={subjects_lower}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TC-05 — End-to-end Banff scenario (integration)
#
# 3-turn Banff conversation. After ingestion, query the DB directly for
# subject="Sam" triples and check whether the 3 retrieval questions can be
# answered from stored data:
#   Q1: "Where did Sam go?"   -> expects object containing "Banff"
#   Q2: "Who did Sam hike with?" -> expects object containing "Arjun"
#   Q3: "What happened to Arjun?" -> expects object containing "ankle"
# ─────────────────────────────────────────────────────────────────────────────

def test_tc05_banff_end_to_end(engine, monkeypatch):
    """3-turn Banff conversation resolves 'you' -> Sam; retrieval Q answers pass."""
    USER_ID = 105

    # Synthetic triples that approximate what T5 would produce from each turn.
    TURN_EXTRACTS = {
        # Turn 1 model_response
        "You recently returned from a trip to Banff": [
            ("you", "returned_from", "Banff", False),
            ("you", "visited", "Banff Canada", False),
        ],
        # Turn 2 model_response
        "You hiked Sulphur Mountain with your friend Arjun": [
            ("you", "hiked", "Sulphur Mountain", False),
            ("you", "hiked_with", "Arjun", False),
        ],
        # Turn 3 model_response
        "Arjun twisted his ankle during your hike": [
            ("Arjun", "twisted", "ankle", False),
            ("your", "hike", "Sulphur Mountain", False),
        ],
        # User texts (minimal, we care about model_response path)
        "just got back from Banff": [("i", "returned_from", "Banff", False)],
        "hiked Sulphur Mountain with Arjun": [("i", "hiked_with", "Arjun", False)],
        "Arjun twisted his ankle on the way down": [("Arjun", "twisted", "ankle", False)],
    }

    def _fake_extract(text):
        text_stripped = text.strip()
        for key, triples in TURN_EXTRACTS.items():
            if key.lower() in text_stripped.lower() or text_stripped.lower() in key.lower():
                return triples
        # Fallback: no triples if we don't recognise the text
        return []

    monkeypatch.setattr(engine, "extract", _fake_extract)
    monkeypatch.setattr(engine, "clean", lambda text: text)

    TURNS: List[Tuple[str, str]] = [
        (
            "just got back from Banff",
            "You recently returned from a trip to Banff, Canada",
        ),
        (
            "hiked Sulphur Mountain with Arjun",
            "You hiked Sulphur Mountain with your friend Arjun",
        ),
        (
            "Arjun twisted his ankle on the way down",
            "Arjun twisted his ankle during your hike",
        ),
    ]

    for user_text, model_resp in TURNS:
        engine.ingest_text(
            user_id=USER_ID,
            text=user_text,
            model_response=model_resp,
            speaker="Sam",
        )

    rels = engine.get_relationships(USER_ID)
    all_triples = [(r.subject, r.predicate, r.object) for r in rels]

    # ── Retrieval question checks (structural: does the DB contain answers?) ──

    # Q1: "Where did Sam go?" -> any triple subject=Sam, object contains "Banff"
    q1_hit = any(
        r.subject.lower() == "sam" and "banff" in r.object.lower()
        for r in rels
    )

    # Q2: "Who did Sam hike with?" -> any triple subject=Sam, object contains "Arjun"
    q2_hit = any(
        r.subject.lower() == "sam" and "arjun" in r.object.lower()
        for r in rels
    )

    # Q3: "What happened to Arjun?" -> any triple subject=Arjun, object contains "ankle"
    q3_hit = any(
        r.subject.lower() == "arjun" and "ankle" in r.object.lower()
        for r in rels
    )

    passed = sum([q1_hit, q2_hit, q3_hit])

    # Diagnostics always printed for the report
    print(f"\n[TC-05] Stored triples for user {USER_ID}:")
    for s, p, o in all_triples:
        print(f"  ({s!r}, {p!r}, {o!r})")
    print(f"[TC-05] Q1 (Where did Sam go? -> Banff): {'PASS' if q1_hit else 'FAIL'}")
    print(f"[TC-05] Q2 (Who did Sam hike with? -> Arjun): {'PASS' if q2_hit else 'FAIL'}")
    print(f"[TC-05] Q3 (What happened to Arjun? -> ankle): {'PASS' if q3_hit else 'FAIL'}")
    print(f"[TC-05] Score: {passed}/3 retrieval questions answered")

    # Guard: no second-person subject should survive in stored triples
    leaked = [r for r in rels if r.subject.lower() in SECOND_PERSON]
    if leaked:
        print(f"[TC-05] WARNING: unresolved second-person subjects still present: "
              f"{[(r.subject, r.predicate, r.object) for r in leaked]}")

    # The fix specifically addresses the model_comprehension resolution slice.
    # Before the fix: Banff was 2/8 overall; resolution was missing entirely.
    # This test validates the model_comprehension slice — all 3 Qs should pass.
    assert passed == 3, (
        f"TC-05 FAIL: {passed}/3 retrieval questions answered. "
        f"Expected 3/3 after the _resolve() fix. "
        f"Stored triples: {all_triples}"
    )
    assert not leaked, (
        f"TC-05 FAIL: {len(leaked)} second-person subject(s) were not resolved: "
        f"{[(r.subject, r.predicate, r.object) for r in leaked]}"
    )
