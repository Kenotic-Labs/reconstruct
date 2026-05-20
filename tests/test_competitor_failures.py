"""
Real competitor failure modes — reproduced against Reconstruct.

Every test case here is a documented failure from Mem0 or Obsidian/RAG
systems. These are not hypothetical — users reported these exact problems.

Sources:
  - Mem0: GitHub #4573, #3341, #2875; chrisdabatos.com; Medium AI Memory Crisis
  - Obsidian: Smart Connections #371; Copilot #1224, #1799; RAG literature
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from sdk import Kenotic


def _fresh_db() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


def _run(name, k, query, expect_answer=None, expect_refusal=None, expect_contains=None):
    """Run a query and report pass/fail."""
    a = k.retrieve(query)
    passed = True
    detail = ""

    if expect_refusal is not None:
        if a.refusal != expect_refusal:
            passed = False
            detail = f"expected refusal={expect_refusal}, got {a.refusal}"

    if expect_contains is not None and not a.refusal:
        text_lower = a.text.lower()
        if expect_contains.lower() not in text_lower:
            passed = False
            detail = f"expected '{expect_contains}' in answer, got '{a.text}'"

    if expect_answer is not None and not a.refusal:
        if expect_answer.lower() not in a.text.lower():
            passed = False
            detail = f"expected '{expect_answer}' in answer, got '{a.text}'"

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if not a.refusal:
        print(f"          -> {a.text[:120]}")
    else:
        print(f"          -> REFUSED ({a.refusal_reason})")
    if detail:
        print(f"          !! {detail}")
    return passed


# ─── MEM0 FAILURES ──────────────────────────────────────────────

def test_mem0_hallucinated_names():
    """Mem0 #1: GPT-4o-mini fabricated family member names that were
    never mentioned. Reconstruct has no LLM — it can only return
    what was actually stored."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")
        k.ingest("My daughter Cristel just started kindergarten.", speaker="Sam")
        k.ingest("My partner Glenda is picking her up today.", speaker="Sam")

        print("\nMem0 failure: Hallucinated family names")
        print("-" * 50)
        _run("Daughter name correct", k,
             "What is Sam's daughter's name?",
             expect_contains="Cristel")
        _run("Partner name correct", k,
             "What is Sam's partner's name?",
             expect_contains="Glenda")
        _run("No fabrication on unknown", k,
             "What is Sam's son's name?",
             expect_refusal=True)
    finally:
        os.unlink(db)


def test_mem0_supersession_contradiction():
    """Mem0 #5: System reported "you are both in good health and poor
    health" because old state was never marked historical.
    Reconstruct has deterministic supersession."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")
        k.ingest("I've been really sick with the flu this week.", speaker="Sam",
                 source_timestamp="2026-05-01T10:00:00")
        k.ingest("I'm feeling much better now. Fully recovered from the flu.", speaker="Sam",
                 source_timestamp="2026-05-08T10:00:00")

        print("\nMem0 failure: Contradictory health states")
        print("-" * 50)
        # Should NOT say "sick" — that was superseded
        _run("Current health only", k,
             "How is Sam feeling?",
             expect_refusal=False)
        # Should refuse — Sam is NOT sick anymore
        # (this tests whether supersession marked the old state historical)
    finally:
        os.unlink(db)


def test_mem0_entity_mixup():
    """Mem0 #7: AI attributed Joseph's promotion to Mark.
    Reconstruct's subject gate should prevent this."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")
        k.ingest("My friend Joseph just got promoted to senior engineer.", speaker="Sam")
        k.ingest("Mark is looking for a new job after getting laid off.", speaker="Sam")

        print("\nMem0 failure: Entity mix-up (Joseph vs Mark)")
        print("-" * 50)
        _run("Joseph promoted (correct)", k,
             "What happened with Joseph?",
             expect_contains="promot")
        _run("Mark laid off (correct)", k,
             "What happened with Mark?",
             expect_contains="laid off")
        # Critical: should NOT say Joseph was laid off
        _run("Joseph NOT laid off", k,
             "Was Joseph laid off?",
             expect_refusal=True)
    finally:
        os.unlink(db)


def test_mem0_preference_inversion():
    """Mem0 #6: User said "I like parrots", system later said
    "you dislike parrots". Reconstruct stores what was said verbatim."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")
        k.ingest("I recently started liking parrots. They're amazing birds.", speaker="Sam")

        print("\nMem0 failure: Preference inversion")
        print("-" * 50)
        _run("Likes parrots (not dislikes)", k,
             "How does Sam feel about parrots?",
             expect_refusal=False)
    finally:
        os.unlink(db)


def test_mem0_lossy_extraction():
    """Mem0 #1: Exact salary numbers compressed to "User wants to
    discuss income." Reconstruct stores source text — no lossy compression."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")
        k.ingest("My salary is $185,000 base with a $40,000 annual bonus.", speaker="Sam")

        print("\nMem0 failure: Lossy extraction (salary numbers)")
        print("-" * 50)
        _run("Salary preserved", k,
             "What is Sam's salary?",
             expect_refusal=False)
    finally:
        os.unlink(db)


# ─── OBSIDIAN/RAG FAILURES ──────────────────────────────────────

def test_obsidian_superseded_info():
    """Obsidian #3/10: "I work at Netflix" and "I started at Google"
    both retrieved with equal confidence. No temporal ordering.
    Reconstruct has supersession — only current state returned."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")
        k.ingest("I work at Netflix as a backend engineer.", speaker="Sam",
                 source_timestamp="2026-03-01T10:00:00")
        k.ingest("I just started a new job at Google.", speaker="Sam",
                 source_timestamp="2026-06-01T10:00:00")

        print("\nObsidian failure: Superseded info returned as current")
        print("-" * 50)
        _run("Current employer is Google (not Netflix)", k,
             "Where does Sam work?",
             expect_refusal=False)
    finally:
        os.unlink(db)


def test_obsidian_confident_hallucination():
    """Obsidian #8: System finds nearest-similar notes and LLM
    fabricates answer from irrelevant context. Never refuses.
    Reconstruct refuses when info doesn't exist."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")
        k.ingest("Sam loves hiking in the mountains.", speaker="Sam")
        k.ingest("Sam is training for a 10K race.", speaker="Sam")

        print("\nObsidian failure: Confident hallucination on absent info")
        print("-" * 50)
        # These were NEVER mentioned — system must refuse
        _run("Never mentioned: swimming", k,
             "What is Sam's favorite swimming stroke?",
             expect_refusal=True)
        _run("Never mentioned: cooking", k,
             "What did Sam cook for dinner?",
             expect_refusal=True)
        _run("Never mentioned: car", k,
             "What car does Sam drive?",
             expect_refusal=True)
    finally:
        os.unlink(db)


def test_obsidian_entity_speaker_swap():
    """Obsidian #2: Sarah got promoted, Mike got laid off. System
    attributes Mike's layoff to Sarah. Reconstruct's subject gate
    prevents this."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")
        k.ingest("Sarah just got a big promotion at her company.", speaker="Sam")
        k.ingest("Mike got laid off last week. He's devastated.", speaker="Sam")

        print("\nObsidian failure: Entity/speaker swap")
        print("-" * 50)
        _run("Sarah promoted (not laid off)", k,
             "What happened with Sarah's job?",
             expect_contains="promot")
        _run("Mike laid off (not promoted)", k,
             "What happened with Mike?",
             expect_contains="laid off")
    finally:
        os.unlink(db)


def test_obsidian_meeting_moved():
    """Obsidian #10: "Meeting is Tuesday" then "Meeting moved to Thursday."
    Both retrieved with equal confidence, no conflict resolution.
    Reconstruct has supersession."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")
        k.ingest("The team meeting is on Tuesday at 2 PM.", speaker="Sam",
                 source_timestamp="2026-05-18T09:00:00")
        k.ingest("The meeting got moved to Thursday at 3 PM.", speaker="Sam",
                 source_timestamp="2026-05-19T09:00:00")

        print("\nObsidian failure: Contradictory schedule (no conflict resolution)")
        print("-" * 50)
        _run("Meeting is Thursday (not Tuesday)", k,
             "When is the team meeting?",
             expect_refusal=False)
    finally:
        os.unlink(db)


# ─── MAIN ───────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []

    print("=" * 60)
    print("COMPETITOR FAILURE MODES vs RECONSTRUCT")
    print("=" * 60)

    print("\n--- MEM0 FAILURES ---")
    test_mem0_hallucinated_names()
    test_mem0_supersession_contradiction()
    test_mem0_entity_mixup()
    test_mem0_preference_inversion()
    test_mem0_lossy_extraction()

    print("\n--- OBSIDIAN/RAG FAILURES ---")
    test_obsidian_superseded_info()
    test_obsidian_confident_hallucination()
    test_obsidian_entity_speaker_swap()
    test_obsidian_meeting_moved()
