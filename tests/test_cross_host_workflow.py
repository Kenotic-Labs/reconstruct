"""
Cross-host workflow test — the Reconstruct product demo.

Simulates the exact scenario from the product vision:
  Day 1: User talks to GPT about career stress
  Day 2: User switches to Claude, asks about interview prep
  Day 3: User updates info (interview moved)
  Day 5: User asks "what's going on in my life?"

Tests what Mem0/Obsidian compete with — but cross-host, on-device,
no LLM in the pipeline.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from sdk import Kenotic


def _fresh_db() -> str:
    """Create a temp DB path (cleaned up by OS)."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


def test_cross_host_career():
    """Day 1-5 career scenario from the product vision."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")

        # ── Day 1: User talks to "GPT" about career stress ────
        k.ingest(
            "I have a Google interview next Tuesday at 3 PM.",
            speaker="Sam",
            source_timestamp="2026-05-18T20:00:00",
        )
        k.ingest(
            "I'm really nervous about it. I feel underprepared.",
            speaker="Sam",
            source_timestamp="2026-05-18T20:01:00",
        )
        k.ingest(
            "My mom keeps asking me about it which makes it worse.",
            speaker="Sam",
            source_timestamp="2026-05-18T20:02:00",
        )
        k.ingest(
            "I need to leave by 1:30 because the drive is long.",
            speaker="Sam",
            source_timestamp="2026-05-18T20:03:00",
        )

        # ── Day 2: User switches to "Claude" ──────────────────
        # Claude calls reconstruct — should know about the interview
        # without being told anything
        a1 = k.retrieve("What is Sam preparing for?")
        print(f"Day 2 - What is Sam preparing for?")
        print(f"  Answer: {a1.text}")
        print(f"  Refusal: {a1.refusal}")
        print()

        a2 = k.retrieve("How is Sam feeling?")
        print(f"Day 2 - How is Sam feeling?")
        print(f"  Answer: {a2.text}")
        print(f"  Refusal: {a2.refusal}")
        print()

        # ── Day 3: User updates — interview moved ─────────────
        k.ingest(
            "The interview got moved to Thursday.",
            speaker="Sam",
            source_timestamp="2026-05-20T10:00:00",
        )

        a3 = k.retrieve("When is Sam's interview?")
        print(f"Day 3 - When is Sam's interview?")
        print(f"  Answer: {a3.text}")
        print(f"  Refusal: {a3.refusal}")
        print()

        # ── Day 5: "What's going on in my life?" ──────────────
        s = k.reconstruct("What's going on in Sam's life?")
        print(f"Day 5 - What's going on in Sam's life?")
        print(f"  Situation: {s.narrative}")
        print()

        # ── Cat 5: Wrong speaker — should REFUSE ──────────────
        a5 = k.retrieve("What is Caroline preparing for?")
        print(f"Cat 5 - What is Caroline preparing for?")
        print(f"  Answer: {a5.text}")
        print(f"  Refusal: {a5.refusal} (should be True)")
        print()

        # Summary
        print("=" * 60)
        results = {
            "career_question": not a1.refusal,
            "emotional_question": not a2.refusal,
            "temporal_update": not a3.refusal,
            "situation_reconstruction": bool(s.narrative and len(s.narrative) > 10),
            "adversarial_refusal": a5.refusal,
        }
        passed = sum(results.values())
        total = len(results)
        print(f"Results: {passed}/{total}")
        for name, ok in results.items():
            print(f"  {'PASS' if ok else 'FAIL'}: {name}")

    finally:
        os.unlink(db)


def test_knowledge_accumulation():
    """Obsidian-style: knowledge builds over weeks, connections form."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")

        # Week 1: health
        k.ingest("I started training for a marathon.", speaker="Sam",
                 source_timestamp="2026-05-01T09:00:00")
        k.ingest("My knee has been hurting after long runs.", speaker="Sam",
                 source_timestamp="2026-05-03T18:00:00")

        # Week 2: work
        k.ingest("I got promoted to senior engineer at Stripe.", speaker="Sam",
                 source_timestamp="2026-05-08T11:00:00")
        k.ingest("The new role starts in June.", speaker="Sam",
                 source_timestamp="2026-05-08T11:01:00")

        # Week 3: relationship
        k.ingest("Sarah and I are planning a trip to Japan in August.", speaker="Sam",
                 source_timestamp="2026-05-15T20:00:00")
        k.ingest("We've been together for three years now.", speaker="Sam",
                 source_timestamp="2026-05-15T20:01:00")

        # Week 4: health update (supersession)
        k.ingest("I saw a doctor about my knee. She said it's runner's knee and I should rest.",
                 speaker="Sam", source_timestamp="2026-05-20T14:00:00")

        # Now query across all of it
        print("Knowledge accumulation test")
        print("-" * 40)

        for q in [
            "What sport is Sam training for?",
            "Where does Sam work?",
            "Who is Sam dating?",
            "What's wrong with Sam's knee?",
            "When is Sam's trip?",
            "What did the doctor say about Sam's knee?",
        ]:
            a = k.retrieve(q)
            status = "REFUSE" if a.refusal else "ANSWER"
            print(f"  [{status}] {q}")
            if not a.refusal:
                print(f"          -> {a.text}")

    finally:
        os.unlink(db)


def test_mem0_comparison():
    """Mem0's core promise: "the AI remembers your preferences."
    Reconstruct should handle this AND more."""
    db = _fresh_db()
    try:
        k = Kenotic(user_id=0, db_path=db, embed_device="cpu")

        # Preferences (what Mem0 does)
        k.ingest("I prefer dark mode in all my apps.", speaker="Sam")
        k.ingest("My favorite programming language is Python.", speaker="Sam")
        k.ingest("I'm vegetarian.", speaker="Sam")
        k.ingest("I wake up at 6 AM every day.", speaker="Sam")

        # Context that goes BEYOND preferences (what Mem0 can't do)
        k.ingest("I've been feeling burned out from work lately.", speaker="Sam")
        k.ingest("I'm thinking about taking a sabbatical in September.", speaker="Sam")
        k.ingest("Actually, I decided against the sabbatical. Too risky financially.",
                 speaker="Sam")

        print("Mem0 comparison test")
        print("-" * 40)

        # Preference queries (Mem0 territory)
        for q in [
            "What programming language does Sam prefer?",
            "Is Sam vegetarian?",
            "What time does Sam wake up?",
        ]:
            a = k.retrieve(q)
            status = "REFUSE" if a.refusal else "ANSWER"
            print(f"  [{status}] {q}")
            if not a.refusal:
                print(f"          -> {a.text}")

        print()

        # Beyond-preference queries (Reconstruct territory)
        for q in [
            "How is Sam feeling about work?",
            "Is Sam taking a sabbatical?",  # should say no — superseded
        ]:
            a = k.retrieve(q)
            status = "REFUSE" if a.refusal else "ANSWER"
            print(f"  [{status}] {q}")
            if not a.refusal:
                print(f"          -> {a.text}")

    finally:
        os.unlink(db)


if __name__ == "__main__":
    print("=" * 60)
    print("CROSS-HOST WORKFLOW TEST")
    print("=" * 60)
    print()
    test_cross_host_career()
    print()
    print("=" * 60)
    print("KNOWLEDGE ACCUMULATION TEST")
    print("=" * 60)
    print()
    test_knowledge_accumulation()
    print()
    print("=" * 60)
    print("MEM0 COMPARISON TEST")
    print("=" * 60)
    print()
    test_mem0_comparison()
