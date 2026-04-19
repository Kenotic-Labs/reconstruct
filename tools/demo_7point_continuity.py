#!/usr/bin/env python3
"""
7-Point Continuity Demo — Kenotic Labs

Proves 7 distinct continuity properties using YAML-bypass ingest
(hand-authored triples via MemoryEngine.store()), then queries via
RetrievalEngine.retrieve() and RetrievalEngine.reconstruct().

No MCP server. No LLM at read time. Deterministic output.

Usage:
    py -3.10 tools/demo_7point_continuity.py
"""
import os

os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES", "GPU-33ef6337-3850-1211-4834-097b0c5873a5"
)
os.environ.setdefault("RAYA_EMBED_DEVICE", "cuda")
os.environ.setdefault("HF_HOME", "D:/Nura/Env/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/Nura/Env/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "D:/Nura/Env/hf_cache/hub")
os.environ["RAYA_SENTENCE_POLISH"] = "0"

import io
import sqlite3
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEMO_DB = str(PROJECT_ROOT / "Memory Storage" / "demo_7point.db")

from config.settings import settings

settings.sqlite_path = DEMO_DB

from app.db.models import MIGRATIONS, run_schema_upgrades
from app.engines.memory import MemoryEngine
from app.engines.temporal import TemporalEngine
from app.engines.retrieval import RetrievalEngine, Answer, Situation, StructuralRefusal

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

SEP = "\n" + "=" * 55
THIN = "-" * 55


def init_db():
    """Wipe and re-create the demo database."""
    Path(DEMO_DB).parent.mkdir(parents=True, exist_ok=True)
    try:
        Path(DEMO_DB).unlink(missing_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(DEMO_DB)
    conn.executescript(MIGRATIONS)
    run_schema_upgrades(conn)
    try:
        conn.execute("ALTER TABLE relationships ADD COLUMN sequence_number INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_seq ON relationships(user_id, sequence_number)"
        )
    except Exception:
        pass
    conn.commit()
    conn.close()


def fresh_engines():
    """Create fresh engine instances (simulates process restart).

    We bypass the module-level singletons so each call returns
    genuinely new objects, but they all point at the same DB file.
    """
    mem = MemoryEngine()
    tmp = TemporalEngine()
    tmp.bind_memory(mem)
    ret = RetrievalEngine(memory_engine=mem, temporal_engine=tmp)
    return mem, tmp, ret


def store_triples(memory, user_id, triples, agent_label="Agent A"):
    """Store a list of (subj, pred, obj, source_text) tuples."""
    n = 0
    for t in triples:
        subj, pred, obj = t[0], t[1], t[2]
        src = t[3] if len(t) > 3 else ""
        rel_id = memory.store(
            user_id=user_id,
            subject=subj,
            predicate=pred,
            object=obj,
            source_text=src,
            confidence=0.95,
        )
        if rel_id:
            n += 1
    print(f"    [check] {n} triples stored")
    return n


def ask(retrieval, user_id, question, mode="lookup"):
    """Query and return (answer_text, raw_result)."""
    if mode == "reconstruct":
        result = retrieval.reconstruct(user_id, question)
    else:
        result = retrieval.retrieve(user_id, question)

    if isinstance(result, Answer):
        return result.text or "(empty)", result
    elif isinstance(result, Situation):
        return result.narrative or "(empty narrative)", result
    elif isinstance(result, StructuralRefusal):
        return f"(refusal: {result.reason})", result
    else:
        return f"(unknown: {type(result).__name__})", result


def print_qa(retrieval, user_id, question, mode="lookup"):
    """Print a formatted Q/A pair. Returns the answer text."""
    text, _ = ask(retrieval, user_id, question, mode)
    print(f"  Q: {question}")
    print(f"  A: {text}")
    print()
    return text


# ═══════════════════════════════════════════════════════════════════════
# POINT 1: Core Continuity
# ═══════════════════════════════════════════════════════════════════════

def point1_core_continuity():
    uid = 1
    print(SEP)
    print("  POINT 1: CORE CONTINUITY")
    print("  Store -> Restart -> Retrieve")
    print(SEP)
    print()

    mem, tmp, ret = fresh_engines()

    triples = [
        ("user", "has_event", "job interview at Conduit AI",
         "I have a job interview at Conduit AI"),
        ("job interview", "scheduled_for", "Thursday 2pm",
         "The job interview is scheduled for Thursday at 2pm"),
        ("user", "feels", "nervous about the interview",
         "I feel nervous about the interview"),
        ("user", "studying", "system design questions for the interview",
         "I am studying system design questions for the interview"),
        ("Conduit AI", "located_in", "downtown Portland",
         "Conduit AI is located in downtown Portland"),
        ("user", "needs_to_leave_by", "12:30pm for the drive",
         "I need to leave by 12:30pm for the drive"),
        ("Sam", "helping_with", "mock interview prep",
         "Sam is helping with mock interview prep"),
    ]

    print(f"  [Agent A -- Session 1] Storing situation...")
    store_triples(mem, uid, triples)
    print()

    print(f"  --- RESTART (new engine instance) ---")
    print()

    _, _, ret2 = fresh_engines()

    print(f"  [Agent B -- Session 2] Querying from fresh session...")
    print()

    print_qa(ret2, uid, "What event do I have coming up?")
    print_qa(ret2, uid, "When is the interview?")
    print_qa(ret2, uid, "Who is helping me with mock prep?")
    print_qa(ret2, uid, "How am I feeling?")

    print(f"  RESULT: Point 1 complete")
    print()


# ═══════════════════════════════════════════════════════════════════════
# POINT 2: Update Continuity
# ═══════════════════════════════════════════════════════════════════════

def point2_update_continuity():
    uid = 2
    print(SEP)
    print("  POINT 2: UPDATE CONTINUITY")
    print("  Store -> Update -> Restart -> Retrieve history")
    print(SEP)
    print()

    mem, tmp, ret = fresh_engines()

    original = [
        ("user", "has_event", "team presentation",
         "I have a team presentation"),
        ("team presentation", "originally_set_for", "Monday at 10am",
         "The team presentation was originally set for Monday at 10am"),
        ("user", "feels", "confident about presenting",
         "I feel confident about presenting"),
    ]

    print(f"  [Agent A -- Session 1] Storing original situation...")
    store_triples(mem, uid, original)
    print()

    updates = [
        ("team presentation", "now_scheduled_for", "Wednesday at 3pm",
         "The team presentation is now scheduled for Wednesday at 3pm"),
        ("user", "now_feels", "stressed about the delay",
         "I now feel stressed about the delay"),
    ]

    print(f"  [Agent A -- Session 1] Storing updates...")
    store_triples(mem, uid, updates)
    print()

    print(f"  --- RESTART (new engine instance) ---")
    print()

    _, _, ret2 = fresh_engines()

    print(f"  [Agent B -- Session 2] Querying updated state...")
    print()

    print_qa(ret2, uid, "When is the team presentation now?")
    print_qa(ret2, uid, "How do I feel now?")
    print_qa(ret2, uid, "What was the original date for the presentation?")

    print(f"  RESULT: Point 2 complete")
    print()


# ═══════════════════════════════════════════════════════════════════════
# POINT 3: Disambiguation Continuity
# ═══════════════════════════════════════════════════════════════════════

def point3_disambiguation():
    uid = 3
    print(SEP)
    print("  POINT 3: DISAMBIGUATION CONTINUITY")
    print("  Two similar situations -> Restart -> Distinguish")
    print(SEP)
    print()

    mem, tmp, ret = fresh_engines()

    triples = [
        # My interview
        ("user", "has_interview_at", "Conduit AI",
         "I have an interview at Conduit AI"),
        ("user interview", "scheduled_for", "Thursday at 2pm",
         "My interview is scheduled for Thursday at 2pm"),
        ("user", "feels_about_interview", "nervous but excited",
         "I feel nervous but excited about my interview"),
        # Marcus's interview
        ("Marcus", "has_interview_at", "Palantir",
         "Marcus has an interview at Palantir"),
        ("Marcus interview", "takes_place", "Friday at 11am",
         "Marcus's interview takes place on Friday at 11am"),
        ("Marcus", "feels_about_interview", "very confident",
         "Marcus feels very confident about his interview"),
    ]

    print(f"  [Agent A -- Session 1] Storing two interview situations...")
    store_triples(mem, uid, triples)
    print()

    print(f"  --- RESTART (new engine instance) ---")
    print()

    _, _, ret2 = fresh_engines()

    print(f"  [Agent B -- Session 2] Disambiguating...")
    print()

    print_qa(ret2, uid, "Where is my interview?")
    print_qa(ret2, uid, "Where is Marcus's interview?")
    print_qa(ret2, uid, "When is my interview?")
    print_qa(ret2, uid, "When is Marcus's interview?")
    print_qa(ret2, uid, "How do I feel about my interview?")
    print_qa(ret2, uid, "How does Marcus feel about his interview?")

    print(f"  RESULT: Point 3 complete")
    print()


# ═══════════════════════════════════════════════════════════════════════
# POINT 4: Multi-hop Reconstruction
# ═══════════════════════════════════════════════════════════════════════

def point4_multihop():
    uid = 4
    print(SEP)
    print("  POINT 4: MULTI-HOP RECONSTRUCTION")
    print("  Separate facts -> Restart -> Reconstruct situation")
    print(SEP)
    print()

    mem, tmp, ret = fresh_engines()

    triples = [
        ("user", "works_as", "backend engineer at Stripe",
         "I work as a backend engineer at Stripe"),
        ("user", "preparing_for", "system design interview at Conduit AI",
         "I am preparing for a system design interview at Conduit AI"),
        ("user", "anxious_because", "never done a startup interview before",
         "I am anxious because I have never done a startup interview before"),
        ("system design interview", "scheduled_for", "next Thursday",
         "The system design interview is scheduled for next Thursday"),
        ("user", "studying", "distributed systems and API design",
         "I am studying distributed systems and API design"),
        ("Sam", "recommended", "practicing whiteboard problems",
         "Sam recommended practicing whiteboard problems"),
    ]

    print(f"  [Agent A -- Session 1] Storing separate facts...")
    store_triples(mem, uid, triples)
    print()

    print(f"  --- RESTART (new engine instance) ---")
    print()

    _, _, ret2 = fresh_engines()

    print(f"  [Agent B -- Session 2] Reconstructing situation...")
    print()

    print_qa(ret2, uid, "What am I preparing for?")
    print_qa(ret2, uid, "Why am I anxious?")
    print_qa(ret2, uid, "Summarize my current situation", mode="reconstruct")

    print(f"  RESULT: Point 4 complete")
    print()


# ═══════════════════════════════════════════════════════════════════════
# POINT 5: Model-agnostic Continuity
# ═══════════════════════════════════════════════════════════════════════

def point5_model_agnostic():
    uid = 5
    print(SEP)
    print("  POINT 5: MODEL-AGNOSTIC CONTINUITY")
    print('  "Claude" stores -> "GPT" queries same DB')
    print(SEP)
    print()

    mem, tmp, ret = fresh_engines()

    triples = [
        ("user", "prefers", "morning meetings before 10am",
         "I prefer morning meetings before 10am"),
        ("user", "allergic_to", "shellfish",
         "I am allergic to shellfish"),
        ("user", "working_on", "quarterly OKR review",
         "I am working on the quarterly OKR review"),
        ("user", "deadline", "OKR draft due Friday",
         "My OKR draft is due Friday"),
        ("user", "wants", "to switch to the platform team",
         "I want to switch to the platform team"),
    ]

    print(f'  [Agent A -- "Claude"] Storing memory...')
    store_triples(mem, uid, triples, agent_label="Claude")
    print()

    print(f"  --- MODEL SWITCH (same DB, new engine instance) ---")
    print()

    _, _, ret2 = fresh_engines()

    print(f'  [Agent B -- "GPT"] Querying same memory...')
    print()

    print_qa(ret2, uid, "When do I prefer meetings?")
    print_qa(ret2, uid, "What am I allergic to?")
    print_qa(ret2, uid, "What am I working on?")
    print_qa(ret2, uid, "When is my deadline?")
    print_qa(ret2, uid, "What team change do I want?")

    print(f"  Memory survives model change. Same DB, different agent.")
    print(f"  RESULT: Point 5 complete")
    print()


# ═══════════════════════════════════════════════════════════════════════
# POINT 6: Institutional Continuity
# ═══════════════════════════════════════════════════════════════════════

def point6_institutional():
    uid = 6
    print(SEP)
    print("  POINT 6: INSTITUTIONAL CONTINUITY")
    print("  Repeated workflow -> Restart -> Continue without re-asking")
    print(SEP)
    print()

    mem, tmp, ret = fresh_engines()

    triples = [
        ("user", "booked", "conference room B for Friday standup",
         "I booked conference room B for the Friday standup"),
        ("conference room B", "requires", "HDMI adapter for projector",
         "Conference room B requires an HDMI adapter for the projector"),
        ("Friday standup", "time_window", "9:00am to 9:30am",
         "The Friday standup is from 9:00am to 9:30am"),
        ("user", "previously_requested", "room A but it was unavailable",
         "I previously requested room A but it was unavailable"),
        ("IT department", "confirmed", "projector is working in room B",
         "IT department confirmed the projector is working in room B"),
        ("user", "needs", "whiteboard markers for the session",
         "I need whiteboard markers for the session"),
    ]

    print(f"  [Agent A -- Session 1] Storing workflow context...")
    store_triples(mem, uid, triples)
    print()

    print(f"  --- RESTART (new engine instance) ---")
    print()

    _, _, ret2 = fresh_engines()

    print(f"  [Agent B -- Session 2] Continuing workflow...")
    print()

    print_qa(ret2, uid, "Which conference room did I book?")
    print_qa(ret2, uid, "What equipment does the room need?")
    print_qa(ret2, uid, "What time is the standup?")
    print_qa(ret2, uid, "What room did I originally want?")
    print_qa(ret2, uid, "What supplies do I still need?")

    print(f"  RESULT: Point 6 complete")
    print()


# ═══════════════════════════════════════════════════════════════════════
# POINT 7: Physical / Operational Continuity
# ═══════════════════════════════════════════════════════════════════════

def point7_physical_operational():
    uid = 7
    print(SEP)
    print("  POINT 7: PHYSICAL / OPERATIONAL CONTINUITY")
    print("  Robot/machine context -> Restart -> Recall operations")
    print(SEP)
    print()

    mem, tmp, ret = fresh_engines()

    triples = [
        ("delivery route", "status", "route 7A was blocked by construction",
         "Route 7A was blocked by construction"),
        ("delivery robot", "used", "backup route 7B through Oak Street",
         "The delivery robot used backup route 7B through Oak Street"),
        ("route 7B", "caused", "12 minute delay on delivery",
         "Route 7B caused a 12 minute delay on delivery"),
        ("delivery robot", "prefers", "route 7A when available",
         "The delivery robot prefers route 7A when available"),
        ("customer 14", "received", "package 18 minutes late",
         "Customer 14 received the package 18 minutes late"),
        ("delivery robot", "should_avoid", "Oak Street during rush hour",
         "The delivery robot should avoid Oak Street during rush hour"),
    ]

    print(f"  [Agent A -- Session 1] Storing operational context...")
    store_triples(mem, uid, triples)
    print()

    print(f"  --- RESTART (new engine instance) ---")
    print()

    _, _, ret2 = fresh_engines()

    print(f"  [Agent B -- Session 2] Querying operational memory...")
    print()

    print_qa(ret2, uid, "What route failed?")
    print_qa(ret2, uid, "What backup route was used?")
    print_qa(ret2, uid, "How much delay occurred?")
    print_qa(ret2, uid, "What should I avoid during rush hour?")
    print_qa(ret2, uid, "What is the preferred route?")

    print(f"  RESULT: Point 7 complete")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    print()
    print("=" * 55)
    print("  KENOTIC LABS — 7-POINT CONTINUITY DEMO")
    print("  YAML-bypass ingest -> structural retrieval")
    print("  No LLM at read time. Deterministic.")
    print("=" * 55)
    print()

    init_db()

    try:
        point1_core_continuity()
        point2_update_continuity()
        point3_disambiguation()
        point4_multihop()
        point5_model_agnostic()
        point6_institutional()
        point7_physical_operational()
    except Exception as e:
        print(f"\n  *** ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 55)
    print("  ALL 7 POINTS EXECUTED")
    print("=" * 55)
    return 0


if __name__ == "__main__":
    sys.exit(main())
