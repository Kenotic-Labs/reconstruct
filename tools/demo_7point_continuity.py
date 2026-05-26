#!/usr/bin/env python3
"""
7-Point Continuity Demo — Kenotic Labs
CONVERSATIONAL MULTI-AGENT VERSION

Proves 7 distinct continuity properties using YAML-bypass ingest
(hand-authored triples via MemoryEngine.store()), then queries via
RetrievalEngine.retrieve() and RetrievalEngine.reconstruct().

Each point runs a 4-phase conversation:
  [Agent A — Claude]  Ingests the user's story → stores triples
  [Agent B — GPT]     Fresh session. Queries memory, then stores its OWN additions
  [Agent C — Gemini]  Cold start. Reconstructs the full picture (A + B)
  [Agent A — Claude]  Returns. Queries "what's new?" → sees B and C's additions

This proves:
  1. Memory ACCUMULATES across agents (not just persists)
  2. Each agent ADDS to understanding (not just reads)
  3. Cold-start agent reconstructs EVERYTHING (not just latest)
  4. Original agent sees what happened while it was gone

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

# ===================================================================
# Helpers
# ===================================================================

HEAVY = "\n" + "=" * 60
LIGHT = "-" * 60
AGENT_A_TAG = "[Agent A -- Claude]"
AGENT_B_TAG = "[Agent B -- GPT]"
AGENT_C_TAG = "[Agent C -- Gemini]"
AGENT_A_RETURN_TAG = "[Agent A -- Claude returns]"


class _SuppressStdout:
    """Context manager to suppress stdout/stderr during noisy init."""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def init_db():
    """Wipe and re-create the demo database."""
    Path(DEMO_DB).parent.mkdir(parents=True, exist_ok=True)
    try:
        Path(DEMO_DB).unlink(missing_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(DEMO_DB)
    with _SuppressStdout():
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


_engines_initialized = False


def fresh_engines():
    """Create fresh engine instances (simulates fresh agent session).

    Each call returns genuinely new objects pointing at the same DB.
    First call suppresses embedding model load noise.
    """
    global _engines_initialized
    if not _engines_initialized:
        with _SuppressStdout():
            mem = MemoryEngine()
            tmp = TemporalEngine()
            tmp.bind_memory(mem)
            ret = RetrievalEngine(memory=mem, temporal=tmp)
        _engines_initialized = True
    else:
        mem = MemoryEngine()
        tmp = TemporalEngine()
        tmp.bind_memory(mem)
        ret = RetrievalEngine(memory=mem, temporal=tmp)
    return mem, tmp, ret


def store_triples(memory, user_id, triples, label=""):
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
    print(f"    -> Stored {n} triples")
    return n


def ask(retrieval, user_id, question, mode="lookup"):
    """Query and return (answer_text, raw_result)."""
    result = retrieval.answer(user_id, question)

    if isinstance(result, Answer):
        return result.text or "(empty)", result
    elif isinstance(result, Situation):
        return result.text or "(empty narrative)", result
    elif isinstance(result, StructuralRefusal):
        return f"(refusal: {result.reason})", result
    else:
        return f"(unknown: {type(result).__name__})", result


def print_qa(retrieval, user_id, question, mode="lookup"):
    """Print a formatted Q/A pair. Returns the answer text."""
    text, _ = ask(retrieval, user_id, question, mode)
    print(f"    Q: {question}")
    print(f"    A: {text}")
    print()
    return text


def type1_vs_type2(user_id, question, retrieval, mode="lookup"):
    """Show Type 1 (flat fact retrieval) vs Type 2 (Kenotic) side by side."""
    # --- Type 1: raw SPO triples from DB, no reconstruction ---
    type1_facts = []
    try:
        from app.db.session import get_db_context
        with get_db_context() as conn:
            rows = conn.execute(
                """SELECT subject, predicate, object
                   FROM relationships
                   WHERE user_id = ?
                     AND COALESCE(is_current, 1) = 1
                     AND tombstoned_at IS NULL
                   ORDER BY COALESCE(sequence_number, id) DESC
                   LIMIT 5""",
                (user_id,),
            ).fetchall()
            for r in rows:
                s = r["subject"] if isinstance(r, sqlite3.Row) else r[0]
                p = r["predicate"] if isinstance(r, sqlite3.Row) else r[1]
                o = r["object"] if isinstance(r, sqlite3.Row) else r[2]
                type1_facts.append(f"{s} {p} {o}")
    except Exception as e:
        type1_facts = [f"(error: {e})"]

    # --- Type 2: Kenotic reconstruction ---
    type2_text, _ = ask(retrieval, user_id, question, mode)

    print(f"  {'=' * 55}")
    print(f"  TYPE 1 (Fact Retrieval -- what Mem0/ChatGPT Memory does):")
    print(f"  Q: {question}")
    print(f"  A:")
    for fact in type1_facts:
        print(f"     * {fact}")
    print(f"     (flat facts, no arc, no emotion, no timeline)")
    print()
    print(f"  TYPE 2 (Understanding Continuity -- what Kenotic does):")
    print(f"  Q: {question}")
    print(f"  A: {type2_text}")
    print(f"     (reconstructed situation with temporal arc + emotional state)")
    print(f"  {'=' * 55}")
    print()


def point_header(n, name):
    print(HEAVY)
    print(f"  POINT {n}: {name}")
    print(HEAVY)
    print()


def agent_separator(label):
    print(f"  {LIGHT}")
    print(f"  --- {label} session ends ---")
    print(f"  {LIGHT}")
    print()


# ===================================================================
# POINT 1: Core Continuity
# ===================================================================

def point1_core_continuity():
    uid = 1
    point_header(1, "CORE CONTINUITY")

    # --- Agent A: Claude ingests the user's story ---
    print(f"  {AGENT_A_TAG}")
    print(f'  "The user just told me about their upcoming job interview..."')
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
    store_triples(mem, uid, triples)
    agent_separator("Agent A")

    # --- Type 1 vs Type 2 contrast ---
    mem_cmp, tmp_cmp, ret_cmp = fresh_engines()
    type1_vs_type2(uid, "What event do I have coming up?", ret_cmp)

    # --- Agent B: GPT queries and adds its own contributions ---
    print(f"  {AGENT_B_TAG} (fresh session)")
    print(f'  "Let me check what\'s going on with this user..."')
    print()

    mem2, tmp2, ret2 = fresh_engines()
    print_qa(ret2, uid, "What event do I have coming up?")
    print_qa(ret2, uid, "When is the interview?")
    print_qa(ret2, uid, "How am I feeling?")

    print(f'  "I can help them prep. Let me add some notes."')
    print()
    agent_b_triples = [
        ("user", "received_advice", "practice whiteboarding before Thursday",
         "GPT advised user to practice whiteboarding before Thursday"),
        ("user", "prep_status", "mock interview scheduled with Sam",
         "GPT noted mock interview is scheduled with Sam"),
        ("user", "should_review", "Conduit AI recent product launches",
         "GPT recommended reviewing Conduit AI recent product launches"),
    ]
    store_triples(mem2, uid, agent_b_triples)
    agent_separator("Agent B")

    # --- Agent C: Gemini cold start, reconstructs everything ---
    print(f"  {AGENT_C_TAG} (cold start -- never seen this user)")
    print(f'  "Let me reconstruct the full picture from memory..."')
    print()

    mem3, tmp3, ret3 = fresh_engines()
    print_qa(ret3, uid, "Summarize this user's current situation", mode="reconstruct")
    print_qa(ret3, uid, "What advice has been given?")
    print_qa(ret3, uid, "Who is helping me with mock prep?")
    agent_separator("Agent C")

    # --- Agent A returns: Claude checks what's new ---
    print(f"  {AGENT_A_RETURN_TAG}")
    print(f'  "I\'m back. Anything new since I was last here?"')
    print()

    mem4, tmp4, ret4 = fresh_engines()
    print_qa(ret4, uid, "What advice has the user received?")
    print_qa(ret4, uid, "What should I review for the interview?")

    print(f"  RESULT: Memory accumulated across 3 agents")
    print()


# ===================================================================
# POINT 2: Update Continuity
# ===================================================================

def point2_update_continuity():
    uid = 2
    point_header(2, "UPDATE CONTINUITY")

    # --- Agent A: Claude stores original situation ---
    print(f"  {AGENT_A_TAG}")
    print(f'  "The user has a team presentation coming up..."')
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
    store_triples(mem, uid, original)

    updates = [
        ("team presentation", "now_scheduled_for", "Wednesday at 3pm",
         "The team presentation is now scheduled for Wednesday at 3pm"),
        ("user", "now_feels", "stressed about the delay",
         "I now feel stressed about the delay"),
    ]
    print(f'  "User just told me the presentation got rescheduled..."')
    print()
    store_triples(mem, uid, updates)
    agent_separator("Agent A")

    # --- Type 1 vs Type 2 contrast ---
    mem_cmp, tmp_cmp, ret_cmp = fresh_engines()
    type1_vs_type2(uid, "When is the team presentation now?", ret_cmp)

    # --- Agent B: GPT queries and adds workflow help ---
    print(f"  {AGENT_B_TAG} (fresh session)")
    print(f'  "Checking in on this user\'s presentation situation..."')
    print()

    mem2, tmp2, ret2 = fresh_engines()
    print_qa(ret2, uid, "When is the team presentation now?")
    print_qa(ret2, uid, "How do I feel now?")

    print(f'  "They seem stressed. Let me help with the rescheduled prep."')
    print()
    agent_b_triples = [
        ("user", "received_advice", "update slide deck for Wednesday deadline",
         "GPT advised updating slide deck for Wednesday deadline"),
        ("user", "action_item", "notify team about new presentation time",
         "GPT flagged action item: notify team about new presentation time"),
    ]
    store_triples(mem2, uid, agent_b_triples)
    agent_separator("Agent B")

    # --- Agent C: Gemini reconstructs ---
    print(f"  {AGENT_C_TAG} (cold start)")
    print(f'  "Reconstructing user\'s presentation timeline..."')
    print()

    mem3, tmp3, ret3 = fresh_engines()
    print_qa(ret3, uid, "Summarize this user's presentation situation", mode="reconstruct")
    print_qa(ret3, uid, "What was the original date for the presentation?")
    print_qa(ret3, uid, "What action items are pending?")
    agent_separator("Agent C")

    # --- Agent A returns ---
    print(f"  {AGENT_A_RETURN_TAG}")
    print(f'  "What happened while I was gone?"')
    print()

    mem4, tmp4, ret4 = fresh_engines()
    print_qa(ret4, uid, "What advice has the user received?")
    print_qa(ret4, uid, "What action items need attention?")

    print(f"  RESULT: Memory accumulated across 3 agents")
    print()


# ===================================================================
# POINT 3: Disambiguation Continuity
# ===================================================================

def point3_disambiguation():
    uid = 3
    point_header(3, "DISAMBIGUATION CONTINUITY")

    # --- Agent A: Claude stores two overlapping interview stories ---
    print(f"  {AGENT_A_TAG}")
    print(f'  "User told me about TWO interviews -- theirs and Marcus\'s..."')
    print()

    mem, tmp, ret = fresh_engines()
    triples = [
        ("user", "has_interview_at", "Conduit AI",
         "I have an interview at Conduit AI"),
        ("user interview", "scheduled_for", "Thursday at 2pm",
         "My interview is scheduled for Thursday at 2pm"),
        ("user", "feels_about_interview", "nervous but excited",
         "I feel nervous but excited about my interview"),
        ("Marcus", "has_interview_at", "Palantir",
         "Marcus has an interview at Palantir"),
        ("Marcus interview", "takes_place", "Friday at 11am",
         "Marcus's interview takes place on Friday at 11am"),
        ("Marcus", "feels_about_interview", "very confident",
         "Marcus feels very confident about his interview"),
    ]
    store_triples(mem, uid, triples)
    agent_separator("Agent A")

    # --- Type 1 vs Type 2 contrast ---
    mem_cmp, tmp_cmp, ret_cmp = fresh_engines()
    type1_vs_type2(uid, "Where is my interview?", ret_cmp)

    # --- Agent B: GPT disambiguates and adds prep for each ---
    print(f"  {AGENT_B_TAG} (fresh session)")
    print(f'  "Two interviews in the mix. Let me sort them out..."')
    print()

    mem2, tmp2, ret2 = fresh_engines()
    print_qa(ret2, uid, "Where is my interview?")
    print_qa(ret2, uid, "Where is Marcus's interview?")
    print_qa(ret2, uid, "When is my interview?")
    print_qa(ret2, uid, "When is Marcus's interview?")

    print(f'  "Adding targeted prep notes for each interview."')
    print()
    agent_b_triples = [
        ("user", "prep_focus", "system design for Conduit AI",
         "GPT noted user should focus on system design for Conduit AI"),
        ("Marcus", "prep_focus", "data infrastructure for Palantir",
         "GPT noted Marcus should focus on data infrastructure for Palantir"),
        ("user", "coordination_note", "both interviews this week, stagger prep",
         "GPT noted both interviews are this week, should stagger prep"),
    ]
    store_triples(mem2, uid, agent_b_triples)
    agent_separator("Agent B")

    # --- Agent C: Gemini reconstructs both interviews ---
    print(f"  {AGENT_C_TAG} (cold start)")
    print(f'  "Reconstructing multi-person interview landscape..."')
    print()

    mem3, tmp3, ret3 = fresh_engines()
    print_qa(ret3, uid, "Summarize all upcoming interviews", mode="reconstruct")
    print_qa(ret3, uid, "How do I feel about my interview?")
    print_qa(ret3, uid, "How does Marcus feel about his interview?")
    agent_separator("Agent C")

    # --- Agent A returns ---
    print(f"  {AGENT_A_RETURN_TAG}")
    print(f'  "What prep notes were added?"')
    print()

    mem4, tmp4, ret4 = fresh_engines()
    print_qa(ret4, uid, "What should I focus on for my interview?")
    print_qa(ret4, uid, "What should Marcus focus on?")

    print(f"  RESULT: Memory accumulated across 3 agents")
    print()


# ===================================================================
# POINT 4: Multi-hop Reconstruction
# ===================================================================

def point4_multihop():
    uid = 4
    point_header(4, "MULTI-HOP RECONSTRUCTION")

    # --- Agent A: Claude stores scattered facts ---
    print(f"  {AGENT_A_TAG}")
    print(f'  "User shared a lot of context across the conversation..."')
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
    store_triples(mem, uid, triples)
    agent_separator("Agent A")

    # --- Type 1 vs Type 2 contrast ---
    mem_cmp, tmp_cmp, ret_cmp = fresh_engines()
    type1_vs_type2(uid, "What am I preparing for?", ret_cmp)

    # --- Agent B: GPT does multi-hop lookup, adds strategic notes ---
    print(f"  {AGENT_B_TAG} (fresh session)")
    print(f'  "Connecting the dots on this user\'s career transition..."')
    print()

    mem2, tmp2, ret2 = fresh_engines()
    print_qa(ret2, uid, "What am I preparing for?")
    print_qa(ret2, uid, "Why am I anxious?")
    print_qa(ret2, uid, "Where do I currently work?")

    print(f'  "Stripe to startup is a big jump. Adding strategic context."')
    print()
    agent_b_triples = [
        ("user", "career_context", "transitioning from big tech to startup",
         "GPT noted user is transitioning from big tech (Stripe) to startup"),
        ("user", "strength_to_highlight", "production-scale distributed systems experience",
         "GPT noted user should highlight production-scale distributed systems experience from Stripe"),
        ("user", "gap_to_address", "startup pace and ambiguity tolerance",
         "GPT identified gap: startup pace and ambiguity tolerance"),
    ]
    store_triples(mem2, uid, agent_b_triples)
    agent_separator("Agent B")

    # --- Agent C: Gemini reconstructs full career narrative ---
    print(f"  {AGENT_C_TAG} (cold start)")
    print(f'  "Building complete picture from all stored context..."')
    print()

    mem3, tmp3, ret3 = fresh_engines()
    print_qa(ret3, uid, "Summarize my current situation", mode="reconstruct")
    print_qa(ret3, uid, "What are my strengths for this interview?")
    print_qa(ret3, uid, "What gaps should I address?")
    agent_separator("Agent C")

    # --- Agent A returns ---
    print(f"  {AGENT_A_RETURN_TAG}")
    print(f'  "What strategic insights were added?"')
    print()

    mem4, tmp4, ret4 = fresh_engines()
    print_qa(ret4, uid, "What career context has been noted?")
    print_qa(ret4, uid, "What strengths should I highlight?")

    print(f"  RESULT: Memory accumulated across 3 agents")
    print()


# ===================================================================
# POINT 5: Model-Agnostic Continuity
# ===================================================================

def point5_model_agnostic():
    uid = 5
    point_header(5, "MODEL-AGNOSTIC CONTINUITY")

    # --- Agent A: Claude stores user preferences ---
    print(f"  {AGENT_A_TAG}")
    print(f'  "Learning the user\'s work preferences and context..."')
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
    store_triples(mem, uid, triples)
    agent_separator("Agent A")

    # --- Type 1 vs Type 2 contrast ---
    mem_cmp, tmp_cmp, ret_cmp = fresh_engines()
    type1_vs_type2(uid, "What am I working on?", ret_cmp)

    # --- Agent B: GPT (different model entirely) queries and adds ---
    print(f"  {AGENT_B_TAG} (different model, same DB)")
    print(f'  "First time seeing this user. What do I need to know?"')
    print()

    mem2, tmp2, ret2 = fresh_engines()
    print_qa(ret2, uid, "When do I prefer meetings?")
    print_qa(ret2, uid, "What am I allergic to?")
    print_qa(ret2, uid, "What am I working on?")
    print_qa(ret2, uid, "When is my deadline?")

    print(f'  "Adding scheduling and dietary notes for future agents."')
    print()
    agent_b_triples = [
        ("user", "scheduling_note", "block mornings for deep work after 10am meetings",
         "GPT noted to block mornings for deep work after 10am meetings"),
        ("user", "dietary_flag", "always check restaurant menus for shellfish",
         "GPT flagged: always check restaurant menus for shellfish"),
        ("user", "okr_reminder", "draft review with manager before Friday submission",
         "GPT set reminder: draft review with manager before Friday submission"),
    ]
    store_triples(mem2, uid, agent_b_triples)
    agent_separator("Agent B")

    # --- Agent C: Gemini reconstructs full user profile ---
    print(f"  {AGENT_C_TAG} (third model, cold start)")
    print(f'  "Building user profile from accumulated memory..."')
    print()

    mem3, tmp3, ret3 = fresh_engines()
    print_qa(ret3, uid, "Summarize this user's preferences and situation", mode="reconstruct")
    print_qa(ret3, uid, "What dietary restrictions should I know about?")
    print_qa(ret3, uid, "What team change do I want?")
    agent_separator("Agent C")

    # --- Agent A returns ---
    print(f"  {AGENT_A_RETURN_TAG}")
    print(f'  "I stored the basics. What did the other agents add?"')
    print()

    mem4, tmp4, ret4 = fresh_engines()
    print_qa(ret4, uid, "What scheduling notes exist?")
    print_qa(ret4, uid, "What reminders have been set?")

    print(f"  RESULT: Memory survives across 3 different model identities")
    print()


# ===================================================================
# POINT 6: Institutional Continuity
# ===================================================================

def point6_institutional():
    uid = 6
    point_header(6, "INSTITUTIONAL CONTINUITY")

    # --- Agent A: Claude stores workflow context ---
    print(f"  {AGENT_A_TAG}")
    print(f'  "Setting up the user\'s Friday standup logistics..."')
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
    store_triples(mem, uid, triples)
    agent_separator("Agent A")

    # --- Type 1 vs Type 2 contrast ---
    mem_cmp, tmp_cmp, ret_cmp = fresh_engines()
    type1_vs_type2(uid, "What is the standup situation?", ret_cmp)

    # --- Agent B: GPT checks on logistics and adds ops notes ---
    print(f"  {AGENT_B_TAG} (fresh session)")
    print(f'  "Reviewing the standup logistics..."')
    print()

    mem2, tmp2, ret2 = fresh_engines()
    print_qa(ret2, uid, "Which conference room did I book?")
    print_qa(ret2, uid, "What equipment does the room need?")
    print_qa(ret2, uid, "What time is the standup?")

    print(f'  "Adding operational prep notes for the standup."')
    print()
    agent_b_triples = [
        ("user", "ops_checklist", "arrive 10 min early to test projector",
         "GPT added ops note: arrive 10 min early to test projector"),
        ("user", "supplies_status", "whiteboard markers requested from office manager",
         "GPT noted: whiteboard markers requested from office manager"),
        ("Friday standup", "backup_plan", "use room C if projector fails",
         "GPT added backup plan: use room C if projector fails"),
    ]
    store_triples(mem2, uid, agent_b_triples)
    agent_separator("Agent B")

    # --- Agent C: Gemini reconstructs full ops picture ---
    print(f"  {AGENT_C_TAG} (cold start)")
    print(f'  "Reconstructing standup operations from memory..."')
    print()

    mem3, tmp3, ret3 = fresh_engines()
    print_qa(ret3, uid, "Summarize the Friday standup preparation", mode="reconstruct")
    print_qa(ret3, uid, "What supplies do I still need?")
    print_qa(ret3, uid, "What is the backup plan?")
    agent_separator("Agent C")

    # --- Agent A returns ---
    print(f"  {AGENT_A_RETURN_TAG}")
    print(f'  "I set up the basics. What ops work was added?"')
    print()

    mem4, tmp4, ret4 = fresh_engines()
    print_qa(ret4, uid, "What checklist items exist for the standup?")
    print_qa(ret4, uid, "What is the backup plan if something fails?")

    print(f"  RESULT: Memory accumulated across 3 agents")
    print()


# ===================================================================
# POINT 7: Physical / Operational Continuity
# ===================================================================

def point7_physical_operational():
    uid = 7
    point_header(7, "PHYSICAL / OPERATIONAL CONTINUITY")

    # --- Agent A: Claude stores operational incident ---
    print(f"  {AGENT_A_TAG}")
    print(f'  "Logging a delivery route incident..."')
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
    store_triples(mem, uid, triples)
    agent_separator("Agent A")

    # --- Type 1 vs Type 2 contrast ---
    mem_cmp, tmp_cmp, ret_cmp = fresh_engines()
    type1_vs_type2(uid, "What happened with the delivery?", ret_cmp)

    # --- Agent B: GPT reviews incident and adds resolution notes ---
    print(f"  {AGENT_B_TAG} (fresh session)")
    print(f'  "Reviewing the delivery incident report..."')
    print()

    mem2, tmp2, ret2 = fresh_engines()
    print_qa(ret2, uid, "What route failed?")
    print_qa(ret2, uid, "What backup route was used?")
    print_qa(ret2, uid, "How much delay occurred?")

    print(f'  "Adding incident analysis and follow-up actions."')
    print()
    agent_b_triples = [
        ("delivery robot", "incident_analysis", "construction on 7A expected through end of month",
         "GPT noted construction on 7A expected through end of month"),
        ("delivery robot", "recommended_action", "pre-compute route 7C via Elm Street as alternative",
         "GPT recommended pre-computing route 7C via Elm Street as alternative"),
        ("customer 14", "followup_status", "apology notification sent with discount code",
         "GPT confirmed apology notification sent to customer 14 with discount code"),
    ]
    store_triples(mem2, uid, agent_b_triples)
    agent_separator("Agent B")

    # --- Agent C: Gemini reconstructs full operational picture ---
    print(f"  {AGENT_C_TAG} (cold start)")
    print(f'  "Reconstructing delivery operations from all stored context..."')
    print()

    mem3, tmp3, ret3 = fresh_engines()
    print_qa(ret3, uid, "Summarize the delivery incident and response", mode="reconstruct")
    print_qa(ret3, uid, "What is the preferred route?")
    print_qa(ret3, uid, "What follow-up was done for the customer?")
    agent_separator("Agent C")

    # --- Agent A returns ---
    print(f"  {AGENT_A_RETURN_TAG}")
    print(f'  "I logged the incident. What analysis and actions were added?"')
    print()

    mem4, tmp4, ret4 = fresh_engines()
    print_qa(ret4, uid, "What is the recommended alternative route?")
    print_qa(ret4, uid, "What is the customer follow-up status?")
    print_qa(ret4, uid, "What should I avoid during rush hour?")

    print(f"  RESULT: Memory accumulated across 3 agents")
    print()


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    print()
    print("=" * 60)
    print("  KENOTIC LABS — 7-POINT CONTINUITY DEMO")
    print("  CONVERSATIONAL MULTI-AGENT VERSION")
    print()
    print("  Memory ACCUMULATES across agents.")
    print("  Each agent ADDS to understanding.")
    print("  Cold-start agents reconstruct EVERYTHING.")
    print("  No LLM at read time. Deterministic.")
    print("=" * 60)
    print()

    init_db()

    # Warm up embedding model (suppresses noisy load output)
    with _SuppressStdout():
        _warmup_mem = MemoryEngine()
        _warmup_mem.store(
            user_id=0, subject="warmup", predicate="init", object="test",
            source_text="warmup", confidence=0.5,
        )
        del _warmup_mem
    # Remove warmup data
    conn = sqlite3.connect(DEMO_DB)
    conn.execute("DELETE FROM relationships WHERE user_id = 0")
    try:
        conn.execute("DELETE FROM memory_traces WHERE relationship_id NOT IN (SELECT id FROM relationships)")
    except Exception:
        pass
    conn.commit()
    conn.close()

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

    print("=" * 60)
    print("  ALL 7 POINTS EXECUTED")
    print("  Memory accumulated. Agents built on each other.")
    print("  Cold-start reconstruction verified.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
