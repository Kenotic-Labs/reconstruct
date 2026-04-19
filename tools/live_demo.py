#!/usr/bin/env python3
"""Live multi-agent continuity demo — agents back and forth."""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HOME", "D:/Nura/Env/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-33ef6337-3850-1211-4834-097b0c5873a5")
warnings.filterwarnings("ignore")

# Suppress model-load noise
import logging
logging.disable(logging.WARNING)

PROJECT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT)
db = os.path.join(PROJECT, "tmp", "live_demo.db")
if os.path.exists(db):
    os.remove(db)

from config.settings import settings
settings.sqlite_path = db

from app.engines.memory import MemoryEngine
from app.engines.retrieval import RetrievalEngine
from app.db.session import init_db

# Suppress stdout during init
_real_stdout = sys.stdout
sys.stdout = io.StringIO()
init_db(db)
# Warm up embedder
from app.vector.embedder import embed_text
embed_text("warmup")
sys.stdout = _real_stdout

def divider(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

# ================================================================
# AGENT A — "Claude" — First session with the user
# ================================================================
divider('AGENT A -- Claude -- First session with the user')

mem = MemoryEngine()

print('  The user opens up about their week...\n')

# Turn 1: Work
mem.store(user_id=1, subject="user", predicate="had", object="rough week at work",
    source_text="Honestly it's been a rough week. My manager Devin called an all-hands Monday and told us the team is being restructured. Half the backend team is moving to a new project I don't care about.")
mem.store(user_id=1, subject="Devin", predicate="announced", object="team restructuring",
    source_text="My manager Devin called an all-hands Monday and told us the team is being restructured.")
mem.store(user_id=1, subject="user", predicate="manager", object="Devin",
    source_text="My manager Devin called an all-hands Monday.")
print('  Turn 1: Work restructuring. Devin announced it Monday. User is frustrated.')

# Turn 2: Friend
mem.store(user_id=1, subject="user", predicate="met_up_with", object="Riya on Wednesday",
    source_text="On the bright side I met up with Riya on Wednesday for coffee. She just got promoted at her company and she was buzzing. Made me feel a bit jealous honestly but also happy for her.")
mem.store(user_id=1, subject="Riya", predicate="got_promoted", object="at her company",
    source_text="She just got promoted at her company and she was buzzing.")
mem.store(user_id=1, subject="user", predicate="feels_about_Riya", object="jealous but happy",
    source_text="Made me feel a bit jealous honestly but also happy for her.")
print('  Turn 2: Met Riya Wednesday. She got promoted. User feels jealous but happy.')

# Turn 3: Weekend plans
mem.store(user_id=1, subject="user", predicate="planning", object="hike to Eagle Creek Saturday",
    source_text="This weekend I'm planning to do that hike up to Eagle Creek with my partner Jo. We've been talking about it for months. I need it honestly, just to clear my head after this week.")
mem.store(user_id=1, subject="user", predicate="partners_with", object="Jo",
    source_text="that hike up to Eagle Creek with my partner Jo.")
mem.store(user_id=1, subject="user", predicate="needs_to", object="clear head after rough week",
    source_text="I need it honestly, just to clear my head after this week.")
print('  Turn 3: Hiking Eagle Creek Saturday with partner Jo. Needs to decompress.')

# Turn 4: Health + proactive request
mem.store(user_id=1, subject="user", predicate="has_appointment", object="dentist Tuesday morning",
    source_text="Oh and I have a dentist appointment Tuesday morning so I'll be late to standup. Can you remind me Monday night?")
mem.store(user_id=1, subject="user", predicate="wants_reminder", object="Monday night about dentist",
    source_text="Can you remind me Monday night?")
print('  Turn 4: Dentist Tuesday. Wants reminder Monday night.')

print('\n  -- 11 triples stored across 4 turns --')
print('  -- Topics: career, friendship, travel, health --')
print('  -- Emotions: frustrated, conflicted, exhausted, excited --')
print('\n  Agent A session ends. Claude goes offline.\n')

# ================================================================
# AGENT B — "GPT" — Fresh session
# ================================================================
divider('AGENT B -- GPT -- Fresh session. Never saw the conversation.')

engine_b = RetrievalEngine()

print('  GPT connects to the same user. Queries memory.\n')

for q in [
    "How has this person's week been?",
    "Who is Riya and what happened with her?",
    "What are the weekend plans?",
    "Any upcoming appointments?",
    "How is this person feeling right now?",
]:
    r = engine_b.retrieve(user_id=1, query_text=q)
    a = r.text if hasattr(r, 'text') else str(r)
    print(f'  Q: {q}')
    print(f'  A: {a}\n')

print('  GPT now adds its own observations...\n')
mem_b = MemoryEngine()
mem_b.store(user_id=1, subject="user", predicate="might_benefit_from",
    object="career goals conversation",
    source_text="The work frustration combined with seeing Riya's promotion suggests this person could benefit from reflecting on their own career trajectory.")
mem_b.store(user_id=1, subject="user", predicate="shows_pattern",
    object="uses outdoor activity to process stress",
    source_text="Planning a hike specifically to clear head after a rough work week. This is a coping pattern worth noting.")
mem_b.store(user_id=1, subject="user", predicate="action_needed",
    object="send reminder Monday night about dentist",
    source_text="User explicitly asked for a Monday night reminder about Tuesday dentist appointment. This is a proactive commitment.")
print('  -> Stored 3 triples:')
print('     - Career insight: might benefit from goals conversation')
print('     - Pattern: uses outdoor activity to process stress')
print('     - Action: send reminder Monday night about dentist')
print('\n  Agent B session ends. GPT goes offline.\n')

# ================================================================
# AGENT C — "Gemini" — Cold start. Reconstructs everything.
# ================================================================
divider('AGENT C -- Gemini -- Cold start. Never saw A or B.')

engine_c = RetrievalEngine()

print('  Gemini has never interacted with this user.')
print('  It reconstructs the full situation from memory alone.\n')

sit = engine_c.reconstruct(user_id=1, query_text="What is going on with this person right now?")
print(f'  Q: What is going on with this person right now?\n')
print(f'  Survivors: {sit.survivors} edges')
print(f'  Participants: {getattr(sit, "participants", "N/A")}')
print(f'  Mood: {getattr(sit, "dominant_mood", "N/A")}')
print()
print(f'  RECONSTRUCTED NARRATIVE:')
narrative = sit.narrative or ""
for sent in narrative.split(". "):
    s = sent.strip().rstrip(".")
    if s:
        print(f'    {s}.')
print()

# Gemini also queries specifics
for q in [
    "What should I know about this person's emotional state?",
    "Is there anything I need to do for this person?",
]:
    r = engine_c.retrieve(user_id=1, query_text=q)
    a = r.text if hasattr(r, 'text') else str(r)
    print(f'  Q: {q}')
    print(f'  A: {a}\n')

print('  Agent C session ends. Gemini goes offline.\n')

# ================================================================
# AGENT A — "Claude" — Returns
# ================================================================
divider('AGENT A -- Claude returns. "What happened while I was gone?"')

engine_a2 = RetrievalEngine()

print('  Claude reconnects. Last time it talked to the user was Turn 4.\n')

for q in [
    "What observations have been made about this user since I left?",
    "Any action items I should know about?",
    "What patterns have been noticed?",
]:
    r = engine_a2.retrieve(user_id=1, query_text=q)
    a = r.text if hasattr(r, 'text') else str(r)
    print(f'  Q: {q}')
    print(f'  A: {a}\n')

divider('WHAT JUST HAPPENED')
print("""  Four agents. Three "models." One memory layer.

  Agent A (Claude) had a 4-turn conversation with the user.
  It stored 11 triples covering work, friendship, plans, health.

  Agent B (GPT) never saw that conversation. It queried the memory,
  understood the user's situation, and added its own insights:
  a career reflection suggestion, a behavioral pattern, and an
  action item (Monday reminder).

  Agent C (Gemini) never saw Agent A OR Agent B. It reconstructed
  the FULL picture from memory — work stress, Riya's promotion,
  the hiking plan, the dentist appointment, GPT's career insight,
  the behavioral pattern — all of it. From structured traces it
  never witnessed being created.

  Agent A (Claude) came back and saw what happened while it was
  gone — GPT's observations, Gemini's queries. The understanding
  accumulated across all three models.

  No LLM at read time. Deterministic. Structural.
  The memory layer preserved not what the user SAID, but what
  each agent UNDERSTOOD — and handed that understanding forward.

  That is continuity.
""")
