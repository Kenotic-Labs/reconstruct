#!/usr/bin/env python3
"""Retrieval Isolation Test.

Bypasses grammar engine and memory engine entirely.
Inserts 25 edges directly into SQLite with known-correct values,
then runs queries against the retrieval engine and scores with F1.
"""
import os
import sys
import json
import hashlib

# ── Step 0: Set DB path BEFORE any imports ──
TEST_DB_PATH = "Memory Storage/test_retrieval_isolation.db"
os.environ["NURA_SQLITE_PATH"] = TEST_DB_PATH
os.environ["RAYA_EMBED_DEVICE"] = "cpu"

# Must set cwd to project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Step 1: Delete old DB ──
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)
    print(f"[SETUP] Deleted old DB: {TEST_DB_PATH}")
for suffix in ("-wal", "-shm"):
    p = TEST_DB_PATH + suffix
    if os.path.exists(p):
        os.remove(p)

# ── Step 2: Patch settings before any other import ──
from config.settings import settings
settings.sqlite_path = TEST_DB_PATH

# ── Step 3: Create fresh DB with migrations ──
from app.db.session import init_db
init_db(TEST_DB_PATH)
print(f"[SETUP] Created fresh DB: {TEST_DB_PATH}")

# ── Step 4: Import what we need ──
import sqlite3
import numpy as np
from app.vector.embedder import embed_text
from app.engines.retrieval import RetrievalEngine, Answer, StructuralRefusal

# ── Step 5: Define edges ──
EDGES = [
    # --- SAM (user_id=1) ---
    {
        "user_id": 1, "subject": "user", "predicate": "work_at", "object": "Google",
        "source_text": "I work at Google as a software engineer.",
        "edge_schematic_category": "career", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Google", "Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "ORG",
    },
    {
        "user_id": 1, "subject": "user", "predicate": "live_in", "object": "Portland, Oregon",
        "source_text": "I live in Portland, Oregon.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Portland", "Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "LOCATION",
    },
    {
        "user_id": 1, "subject": "user", "predicate": "live_in", "object": "Boston",
        "source_text": "I used to live in Boston.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 1,
        "relational_entities": json.dumps(["Boston", "Sam"]),
        "is_current": 0, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "LOCATION",
    },
    {
        "user_id": 1, "subject": "Linda", "predicate": "be_mother_of", "object": "user",
        "source_text": "My mom's name is Linda.",
        "edge_schematic_category": "family", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Linda", "Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "PERSON",
    },
    {
        "user_id": 1, "subject": "user", "predicate": "be_allergic_to", "object": "peanuts",
        "source_text": "I'm allergic to peanuts.",
        "edge_schematic_category": "health", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    {
        "user_id": 1, "subject": "user", "predicate": "have", "object": "a cat named Whiskers",
        "source_text": "My cat Whiskers is 3 years old.",
        "edge_schematic_category": "family", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Whiskers", "Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "ANIMAL",
    },
    {
        "user_id": 1, "subject": "user", "predicate": "love", "object": "playing guitar",
        "source_text": "I love playing guitar in my free time.",
        "edge_schematic_category": "hobby", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    {
        "user_id": 1, "subject": "user", "predicate": "get_married_to", "object": "Sarah",
        "source_text": "I got married to Sarah in June 2024.",
        "edge_schematic_category": "family", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": "June 2024", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Sarah", "Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "PERSON",
    },
    {
        "user_id": 1, "subject": "user", "predicate": "graduate_from", "object": "Stanford",
        "source_text": "I graduated from Stanford with a degree in computer science.",
        "edge_schematic_category": "education", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Stanford", "Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "ORG",
    },
    {
        "user_id": 1, "subject": "user", "predicate": "feel", "object": "nervous",
        "source_text": "I'm really nervous about my interview at Apple next Tuesday.",
        "edge_schematic_category": "career", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "next Tuesday", "edge_emotional_label": "nervous",
        "emotional_target": "interview", "is_historical": 0,
        "relational_entities": json.dumps(["Apple", "Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.3,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    {
        "user_id": 1, "subject": "user", "predicate": "feel", "object": "optimistic",
        "source_text": "I feel pretty optimistic about this year.",
        "edge_schematic_category": "identity", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": "optimistic",
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.8,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    {
        "user_id": 1, "subject": "user", "predicate": "train_for", "object": "a marathon",
        "source_text": "I've been training for a marathon since January.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "since January", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    {
        "user_id": 1, "subject": "Emily", "predicate": "have", "object": "a baby girl named Lily",
        "source_text": "My sister Emily just had a baby girl named Lily.",
        "edge_schematic_category": "family", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Emily", "Lily", "Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "PERSON",
    },
    {
        "user_id": 1, "subject": "Sam's dad", "predicate": "retire_from", "object": "Boeing",
        "source_text": "My dad retired last year after 30 years at Boeing.",
        "edge_schematic_category": "family", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": "last year", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Boeing", "Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "ORG",
    },
    {
        "user_id": 1, "subject": "user", "predicate": "sign_up_for", "object": "a pottery class",
        "source_text": "I signed up for a pottery class that starts next month.",
        "edge_schematic_category": "hobby", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "next month", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Sam"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # --- ALEX (user_id=2) ---
    {
        "user_id": 2, "subject": "user", "predicate": "work_as", "object": "an art director at a design agency",
        "source_text": "I work as an art director at a design agency.",
        "edge_schematic_category": "career", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Alex"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    {
        "user_id": 2, "subject": "user", "predicate": "live_in", "object": "Seattle",
        "source_text": "I live in Seattle with my partner Jordan.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Seattle", "Jordan", "Alex"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "LOCATION",
    },
    {
        "user_id": 2, "subject": "user", "predicate": "be_best_friend_with", "object": "Bella",
        "source_text": "My best friend Bella and I meet every Thursday for dinner.",
        "edge_schematic_category": "social", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "every Thursday", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Bella", "Alex"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "PERSON",
    },
    {
        "user_id": 2, "subject": "user", "predicate": "adopt", "object": "a rescue dog named Biscuit",
        "source_text": "I adopted a rescue dog named Biscuit last month.",
        "edge_schematic_category": "family", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": "last month", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Biscuit", "Alex"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "ANIMAL",
    },
    {
        "user_id": 2, "subject": "user", "predicate": "take", "object": "an online photography course",
        "source_text": "I'm taking an online photography course.",
        "edge_schematic_category": "education", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Alex"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    {
        "user_id": 2, "subject": "user", "predicate": "move_to", "object": "Seattle",
        "source_text": "I moved to Seattle from Chicago three years ago.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": "three years ago", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Seattle", "Chicago", "Alex"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "LOCATION",
    },
    {
        "user_id": 2, "subject": "user", "predicate": "feel", "object": "stressed",
        "source_text": "I feel stressed about the project deadline this Friday.",
        "edge_schematic_category": "career", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "this Friday", "edge_emotional_label": "stressed",
        "emotional_target": "project deadline", "is_historical": 0,
        "relational_entities": json.dumps(["Alex"]),
        "is_current": 1, "edge_emotional_valence": 0.3,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    {
        "user_id": 2, "subject": "Marcus", "predicate": "work_in", "object": "finance",
        "source_text": "My brother Marcus lives in Denver and works in finance.",
        "edge_schematic_category": "family", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Marcus", "Denver", "Alex"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    {
        "user_id": 2, "subject": "user", "predicate": "do", "object": "yoga every morning",
        "source_text": "I've been doing yoga every morning for the past six months.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "six months", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Alex"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    {
        "user_id": 2, "subject": "user", "predicate": "love", "object": "cooking Thai food",
        "source_text": "I love cooking Thai food, especially pad thai.",
        "edge_schematic_category": "hobby", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Alex"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
]

# Predicted queries per edge (0-indexed edge list)
PQS = {
    0: [("Where does Sam work?", "Google"),
        ("What does Sam do for work?", "Google"),
        ("What is Sam's job?", "Google"),
        ("What is Sam's occupation?", "Google")],
    1: [("Where does Sam live?", "Portland, Oregon"),
        ("Where is Sam located?", "Portland, Oregon"),
        ("What city does Sam live in?", "Portland, Oregon")],
    2: [("Where did Sam used to live?", "Boston"),
        ("Where did Sam live before?", "Boston")],
    3: [("Who is Sam's mom?", "Linda"),
        ("Who is Sam's mother?", "Linda"),
        ("What is Sam's mom's name?", "Linda"),
        ("What is Linda to Sam?", "mother")],
    4: [("What is Sam allergic to?", "peanuts"),
        ("Does Sam have any allergies?", "peanuts"),
        ("What food allergies does Sam have?", "peanuts")],
    5: [("What is Sam's cat's name?", "Whiskers"),
        ("Does Sam have a pet?", "a cat named Whiskers"),
        ("What pet does Sam have?", "a cat named Whiskers")],
    6: [("What is Sam's hobby?", "playing guitar"),
        ("What does Sam do for fun?", "playing guitar"),
        ("What does Sam enjoy?", "playing guitar"),
        ("What does Sam love doing?", "playing guitar")],
    7: [("Who did Sam marry?", "Sarah"),
        ("Who is Sam married to?", "Sarah"),
        ("When did Sam get married?", "June 2024"),
        ("Who is Sam's wife?", "Sarah")],
    8: [("Where did Sam graduate from?", "Stanford"),
        ("Where did Sam go to school?", "Stanford"),
        ("What is Sam's education?", "Stanford")],
    9: [("How does Sam feel about the interview?", "nervous"),
        ("How does Sam feel?", "nervous"),
        ("Is Sam nervous?", "nervous")],
    10: [("How does Sam feel about this year?", "optimistic"),
         ("Is Sam optimistic?", "optimistic")],
    11: [("What is Sam training for?", "a marathon"),
         ("What exercise does Sam do?", "a marathon"),
         ("Is Sam training for something?", "a marathon"),
         ("How is the marathon training going?", "a marathon")],
    12: [("What happened with Sam's sister?", "a baby girl named Lily"),
         ("Who is Emily's baby?", "Lily"),
         ("Did Emily have a baby?", "a baby girl named Lily")],
    13: [("Where did Sam's dad work?", "Boeing"),
         ("When did Sam's dad retire?", "last year"),
         ("What did Sam's dad do?", "Boeing")],
    14: [("What class is Sam taking?", "a pottery class"),
         ("What is Sam signed up for?", "a pottery class"),
         ("What new activity is Sam doing?", "a pottery class")],
    15: [("What does Alex do for work?", "an art director at a design agency"),
         ("What is Alex's job?", "an art director at a design agency"),
         ("Where does Alex work?", "an art director at a design agency")],
    16: [("Where does Alex live?", "Seattle"),
         ("What city does Alex live in?", "Seattle")],
    17: [("Who is Alex's best friend?", "Bella"),
         ("Who does Alex meet for dinner?", "Bella")],
    18: [("What kind of dog does Alex have?", "a rescue dog named Biscuit"),
         ("What is Alex's dog's name?", "Biscuit"),
         ("When did Alex adopt a dog?", "last month")],
    19: [("What is Alex studying?", "an online photography course"),
         ("What course is Alex taking?", "an online photography course"),
         ("Is Alex learning something new?", "an online photography course")],
    20: [("When did Alex move to Seattle?", "three years ago"),
         ("Where did Alex move from?", "Chicago"),
         ("How long has Alex lived in Seattle?", "three years ago")],
    21: [("How does Alex feel?", "stressed"),
         ("Is Alex stressed?", "stressed"),
         ("How does Alex feel about work?", "stressed")],
    22: [("What does Alex's brother do?", "finance"),
         ("Who is Marcus?", "finance"),
         ("Where does Alex's brother work?", "finance"),
         ("What does Marcus do for work?", "finance")],
    23: [("What exercise does Alex do?", "yoga every morning"),
         ("Does Alex do yoga?", "yoga every morning"),
         ("What is Alex's morning routine?", "yoga every morning")],
    24: [("What does Alex love cooking?", "cooking Thai food"),
         ("What is Alex's hobby?", "cooking Thai food"),
         ("What food does Alex like to cook?", "cooking Thai food")],
}

# Facts to insert
FACTS = [
    (1, "career::WORK::Sam", "Google"),
    (1, "housing::LOCATION::Sam", "Portland, Oregon"),
    (1, "family::BE::Sam::mother", "Linda"),
    (1, "health::BE::Sam::allergy", "peanuts"),
    (1, "hobby::PREFERENCE::Sam", "playing guitar"),
    (2, "career::WORK::Alex", "an art director at a design agency"),
    (2, "housing::LOCATION::Alex", "Seattle"),
]

# Entities to insert
ENTITIES = [
    (1, "Sam", "PERSON"),
    (1, "Google", "ORG"),
    (1, "Linda", "PERSON"),
    (1, "Stanford", "ORG"),
    (1, "Apple", "ORG"),
    (1, "Emily", "PERSON"),
    (1, "Lily", "PERSON"),
    (1, "Boeing", "ORG"),
    (1, "Whiskers", "ANIMAL"),
    (1, "Sarah", "PERSON"),
    (2, "Alex", "PERSON"),
    (2, "Seattle", "LOCATION"),
    (2, "Jordan", "PERSON"),
    (2, "Bella", "PERSON"),
    (2, "Biscuit", "ANIMAL"),
    (2, "Marcus", "PERSON"),
    (2, "Denver", "LOCATION"),
]

# ── Step 6: Insert data ──
def insert_all():
    conn = sqlite3.connect(TEST_DB_PATH)
    conn.row_factory = sqlite3.Row

    # Insert edges
    edge_ids = []
    for seq, edge in enumerate(EDGES, start=1):
        source_hash = hashlib.sha256(edge["source_text"].encode()).hexdigest()
        edge_emb = embed_text(edge["source_text"]).tobytes()
        pred_emb = embed_text(edge["predicate"].replace("_", " ")).tobytes()

        conn.execute("""
            INSERT INTO relationships (
                user_id, subject, predicate, object,
                source_text, source_text_hash,
                edge_schematic_category, edge_episodic_significance,
                edge_mood, edge_temporal_context,
                temporal_expression, edge_emotional_label,
                emotional_target, is_historical,
                relational_entities, is_current,
                edge_embedding, predicate_embedding,
                edge_emotional_valence, edge_relational_type,
                confidence, sequence_number,
                subject_type, object_type
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            edge["user_id"], edge["subject"], edge["predicate"], edge["object"],
            edge["source_text"], source_hash,
            edge["edge_schematic_category"], edge["edge_episodic_significance"],
            edge["edge_mood"], edge["edge_temporal_context"],
            edge.get("temporal_expression"), edge.get("edge_emotional_label"),
            edge.get("emotional_target"), edge["is_historical"],
            edge["relational_entities"], edge["is_current"],
            edge_emb, pred_emb,
            edge["edge_emotional_valence"], edge["edge_relational_type"],
            edge["confidence"], seq,
            edge.get("subject_type"), edge.get("object_type"),
        ))
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        edge_ids.append(rid)

        # FTS5 sync
        conn.execute("""
            INSERT INTO relationships_fts(rowid, subject, predicate, object, source_text)
            VALUES (?, ?, ?, ?, ?)
        """, (rid, edge["subject"], edge["predicate"].replace("_", " "),
              edge["object"], edge["source_text"]))

    conn.commit()
    print(f"[SETUP] Inserted {len(edge_ids)} edges")

    # Insert predicted queries
    pq_count = 0
    for edge_idx, pqs in PQS.items():
        rid = edge_ids[edge_idx]
        uid = EDGES[edge_idx]["user_id"]
        for question, answer in pqs:
            q_emb = embed_text(question).tobytes()
            conn.execute("""
                INSERT INTO predicted_queries (
                    relationship_id, user_id, predicted_question,
                    answer_text, answer_subject, question_embedding, confidence
                ) VALUES (?,?,?,?,?,?,?)
            """, (rid, uid, question, answer, EDGES[edge_idx]["subject"],
                  q_emb, 0.9))
            pq_count += 1
    conn.commit()
    print(f"[SETUP] Inserted {pq_count} predicted queries")

    # Insert facts
    for uid, key, value in FACTS:
        fact_emb = embed_text(f"{key} {value}").tobytes()
        history = None
        if key == "housing::LOCATION::Sam":
            history = json.dumps([{"value": "Boston", "at": "2025-01-01", "type": "update"}])
        conn.execute("""
            INSERT OR REPLACE INTO facts (user_id, key, value, confidence, embedding, history)
            VALUES (?,?,?,0.9,?,?)
        """, (uid, key, value, fact_emb, history))
    conn.commit()
    print(f"[SETUP] Inserted {len(FACTS)} facts")

    # Insert entities
    for uid, name, etype in ENTITIES:
        ent_emb = embed_text(name).tobytes()
        conn.execute("""
            INSERT OR REPLACE INTO entities (user_id, name, entity_type, embedding, mention_count)
            VALUES (?,?,?,?,1)
        """, (uid, name, etype, ent_emb))
    conn.commit()
    print(f"[SETUP] Inserted {len(ENTITIES)} entities")

    conn.close()

print("\n" + "=" * 70)
print("INSERTING DATA")
print("=" * 70)
insert_all()

# ── Step 7: Run queries ──
queries = [
    ("What does Sam do for work?", 1, "Google software engineer"),
    ("Where does Sam live?", 1, "Portland Oregon"),
    ("What is Sam's hobby?", 1, "playing guitar"),
    ("Who is Sam's mom?", 1, "Linda"),
    ("What is Sam allergic to?", 1, "peanuts"),
    ("What is Sam's cat's name?", 1, "Whiskers cat"),
    ("Where does Alex work?", 2, "art director design agency"),
    ("Who is Alex's best friend?", 2, "Bella"),
    ("What kind of dog does Alex have?", 2, "rescue dog Biscuit"),
    ("When did Sam get married?", 1, "June 2024"),
    ("When is Sam's interview?", 1, "next Tuesday"),
    ("When did Alex move to Seattle?", 2, "three years ago"),
    ("What does Alex do for work?", 1, "REFUSE"),
    ("Where does Sam live?", 2, "REFUSE"),
    ("Who is Sam's best friend?", 2, "REFUSE"),
    ("How does Sam feel about the interview?", 1, "nervous"),
    ("How does Alex feel?", 2, "stressed"),
    ("Where is Sam's mom?", 1, "Linda"),
    ("What does Alex's brother do?", 2, "finance"),
    ("What does Sam do for fun?", 1, "playing guitar"),
    ("What exercise does Sam do?", 1, "marathon"),
    ("What is Alex studying?", 2, "photography course"),
    ("Where did Sam used to live?", 1, "Boston"),
    ("What's going on in Sam's life?", 1, "NONEMPTY"),
    ("Where did Sam graduate from?", 1, "Stanford"),
]


def f1(pred, gold):
    if not pred or not gold:
        return 0.0
    p_tokens = set(pred.lower().split())
    g_tokens = set(gold.lower().split())
    overlap = p_tokens & g_tokens
    if not overlap:
        return 0.0
    prec = len(overlap) / len(p_tokens)
    rec = len(overlap) / len(g_tokens)
    return 2 * prec * rec / (prec + rec)


# Reset singleton so we get a fresh engine
import app.engines.retrieval as _ret_mod
_ret_mod._singleton = None

engine = RetrievalEngine(None, None)

print("\n" + "=" * 70)
print("RUNNING 25 QUERIES")
print("=" * 70 + "\n")

passed = 0
failed = 0
results = []

for i, (query, uid, gold) in enumerate(queries, 1):
    result = engine.retrieve(uid, query)
    is_refusal = isinstance(result, StructuralRefusal)
    answer_text = result.text if result.text else ""

    if gold == "REFUSE":
        ok = is_refusal
        status = "PASS" if ok else "FAIL"
        detail = f"got {'StructuralRefusal' if is_refusal else 'Answer'}: {answer_text[:80]}"
    elif gold == "NONEMPTY":
        ok = not is_refusal and len(answer_text.strip()) > 3
        status = "PASS" if ok else "FAIL"
        detail = f"len={len(answer_text)}, text={answer_text[:80]}"
    else:
        if is_refusal:
            ok = False
            status = "FAIL"
            score = 0.0
            detail = f"got StructuralRefusal({result.reason})"
        else:
            score = f1(answer_text, gold)
            ok = score >= 0.3
            status = "PASS" if ok else "FAIL"
            detail = f"F1={score:.3f} answer={answer_text[:80]} gold={gold}"

    if ok:
        passed += 1
    else:
        failed += 1

    tag = "+" if ok else "X"
    print(f"  [{tag}] Q{i:02d} ({status}) uid={uid} | {query}")
    print(f"         {detail}")
    if hasattr(result, 'convergence_details') and result.convergence_details:
        cd = result.convergence_details
        extras = []
        if 'entry_cosine' in cd:
            extras.append(f"entry={cd['entry_cosine']:.3f}")
        if 'exit_cosine' in cd:
            extras.append(f"exit={cd['exit_cosine']:.3f}")
        if 'source_stages' in cd:
            extras.append(f"stages={cd['source_stages']}")
        if cd.get('validate_warning'):
            extras.append(f"warn={cd['validate_warning']}")
        if extras:
            print(f"         {' | '.join(extras)}")
    if hasattr(result, 'source'):
        print(f"         source={result.source}")
    print()

# ── Step 8: Report ──
total = len(queries)
pct = (passed / total * 100) if total else 0

print("=" * 70)
print(f"RESULTS: {passed}/{total} passed ({pct:.1f}%)")
print(f"         {failed}/{total} failed")
print("=" * 70)

if pct >= 80:
    print("\n>>> RETRIEVAL ENGINE IS WORKING WELL <<<")
elif pct >= 60:
    print("\n>>> RETRIEVAL ENGINE NEEDS IMPROVEMENT <<<")
else:
    print("\n>>> RETRIEVAL ENGINE HAS CRITICAL ISSUES <<<")
