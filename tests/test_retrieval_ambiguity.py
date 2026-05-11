#!/usr/bin/env python3
"""Retrieval Ambiguity Stress Test.

Bypasses grammar engine and memory engine entirely.
Inserts 30 edges directly into SQLite with deliberate ambiguity:
overlapping entities, similar predicates, temporal confusion,
and cross-domain edges that share keywords.
The retrieval engine must pick the RIGHT edge.
"""
import os
import sys
import json
import hashlib

# ── Step 0: Set DB path BEFORE any imports ──
TEST_DB_PATH = "Memory Storage/test_retrieval_ambiguity.db"
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
# Maya = user_id 1, Kai = user_id 2
# Historical edges get LOW sequence numbers, current edges get HIGH ones.

EDGES = [
    # --- Scenario 1: Two jobs, different times ---
    # Edge A: current job (Stripe) — seq will be HIGHER
    {
        "user_id": 1, "subject": "user", "predicate": "work_at", "object": "Stripe",
        "source_text": "I work at Stripe as a backend engineer.",
        "edge_schematic_category": "career", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Stripe", "Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "ORG",
        "_seq": 20,  # current = higher seq
    },
    # Edge B: old job (Shopify) — seq will be LOWER
    {
        "user_id": 1, "subject": "user", "predicate": "work_at", "object": "Shopify",
        "source_text": "I used to work at Shopify as a frontend developer.",
        "edge_schematic_category": "career", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 1,
        "relational_entities": json.dumps(["Shopify", "Maya"]),
        "is_current": 0, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "ORG",
        "_seq": 5,  # historical = lower seq
    },

    # --- Scenario 2: Two locations, superseded ---
    # Edge C: current location (Austin)
    {
        "user_id": 1, "subject": "user", "predicate": "live_in", "object": "Austin, Texas",
        "source_text": "I live in Austin, Texas.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Austin", "Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "LOCATION",
        "_seq": 21,
    },
    # Edge D: old location (Denver)
    {
        "user_id": 1, "subject": "user", "predicate": "live_in", "object": "Denver",
        "source_text": "I used to live in Denver.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 1,
        "relational_entities": json.dumps(["Denver", "Maya"]),
        "is_current": 0, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "LOCATION",
        "_seq": 6,
    },

    # --- Scenario 3: Two friends, different contexts ---
    # Edge E: best friend from college (Priya)
    {
        "user_id": 1, "subject": "user", "predicate": "be_friend_with", "object": "Priya",
        "source_text": "Priya is my best friend from college.",
        "edge_schematic_category": "social", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Priya", "Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "PERSON",
        "_seq": 22,
    },
    # Edge F: work friend (Nadia)
    {
        "user_id": 1, "subject": "user", "predicate": "be_friend_with", "object": "Nadia",
        "source_text": "Nadia is my closest friend at work.",
        "edge_schematic_category": "social", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Nadia", "Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "PERSON",
        "_seq": 23,
    },

    # --- Scenario 4: Similar emotions, different targets ---
    # Edge G: anxious about interview
    {
        "user_id": 1, "subject": "user", "predicate": "feel", "object": "anxious",
        "source_text": "I feel anxious about the job interview tomorrow.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "tomorrow", "edge_emotional_label": "anxious",
        "emotional_target": "job interview", "is_historical": 0,
        "relational_entities": json.dumps(["Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.3,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
        "_seq": 24,
    },
    # Edge H: anxious about presentation
    {
        "user_id": 1, "subject": "user", "predicate": "feel", "object": "anxious",
        "source_text": "I'm anxious about the big presentation next week.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "next week", "edge_emotional_label": "anxious",
        "emotional_target": "presentation", "is_historical": 0,
        "relational_entities": json.dumps(["Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.3,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
        "_seq": 25,
    },
    # Edge I: excited about vacation
    {
        "user_id": 1, "subject": "user", "predicate": "feel", "object": "excited",
        "source_text": "I'm so excited about our vacation to Japan.",
        "edge_schematic_category": "identity", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": "excited",
        "emotional_target": "vacation", "is_historical": 0,
        "relational_entities": json.dumps(["Maya", "Japan"]),
        "is_current": 1, "edge_emotional_valence": 0.8,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
        "_seq": 26,
    },

    # --- Scenario 5: Overlapping keywords across domains ---
    # Edge J: run a 10K race (health)
    {
        "user_id": 1, "subject": "user", "predicate": "run", "object": "a 10K race",
        "source_text": "I ran a 10K race last Saturday.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": "last Saturday", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
        "_seq": 27,
    },
    # Edge K: run a small business (career)
    {
        "user_id": 1, "subject": "user", "predicate": "run", "object": "a small online business selling pottery",
        "source_text": "I run a small online business selling pottery.",
        "edge_schematic_category": "career", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
        "_seq": 28,
    },

    # --- Scenario 6: Same entity, different roles ---
    # Edge L: sister Zara
    {
        "user_id": 1, "subject": "user", "predicate": "sister", "object": "Zara",
        "source_text": "My sister Zara is a doctor in Chicago.",
        "edge_schematic_category": "family", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Zara", "Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "PERSON",
        "_seq": 29,
    },
    # Edge M: Zara works as doctor
    {
        "user_id": 1, "subject": "Zara", "predicate": "work_as", "object": "a doctor",
        "source_text": "Zara works as a doctor at Northwestern Hospital.",
        "edge_schematic_category": "career", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Zara", "Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
        "_seq": 30,
    },
    # Edge N: Zara lives in Chicago
    {
        "user_id": 1, "subject": "Zara", "predicate": "live_in", "object": "Chicago",
        "source_text": "Zara lives in Chicago.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Zara", "Chicago", "Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "LOCATION",
        "_seq": 31,
    },

    # --- Scenario 7: Temporal education ambiguity ---
    # Edge O: graduated from MIT
    {
        "user_id": 1, "subject": "user", "predicate": "graduate_from", "object": "MIT",
        "source_text": "I graduated from MIT in 2019.",
        "edge_schematic_category": "education", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": "2019", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["MIT", "Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "ORG",
        "_seq": 32,
    },
    # Edge P: starting masters at Stanford
    {
        "user_id": 1, "subject": "user", "predicate": "start", "object": "a master's program at Stanford",
        "source_text": "I'm starting a master's program at Stanford this fall.",
        "edge_schematic_category": "education", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "this fall", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Stanford", "Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "ORG",
        "_seq": 33,
    },

    # --- Scenario 8: Cross-user isolation (Kai = user_id 2) ---
    # Edge Q: Kai works at Google
    {
        "user_id": 2, "subject": "user", "predicate": "work_at", "object": "Google",
        "source_text": "I work at Google as a designer.",
        "edge_schematic_category": "career", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Google", "Kai"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "ORG",
        "_seq": 34,
    },
    # Edge R: Kai lives in Portland
    {
        "user_id": 2, "subject": "user", "predicate": "live_in", "object": "Portland",
        "source_text": "I live in Portland.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Portland", "Kai"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "LOCATION",
        "_seq": 35,
    },

    # --- Additional edges for richness (user_id=1, Maya) ---
    # Edge S: allergic to shellfish
    {
        "user_id": 1, "subject": "user", "predicate": "allergic_to", "object": "shellfish",
        "source_text": "I'm allergic to shellfish.",
        "edge_schematic_category": "health", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
        "_seq": 36,
    },
    # Edge T: golden retriever named Mochi
    {
        "user_id": 1, "subject": "user", "predicate": "have", "object": "a golden retriever named Mochi",
        "source_text": "I have a golden retriever named Mochi.",
        "edge_schematic_category": "family", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Mochi", "Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "ANIMAL",
        "_seq": 37,
    },
    # Edge U: love rock climbing
    {
        "user_id": 1, "subject": "user", "predicate": "love", "object": "rock climbing",
        "source_text": "I love rock climbing on weekends.",
        "edge_schematic_category": "hobby", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "GENERIC",
        "_seq": 38,
    },
    # Edge V: married to Jordan
    {
        "user_id": 1, "subject": "user", "predicate": "married_to", "object": "Jordan",
        "source_text": "I married Jordan in March 2023.",
        "edge_schematic_category": "family", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": "March 2023", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0,
        "relational_entities": json.dumps(["Jordan", "Maya"]),
        "is_current": 1, "edge_emotional_valence": 0.5,
        "edge_relational_type": "personal", "confidence": 0.9,
        "subject_type": "PERSON", "object_type": "PERSON",
        "_seq": 39,
    },
]

# Predicted queries per edge (0-indexed)
PQS = {
    # Edge A (0): Stripe current
    0: [("Where does Maya work?", "Stripe"),
        ("What does Maya do for work?", "Stripe"),
        ("What is Maya's job?", "Stripe")],
    # Edge B (1): Shopify historical
    1: [("Where did Maya used to work?", "Shopify"),
        ("Where did Maya work before?", "Shopify")],
    # Edge C (2): Austin current
    2: [("Where does Maya live?", "Austin, Texas"),
        ("What city does Maya live in?", "Austin, Texas")],
    # Edge D (3): Denver historical
    3: [("Where did Maya used to live?", "Denver"),
        ("Where did Maya live before?", "Denver")],
    # Edge E (4): Priya best friend
    4: [("Who is Maya's best friend?", "Priya"),
        ("Who is Priya to Maya?", "best friend from college")],
    # Edge F (5): Nadia work friend
    5: [("Who is Maya's work friend?", "Nadia"),
        ("Who is Nadia to Maya?", "closest friend at work")],
    # Edge G (6): anxious about interview
    6: [("How does Maya feel about the interview?", "anxious"),
        ("Is Maya anxious?", "anxious")],
    # Edge H (7): anxious about presentation
    7: [("How does Maya feel about the presentation?", "anxious"),
        ("Is Maya worried about the presentation?", "anxious")],
    # Edge I (8): excited about vacation
    8: [("How does Maya feel about the vacation?", "excited"),
        ("Is Maya excited about Japan?", "excited")],
    # Edge J (9): 10K race
    9: [("What race did Maya run?", "a 10K race"),
        ("What exercise does Maya do?", "a 10K race")],
    # Edge K (10): small business
    10: [("What business does Maya run?", "a small online business selling pottery"),
         ("Does Maya have a business?", "a small online business selling pottery")],
    # Edge L (11): sister Zara
    11: [("Who is Maya's sister?", "Zara"),
         ("Who is Zara?", "sister")],
    # Edge M (12): Zara doctor
    12: [("What does Maya's sister do?", "a doctor"),
         ("What does Zara do for work?", "a doctor")],
    # Edge N (13): Zara Chicago
    13: [("Where does Maya's sister live?", "Chicago"),
         ("Where does Zara live?", "Chicago")],
    # Edge O (14): MIT graduation
    14: [("Where did Maya graduate from?", "MIT"),
         ("When did Maya graduate?", "2019")],
    # Edge P (15): Stanford masters
    15: [("Where is Maya going for grad school?", "Stanford"),
         ("What is Maya studying?", "a master's program at Stanford")],
    # Edge Q (16): Kai Google
    16: [("Where does Kai work?", "Google"),
         ("What does Kai do?", "Google")],
    # Edge R (17): Kai Portland
    17: [("Where does Kai live?", "Portland")],
    # Edge S (18): shellfish allergy
    18: [("What is Maya allergic to?", "shellfish"),
         ("Does Maya have any allergies?", "shellfish")],
    # Edge T (19): dog Mochi
    19: [("What is Maya's dog's name?", "Mochi"),
         ("Does Maya have a pet?", "a golden retriever named Mochi")],
    # Edge U (20): rock climbing
    20: [("What is Maya's hobby?", "rock climbing"),
         ("What does Maya do for fun?", "rock climbing")],
    # Edge V (21): married to Jordan
    21: [("Who is Maya married to?", "Jordan"),
         ("When did Maya get married?", "March 2023")],
}

# Facts to insert
FACTS = [
    (1, "career::WORK::Maya", "Stripe"),
    (1, "housing::LOCATION::Maya", "Austin, Texas"),
    (1, "health::BE::Maya::allergy", "shellfish"),
    (1, "hobby::PREFERENCE::Maya", "rock climbing"),
    (2, "career::WORK::Kai", "Google"),
    (2, "housing::LOCATION::Kai", "Portland"),
]

# Entities to insert
ENTITIES = [
    (1, "Maya", "PERSON"),
    (1, "Stripe", "ORG"),
    (1, "Shopify", "ORG"),
    (1, "Austin", "LOCATION"),
    (1, "Denver", "LOCATION"),
    (1, "Priya", "PERSON"),
    (1, "Nadia", "PERSON"),
    (1, "Japan", "LOCATION"),
    (1, "Zara", "PERSON"),
    (1, "Chicago", "LOCATION"),
    (1, "MIT", "ORG"),
    (1, "Stanford", "ORG"),
    (1, "Mochi", "ANIMAL"),
    (1, "Jordan", "PERSON"),
    (2, "Kai", "PERSON"),
    (2, "Google", "ORG"),
    (2, "Portland", "LOCATION"),
]

# ── Step 6: Insert data ──
def insert_all():
    conn = sqlite3.connect(TEST_DB_PATH)
    conn.row_factory = sqlite3.Row

    # Insert edges
    edge_ids = []
    for edge in EDGES:
        seq = edge["_seq"]
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
        if key == "housing::LOCATION::Maya":
            history = json.dumps([{"value": "Denver", "at": "2025-01-01", "type": "update"}])
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
    # Temporal disambiguation
    ("Where does Maya work?", 1, "Stripe"),
    ("Where did Maya work before?", 1, "Shopify"),
    ("Where does Maya live?", 1, "Austin"),
    ("Where did Maya used to live?", 1, "Denver"),

    # Social disambiguation
    ("Who is Maya's best friend?", 1, "Priya"),
    ("Who is Maya's friend at work?", 1, "Nadia"),

    # Emotional disambiguation
    ("How does Maya feel about the interview?", 1, "anxious"),
    ("How does Maya feel about the presentation?", 1, "anxious"),
    ("How does Maya feel about the vacation?", 1, "excited"),

    # Cross-domain keyword overlap
    ("What exercise does Maya do?", 1, "10K race"),
    ("What business does Maya run?", 1, "online business pottery"),

    # Multi-entity same family
    ("Who is Maya's sister?", 1, "Zara"),
    ("What does Maya's sister do?", 1, "doctor"),
    ("Where does Maya's sister live?", 1, "Chicago"),

    # Temporal education
    ("Where did Maya graduate from?", 1, "MIT"),
    ("Where is Maya going for grad school?", 1, "Stanford"),

    # Cross-user isolation
    ("Where does Kai work?", 1, "REFUSE"),
    ("Where does Kai work?", 2, "Google"),
    ("Where does Kai live?", 2, "Portland"),

    # Standard facts (should still work)
    ("What is Maya allergic to?", 1, "shellfish"),
    ("What is Maya's dog's name?", 1, "Mochi"),
    ("What is Maya's hobby?", 1, "rock climbing"),
    ("Who is Maya married to?", 1, "Jordan"),
    ("When did Maya get married?", 1, "March 2023"),
    ("What does Maya do for fun?", 1, "rock climbing"),
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

engine = RetrievalEngine(memory_engine=None, temporal_engine=None)

print("\n" + "=" * 70)
print("RUNNING 25 AMBIGUITY QUERIES")
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
    print("\n>>> RETRIEVAL ENGINE HANDLES AMBIGUITY WELL <<<")
elif pct >= 60:
    print("\n>>> RETRIEVAL ENGINE NEEDS IMPROVEMENT ON AMBIGUITY <<<")
else:
    print("\n>>> RETRIEVAL ENGINE HAS CRITICAL AMBIGUITY ISSUES <<<")
