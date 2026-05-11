#!/usr/bin/env python3
"""Retrieval Ambiguity Stress Test.

Inserts 50 edges for ONE user with deliberately overlapping words,
competing entities, and confusable contexts. Then runs 30 queries
where the retrieval engine must pick the right edge from near-duplicates.
"""
import os
import sys
import json
import hashlib

# ── Step 0: Set DB path BEFORE any imports ──
TEST_DB_PATH = "Memory Storage/test_retrieval_ambiguity_hard.db"
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
from app.engines.predicted_queries import generate_predicted_queries

# ── Step 5: Define 50 edges ──
# All user_id=1, speaker="Sam"
EDGES = [
    # ── CAREER (8 edges) — all mention "work" in different senses ──
    # 1
    {
        "user_id": 1, "subject": "user", "predicate": "work_at", "object": "Google",
        "source_text": "I work at Google as a software engineer.",
        "edge_schematic_category": "career", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Google", "Sam"]),
        "subject_type": "PERSON", "object_type": "ORG",
    },
    # 2
    {
        "user_id": 1, "subject": "user", "predicate": "work_on", "object": "a big project",
        "source_text": "I've been working really hard on this project lately.",
        "edge_schematic_category": "career", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 3
    {
        "user_id": 1, "subject": "user", "predicate": "work_at", "object": "Netflix",
        "source_text": "I worked at Netflix before Google.",
        "edge_schematic_category": "career", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 1, "is_current": 0,
        "relational_entities": json.dumps(["Netflix", "Sam"]),
        "subject_type": "PERSON", "object_type": "ORG",
    },
    # 4
    {
        "user_id": 1, "subject": "user", "predicate": "do", "object": "an intense workout",
        "source_text": "My workout routine has been intense this week.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 5
    {
        "user_id": 1, "subject": "user", "predicate": "work_from", "object": "home",
        "source_text": "Working from home has been great for my productivity.",
        "edge_schematic_category": "career", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 6
    {
        "user_id": 1, "subject": "user", "predicate": "take", "object": "a woodworking class",
        "source_text": "The woodworking class I took was really fun.",
        "edge_schematic_category": "hobby", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 7
    {
        "user_id": 1, "subject": "user", "predicate": "work_on", "object": "relationship with dad",
        "source_text": "I need to work on my relationship with my dad.",
        "edge_schematic_category": "family", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 8
    {
        "user_id": 1, "subject": "Jake", "predicate": "get_promoted", "object": "at work",
        "source_text": "My coworker Jake is being promoted.",
        "edge_schematic_category": "career", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Jake", "Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },

    # ── FAMILY (8 edges) — overlapping names ──
    # 9
    {
        "user_id": 1, "subject": "Linda", "predicate": "live_in", "object": "Boston",
        "source_text": "My mom Linda lives in Boston.",
        "edge_schematic_category": "family", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Linda", "Boston", "Sam"]),
        "subject_type": "PERSON", "object_type": "LOCATION",
    },
    # 10
    {
        "user_id": 1, "subject": "Linda", "predicate": "recommend", "object": "a restaurant",
        "source_text": "Linda from work recommended a great restaurant.",
        "edge_schematic_category": "social", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Linda", "Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 11
    {
        "user_id": 1, "subject": "user", "predicate": "call_with", "object": "mom about Thanksgiving",
        "source_text": "My mom called me yesterday about Thanksgiving.",
        "edge_schematic_category": "family", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 12
    {
        "user_id": 1, "subject": "user", "predicate": "tell", "object": "mom about promotion",
        "source_text": "I told my mom I got the promotion.",
        "edge_schematic_category": "family", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 13
    {
        "user_id": 1, "subject": "Sarah", "predicate": "have", "object": "a baby",
        "source_text": "My sister Sarah had a baby.",
        "edge_schematic_category": "family", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sarah", "Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 14
    {
        "user_id": 1, "subject": "Sarah", "predicate": "drop_out", "object": "the race",
        "source_text": "Sarah from my running club dropped out of the race.",
        "edge_schematic_category": "social", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sarah", "Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 15
    {
        "user_id": 1, "subject": "Sam's dad", "predicate": "retire_from", "object": "Boeing",
        "source_text": "My dad retired from Boeing last year.",
        "edge_schematic_category": "family", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": "last year", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Boeing", "Sam"]),
        "subject_type": "PERSON", "object_type": "ORG",
    },
    # 16
    {
        "user_id": 1, "subject": "user", "predicate": "argue_with", "object": "dad",
        "source_text": "My dad and I haven't spoken since the argument.",
        "edge_schematic_category": "family", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": "upset",
        "emotional_target": "dad", "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "PERSON",
    },

    # ── HEALTH (6 edges) — "running" in 6 different senses ──
    # 17
    {
        "user_id": 1, "subject": "user", "predicate": "run", "object": "every morning",
        "source_text": "I've been running every morning for the marathon.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 18
    {
        "user_id": 1, "subject": "user", "predicate": "be", "object": "late for the interview",
        "source_text": "I'm running late for the interview.",
        "edge_schematic_category": "career", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 19
    {
        "user_id": 1, "subject": "user", "predicate": "organize", "object": "the charity event",
        "source_text": "I ran the charity event last weekend.",
        "edge_schematic_category": "social", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 20
    {
        "user_id": 1, "subject": "user", "predicate": "have", "object": "a runny nose",
        "source_text": "My nose has been running all week.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 21
    {
        "user_id": 1, "subject": "user", "predicate": "lose_patience", "object": "with project",
        "source_text": "I'm running out of patience with this project.",
        "edge_schematic_category": "career", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": "frustrated",
        "emotional_target": "project", "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 22
    {
        "user_id": 1, "subject": "user", "predicate": "meet", "object": "Jake",
        "source_text": "I ran into Jake at the coffee shop.",
        "edge_schematic_category": "social", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Jake", "Sam"]),
        "subject_type": "PERSON", "object_type": "PERSON",
    },

    # ── EMOTIONAL (6 edges) — same emotion, different targets + negation ──
    # 23
    {
        "user_id": 1, "subject": "user", "predicate": "feel", "object": "nervous",
        "source_text": "I'm nervous about the Google interview on Tuesday.",
        "edge_schematic_category": "career", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "Tuesday", "edge_emotional_label": "nervous",
        "emotional_target": "Google interview", "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Google", "Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 24
    {
        "user_id": 1, "subject": "user", "predicate": "feel", "object": "nervous",
        "source_text": "I'm nervous about the presentation to the board.",
        "edge_schematic_category": "career", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": "nervous",
        "emotional_target": "presentation", "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 25
    {
        "user_id": 1, "subject": "user", "predicate": "feel", "object": "nervous",
        "source_text": "I'm nervous about introducing Sarah to my parents.",
        "edge_schematic_category": "family", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": "nervous",
        "emotional_target": "introducing Sarah", "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sarah", "Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 26
    {
        "user_id": 1, "subject": "user", "predicate": "feel", "object": "nervous",
        "source_text": "I feel nervous driving in the rain.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": "nervous",
        "emotional_target": "driving", "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 27
    {
        "user_id": 1, "subject": "user", "predicate": "feel", "object": "nervous",
        "source_text": "The nervousness before the marathon is normal.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": "nervous",
        "emotional_target": "marathon", "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 28 — negation
    {
        "user_id": 1, "subject": "user", "predicate": "feel", "object": "not nervous",
        "source_text": "I'm not nervous about the surgery anymore.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": "surgery", "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
        "edge_negated": 1,
    },

    # ── TEMPORAL (6 edges) — same time words, different events ──
    # 29
    {
        "user_id": 1, "subject": "user", "predicate": "move_to", "object": "Portland",
        "source_text": "I'm moving to Portland next month.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "next month", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Portland", "Sam"]),
        "subject_type": "PERSON", "object_type": "LOCATION",
    },
    # 30
    {
        "user_id": 1, "subject": "user", "predicate": "start", "object": "a new diet",
        "source_text": "I'm starting the new diet next month.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "next month", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 31
    {
        "user_id": 1, "subject": "user", "predicate": "interview_at", "object": "Google",
        "source_text": "Last Tuesday I had the interview at Google.",
        "edge_schematic_category": "career", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": "last Tuesday", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Google", "Sam"]),
        "subject_type": "PERSON", "object_type": "ORG",
    },
    # 32
    {
        "user_id": 1, "subject": "user", "predicate": "go_to", "object": "the dentist",
        "source_text": "Last Tuesday I went to the dentist.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": "last Tuesday", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 33
    {
        "user_id": 1, "subject": "user", "predicate": "have", "object": "a headache",
        "source_text": "I've had this headache since Monday.",
        "edge_schematic_category": "health", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "since Monday", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 34
    {
        "user_id": 1, "subject": "user", "predicate": "work_at", "object": "this company",
        "source_text": "I've been at this company since 2020.",
        "edge_schematic_category": "career", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": "since 2020", "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "ORG",
    },

    # ── CONTRADICTIONS (4 edges) — test supersession ──
    # 35 — Google superseded
    {
        "user_id": 1, "subject": "user", "predicate": "work_at", "object": "Google",
        "source_text": "I work at Google.",
        "edge_schematic_category": "career", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 1, "is_current": 0,
        "relational_entities": json.dumps(["Google", "Sam"]),
        "subject_type": "PERSON", "object_type": "ORG",
    },
    # 36 — Apple is most recent
    {
        "user_id": 1, "subject": "user", "predicate": "start_at", "object": "Apple",
        "source_text": "I just started at Apple.",
        "edge_schematic_category": "career", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Apple", "Sam"]),
        "subject_type": "PERSON", "object_type": "ORG",
    },
    # 37 — Portland current
    {
        "user_id": 1, "subject": "user", "predicate": "live_in", "object": "Portland",
        "source_text": "I live in Portland.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Portland", "Sam"]),
        "subject_type": "PERSON", "object_type": "LOCATION",
    },
    # 38 — Portland historical
    {
        "user_id": 1, "subject": "user", "predicate": "live_in", "object": "Portland",
        "source_text": "I used to live in Portland.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 1, "is_current": 0,
        "relational_entities": json.dumps(["Portland", "Sam"]),
        "subject_type": "PERSON", "object_type": "LOCATION",
    },

    # ── VAGUE / SHARED WORDS (6 edges) — "love" in different domains ──
    # 39
    {
        "user_id": 1, "subject": "user", "predicate": "love", "object": "playing guitar",
        "source_text": "I love playing guitar.",
        "edge_schematic_category": "hobby", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 40
    {
        "user_id": 1, "subject": "user", "predicate": "love", "object": "Whiskers",
        "source_text": "I love my cat Whiskers.",
        "edge_schematic_category": "family", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Whiskers", "Sam"]),
        "subject_type": "PERSON", "object_type": "ANIMAL",
    },
    # 41
    {
        "user_id": 1, "subject": "user", "predicate": "love", "object": "Thai food",
        "source_text": "I love Thai food.",
        "edge_schematic_category": "hobby", "edge_episodic_significance": "stative",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 42
    {
        "user_id": 1, "subject": "user", "predicate": "love", "object": "my job",
        "source_text": "I love what I do for work.",
        "edge_schematic_category": "career", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 43
    {
        "user_id": 1, "subject": "user", "predicate": "frustrate", "object": "situation with Jake",
        "source_text": "The thing with Jake was really frustrating.",
        "edge_schematic_category": "social", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": "frustrated",
        "emotional_target": "Jake", "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Jake", "Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 44
    {
        "user_id": 1, "subject": "user", "predicate": "tell", "object": "about boss situation",
        "source_text": "I told you about that situation with my boss.",
        "edge_schematic_category": "career", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },

    # ── META — Jake disambiguation (6 edges with different Jakes) ──
    # 45
    {
        "user_id": 1, "subject": "Jake", "predicate": "get_promoted", "object": "at work",
        "source_text": "Jake from work got promoted.",
        "edge_schematic_category": "career", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Jake", "Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
    # 46
    {
        "user_id": 1, "subject": "user", "predicate": "hike_with", "object": "Jake",
        "source_text": "Jake and I went hiking.",
        "edge_schematic_category": "social", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Jake", "Sam"]),
        "subject_type": "PERSON", "object_type": "PERSON",
    },
    # 47
    {
        "user_id": 1, "subject": "Jake", "predicate": "move_to", "object": "Denver",
        "source_text": "Jake said he's moving to Denver.",
        "edge_schematic_category": "housing", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Jake", "Denver", "Sam"]),
        "subject_type": "PERSON", "object_type": "LOCATION",
    },
    # 48
    {
        "user_id": 1, "subject": "user", "predicate": "have", "object": "a dog named Jake",
        "source_text": "My dog Jake needs to go to the vet.",
        "edge_schematic_category": "family", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Jake", "Sam"]),
        "subject_type": "PERSON", "object_type": "ANIMAL",
    },
    # 49
    {
        "user_id": 1, "subject": "user", "predicate": "meet", "object": "a guy named Jake",
        "source_text": "I met a guy named Jake at the conference.",
        "edge_schematic_category": "social", "edge_episodic_significance": "routine",
        "edge_mood": "indicative", "edge_temporal_context": "past",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Jake", "Sam"]),
        "subject_type": "PERSON", "object_type": "PERSON",
    },
    # 50
    {
        "user_id": 1, "subject": "Jake", "predicate": "expect", "object": "a baby",
        "source_text": "Jake's wife is expecting.",
        "edge_schematic_category": "family", "edge_episodic_significance": "milestone",
        "edge_mood": "indicative", "edge_temporal_context": "present",
        "temporal_expression": None, "edge_emotional_label": None,
        "emotional_target": None, "is_historical": 0, "is_current": 1,
        "relational_entities": json.dumps(["Jake", "Sam"]),
        "subject_type": "PERSON", "object_type": "GENERIC",
    },
]

# Facts to insert
FACTS = [
    (1, "career::WORK::Sam", "Apple"),
    (1, "housing::LOCATION::Sam", "Portland"),
    (1, "hobby::PREFERENCE::Sam", "playing guitar"),
    (1, "family::BE::Sam::mother", "Linda"),
]

# Entities to insert
ENTITIES = [
    (1, "Sam", "PERSON"),
    (1, "Jake", "PERSON"),
    (1, "Linda", "PERSON"),
    (1, "Sarah", "PERSON"),
    (1, "Google", "ORG"),
    (1, "Netflix", "ORG"),
    (1, "Apple", "ORG"),
    (1, "Boeing", "ORG"),
    (1, "Portland", "LOCATION"),
    (1, "Boston", "LOCATION"),
    (1, "Denver", "LOCATION"),
    (1, "Whiskers", "ANIMAL"),
]


# ── Step 6: Insert data ──
def insert_all():
    conn = sqlite3.connect(TEST_DB_PATH)
    conn.row_factory = sqlite3.Row

    # Check which columns exist on relationships
    rel_cols = {row[1] for row in conn.execute("PRAGMA table_info(relationships)").fetchall()}

    edge_ids = []
    for seq, edge in enumerate(EDGES, start=1):
        source_hash = hashlib.sha256(edge["source_text"].encode()).hexdigest()
        edge_emb = embed_text(edge["source_text"]).tobytes()
        pred_emb = embed_text(edge["predicate"].replace("_", " ")).tobytes()

        # Build column list dynamically
        cols = [
            "user_id", "subject", "predicate", "object",
            "source_text", "source_text_hash",
            "edge_schematic_category", "edge_episodic_significance",
            "edge_mood", "edge_temporal_context",
            "temporal_expression", "edge_emotional_label",
            "emotional_target", "is_historical",
            "relational_entities", "is_current",
            "edge_embedding", "predicate_embedding",
            "edge_emotional_valence", "edge_relational_type",
            "confidence", "sequence_number",
            "subject_type", "object_type",
        ]
        vals = [
            edge["user_id"], edge["subject"], edge["predicate"], edge["object"],
            edge["source_text"], source_hash,
            edge["edge_schematic_category"], edge["edge_episodic_significance"],
            edge["edge_mood"], edge["edge_temporal_context"],
            edge.get("temporal_expression"), edge.get("edge_emotional_label"),
            edge.get("emotional_target"), edge["is_historical"],
            edge["relational_entities"], edge["is_current"],
            edge_emb, pred_emb,
            0.5, "personal",
            0.9, seq,
            edge.get("subject_type"), edge.get("object_type"),
        ]

        # Add edge_negated if column exists and edge has it
        if "edge_negated" in rel_cols and edge.get("edge_negated"):
            cols.append("edge_negated")
            vals.append(edge["edge_negated"])

        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)

        conn.execute(f"""
            INSERT INTO relationships ({col_str})
            VALUES ({placeholders})
        """, vals)
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

    # Insert predicted queries via generate_predicted_queries
    pq_count = 0
    for edge_idx, edge in enumerate(EDGES):
        rid = edge_ids[edge_idx]
        uid = edge["user_id"]
        pqs = generate_predicted_queries(
            edge["subject"],
            edge["predicate"],
            edge["object"],
            edge.get("subject_type", "GENERIC"),
            edge.get("object_type", "GENERIC"),
        )
        for question, q_emb in pqs:
            conn.execute("""
                INSERT INTO predicted_queries (
                    relationship_id, user_id, predicted_question,
                    answer_text, answer_subject, question_embedding, confidence
                ) VALUES (?,?,?,?,?,?,?)
            """, (rid, uid, question, edge["object"], edge["subject"],
                  q_emb.tobytes(), 0.9))
            pq_count += 1
    conn.commit()
    print(f"[SETUP] Inserted {pq_count} predicted queries")

    # Insert facts
    for uid, key, value in FACTS:
        fact_emb = embed_text(f"{key} {value}").tobytes()
        conn.execute("""
            INSERT OR REPLACE INTO facts (user_id, key, value, confidence, embedding, history)
            VALUES (?,?,?,0.9,?,?)
        """, (uid, key, value, fact_emb, None))
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
print("INSERTING 50 EDGES (AMBIGUITY STRESS TEST)")
print("=" * 70)
insert_all()

# ── Step 7: Define 30 queries ──
# (query, gold_keywords, description)
queries = [
    # Career — "work" appears in 8+ edges
    ("Where does Sam work?", "Apple", "most recent career, not Google"),
    ("What did Sam do before Apple?", "Google", "historical career"),
    ("What hobby class did Sam take?", "woodworking", "hobby not career"),
    ("Who got promoted at work?", "Jake", "coworker Jake"),

    # Family — two Lindas, two Sarahs
    ("Who is Sam's mom?", "Linda", "mom-Linda not work-Linda"),
    ("Where does Sam's mom live?", "Boston", "multi-hop"),
    ("Who is Sam's sister?", "Sarah", "sister-Sarah not club-Sarah"),
    ("What happened with Sam's sister?", "baby", "sister milestone"),

    # Running — 6 different meanings
    ("What exercise does Sam do?", "running marathon", "literal running"),
    ("What event did Sam organize?", "charity", "ran=organized"),
    ("Is Sam feeling sick?", "nose running runny", "symptom"),

    # Emotional targets — 5 nervous edges
    ("How does Sam feel about the interview?", "nervous", "interview nervous"),
    ("How does Sam feel about the presentation?", "nervous", "presentation nervous"),
    ("How does Sam feel about the surgery?", "not nervous", "negation"),

    # Temporal — same time refs
    ("What is Sam doing next month?", "moving Portland", "not the diet"),
    ("What happened last Tuesday?", "interview Google", "not the dentist"),
    ("How long has Sam been at the company?", "since 2020", "duration"),

    # Contradictions
    ("Where does Sam work now?", "Apple", "supersedes Google"),
    ("Does Sam live in Portland?", "Portland", "current"),

    # Love disambiguation
    ("What is Sam's hobby?", "guitar", "love=hobby not food/pet"),
    ("What food does Sam like?", "Thai", "love=food"),
    ("What pet does Sam have?", "Whiskers cat", "love=pet"),

    # Jake disambiguation
    ("Who got promoted?", "Jake", "work Jake"),
    ("What did Sam do with Jake?", "hiking", "friend Jake"),
    ("Who is moving to Denver?", "Jake", "friend Jake"),
    ("What is Sam's dog's name?", "Jake dog", "dog Jake"),

    # Cross-entity (should refuse or return Sam's edge)
    ("Where does Jake live?", "REFUSE", "Jake not the user"),

    # Vague
    ("What happened with Jake?", "frustrating", "the frustration"),
    ("What's going on at Sam's work?", "started Apple", "recent career"),
    ("What is Sam's relationship with his dad like?", "argument", "emotional family"),
]

# ── F1 scoring ──
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


# ── Step 8: Run queries ──

# Reset singleton so we get a fresh engine
import app.engines.retrieval as _ret_mod
_ret_mod._singleton = None

engine = RetrievalEngine(memory_engine=None, temporal_engine=None)

print("\n" + "=" * 70)
print("RUNNING 30 QUERIES (AMBIGUITY STRESS TEST)")
print("=" * 70 + "\n")

passed = 0
failed = 0
results_by_category = {
    "Career": {"pass": 0, "total": 0},
    "Family": {"pass": 0, "total": 0},
    "Running": {"pass": 0, "total": 0},
    "Emotional": {"pass": 0, "total": 0},
    "Temporal": {"pass": 0, "total": 0},
    "Contradictions": {"pass": 0, "total": 0},
    "Love": {"pass": 0, "total": 0},
    "Jake": {"pass": 0, "total": 0},
    "Cross-entity": {"pass": 0, "total": 0},
    "Vague": {"pass": 0, "total": 0},
}

# Map query index to category
category_ranges = [
    (0, 4, "Career"),
    (4, 8, "Family"),
    (8, 11, "Running"),
    (11, 14, "Emotional"),
    (14, 17, "Temporal"),
    (17, 19, "Contradictions"),
    (19, 22, "Love"),
    (22, 26, "Jake"),
    (26, 27, "Cross-entity"),
    (27, 30, "Vague"),
]

def get_category(idx):
    for start, end, cat in category_ranges:
        if start <= idx < end:
            return cat
    return "Unknown"


# Get top-3 candidates helper
def get_top3_info(result):
    """Extract top-3 candidate info from convergence_details or Answer."""
    top3 = []
    if hasattr(result, 'convergence_details') and result.convergence_details:
        cd = result.convergence_details
        exit_cos = cd.get('exit_cosine', 0.0)
        obj = result.object if hasattr(result, 'object') else None
        top3.append((obj or "?", exit_cos))
    return top3


for i, (query, gold, desc) in enumerate(queries):
    uid = 1
    cat = get_category(i)
    results_by_category[cat]["total"] += 1

    result = engine.retrieve(uid, query)
    is_refusal = isinstance(result, StructuralRefusal)
    answer_text = result.text if result.text else ""

    # Get top-3 candidates by re-running pipeline stages for analysis
    # (we use the convergence details from the main result)
    top3_display = []
    if hasattr(result, 'convergence_details') and result.convergence_details:
        cd = result.convergence_details
        top3_display.append((
            result.object if hasattr(result, 'object') and result.object else "?",
            cd.get('exit_cosine', 0.0)
        ))

    if gold == "REFUSE":
        ok = is_refusal
        status = "PASS" if ok else "FAIL"
        score = 1.0 if ok else 0.0
        detail_answer = f"{'StructuralRefusal' if is_refusal else 'Answer'}: {answer_text[:80]}"
    else:
        if is_refusal:
            ok = False
            status = "FAIL"
            score = 0.0
            detail_answer = f"StructuralRefusal({result.reason})"
        else:
            score = f1(answer_text, gold)
            ok = score >= 0.3
            status = "PASS" if ok else "FAIL"
            detail_answer = answer_text[:80]

    if ok:
        passed += 1
        results_by_category[cat]["pass"] += 1
    else:
        failed += 1

    tag = "PASS" if ok else "FAIL"
    print(f"Q{i+1:02d} [{tag}] \"{query}\"")
    print(f"  answer: \"{detail_answer}\"  gold: \"{gold}\"  F1: {score:.3f}")
    if hasattr(result, 'convergence_details') and result.convergence_details:
        cd = result.convergence_details
        obj_got = result.object if hasattr(result, 'object') and result.object else "?"
        exit_cos = cd.get('exit_cosine', 0.0)
        src_text = cd.get('source_text', '')[:60]
        print(f"  top-1: (\"{obj_got}\", {exit_cos:.3f})  src: \"{src_text}\"")
    print(f"  [{desc}]")
    print()


# ── Step 9: Report ──
total = len(queries)
pct = (passed / total * 100) if total else 0

print("=" * 70)
print(f"SUMMARY: {passed}/{total} passed ({pct:.1f}%)")
print(f"         {failed}/{total} failed")
print("=" * 70)
print()
print("BREAKDOWN BY CATEGORY:")
for cat, counts in results_by_category.items():
    t = counts["total"]
    p = counts["pass"]
    cat_pct = (p / t * 100) if t else 0
    bar = "+" * p + "-" * (t - p)
    print(f"  {cat:15s}: {p}/{t} ({cat_pct:5.1f}%)  [{bar}]")

print()
if pct >= 80:
    print(">>> RETRIEVAL ENGINE HANDLES AMBIGUITY WELL <<<")
elif pct >= 60:
    print(">>> RETRIEVAL ENGINE NEEDS AMBIGUITY IMPROVEMENTS <<<")
elif pct >= 40:
    print(">>> RETRIEVAL ENGINE HAS SIGNIFICANT AMBIGUITY ISSUES <<<")
else:
    print(">>> RETRIEVAL ENGINE HAS CRITICAL AMBIGUITY FAILURES <<<")
