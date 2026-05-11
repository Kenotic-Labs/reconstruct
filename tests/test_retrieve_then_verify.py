#!/usr/bin/env python3
"""Retrieve-Then-Verify loop test.

ROOT CAUSE: The moat pipeline scores 6/30 (20%) on the hard ambiguity test
because exit_cosine = max(pq_cosine, edge_cosine, predicate_cosine) produces
near-identical scores when 50 edges share overlapping vocabulary. The pipeline
picks top-1 by cosine with no check that the answer is grounded in the DB.

FIX UNDER TEST: After ranking, verify each candidate against the DB using
structured lookups (facts table, triple existence, schema+entity). Return
the first verified candidate, or StructuralRefusal if none verify.

Research grounding:
  - BASEBALL (1966): DB lookup is truth
  - TREC QA (1999): answer validation = recognizing when answer isn't in corpus
  - CLEF AVE (2006): (Q, A, Text) -> verified?
  - FactKG (ACL 2023): one-hop triple existence check
  - RVR (2025): retrieve-verify-retrieve multi-round loop

This is a TEST SCRIPT, not production code.
"""
import os
import sys
import json
import hashlib
import sqlite3

# ── Step 0: Set DB path BEFORE any imports ──
TEST_DB_PATH = "Memory Storage/test_ambiguity_hard.db"
os.environ["NURA_SQLITE_PATH"] = TEST_DB_PATH
os.environ["RAYA_EMBED_DEVICE"] = "cpu"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Step 1: Delete old DB and recreate ──
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)
    print(f"[SETUP] Deleted old DB: {TEST_DB_PATH}")
for suffix in ("-wal", "-shm"):
    p = TEST_DB_PATH + suffix
    if os.path.exists(p):
        os.remove(p)

from config.settings import settings
settings.sqlite_path = TEST_DB_PATH

from app.db.session import init_db
init_db(TEST_DB_PATH)
print(f"[SETUP] Created fresh DB: {TEST_DB_PATH}")

# ── Step 2: Import engines ──
import numpy as np
from app.vector.embedder import embed_text
from app.engines.retrieval import RetrievalEngine, Answer, StructuralRefusal
from app.engines.retrieval_types import Candidate
from app.engines.predicted_queries import generate_predicted_queries

# ── Step 3: 50 edges (same dataset as test_retrieval_ambiguity_hard.py) ──

EDGES = [
    {"user_id": 1, "subject": "user", "predicate": "work_at", "object": "Google",
     "source_text": "I work at Google as a software engineer.",
     "edge_schematic_category": "career", "edge_episodic_significance": "stative",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Google", "Sam"]),
     "subject_type": "PERSON", "object_type": "ORG"},
    {"user_id": 1, "subject": "user", "predicate": "work_on", "object": "a big project",
     "source_text": "I've been working really hard on this project lately.",
     "edge_schematic_category": "career", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "work_at", "object": "Netflix",
     "source_text": "I worked at Netflix before Google.",
     "edge_schematic_category": "career", "edge_episodic_significance": "stative",
     "edge_mood": "indicative", "edge_temporal_context": "past",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 1, "is_current": 0,
     "relational_entities": json.dumps(["Netflix", "Sam"]),
     "subject_type": "PERSON", "object_type": "ORG"},
    {"user_id": 1, "subject": "user", "predicate": "do", "object": "an intense workout",
     "source_text": "My workout routine has been intense this week.",
     "edge_schematic_category": "health", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "work_from", "object": "home",
     "source_text": "Working from home has been great for my productivity.",
     "edge_schematic_category": "career", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "take", "object": "a woodworking class",
     "source_text": "The woodworking class I took was really fun.",
     "edge_schematic_category": "hobby", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "work_on", "object": "relationship with dad",
     "source_text": "I need to work on my relationship with my dad.",
     "edge_schematic_category": "family", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "Jake", "predicate": "get_promoted", "object": "at work",
     "source_text": "My coworker Jake is being promoted.",
     "edge_schematic_category": "career", "edge_episodic_significance": "milestone",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Jake", "Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "Linda", "predicate": "live_in", "object": "Boston",
     "source_text": "My mom Linda lives in Boston.",
     "edge_schematic_category": "family", "edge_episodic_significance": "stative",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Linda", "Boston", "Sam"]),
     "subject_type": "PERSON", "object_type": "LOCATION"},
    {"user_id": 1, "subject": "Linda", "predicate": "recommend", "object": "a restaurant",
     "source_text": "Linda from work recommended a great restaurant.",
     "edge_schematic_category": "social", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Linda", "Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "call_with", "object": "mom about Thanksgiving",
     "source_text": "My mom called me yesterday about Thanksgiving.",
     "edge_schematic_category": "family", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "tell", "object": "mom about promotion",
     "source_text": "I told my mom I got the promotion.",
     "edge_schematic_category": "family", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "Sarah", "predicate": "have", "object": "a baby",
     "source_text": "My sister Sarah had a baby.",
     "edge_schematic_category": "family", "edge_episodic_significance": "milestone",
     "edge_mood": "indicative", "edge_temporal_context": "past",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sarah", "Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "Sarah", "predicate": "drop_out", "object": "the race",
     "source_text": "Sarah from my running club dropped out of the race.",
     "edge_schematic_category": "social", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sarah", "Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "Sam's dad", "predicate": "retire_from", "object": "Boeing",
     "source_text": "My dad retired from Boeing last year.",
     "edge_schematic_category": "family", "edge_episodic_significance": "milestone",
     "edge_mood": "indicative", "edge_temporal_context": "past",
     "temporal_expression": "last year", "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Boeing", "Sam"]),
     "subject_type": "PERSON", "object_type": "ORG"},
    {"user_id": 1, "subject": "user", "predicate": "argue_with", "object": "dad",
     "source_text": "My dad and I haven't spoken since the argument.",
     "edge_schematic_category": "family", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": "upset",
     "emotional_target": "dad", "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "PERSON"},
    {"user_id": 1, "subject": "user", "predicate": "run", "object": "every morning",
     "source_text": "I've been running every morning for the marathon.",
     "edge_schematic_category": "health", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "be", "object": "late for the interview",
     "source_text": "I'm running late for the interview.",
     "edge_schematic_category": "career", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "organize", "object": "the charity event",
     "source_text": "I ran the charity event last weekend.",
     "edge_schematic_category": "social", "edge_episodic_significance": "milestone",
     "edge_mood": "indicative", "edge_temporal_context": "past",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "have", "object": "a runny nose",
     "source_text": "My nose has been running all week.",
     "edge_schematic_category": "health", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "lose_patience", "object": "with project",
     "source_text": "I'm running out of patience with this project.",
     "edge_schematic_category": "career", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": "frustrated",
     "emotional_target": "project", "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "meet", "object": "Jake",
     "source_text": "I ran into Jake at the coffee shop.",
     "edge_schematic_category": "social", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Jake", "Sam"]),
     "subject_type": "PERSON", "object_type": "PERSON"},
    {"user_id": 1, "subject": "user", "predicate": "feel", "object": "nervous",
     "source_text": "I'm nervous about the Google interview on Tuesday.",
     "edge_schematic_category": "career", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": "Tuesday", "edge_emotional_label": "nervous",
     "emotional_target": "Google interview", "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Google", "Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "feel", "object": "nervous",
     "source_text": "I'm nervous about the presentation to the board.",
     "edge_schematic_category": "career", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": "nervous",
     "emotional_target": "presentation", "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "feel", "object": "nervous",
     "source_text": "I'm nervous about introducing Sarah to my parents.",
     "edge_schematic_category": "family", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": "nervous",
     "emotional_target": "introducing Sarah", "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sarah", "Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "feel", "object": "nervous",
     "source_text": "I feel nervous driving in the rain.",
     "edge_schematic_category": "health", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": "nervous",
     "emotional_target": "driving", "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "feel", "object": "nervous",
     "source_text": "The nervousness before the marathon is normal.",
     "edge_schematic_category": "health", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": "nervous",
     "emotional_target": "marathon", "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "feel", "object": "not nervous",
     "source_text": "I'm not nervous about the surgery anymore.",
     "edge_schematic_category": "health", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": "surgery", "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC",
     "edge_negated": 1},
    {"user_id": 1, "subject": "user", "predicate": "move_to", "object": "Portland",
     "source_text": "I'm moving to Portland next month.",
     "edge_schematic_category": "housing", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": "next month", "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Portland", "Sam"]),
     "subject_type": "PERSON", "object_type": "LOCATION"},
    {"user_id": 1, "subject": "user", "predicate": "start", "object": "a new diet",
     "source_text": "I'm starting the new diet next month.",
     "edge_schematic_category": "health", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": "next month", "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "interview_at", "object": "Google",
     "source_text": "Last Tuesday I had the interview at Google.",
     "edge_schematic_category": "career", "edge_episodic_significance": "milestone",
     "edge_mood": "indicative", "edge_temporal_context": "past",
     "temporal_expression": "last Tuesday", "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Google", "Sam"]),
     "subject_type": "PERSON", "object_type": "ORG"},
    {"user_id": 1, "subject": "user", "predicate": "go_to", "object": "the dentist",
     "source_text": "Last Tuesday I went to the dentist.",
     "edge_schematic_category": "health", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "past",
     "temporal_expression": "last Tuesday", "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "have", "object": "a headache",
     "source_text": "I've had this headache since Monday.",
     "edge_schematic_category": "health", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": "since Monday", "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "work_at", "object": "this company",
     "source_text": "I've been at this company since 2020.",
     "edge_schematic_category": "career", "edge_episodic_significance": "stative",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": "since 2020", "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "ORG"},
    {"user_id": 1, "subject": "user", "predicate": "work_at", "object": "Google",
     "source_text": "I work at Google.",
     "edge_schematic_category": "career", "edge_episodic_significance": "stative",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 1, "is_current": 0,
     "relational_entities": json.dumps(["Google", "Sam"]),
     "subject_type": "PERSON", "object_type": "ORG"},
    {"user_id": 1, "subject": "user", "predicate": "start_at", "object": "Apple",
     "source_text": "I just started at Apple.",
     "edge_schematic_category": "career", "edge_episodic_significance": "milestone",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Apple", "Sam"]),
     "subject_type": "PERSON", "object_type": "ORG"},
    {"user_id": 1, "subject": "user", "predicate": "live_in", "object": "Portland",
     "source_text": "I live in Portland.",
     "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Portland", "Sam"]),
     "subject_type": "PERSON", "object_type": "LOCATION"},
    {"user_id": 1, "subject": "user", "predicate": "live_in", "object": "Portland",
     "source_text": "I used to live in Portland.",
     "edge_schematic_category": "housing", "edge_episodic_significance": "stative",
     "edge_mood": "indicative", "edge_temporal_context": "past",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 1, "is_current": 0,
     "relational_entities": json.dumps(["Portland", "Sam"]),
     "subject_type": "PERSON", "object_type": "LOCATION"},
    {"user_id": 1, "subject": "user", "predicate": "love", "object": "playing guitar",
     "source_text": "I love playing guitar.",
     "edge_schematic_category": "hobby", "edge_episodic_significance": "stative",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "love", "object": "Whiskers",
     "source_text": "I love my cat Whiskers.",
     "edge_schematic_category": "family", "edge_episodic_significance": "stative",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Whiskers", "Sam"]),
     "subject_type": "PERSON", "object_type": "ANIMAL"},
    {"user_id": 1, "subject": "user", "predicate": "love", "object": "Thai food",
     "source_text": "I love Thai food.",
     "edge_schematic_category": "hobby", "edge_episodic_significance": "stative",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "love", "object": "my job",
     "source_text": "I love what I do for work.",
     "edge_schematic_category": "career", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "frustrate", "object": "situation with Jake",
     "source_text": "The thing with Jake was really frustrating.",
     "edge_schematic_category": "social", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": "frustrated",
     "emotional_target": "Jake", "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Jake", "Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "tell", "object": "about boss situation",
     "source_text": "I told you about that situation with my boss.",
     "edge_schematic_category": "career", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "Jake", "predicate": "get_promoted", "object": "at work",
     "source_text": "Jake from work got promoted.",
     "edge_schematic_category": "career", "edge_episodic_significance": "milestone",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Jake", "Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
    {"user_id": 1, "subject": "user", "predicate": "hike_with", "object": "Jake",
     "source_text": "Jake and I went hiking.",
     "edge_schematic_category": "social", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "past",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Jake", "Sam"]),
     "subject_type": "PERSON", "object_type": "PERSON"},
    {"user_id": 1, "subject": "Jake", "predicate": "move_to", "object": "Denver",
     "source_text": "Jake said he's moving to Denver.",
     "edge_schematic_category": "housing", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Jake", "Denver", "Sam"]),
     "subject_type": "PERSON", "object_type": "LOCATION"},
    {"user_id": 1, "subject": "user", "predicate": "have", "object": "a dog named Jake",
     "source_text": "My dog Jake needs to go to the vet.",
     "edge_schematic_category": "family", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Jake", "Sam"]),
     "subject_type": "PERSON", "object_type": "ANIMAL"},
    {"user_id": 1, "subject": "user", "predicate": "meet", "object": "a guy named Jake",
     "source_text": "I met a guy named Jake at the conference.",
     "edge_schematic_category": "social", "edge_episodic_significance": "routine",
     "edge_mood": "indicative", "edge_temporal_context": "past",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Jake", "Sam"]),
     "subject_type": "PERSON", "object_type": "PERSON"},
    {"user_id": 1, "subject": "Jake", "predicate": "expect", "object": "a baby",
     "source_text": "Jake's wife is expecting.",
     "edge_schematic_category": "family", "edge_episodic_significance": "milestone",
     "edge_mood": "indicative", "edge_temporal_context": "present",
     "temporal_expression": None, "edge_emotional_label": None,
     "emotional_target": None, "is_historical": 0, "is_current": 1,
     "relational_entities": json.dumps(["Jake", "Sam"]),
     "subject_type": "PERSON", "object_type": "GENERIC"},
]

FACTS = [
    (1, "career::WORK::Sam", "Apple"),
    (1, "housing::LOCATION::Sam", "Portland"),
    (1, "hobby::PREFERENCE::Sam", "playing guitar"),
    (1, "family::BE::Sam::mother", "Linda"),
]

ENTITIES = [
    (1, "Sam", "PERSON"), (1, "Jake", "PERSON"), (1, "Linda", "PERSON"),
    (1, "Sarah", "PERSON"), (1, "Google", "ORG"), (1, "Netflix", "ORG"),
    (1, "Apple", "ORG"), (1, "Boeing", "ORG"), (1, "Portland", "LOCATION"),
    (1, "Boston", "LOCATION"), (1, "Denver", "LOCATION"), (1, "Whiskers", "ANIMAL"),
]


def insert_all():
    conn = sqlite3.connect(TEST_DB_PATH)
    conn.row_factory = sqlite3.Row
    rel_cols = {row[1] for row in conn.execute("PRAGMA table_info(relationships)").fetchall()}
    edge_ids = []
    for seq, edge in enumerate(EDGES, start=1):
        source_hash = hashlib.sha256(edge["source_text"].encode()).hexdigest()
        edge_emb = embed_text(edge["source_text"]).tobytes()
        pred_emb = embed_text(edge["predicate"].replace("_", " ")).tobytes()
        cols = [
            "user_id", "subject", "predicate", "object", "source_text", "source_text_hash",
            "edge_schematic_category", "edge_episodic_significance", "edge_mood",
            "edge_temporal_context", "temporal_expression", "edge_emotional_label",
            "emotional_target", "is_historical", "relational_entities", "is_current",
            "edge_embedding", "predicate_embedding", "edge_emotional_valence",
            "edge_relational_type", "confidence", "sequence_number", "subject_type", "object_type",
        ]
        vals = [
            edge["user_id"], edge["subject"], edge["predicate"], edge["object"],
            edge["source_text"], source_hash, edge["edge_schematic_category"],
            edge["edge_episodic_significance"], edge["edge_mood"], edge["edge_temporal_context"],
            edge.get("temporal_expression"), edge.get("edge_emotional_label"),
            edge.get("emotional_target"), edge["is_historical"], edge["relational_entities"],
            edge["is_current"], edge_emb, pred_emb, 0.5, "personal", 0.9, seq,
            edge.get("subject_type"), edge.get("object_type"),
        ]
        if "edge_negated" in rel_cols and edge.get("edge_negated"):
            cols.append("edge_negated")
            vals.append(edge["edge_negated"])
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        conn.execute(f"INSERT INTO relationships ({col_str}) VALUES ({placeholders})", vals)
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        edge_ids.append(rid)
        conn.execute(
            "INSERT INTO relationships_fts(rowid, subject, predicate, object, source_text) VALUES (?, ?, ?, ?, ?)",
            (rid, edge["subject"], edge["predicate"].replace("_", " "), edge["object"], edge["source_text"]),
        )
    conn.commit()
    print(f"[SETUP] Inserted {len(edge_ids)} edges")

    pq_count = 0
    for edge_idx, edge in enumerate(EDGES):
        rid = edge_ids[edge_idx]
        pqs = generate_predicted_queries(
            edge["subject"], edge["predicate"], edge["object"],
            edge.get("subject_type", "GENERIC"), edge.get("object_type", "GENERIC"),
        )
        for question, q_emb in pqs:
            conn.execute(
                "INSERT INTO predicted_queries (relationship_id, user_id, predicted_question, answer_text, answer_subject, question_embedding, confidence) VALUES (?,?,?,?,?,?,?)",
                (rid, edge["user_id"], question, edge["object"], edge["subject"], q_emb.tobytes(), 0.9),
            )
            pq_count += 1
    conn.commit()
    print(f"[SETUP] Inserted {pq_count} predicted queries")

    for uid, key, value in FACTS:
        fact_emb = embed_text(f"{key} {value}").tobytes()
        conn.execute(
            "INSERT OR REPLACE INTO facts (user_id, key, value, confidence, embedding, history) VALUES (?,?,?,0.9,?,?)",
            (uid, key, value, fact_emb, None),
        )
    conn.commit()
    print(f"[SETUP] Inserted {len(FACTS)} facts")

    for uid, name, etype in ENTITIES:
        ent_emb = embed_text(name).tobytes()
        conn.execute(
            "INSERT OR REPLACE INTO entities (user_id, name, entity_type, embedding, mention_count) VALUES (?,?,?,?,1)",
            (uid, name, etype, ent_emb),
        )
    conn.commit()
    print(f"[SETUP] Inserted {len(ENTITIES)} entities")
    conn.close()


QUERIES = [
    ("Where does Sam work?", "Apple", "most recent career, not Google"),
    ("What did Sam do before Apple?", "Google", "historical career"),
    ("What hobby class did Sam take?", "woodworking", "hobby not career"),
    ("Who got promoted at work?", "Jake", "coworker Jake"),
    ("Who is Sam's mom?", "Linda", "mom-Linda not work-Linda"),
    ("Where does Sam's mom live?", "Boston", "multi-hop"),
    ("Who is Sam's sister?", "Sarah", "sister-Sarah not club-Sarah"),
    ("What happened with Sam's sister?", "baby", "sister milestone"),
    ("What exercise does Sam do?", "running marathon", "literal running"),
    ("What event did Sam organize?", "charity", "ran=organized"),
    ("Is Sam feeling sick?", "nose running runny", "symptom"),
    ("How does Sam feel about the interview?", "nervous", "interview nervous"),
    ("How does Sam feel about the presentation?", "nervous", "presentation nervous"),
    ("How does Sam feel about the surgery?", "not nervous", "negation"),
    ("What is Sam doing next month?", "moving Portland", "not the diet"),
    ("What happened last Tuesday?", "interview Google", "not the dentist"),
    ("How long has Sam been at the company?", "since 2020", "duration"),
    ("Where does Sam work now?", "Apple", "supersedes Google"),
    ("Does Sam live in Portland?", "Portland", "current"),
    ("What is Sam's hobby?", "guitar", "love=hobby not food/pet"),
    ("What food does Sam like?", "Thai", "love=food"),
    ("What pet does Sam have?", "Whiskers cat", "love=pet"),
    ("Who got promoted?", "Jake", "work Jake"),
    ("What did Sam do with Jake?", "hiking", "friend Jake"),
    ("Who is moving to Denver?", "Jake", "friend Jake"),
    ("What is Sam's dog's name?", "Jake dog", "dog Jake"),
    ("Where does Jake live?", "REFUSE", "Jake not the user"),
    ("What happened with Jake?", "frustrating", "the frustration"),
    ("What's going on at Sam's work?", "started Apple", "recent career"),
    ("What is Sam's relationship with his dad like?", "argument", "emotional family"),
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


# ============================================================================
# VERIFICATION -- pure structured DB lookups, no cosine, no LLM
# ============================================================================

def verify_candidate(conn, user_id, candidate_edge, query_info):
    """Verify a candidate answer is grounded in stored facts.

    ROOT CAUSE OF PRIOR 0-DELTA: Strategy 2 used
        SELECT COUNT(*) FROM relationships WHERE id = ?
    which is vacuous -- every candidate IS a real DB row. The loop
    always verified rank-0 and never tried alternatives.

    FIX: All strategies now check CONTENT ALIGNMENT via SQL:

    Strategy 1: FACTS TABLE -- look up facts by query's inferred schema.
                If a fact exists, candidate's object must match its value.

    Strategy 2: SCHEMA-SCOPED TRIPLE -- candidate's row ID must exist
                in a relationship whose schema matches the QUERY's schema.
                SQL: WHERE id = ? AND edge_schematic_category = ?

    Strategy 3: ENTITY-SCOPED EXISTENCE -- candidate must reference the
                query's resolved entity in subject/object/relational_entities.
    """
    subj = (candidate_edge.get("subject") or "").strip()
    pred = (candidate_edge.get("predicate") or "").strip()
    obj = (candidate_edge.get("object") or "").strip()
    cand_schema = (candidate_edge.get("edge_schematic_category") or "").strip()
    source_text = (candidate_edge.get("source_text") or "").strip()
    rid = candidate_edge.get("id") or candidate_edge.get("relationship_id")

    q_schema = (query_info.get("match_schema") or "").strip()
    q_entity = (query_info.get("match_entity") or "").strip()
    q_object = (query_info.get("match_object") or "").strip()

    # Diagnosis from runs #1-#3:
    # Run #1: triple_existence vacuous (every candidate IS a DB row) -> 0 delta
    # Run #2: facts_contradict rejected correct candidates (Q04/Q06/Q23/Q25)
    # Run #3: schema_mismatch rejected correct candidates because
    #   classify_query infers wrong schemas (hobby for career promotions)
    #
    # Root cause: REJECTION-based verification fails because:
    #   (a) classify_query schema inference is wrong ~40% of the time
    #   (b) facts may answer different questions in the same schema
    # Only POSITIVE verification works reliably.

    # ── Strategy 1: FACTS TABLE -- positive match only ──
    # Search ALL facts for this user. If a fact value appears in the
    # candidate's object or source_text AND the fact's schema matches
    # the candidate's schema, the candidate is positively verified.
    fact_rows = conn.execute(
        "SELECT key, value FROM facts WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    for frow in fact_rows:
        fv = (frow[1] or "").lower()
        fk = (frow[0] or "").lower()
        if not fv:
            continue
        obj_l = obj.lower()
        src_l = source_text.lower()
        if fv in obj_l or fv in src_l or obj_l in fv:
            # Require the fact's schema prefix to match the candidate's schema
            fact_schema = fk.split("::")[0] if "::" in fk else ""
            if fact_schema and cand_schema and fact_schema == cand_schema:
                return True, "facts_table"

    # ── Strategy 2: SCHEMA ALIGNMENT -- positive only ──
    # If query schema matches candidate schema, that's a positive signal.
    # Never reject for mismatch (schema inference unreliable).
    if q_schema and cand_schema and q_schema == cand_schema and rid:
        row = conn.execute(
            """SELECT COUNT(*) FROM relationships
               WHERE id = ? AND user_id = ?
               AND COALESCE(is_current, 1) = 1
               AND tombstoned_at IS NULL""",
            (rid, user_id),
        ).fetchone()
        if row and row[0] > 0:
            return True, "schema_match"

    # ── Strategy 3: ENTITY-SCOPED -- positive only ──
    # If query mentions an entity and candidate references it.
    if q_entity and rid:
        row = conn.execute(
            """SELECT COUNT(*) FROM relationships
               WHERE id = ? AND user_id = ?
               AND (subject = ? OR object = ?
                    OR relational_entities LIKE ?)
               AND COALESCE(is_current, 1) = 1
               AND tombstoned_at IS NULL""",
            (rid, user_id, q_entity, q_entity, f"%{q_entity}%"),
        ).fetchone()
        if row and row[0] > 0:
            return True, "entity_scoped"

    # ── No positive verification -- return False ──
    # Loop will try the next candidate. If none verify, StructuralRefusal.
    return False, "none"


def extract_query_info(query_text):
    try:
        from app.engines.grammar_engine import classify_query
        qd = classify_query(query_text)
        return {
            "wh_word": qd.wh_word, "match_subject": qd.match_subject,
            "match_predicate": qd.match_predicate, "match_object": qd.match_object,
            "match_entity": qd.match_entity, "match_schema": qd.match_schema,
            "return_field": qd.return_field,
        }
    except Exception:
        return {"wh_word": None, "match_subject": None, "match_predicate": None,
                "match_object": None, "match_entity": None, "match_schema": None,
                "return_field": "episodic"}


def extract_answer_text(query_text, candidate_edge):
    from app.engines.wh_type import parse_expected_answer_type
    subj = candidate_edge.get("subject") or ""
    obj = candidate_edge.get("object") or ""
    source_text = (candidate_edge.get("source_text") or "").strip()
    expected_type = parse_expected_answer_type(query_text)
    if expected_type == "TIME":
        answer_text = candidate_edge.get("temporal_expression") or candidate_edge.get("resolved_event_date") or obj
    elif expected_type == "PERSON":
        answer_text = obj if subj.lower() == "user" else subj
    elif expected_type == "LOCATION":
        answer_text = obj or subj
    else:
        answer_text = obj
    if not answer_text or len(answer_text.strip()) <= 1:
        answer_text = source_text
    return answer_text


# ============================================================================
# RETRIEVE-THEN-VERIFY LOOP
# ============================================================================

def retrieve_with_verification(engine, user_id, query_text, conn):
    """Run stages 1-5, then verify each candidate against the DB."""
    from app.vector import embedder as _embedder_module

    query_text = (query_text or "").strip()
    if not query_text:
        return StructuralRefusal(reason="empty_query"), None, "none", -1

    fact_answer = engine._try_facts_lookup(user_id, query_text)
    if fact_answer is not None:
        return fact_answer, None, "facts_fast_path", 0

    try:
        q_emb = _embedder_module.embed_text(query_text)
    except Exception:
        return StructuralRefusal(reason="embed_failed"), None, "none", -1

    candidates = engine._stage1_entry(user_id, q_emb)
    candidates = engine._stage2_expand(user_id, query_text, candidates)
    if not candidates:
        return StructuralRefusal(reason="no_candidates"), None, "none", -1
    candidates = engine._stage3_group(candidates)
    candidates = engine._stage4_relate(user_id, query_text, candidates)
    if not candidates:
        return StructuralRefusal(reason="no_structural_match"), None, "none", -1
    candidates = engine._stage5_exit(q_emb, candidates)
    candidates.sort(key=lambda c: (-c.exit_cosine, -(c.edge.get("sequence_number") or 0)))

    query_info = extract_query_info(query_text)

    for rank, cand in enumerate(candidates):
        verified, strategy = verify_candidate(conn, user_id, cand.edge, query_info)
        if verified:
            answer_text = extract_answer_text(query_text, cand.edge)
            return Answer(
                text=answer_text,
                subject=cand.edge.get("subject") or "",
                predicate=cand.edge.get("predicate") or "",
                object=cand.edge.get("object") or "",
                confidence=1.0, source="verified_pipeline",
                survivors=len(candidates),
                convergence_details={
                    "query": query_text, "exit_cosine": cand.exit_cosine,
                    "verification_strategy": strategy, "verification_rank": rank,
                    "source_text": (cand.edge.get("source_text") or "").strip(),
                    "total_candidates": len(candidates),
                },
            ), cand, strategy, rank

    return StructuralRefusal(
        reason="no_verified_candidate",
        convergence_details={"query": query_text, "candidates_checked": len(candidates)},
    ), None, "none", -1


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("INSERTING 50 EDGES (AMBIGUITY STRESS TEST)")
    print("=" * 70)
    insert_all()

    import app.engines.retrieval as _ret_mod
    _ret_mod._singleton = None
    engine = RetrievalEngine(memory_engine=None, temporal_engine=None)
    conn = sqlite3.connect(TEST_DB_PATH)
    conn.row_factory = sqlite3.Row

    print("\n" + "=" * 70)
    print("RUNNING 30 QUERIES: MOAT-ONLY vs RETRIEVE-THEN-VERIFY")
    print("=" * 70 + "\n")

    cat_ranges = [
        (0, 4, "Career"), (4, 8, "Family"), (8, 11, "Running"),
        (11, 14, "Emotional"), (14, 17, "Temporal"), (17, 19, "Contradictions"),
        (19, 22, "Love"), (22, 26, "Jake"), (26, 27, "Cross-entity"), (27, 30, "Vague"),
    ]

    def get_cat(idx):
        for s, e, c in cat_ranges:
            if s <= idx < e:
                return c
        return "Unknown"

    moat_passed = 0
    verify_passed = 0
    total = len(QUERIES)
    detailed = []

    print(f"{'#':>3} | {'Query':<45} | {'Moat top-1':<25} | {'V?':>2} | {'Verified answer':<25} | {'OK':>4} | {'Strategy':<18}")
    print("-" * 140)

    for i, (query, gold, desc) in enumerate(QUERIES):
        uid = 1
        cat = get_cat(i)

        moat_result = engine.retrieve(uid, query)
        moat_refusal = isinstance(moat_result, StructuralRefusal)
        moat_text = moat_result.text if moat_result.text else ""
        if gold == "REFUSE":
            moat_ok = moat_refusal
        elif moat_refusal:
            moat_ok = False
        else:
            moat_ok = f1(moat_text, gold) >= 0.3
        if moat_ok:
            moat_passed += 1

        vr, vc, vs, vrank = retrieve_with_verification(engine, uid, query, conn)
        vr_refusal = isinstance(vr, StructuralRefusal)
        vr_text = vr.text if vr.text else ""
        if gold == "REFUSE":
            vr_ok = vr_refusal
        elif vr_refusal:
            vr_ok = False
        else:
            vr_ok = f1(vr_text, gold) >= 0.3
        if vr_ok:
            verify_passed += 1

        md = "REFUSE" if moat_refusal else moat_text[:25]
        vd = "REFUSE" if vr_refusal else vr_text[:25]
        delta = ""
        if vr_ok and not moat_ok:
            delta = " <<<FIXED"
        elif not vr_ok and moat_ok:
            delta = " <<<REGRESSED"

        print(f"Q{i+1:02d} | {query:<45} | {md:<25} | {'Y' if vr_ok else 'N':>2} | {vd:<25} | {'PASS' if vr_ok else 'FAIL':>4} | {vs:<18}{delta}")

        detailed.append({
            "query_num": i + 1, "query": query, "gold": gold, "category": cat,
            "desc": desc, "moat_answer": moat_text[:60] if not moat_refusal else "REFUSE",
            "moat_ok": moat_ok, "verify_answer": vr_text[:60] if not vr_refusal else "REFUSE",
            "verify_ok": vr_ok, "verify_strategy": vs, "verify_rank": vrank,
        })

    conn.close()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    mp = moat_passed / total * 100
    vp = verify_passed / total * 100
    dc = verify_passed - moat_passed
    print(f"  MOAT-ONLY:             {moat_passed}/{total} ({mp:.1f}%)")
    print(f"  RETRIEVE-THEN-VERIFY:  {verify_passed}/{total} ({vp:.1f}%)")
    print(f"  DELTA:                 {'+' if dc >= 0 else ''}{dc} ({'+' if (vp-mp) >= 0 else ''}{vp-mp:.1f}%)")

    print("\nBY CATEGORY:")
    cm, cv, ct = {}, {}, {}
    for r in detailed:
        c = r["category"]
        cm[c] = cm.get(c, 0) + (1 if r["moat_ok"] else 0)
        cv[c] = cv.get(c, 0) + (1 if r["verify_ok"] else 0)
        ct[c] = ct.get(c, 0) + 1
    for c in dict.fromkeys(r["category"] for r in detailed):
        d = cv[c] - cm[c]
        print(f"  {c:15s}: moat={cm[c]}/{ct[c]}  verify={cv[c]}/{ct[c]}  delta={'+' if d >= 0 else ''}{d}")

    print("\nVERIFICATION STRATEGY USAGE:")
    sc = {}
    for r in detailed:
        sc[r["verify_strategy"]] = sc.get(r["verify_strategy"], 0) + 1
    for s, c in sorted(sc.items(), key=lambda x: -x[1]):
        print(f"  {s:20s}: {c}")

    fixed = [r for r in detailed if r["verify_ok"] and not r["moat_ok"]]
    regressed = [r for r in detailed if not r["verify_ok"] and r["moat_ok"]]
    if fixed:
        print(f"\nFIXED BY VERIFICATION ({len(fixed)}):")
        for r in fixed:
            print(f"  Q{r['query_num']:02d}: {r['query']}  [{r['verify_strategy']}]")
    if regressed:
        print(f"\nREGRESSED BY VERIFICATION ({len(regressed)}):")
        for r in regressed:
            print(f"  Q{r['query_num']:02d}: {r['query']}")

    failures = [r for r in detailed if not r["verify_ok"]]
    if failures:
        print(f"\nFAILURE LAYER ATTRIBUTION ({len(failures)}):")
        for r in failures:
            if r["gold"] == "REFUSE" and not r["verify_ok"]:
                layer = "missing_refusal_gate"
            elif r["verify_strategy"] == "none" and r["verify_answer"] == "REFUSE":
                layer = "false_refusal"
            elif r["verify_rank"] == -1:
                layer = "no_candidate_verified"
            else:
                layer = "wrong_ranking"
            print(f"  Q{r['query_num']:02d}: {r['query'][:40]:<40s}  layer={layer}")


if __name__ == "__main__":
    main()
