"""
Scale test: 1,000 edges across 5 life domains.
Validates trace-scoped retrieval latency and correctness at scale.

Root cause under test: RetrievalEngine uses SQL LIKE pattern matching
on relational_entities (JSON text) and source_text columns via
_scope_by_entity() and _scope_by_entity_and_keyword(). At 1000+ edges,
these LIKE scans could degrade. FTS5 exists but the primary retrieval
path uses raw SQL LIKE, not BM25. This test measures whether the
current design holds under scale (latency < 500ms) and whether
user_id scoping correctly isolates cross-speaker data (Cat 5).

Run: py -3.10 tests/test_scale_1000_edges.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import sqlite3
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Override DB path BEFORE any imports touch settings
TEST_DB_PATH = str(PROJECT_ROOT / "Memory Storage" / "test_scale_1000.db")
os.environ["NURA_SQLITE_PATH"] = TEST_DB_PATH

# ── Imports ──
from config.settings import settings
settings.sqlite_path = TEST_DB_PATH

from app.db.models import MIGRATIONS, run_schema_upgrades
from app.db.session import get_db_context
from app.engines.memory import MemoryEngine
from app.engines.retrieval import RetrievalEngine, Answer, StructuralRefusal


# ============================================================================
# Edge generation: 1,000 realistic conversational edges across 5 domains
# ============================================================================

# User IDs: Sam=1, Alex=2. Each user gets their own edges.
# The retrieval engine scopes by user_id, so cross-speaker isolation
# is tested by querying user_id=1 for Alex-only facts.

SAM_UID = 1
ALEX_UID = 2


def _career_edges_sam() -> List[Dict[str, Any]]:
    """200 career edges for Sam."""
    edges = []
    companies = [
        "Google", "Stripe", "Tesla", "Meta", "Apple", "SpaceX", "Netflix",
        "Amazon", "Uber", "Airbnb", "Salesforce", "Oracle", "IBM", "Intel",
        "Microsoft", "Adobe", "Zoom", "Slack", "GitHub", "Figma",
    ]
    roles = [
        "software engineer", "product manager", "data scientist",
        "engineering manager", "CTO", "VP of Engineering", "tech lead",
        "machine learning engineer", "platform engineer", "DevOps lead",
    ]
    colleagues = [
        "Derek", "Priya", "Marcus", "Leah", "Tomoko", "Jamal", "Elena",
        "Ryan", "Sasha", "Nadia", "Carlos", "Mina", "Dmitri", "Fatima",
        "Liam", "Yuki", "Hassan", "Anya", "Vincent", "Kenji",
    ]
    actions = [
        "got promoted to", "started working as", "left the role of",
        "applied for", "interviewed for", "was offered", "turned down",
        "negotiated salary for", "got a raise as", "transitioned to",
        "completed training for", "mentored someone in", "led a team as",
        "was recruited for", "resigned from", "freelanced as",
        "consulted as", "co-founded a startup as", "pivoted career to",
        "took a sabbatical from",
    ]
    for i in range(200):
        company = companies[i % len(companies)]
        role = roles[i % len(roles)]
        colleague = colleagues[i % len(colleagues)]
        action = actions[i % len(actions)]
        days_ago = 200 - i
        ts = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()

        if i % 5 == 0:
            text = f"Sam {action} {role} at {company}"
        elif i % 5 == 1:
            text = f"Sam works with {colleague} at {company}"
        elif i % 5 == 2:
            text = f"Sam had a meeting about the {role} position at {company}"
        elif i % 5 == 3:
            text = f"{colleague} recommended Sam for the {role} role at {company}"
        else:
            text = f"Sam is preparing for an interview at {company} for {role}"

        edges.append({
            "user_id": SAM_UID,
            "subject": "Sam",
            "predicate": action.replace(" ", "_"),
            "object": f"{role} at {company}" if i % 3 == 0 else company,
            "source_text": text,
            "edge_schematic_category": "career",
            "edge_episodic_significance": "notable" if i % 10 == 0 else "routine",
            "edge_emotional_valence": 0.7 if "promoted" in action else 0.5,
            "edge_emotional_label": "excited" if "promoted" in action else None,
            "edge_relational_type": "professional",
            "edge_temporal_context": "present" if i > 150 else "past",
            "relational_entities": json.dumps(["Sam", colleague]),
            "source_timestamp": ts,
            "confidence": 0.9,
        })
    return edges


def _career_edges_alex() -> List[Dict[str, Any]]:
    """50 career edges for Alex (different companies/roles)."""
    edges = []
    companies = ["Palantir", "Notion", "Vercel", "Supabase", "Cloudflare"]
    roles = ["designer", "UX researcher", "product designer", "design lead", "art director"]
    for i in range(50):
        company = companies[i % len(companies)]
        role = roles[i % len(roles)]
        days_ago = 100 - i
        ts = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
        text = f"Alex works as a {role} at {company}"
        edges.append({
            "user_id": ALEX_UID,
            "subject": "Alex",
            "predicate": "works_as",
            "object": f"{role} at {company}",
            "source_text": text,
            "edge_schematic_category": "career",
            "edge_episodic_significance": "routine",
            "edge_emotional_valence": 0.6,
            "edge_relational_type": "professional",
            "edge_temporal_context": "present",
            "relational_entities": json.dumps(["Alex"]),
            "source_timestamp": ts,
            "confidence": 0.9,
        })
    return edges


def _family_edges_sam() -> List[Dict[str, Any]]:
    """200 family edges for Sam."""
    edges = []
    family = [
        ("Mom", "mother"), ("Dad", "father"), ("Grandma Rose", "grandmother"),
        ("Uncle Raj", "uncle"), ("Aunt Priya", "aunt"), ("Cousin Arjun", "cousin"),
        ("Sister Maya", "sister"), ("Brother Vikram", "brother"),
        ("Niece Ananya", "niece"), ("Nephew Dev", "nephew"),
    ]
    events = [
        "had dinner with", "called", "visited", "celebrated birthday of",
        "went shopping with", "cooked for", "argued with", "hugged",
        "received a gift from", "planned a trip with", "watched a movie with",
        "played cards with", "went to temple with", "picked up from airport",
        "helped move", "drove to doctor with", "shared news with",
        "planned surprise for", "texted", "video called",
    ]
    for i in range(200):
        member, relation = family[i % len(family)]
        event = events[i % len(events)]
        days_ago = 200 - i
        ts = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
        text = f"Sam {event} {member}"
        edges.append({
            "user_id": SAM_UID,
            "subject": "Sam",
            "predicate": event.replace(" ", "_"),
            "object": member,
            "source_text": text,
            "edge_schematic_category": "family",
            "edge_episodic_significance": "notable" if "birthday" in event else "routine",
            "edge_emotional_valence": 0.8 if "hugged" in event else 0.5,
            "edge_emotional_label": "happy" if "birthday" in event else None,
            "edge_relational_type": "family",
            "edge_temporal_context": "past",
            "relational_entities": json.dumps(["Sam", member]),
            "source_timestamp": ts,
            "confidence": 0.9,
        })
    return edges


def _health_edges_sam() -> List[Dict[str, Any]]:
    """200 health edges for Sam."""
    edges = []
    activities = [
        "ran 5 miles", "did yoga for 30 minutes", "went to the gym",
        "swam 20 laps", "did pushups", "walked in the park",
        "cycled 10 miles", "did a HIIT workout", "stretched for 15 minutes",
        "played basketball", "did weightlifting", "ran a half marathon",
        "meditated for 20 minutes", "did pilates", "hiked Mount Tam",
        "did boxing training", "went rock climbing", "played tennis",
        "took a dance class", "practiced tai chi",
    ]
    conditions = [
        "felt a headache", "had back pain", "felt energetic",
        "had trouble sleeping", "felt stressed", "felt calm after exercise",
        "had a sore knee", "felt exhausted", "recovered from a cold",
        "felt motivated",
    ]
    for i in range(200):
        days_ago = 200 - i
        ts = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
        if i % 3 != 2:
            activity = activities[i % len(activities)]
            text = f"Sam {activity}"
            predicate = activity.split()[0]
            obj = " ".join(activity.split()[1:])
        else:
            condition = conditions[i % len(conditions)]
            text = f"Sam {condition}"
            predicate = condition.split()[0]
            obj = " ".join(condition.split()[1:])

        # Marathon training specifically
        if i == 180:
            text = "Sam is training for a marathon in October"
            predicate = "is_training_for"
            obj = "marathon in October"

        edges.append({
            "user_id": SAM_UID,
            "subject": "Sam",
            "predicate": predicate,
            "object": obj,
            "source_text": text,
            "edge_schematic_category": "health",
            "edge_episodic_significance": "routine",
            "edge_emotional_valence": 0.6 if "energetic" in text or "motivated" in text else 0.4,
            "edge_emotional_label": "energetic" if "energetic" in text else None,
            "edge_relational_type": "personal",
            "edge_temporal_context": "present",
            "relational_entities": json.dumps(["Sam"]),
            "source_timestamp": ts,
            "confidence": 0.9,
        })
    return edges


def _social_edges_sam() -> List[Dict[str, Any]]:
    """150 social edges for Sam."""
    edges = []
    friends = [
        "Jake", "Emily", "Ravi", "Sophie", "Chris", "Fatima", "Leo",
        "Nina", "Omar", "Tanya", "Kai", "Iris", "Hugo", "Diana", "Theo",
    ]
    activities = [
        "went to a concert with", "had coffee with", "played video games with",
        "went hiking with", "had dinner at a restaurant with",
        "watched a game with", "went to a museum with", "traveled to Paris with",
        "attended a wedding with", "went surfing with", "threw a party with",
        "went camping with", "ran a race with", "cooked dinner for",
        "planned a road trip with",
    ]
    for i in range(150):
        friend = friends[i % len(friends)]
        activity = activities[i % len(activities)]
        days_ago = 150 - i
        ts = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
        text = f"Sam {activity} {friend}"
        edges.append({
            "user_id": SAM_UID,
            "subject": "Sam",
            "predicate": activity.replace(" ", "_"),
            "object": friend,
            "source_text": text,
            "edge_schematic_category": "social",
            "edge_episodic_significance": "routine",
            "edge_emotional_valence": 0.7,
            "edge_emotional_label": "happy",
            "edge_relational_type": "friendship",
            "edge_temporal_context": "past",
            "relational_entities": json.dumps(["Sam", friend]),
            "source_timestamp": ts,
            "confidence": 0.9,
        })
    return edges


def _social_edges_alex() -> List[Dict[str, Any]]:
    """50 social edges for Alex (different friends)."""
    edges = []
    friends = ["Bella", "Owen", "Nora", "Quinn", "Sienna"]
    for i in range(50):
        friend = friends[i % len(friends)]
        days_ago = 50 - i
        ts = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
        # Alex's best friend is Bella
        if i == 0:
            text = f"Alex considers {friend} to be their best friend"
            predicate = "considers_best_friend"
        else:
            text = f"Alex hung out with {friend}"
            predicate = "hung_out_with"
        edges.append({
            "user_id": ALEX_UID,
            "subject": "Alex",
            "predicate": predicate,
            "object": friend,
            "source_text": text,
            "edge_schematic_category": "social",
            "edge_episodic_significance": "routine",
            "edge_emotional_valence": 0.7,
            "edge_relational_type": "friendship",
            "edge_temporal_context": "past",
            "relational_entities": json.dumps(["Alex", friend]),
            "source_timestamp": ts,
            "confidence": 0.9,
        })
    return edges


def _daily_edges_sam() -> List[Dict[str, Any]]:
    """150 daily/misc edges for Sam."""
    edges = []
    hobbies = [
        "reading science fiction", "building model trains", "playing guitar",
        "photography", "cooking Italian food", "learning Japanese",
        "watching documentaries", "gardening", "woodworking", "sketching",
    ]
    errands = [
        "went grocery shopping", "picked up dry cleaning", "paid the electric bill",
        "took the car for an oil change", "mailed a package", "returned library books",
        "bought new running shoes", "repaired the kitchen faucet",
        "organized the garage", "cleaned the apartment",
    ]
    for i in range(150):
        days_ago = 150 - i
        ts = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
        if i % 3 == 0:
            hobby = hobbies[i % len(hobbies)]
            text = f"Sam spent time {hobby}"
            predicate = "spent_time"
            obj = hobby
        elif i % 3 == 1:
            errand = errands[i % len(errands)]
            text = f"Sam {errand}"
            predicate = errand.split()[0]
            obj = " ".join(errand.split()[1:])
        else:
            text = f"Sam felt relaxed after a quiet evening at home"
            predicate = "felt"
            obj = "relaxed after a quiet evening at home"

        # Specific hobby edge for query testing
        if i == 0:
            text = "Sam enjoys reading science fiction novels"
            predicate = "enjoys"
            obj = "reading science fiction novels"

        edges.append({
            "user_id": SAM_UID,
            "subject": "Sam",
            "predicate": predicate,
            "object": obj,
            "source_text": text,
            "edge_schematic_category": "daily",
            "edge_episodic_significance": "routine",
            "edge_emotional_valence": 0.5,
            "edge_relational_type": "personal",
            "edge_temporal_context": "present",
            "relational_entities": json.dumps(["Sam"]),
            "source_timestamp": ts,
            "confidence": 0.9,
        })
    return edges


def _daily_edges_alex() -> List[Dict[str, Any]]:
    """50 daily/misc edges for Alex."""
    edges = []
    for i in range(50):
        days_ago = 50 - i
        ts = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()

        # Location edge for Alex
        if i == 0:
            text = "Alex lives in Portland, Oregon"
            predicate = "lives_in"
            obj = "Portland, Oregon"
        elif i == 1:
            text = "Alex moved to Portland last year"
            predicate = "moved_to"
            obj = "Portland"
        else:
            text = f"Alex went to the farmers market"
            predicate = "went_to"
            obj = "the farmers market"

        edges.append({
            "user_id": ALEX_UID,
            "subject": "Alex",
            "predicate": predicate,
            "object": obj,
            "source_text": text,
            "edge_schematic_category": "daily",
            "edge_episodic_significance": "routine",
            "edge_emotional_valence": 0.5,
            "edge_relational_type": "personal",
            "edge_temporal_context": "present",
            "relational_entities": json.dumps(["Alex"]),
            "source_timestamp": ts,
            "confidence": 0.9,
        })
    return edges


def _special_edges() -> List[Dict[str, Any]]:
    """Specific edges needed for query correctness validation."""
    now = datetime.utcnow()
    return [
        # Sam's current job (most recent career edge)
        {
            "user_id": SAM_UID,
            "subject": "Sam",
            "predicate": "works_as",
            "object": "software engineer at Kenotic Labs",
            "source_text": "Sam works as a software engineer at Kenotic Labs",
            "edge_schematic_category": "career",
            "edge_episodic_significance": "stative",
            "edge_emotional_valence": 0.8,
            "edge_relational_type": "professional",
            "edge_temporal_context": "present",
            "relational_entities": json.dumps(["Sam"]),
            "source_timestamp": now.isoformat(),
            "confidence": 0.95,
        },
        # Sam's new job start date
        {
            "user_id": SAM_UID,
            "subject": "Sam",
            "predicate": "started_new_job",
            "object": "Kenotic Labs",
            "source_text": "Sam started his new job at Kenotic Labs in January 2026",
            "edge_schematic_category": "career",
            "edge_episodic_significance": "milestone",
            "edge_emotional_valence": 0.8,
            "edge_relational_type": "professional",
            "edge_temporal_context": "past",
            "relational_entities": json.dumps(["Sam"]),
            "source_timestamp": (now - timedelta(days=90)).isoformat(),
            "temporal_expression": "January 2026",
            "resolved_event_date": "2026-01-15",
            "confidence": 0.95,
        },
        # Sam's interview (for ambiguous query testing)
        {
            "user_id": SAM_UID,
            "subject": "Sam",
            "predicate": "had_interview_at",
            "object": "Kenotic Labs",
            "source_text": "Sam had an interview at Kenotic Labs last month",
            "edge_schematic_category": "career",
            "edge_episodic_significance": "notable",
            "edge_emotional_valence": 0.6,
            "edge_emotional_label": "nervous",
            "edge_relational_type": "professional",
            "edge_temporal_context": "past",
            "relational_entities": json.dumps(["Sam"]),
            "source_timestamp": (now - timedelta(days=120)).isoformat(),
            "confidence": 0.9,
        },
        # Alex's interview (different company -- disambiguation test)
        {
            "user_id": ALEX_UID,
            "subject": "Alex",
            "predicate": "had_interview_at",
            "object": "Figma",
            "source_text": "Alex had an interview at Figma last week",
            "edge_schematic_category": "career",
            "edge_episodic_significance": "notable",
            "edge_emotional_valence": 0.6,
            "edge_relational_type": "professional",
            "edge_temporal_context": "past",
            "relational_entities": json.dumps(["Alex"]),
            "source_timestamp": (now - timedelta(days=7)).isoformat(),
            "confidence": 0.9,
        },
        # Sam's emotional state
        {
            "user_id": SAM_UID,
            "subject": "Sam",
            "predicate": "feels",
            "object": "optimistic about the future",
            "source_text": "Sam feels optimistic about the future",
            "edge_schematic_category": "identity",
            "edge_episodic_significance": "routine",
            "edge_emotional_valence": 0.8,
            "edge_emotional_label": "optimistic",
            "edge_relational_type": "personal",
            "edge_temporal_context": "present",
            "relational_entities": json.dumps(["Sam"]),
            "source_timestamp": now.isoformat(),
            "confidence": 0.9,
        },
    ]


# ============================================================================
# DB setup and edge ingestion (direct SQL, bypassing grammar engine)
# ============================================================================

def _create_fresh_db() -> None:
    """Create a fresh test DB with all migrations."""
    db_path = Path(TEST_DB_PATH)
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(MIGRATIONS)
    run_schema_upgrades(conn)
    conn.commit()
    conn.close()
    print(f"[SETUP] Created fresh DB: {db_path}")


def _ingest_edges_direct(edges: List[Dict[str, Any]]) -> float:
    """Insert edges directly into relationships table via SQL.

    Bypasses grammar engine (we are testing retrieval, not extraction).
    Computes embeddings for source_text and predicate.

    Root cause addressed: The UNIQUE constraint on relationships is
    UNIQUE(user_id, source_text_hash, created_at) (models.py line 166).
    Repeated source_text values (e.g. 48 copies of "Alex went to the
    farmers market") produce identical SHA-256 hashes. When batch-inserted
    within the same second, datetime('now') yields the same created_at,
    causing constraint violations. Fix: include a sequence counter in
    the hash input, and set created_at from source_timestamp (which is
    unique per edge by construction in the test data generators).

    Returns wall-clock seconds for the full ingestion.
    """
    from app.vector.embedder import embed_text

    t0 = time.perf_counter()
    seq = 0

    with get_db_context() as conn:
        for edge in edges:
            seq += 1
            source_text = edge["source_text"]
            predicate = edge.get("predicate", "")

            # Compute embeddings
            try:
                edge_emb = embed_text(source_text).tobytes()
            except Exception:
                edge_emb = None
            try:
                pred_emb = embed_text(predicate.replace("_", " ")).tobytes() if predicate else None
            except Exception:
                pred_emb = None

            # Sequence-unique hash to avoid UNIQUE constraint collisions
            unique_hash = hashlib.sha256(
                f"{source_text}:{seq}".encode()
            ).hexdigest()
            created_at = edge.get("source_timestamp") or datetime.utcnow().isoformat()

            conn.execute(
                """INSERT INTO relationships (
                    user_id, subject, predicate, object,
                    source_text, source_text_hash,
                    edge_schematic_category, edge_episodic_significance,
                    edge_emotional_valence, edge_emotional_label,
                    edge_relational_type, edge_temporal_context,
                    relational_entities,
                    source_timestamp, confidence,
                    edge_embedding, predicate_embedding,
                    sequence_number, is_current,
                    temporal_expression, resolved_event_date,
                    edge_mood, created_at
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?,
                    ?, ?,
                    ?, ?,
                    ?, 1,
                    ?, ?,
                    'indicative', ?
                )""",
                (
                    edge["user_id"], edge.get("subject"), predicate, edge.get("object"),
                    source_text, unique_hash,
                    edge.get("edge_schematic_category", "uncategorized"),
                    edge.get("edge_episodic_significance", "routine"),
                    edge.get("edge_emotional_valence", 0.5),
                    edge.get("edge_emotional_label"),
                    edge.get("edge_relational_type", "personal"),
                    edge.get("edge_temporal_context", "present"),
                    edge.get("relational_entities"),
                    edge.get("source_timestamp"),
                    edge.get("confidence", 0.9),
                    edge_emb, pred_emb,
                    seq,
                    edge.get("temporal_expression"),
                    edge.get("resolved_event_date"),
                    created_at,
                ),
            )

            # FTS5 sync
            rel_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                """INSERT INTO relationships_fts(rowid, subject, predicate, object, source_text)
                   VALUES (?, ?, ?, ?, ?)""",
                (rel_id, edge.get("subject", ""), predicate,
                 edge.get("object", ""), source_text),
            )

        conn.commit()

    elapsed = time.perf_counter() - t0
    return elapsed


# ============================================================================
# Query definitions
# ============================================================================

def _f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer.
    LoCoMo-style: lowercase, split on whitespace, compute precision/recall
    of token overlap."""
    if not prediction or not gold:
        return 0.0
    pred_tokens = set(prediction.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = pred_tokens & gold_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _locomo_strict(r, gold: str, f1_threshold: float = 0.3) -> bool:
    """LoCoMo-strict: Answer must have F1 >= threshold against gold."""
    if not isinstance(r, Answer) or not r.text:
        return False
    return _f1(r.text, gold) >= f1_threshold


def _refuse(r):
    """Adversarial: must be StructuralRefusal."""
    return isinstance(r, StructuralRefusal)


# Gold answers based on the test data ground truth.
# F1 threshold 0.3 = at least 30% token overlap with gold.
queries = [
    # Cat 4 -- single-hop factual
    ("What does Sam do for work?", SAM_UID,
     "gold: software engineer at Kenotic Labs", "cat4",
     lambda r: _locomo_strict(r, "software engineer at Kenotic Labs")),

    ("Where does Alex live?", ALEX_UID,
     "gold: Portland Oregon", "cat4",
     lambda r: _locomo_strict(r, "Portland Oregon")),

    ("What is Sam's hobby?", SAM_UID,
     "gold: reading science fiction novels", "cat4",
     lambda r: _locomo_strict(r, "reading science fiction novels")),

    ("Who is Alex's best friend?", ALEX_UID,
     "gold: Bella", "cat4",
     lambda r: _locomo_strict(r, "Bella")),

    # Cat 2 -- temporal
    ("When did Sam start his new job?", SAM_UID,
     "gold: January 2026", "cat2",
     lambda r: _locomo_strict(r, "January 2026")),

    ("When did Alex move?", ALEX_UID,
     "gold: last year moved to Portland", "cat2",
     lambda r: _locomo_strict(r, "last year moved to Portland")),

    # Cat 5 -- adversarial (MUST refuse)
    ("What does Alex do for work?", SAM_UID,
     "MUST refuse -- wrong user_id", "cat5",
     lambda r: _refuse(r)),

    ("Where does Sam live?", ALEX_UID,
     "MUST refuse -- wrong user_id", "cat5",
     lambda r: _refuse(r)),

    # Cat 1 -- aggregation
    ("What kind of exercise does Sam do?", SAM_UID,
     "gold: ran yoga gym swam cycled basketball marathon pushups weightlifting pilates hiking tennis boxing", "cat1",
     lambda r: _locomo_strict(r, "ran yoga gym swam cycled basketball marathon pushups weightlifting pilates hiking tennis boxing")),

    ("What are Alex's hobbies?", ALEX_UID,
     "gold: Bella Owen Nora Quinn Sienna farmers market", "cat1",
     lambda r: _locomo_strict(r, "Bella Owen Nora Quinn Sienna farmers market")),

    # Reconstruction -- must return actual content about the person
    ("What's going on in Sam's life?", SAM_UID,
     "gold: Sam works at Kenotic Labs career family health social", "reconstruct",
     lambda r: isinstance(r, Answer) and bool(r.text) and len(r.text.strip()) > 3),

    ("Tell me about Alex's family", ALEX_UID,
     "Alex has no family edges -- must refuse", "reconstruct",
     lambda r: _refuse(r)),

    # Stress -- ambiguous
    ("Tell me about the interview", SAM_UID,
     "gold: Sam had an interview at Kenotic Labs", "stress",
     lambda r: _locomo_strict(r, "Sam had an interview at Kenotic Labs")),

    ("How is the marathon training going?", SAM_UID,
     "gold: training for a marathon in October", "stress",
     lambda r: _locomo_strict(r, "training for a marathon in October")),

    # Edge cases
    ("What happened last week?", SAM_UID,
     "must return SOME recent event (answer, not refusal)", "edge",
     lambda r: isinstance(r, Answer) and bool(r.text) and len(r.text.strip()) > 2),

    ("How does Sam feel?", SAM_UID,
     "gold: optimistic about the future", "edge",
     lambda r: _locomo_strict(r, "optimistic about the future")),

    # Scale stress -- must return relevant content from the right domain
    ("What did Sam say about work?", SAM_UID,
     "gold: Sam works as a software engineer at Kenotic Labs promoted career", "scale",
     lambda r: _locomo_strict(r, "Sam works as a software engineer at Kenotic Labs promoted career")),

    ("Tell me about family", SAM_UID,
     "gold: Mom Dad Grandma Rose Uncle Raj Sister Maya Brother Vikram family", "scale",
     lambda r: _locomo_strict(r, "Mom Dad Grandma Rose Uncle Raj Sister Maya Brother Vikram family")),

    ("What's new?", SAM_UID,
     "must return recent content (answer, not refusal)", "scale",
     lambda r: isinstance(r, Answer) and bool(r.text) and len(r.text.strip()) > 2),

    ("Everything about health", SAM_UID,
     "gold: ran yoga gym swam marathon exercise health headache knee", "scale",
     lambda r: _locomo_strict(r, "ran yoga gym swam marathon exercise health headache knee")),
]


# ============================================================================
# Run and report
# ============================================================================

def _run_queries(retrieval: RetrievalEngine) -> List[Dict[str, Any]]:
    """Execute all queries and collect timing + correctness."""
    results = []
    for query_text, user_id, description, category, validator in queries:
        t0 = time.perf_counter()
        try:
            result = retrieval.retrieve(user_id, query_text)
            error = None
        except Exception as e:
            result = None
            error = str(e)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Extract metrics
        if isinstance(result, Answer):
            answer_text = result.text or ""
            survivors = result.survivors
            pool_size = result.convergence_details.get("pool_size", survivors)
            result_type = "Answer"
        elif isinstance(result, StructuralRefusal):
            answer_text = result.text or ""
            survivors = 0
            pool_size = 0
            result_type = "StructuralRefusal"
        else:
            answer_text = ""
            survivors = 0
            pool_size = 0
            result_type = "None" if result is None else type(result).__name__

        correct = False
        if error is None:
            try:
                correct = validator(result)
            except Exception:
                correct = False

        # Extract gold from description for F1 computation
        _gold = ""
        if description.startswith("gold:"):
            _gold = description[5:].strip()
        _f1_score = _f1(answer_text, _gold) if _gold and answer_text else None

        results.append({
            "query": query_text,
            "user_id": user_id,
            "description": description,
            "category": category,
            "latency_ms": round(elapsed_ms, 2),
            "result_type": result_type,
            "answer": answer_text[:120],
            "pool_size": pool_size,
            "survivors": survivors,
            "correct": correct,
            "error": error,
            "f1": _f1_score,
        })
    return results


def _print_report(
    edge_count: int,
    ingestion_time: float,
    results: List[Dict[str, Any]],
) -> None:
    """Print structured report."""
    latencies = [r["latency_ms"] for r in results]
    lat_min = min(latencies)
    lat_max = max(latencies)
    lat_median = statistics.median(latencies)
    lat_sorted = sorted(latencies)
    p95_idx = int(len(lat_sorted) * 0.95)
    lat_p95 = lat_sorted[min(p95_idx, len(lat_sorted) - 1)]

    pass_count = sum(1 for r in results if r["correct"])
    fail_count = sum(1 for r in results if not r["correct"])

    # Compute mean F1 across queries that have gold answers
    f1_scores = [r["f1"] for r in results if r["f1"] is not None]
    mean_f1 = statistics.mean(f1_scores) if f1_scores else 0.0

    print("\n" + "=" * 80)
    print("SCALE TEST REPORT: 1,000 edges, 20 queries (LoCoMo-strict F1)")
    print("=" * 80)

    print(f"\n[INGESTION]")
    print(f"  Total edges:      {edge_count}")
    print(f"  Ingestion time:   {ingestion_time:.2f}s")
    print(f"  Per-edge avg:     {(ingestion_time / edge_count * 1000):.1f}ms")

    print(f"\n[LATENCY]")
    print(f"  Min:    {lat_min:.1f}ms")
    print(f"  Median: {lat_median:.1f}ms")
    print(f"  P95:    {lat_p95:.1f}ms")
    print(f"  Max:    {lat_max:.1f}ms")

    slow = [r for r in results if r["latency_ms"] > 500]
    if slow:
        print(f"\n  WARNING: {len(slow)} queries exceeded 500ms:")
        for r in slow:
            print(f"    {r['latency_ms']:.0f}ms  {r['query']}")
    else:
        print(f"  All queries under 500ms.")

    print(f"\n[CORRECTNESS]")
    print(f"  Pass: {pass_count}/{len(results)} ({100*pass_count//len(results)}%)")
    print(f"  Fail: {fail_count}/{len(results)}")
    print(f"  Mean F1: {mean_f1:.3f} (across {len(f1_scores)} queries with gold answers)")

    print(f"\n[PER-QUERY DETAIL]")
    print(f"  {'#':>2}  {'Cat':>10}  {'ms':>7}  {'Pool':>5}  {'F1':>5}  {'OK':>3}  {'Type':>18}  Query")
    print(f"  {'':->2}  {'':->10}  {'':->7}  {'':->5}  {'':->5}  {'':->3}  {'':->18}  {'':->40}")
    for i, r in enumerate(results, 1):
        ok = "Y" if r["correct"] else "N"
        f1_str = f"{r['f1']:.2f}" if r["f1"] is not None else "  --"
        print(f"  {i:>2}  {r['category']:>10}  {r['latency_ms']:>7.1f}  {r['pool_size']:>5}  {f1_str:>5}  {ok:>3}  {r['result_type']:>18}  {r['query']}")

    # Detailed answers
    print(f"\n[ANSWERS]")
    for i, r in enumerate(results, 1):
        status = "PASS" if r["correct"] else "FAIL"
        f1_str = f" F1={r['f1']:.3f}" if r["f1"] is not None else ""
        print(f"  Q{i:02d} [{status}]{f1_str} {r['query']}")
        if r["error"]:
            print(f"       ERROR: {r['error']}")
        else:
            print(f"       -> {r['answer']}")
        print(f"       Expected: {r['description']}")
        print()

    # Category breakdown
    print(f"[CATEGORY BREAKDOWN]")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"pass": 0, "fail": 0, "latencies": []}
        categories[cat]["latencies"].append(r["latency_ms"])
        if r["correct"]:
            categories[cat]["pass"] += 1
        else:
            categories[cat]["fail"] += 1

    for cat, data in sorted(categories.items()):
        total = data["pass"] + data["fail"]
        avg_lat = statistics.mean(data["latencies"])
        print(f"  {cat:>12}: {data['pass']}/{total} pass, avg latency {avg_lat:.1f}ms")

    # Cat 5 adversarial analysis
    print(f"\n[CAT 5 ADVERSARIAL ANALYSIS]")
    cat5 = [r for r in results if r["category"] == "cat5"]
    for r in cat5:
        if r["correct"]:
            print(f"  CORRECT REFUSAL: {r['query']}")
        else:
            print(f"  FAILED REFUSAL:  {r['query']}")
            print(f"    Got: {r['result_type']} -> {r['answer']}")

    # Scale stress analysis
    print(f"\n[SCALE STRESS ANALYSIS]")
    scale = [r for r in results if r["category"] == "scale"]
    for r in scale:
        spike = " ** SPIKE" if r["latency_ms"] > 200 else ""
        print(f"  {r['latency_ms']:>7.1f}ms  pool={r['pool_size']:>4}  {r['query']}{spike}")

    # DB stats
    print(f"\n[DB STATS]")
    with get_db_context() as conn:
        total_rows = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        sam_rows = conn.execute(
            "SELECT COUNT(*) FROM relationships WHERE user_id = ?", (SAM_UID,)
        ).fetchone()[0]
        alex_rows = conn.execute(
            "SELECT COUNT(*) FROM relationships WHERE user_id = ?", (ALEX_UID,)
        ).fetchone()[0]
        fts_rows = conn.execute("SELECT COUNT(*) FROM relationships_fts").fetchone()[0]
        print(f"  Total relationships: {total_rows}")
        print(f"  Sam's edges:         {sam_rows}")
        print(f"  Alex's edges:        {alex_rows}")
        print(f"  FTS5 entries:        {fts_rows}")
        db_size = Path(TEST_DB_PATH).stat().st_size / (1024 * 1024)
        print(f"  DB file size:        {db_size:.2f} MB")

    print("\n" + "=" * 80)
    pct = 100 * pass_count // len(results)
    grade = "PASS" if pct >= 85 else "CLOSE" if pct >= 70 else "NEEDS WORK"
    print(f"VERDICT: {grade} — {pass_count}/{len(results)} ({pct}%), "
          f"mean F1={mean_f1:.3f}, median {lat_median:.0f}ms, p95 {lat_p95:.0f}ms")
    print("=" * 80)


# ============================================================================
# Main
# ============================================================================

def main():
    print("[SCALE TEST] 1,000 edges across 5 life domains")
    print(f"[SCALE TEST] DB path: {TEST_DB_PATH}")

    # Step 1: Create fresh DB
    _create_fresh_db()

    # Step 2: Generate all edges
    print("[SCALE TEST] Generating edges...")
    all_edges = (
        _career_edges_sam()        # 200
        + _career_edges_alex()     # 50
        + _family_edges_sam()      # 200
        + _health_edges_sam()      # 200
        + _social_edges_sam()      # 150
        + _social_edges_alex()     # 50
        + _daily_edges_sam()       # 150
        + _daily_edges_alex()      # 50
        + _special_edges()         # 5
    )
    print(f"[SCALE TEST] Generated {len(all_edges)} edges "
          f"(Sam={sum(1 for e in all_edges if e['user_id'] == SAM_UID)}, "
          f"Alex={sum(1 for e in all_edges if e['user_id'] == ALEX_UID)})")

    # Step 3: Ingest
    print("[SCALE TEST] Ingesting edges (with embeddings)...")
    ingestion_time = _ingest_edges_direct(all_edges)
    print(f"[SCALE TEST] Ingestion complete: {ingestion_time:.2f}s")

    # Step 4: Instantiate engines
    mem = MemoryEngine()
    retrieval = RetrievalEngine(memory_engine=mem)

    # Step 5: Run queries
    print("[SCALE TEST] Running 20 retrieval queries...")
    results = _run_queries(retrieval)

    # Step 6: Report
    _print_report(len(all_edges), ingestion_time, results)

    # Cleanup
    try:
        Path(TEST_DB_PATH).unlink(missing_ok=True)
        # Also clean WAL/SHM files
        for ext in ("-wal", "-shm"):
            p = Path(TEST_DB_PATH + ext)
            if p.exists():
                p.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    main()
