import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.verify_implied_fact import (  # noqa: E402
    LOCOMO_REFUSAL,
    build_implied_fact,
    verify_candidates,
)


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            object_full TEXT,
            source_text TEXT,
            tombstoned_at TEXT,
            is_current INTEGER DEFAULT 1,
            sequence_number INTEGER,
            relational_entities TEXT,
            episodic_fact TEXT,
            pq_1 TEXT,
            pq_2 TEXT,
            pq_3 TEXT,
            pq_4 TEXT,
            vq_1 TEXT,
            vq_2 TEXT
        )"""
    )
    return conn


def _insert_edge(
    conn: sqlite3.Connection,
    *,
    subject: str,
    predicate: str,
    obj: str,
    source_text: str,
) -> int:
    import json
    relational_entities = json.dumps([subject])
    episodic_fact = source_text  # episodic trace = the fact itself
    cur = conn.execute(
        """INSERT INTO edges
              (user_id, subject, predicate, object, source_text,
               relational_entities, episodic_fact, sequence_number)
           VALUES (1, ?, ?, ?, ?, ?, ?, 1)""",
        (subject, predicate, obj, source_text, relational_entities, episodic_fact),
    )
    return int(cur.lastrowid)


def test_build_implied_fact_uses_final_lexical_claim():
    fact = build_implied_fact(
        "Who did the work and researched adoption agencies?",
        "Melanie",
    )

    assert fact.subject == "Melanie"
    assert fact.predicate == "research"
    assert fact.object == "adoption agencies"
    assert fact.statement == "Melanie researched adoption agencies"


def test_rejects_candidate_when_implied_fact_is_not_in_db():
    conn = _make_conn()
    _insert_edge(
        conn,
        subject="Caroline",
        predicate="research",
        obj="adoption agencies",
        source_text="Caroline researched adoption agencies.",
    )
    _insert_edge(
        conn,
        subject="Melanie",
        predicate="work_in",
        obj="adoption agencies",
        source_text="Melanie worked in adoption agencies.",
    )

    result = verify_candidates(
        conn,
        1,
        "Who did the work and researched adoption agencies?",
        ["Melanie"],
    )

    assert result.refusal is True
    assert result.answer == LOCOMO_REFUSAL
    assert result.decisions[0].candidate == "Melanie"
    assert result.decisions[0].implied_fact.statement == (
        "Melanie researched adoption agencies"
    )
    assert result.decisions[0].reason == "not_in_db"


def test_rejection_memory_skips_rejected_candidate_and_verifies_next():
    conn = _make_conn()
    caroline_edge_id = _insert_edge(
        conn,
        subject="Caroline",
        predicate="research",
        obj="adoption agencies",
        source_text="Caroline researched adoption agencies.",
    )
    _insert_edge(
        conn,
        subject="Melanie",
        predicate="work_in",
        obj="adoption agencies",
        source_text="Melanie worked in adoption agencies.",
    )

    rejected = set()
    first = verify_candidates(
        conn,
        1,
        "Who did the work and researched adoption agencies?",
        ["Melanie"],
        rejected=rejected,
    )
    second = verify_candidates(
        conn,
        1,
        "Who did the work and researched adoption agencies?",
        ["Melanie", "Caroline"],
        rejected=rejected,
    )

    assert first.refusal is True
    assert second.refusal is False
    assert second.answer == "Caroline"
    assert second.verified_edge_id == caroline_edge_id
    assert second.decisions[0].reason == "previously_rejected"
    assert second.decisions[1].reason == "verified"


def test_verified_query_is_written_without_replacing_predicted_queries():
    conn = _make_conn()
    edge_id = _insert_edge(
        conn,
        subject="Caroline",
        predicate="research",
        obj="adoption agencies",
        source_text="Caroline researched adoption agencies.",
    )
    conn.execute(
        "UPDATE edges SET pq_1 = ? WHERE id = ?",
        ("Who researched adoption agencies?", edge_id),
    )

    query = "Who did the work and researched adoption agencies?"
    result = verify_candidates(
        conn,
        1,
        query,
        ["Caroline"],
        write_verified_query=True,
    )
    row = conn.execute(
        "SELECT pq_1, vq_1, vq_2 FROM edges WHERE id = ?",
        (edge_id,),
    ).fetchone()

    assert result.refusal is False
    assert row["pq_1"] == "Who researched adoption agencies?"
    assert row["vq_1"] == query
    assert row["vq_2"] is None
