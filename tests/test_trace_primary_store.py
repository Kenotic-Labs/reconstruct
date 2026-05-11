import hashlib
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.models import MIGRATIONS, run_schema_upgrades


@dataclass
class TraceDecomposition:
    episodic_fact: str = ""
    episodic_significance: str = "routine"
    emotional_state: Optional[str] = None
    emotional_valence: Optional[float] = None
    emotional_target: Optional[str] = None
    temporal_direction: str = "present"
    temporal_expression: Optional[str] = None
    temporal_resolved: Optional[str] = None
    relational_subject: str = "user"
    relational_entities: List[str] = field(default_factory=list)
    relational_type: str = "personal"
    schematic_category: str = "uncategorized"
    source_text: str = ""
    utterance_type: int = 0
    mood: str = "indicative"
    negated: bool = False
    is_historical: bool = False
    subject: str = ""
    predicate: str = ""
    object: str = ""
    extraction_rule: str = ""


@pytest.fixture
def mem_engine(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("config.settings.settings.sqlite_path", db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.executescript(MIGRATIONS)
    conn.commit()
    conn.row_factory = sqlite3.Row
    run_schema_upgrades(conn)
    conn.close()
    from app.engines.memory import MemoryEngine
    return MemoryEngine()


@pytest.fixture
def db_conn(tmp_path, mem_engine):
    from config.settings import settings
    conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


USER_ID = 42


def test_01_trace_only_stores(mem_engine, db_conn):
    """TraceDecomposition with all 5 traces but empty S/P/O stores."""
    td = TraceDecomposition(
        episodic_fact="feeling nervous about upcoming event",
        episodic_significance="notable",
        emotional_state="nervous",
        emotional_valence=0.3,
        temporal_direction="future",
        relational_type="personal",
        schematic_category="career",
        source_text="I am so nervous right now",
    )
    rel_id = mem_engine.store(
        user_id=USER_ID,
        trace_decomposition=td,
        source_text="I am so nervous right now",
    )
    assert rel_id > 0
    row = db_conn.execute(
        "SELECT * FROM relationships WHERE id = ?", (rel_id,)
    ).fetchone()
    assert row is not None
    assert row["source_text"] == "I am so nervous right now"
    assert row["source_text_hash"] is not None
    assert len(row["source_text_hash"]) == 64


def test_02_trace_with_spo_stores(mem_engine, db_conn):
    """TraceDecomposition with S/P/O stores both traces and triple."""
    td = TraceDecomposition(
        episodic_fact="user works at Google",
        episodic_significance="notable",
        temporal_direction="present",
        relational_type="professional",
        schematic_category="career",
        source_text="I work at Google",
        subject="user",
        predicate="works_at",
        object="Google",
    )
    rel_id = mem_engine.store(
        user_id=USER_ID,
        trace_decomposition=td,
        source_text="I work at Google",
    )
    assert rel_id > 0
    row = db_conn.execute(
        "SELECT * FROM relationships WHERE id = ?", (rel_id,)
    ).fetchone()
    assert row["subject"] == "user"
    assert row["predicate"] == "works_at"
    assert row["object"] == "Google"
    assert row["source_text_hash"] is not None


def test_03_old_style_store(mem_engine, db_conn):
    """Old callers: store(subject, predicate, object) still works."""
    rel_id = mem_engine.store(
        user_id=USER_ID,
        subject="Maya",
        predicate="is_sister_of",
        object="user",
    )
    assert rel_id > 0
    row = db_conn.execute(
        "SELECT * FROM relationships WHERE id = ?", (rel_id,)
    ).fetchone()
    assert row["subject"] == "Maya"
    assert row["predicate"] == "is_sister_of"
    assert row["object"] == "user"
    assert "Maya" in row["source_text"]
    assert row["source_text_hash"] is not None


def test_04_dedup_on_hash(mem_engine, db_conn):
    """Same source_text deduplicates to same row."""
    source = "My sister Maya is visiting next week"
    rel_id_1 = mem_engine.store(
        user_id=USER_ID,
        subject="Maya",
        predicate="visiting",
        object="next week",
        source_text=source,
    )
    rel_id_2 = mem_engine.store(
        user_id=USER_ID,
        subject="Maya",
        predicate="visiting",
        object="next week",
        source_text=source,
    )
    assert rel_id_1 > 0
    assert rel_id_1 == rel_id_2
    hash_val = hashlib.sha256(source.encode("utf-8")).hexdigest()
    hash_count = db_conn.execute(
        "SELECT COUNT(*) FROM relationships WHERE source_text_hash = ?",
        (hash_val,),
    ).fetchone()[0]
    assert hash_count == 1


def test_05_thesis_example(mem_engine, db_conn):
    """CLAUDE.md thesis example stores with all 5 traces filled."""
    source = "I am nervous because I have a Google interview next Tuesday"
    td = TraceDecomposition(
        episodic_fact="user has interview at Google next Tuesday",
        episodic_significance="notable",
        emotional_state="nervous",
        emotional_valence=0.3,
        emotional_target="Google interview",
        temporal_direction="future",
        temporal_expression="next Tuesday",
        relational_subject="user",
        relational_entities=["Google"],
        relational_type="professional",
        schematic_category="career",
        source_text=source,
        subject="user",
        predicate="has_interview",
        object="Google",
    )
    rel_id = mem_engine.store(
        user_id=USER_ID,
        trace_decomposition=td,
        source_text=source,
    )
    assert rel_id > 0
    row = db_conn.execute(
        "SELECT * FROM relationships WHERE id = ?", (rel_id,)
    ).fetchone()
    assert row["subject"] == "user"
    assert row["predicate"] == "has_interview"
    assert row["object"] == "Google"
    assert row["source_text"] == source
    assert row["source_text_hash"] == hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()
    assert row["edge_schematic_category"] is not None
    assert row["edge_temporal_context"] is not None
    assert row["edge_relational_type"] is not None
    assert row["edge_episodic_significance"] is not None
    assert row["edge_emotional_valence"] is not None


def test_06_schema_trace_primary(tmp_path):
    """New databases get trace-primary schema from MIGRATIONS."""
    db_path = str(tmp_path / "schema_test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(MIGRATIONS)
    conn.commit()

    cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(relationships)"
    ).fetchall()}
    assert "source_text_hash" in cols
    assert "source_text" in cols
    assert "edge_schematic_category" in cols
    assert "edge_temporal_context" in cols
    assert "edge_relational_type" in cols
    assert "edge_episodic_significance" in cols
    assert "edge_emotional_valence" in cols

    for row in conn.execute("PRAGMA table_info(relationships)").fetchall():
        col_name = row[1]
        notnull = row[3]
        if col_name in ("subject", "predicate", "object"):
            assert notnull == 0, f"{col_name} should be nullable"

    create_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='relationships'"
    ).fetchone()[0]
    assert "source_text_hash" in create_sql
    assert "UNIQUE(user_id, subject, predicate, object)" not in create_sql

    index_names = {
        row[1] for row in conn.execute(
            "PRAGMA index_list(relationships)"
        ).fetchall()
    }
    assert "idx_rel_source_hash" in index_names
    assert "idx_rel_schema_cat" in index_names
    assert "idx_rel_subject_schema" in index_names

    conn.close()


def test_07_different_sources_no_collision(mem_engine, db_conn):
    """Different source texts producing same SPO get separate rows."""
    rel_id_1 = mem_engine.store(
        user_id=USER_ID,
        subject="user",
        predicate="feels",
        object="nervous",
        source_text="I am nervous about my Google interview",
    )
    rel_id_2 = mem_engine.store(
        user_id=USER_ID,
        subject="user",
        predicate="feels",
        object="nervous",
        source_text="I am really scared about the Google interview",
    )
    assert rel_id_1 > 0
    assert rel_id_2 > 0
    assert rel_id_1 != rel_id_2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
