"""
kenoticArchitectureV1 Runtime Verifier

Verifies the 5-engine continuity architecture:

  1. INGESTION    (ingestion.py)        — messy → clean text
  2. GRAMMAR      (grammar_engine.py)   — clean → meaning (5 traces, SPO, PQs, types)
  3. TEMPORAL     (temporal.py)          — time truth (event dates, supersession, arcs)
  4. MEMORY       (memory.py)            — THE writer (edges + edge_extraction)
  5. RECONSTRUCTION (reconstruction.py)  — THE reader (retrieval, verification, entity resolution)

Plus:
  - DB schema (edges, edge_extraction, edges_fts, entities, arcs)
  - SDK client (sdk/client.py)

Each engine is checked for:
  - PRESENT  (module importable, key functions exist)
  - CONNECTED (call chains exist)
  - SCHEMA (DB tables and columns exist)

Returns List[Dict[str, str]] where each dict has {check, status, detail}.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)


def verify_kenotic_architecture_v1(
    db_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Run all architecture checks. Returns list of check results."""

    results: List[Dict[str, str]] = []

    def _check(name: str, ok: bool, detail: str = ""):
        results.append({
            "check": name,
            "status": "PASS" if ok else "FAIL",
            "detail": detail,
        })

    # =================================================================
    # ENGINE 1: INGESTION (ingestion.py)
    # =================================================================
    try:
        from app.engines import ingestion

        for fn_name in ["cleanup", "is_available"]:
            has_fn = hasattr(ingestion, fn_name) and callable(
                getattr(ingestion, fn_name)
            )
            _check(
                f"ingestion.{fn_name}",
                has_fn,
                f"{fn_name}() exists" if has_fn else f"{fn_name}() NOT found",
            )
    except Exception as e:
        _check("ingestion.import", False, f"Cannot import ingestion: {e}")

    # =================================================================
    # ENGINE 2: GRAMMAR (grammar_engine.py)
    # =================================================================
    try:
        from app.engines import grammar_engine

        for fn_name in ["process", "classify_query", "classify_verb_class",
                         "generate_predicted_queries"]:
            has_fn = hasattr(grammar_engine, fn_name) and callable(
                getattr(grammar_engine, fn_name)
            )
            _check(
                f"grammar.{fn_name}",
                has_fn,
                f"{fn_name}() exists" if has_fn else f"{fn_name}() NOT found",
            )

        # Check TraceDecomposition dataclass
        has_td = hasattr(grammar_engine, "TraceDecomposition")
        _check(
            "grammar.TraceDecomposition",
            has_td,
            "TraceDecomposition exists" if has_td else "TraceDecomposition NOT found",
        )

        # Check GrammarResult dataclass
        has_gr = hasattr(grammar_engine, "GrammarResult")
        _check(
            "grammar.GrammarResult",
            has_gr,
            "GrammarResult exists" if has_gr else "GrammarResult NOT found",
        )

    except Exception as e:
        _check("grammar.import", False, f"Cannot import grammar_engine: {e}")

    # =================================================================
    # ENGINE 3: TEMPORAL (temporal.py)
    # =================================================================
    try:
        from app.engines import temporal

        has_get = hasattr(temporal, "get_temporal_engine") and callable(
            temporal.get_temporal_engine
        )
        _check(
            "temporal.get_temporal_engine",
            has_get,
            "get_temporal_engine() exists" if has_get else "NOT found",
        )

        if has_get:
            te = temporal.get_temporal_engine()
            for method in ["resolve_event_date", "detect_supersession", "detect_arcs"]:
                has_m = hasattr(te, method) and callable(getattr(te, method))
                _check(
                    f"temporal.{method}",
                    has_m,
                    f"{method}() exists" if has_m else f"{method}() NOT found",
                )

    except Exception as e:
        _check("temporal.import", False, f"Cannot import temporal: {e}")

    # =================================================================
    # ENGINE 4: MEMORY (memory.py) — THE WRITER
    # =================================================================
    try:
        from app.engines import memory

        has_get = hasattr(memory, "get_memory_engine") and callable(
            memory.get_memory_engine
        )
        _check(
            "memory.get_memory_engine",
            has_get,
            "get_memory_engine() exists" if has_get else "NOT found",
        )

        if has_get:
            me = memory.get_memory_engine()
            for method in ["store", "ingest_text", "supersede",
                           "get_relationships", "get_entity", "entity_link",
                           "forget_by_triple_id", "forget_by_entity",
                           "forget_by_time_range", "forget_by_source",
                           "list_by_facet", "summarize"]:
                has_m = hasattr(me, method) and callable(getattr(me, method))
                _check(
                    f"memory.{method}",
                    has_m,
                    f"{method}() exists" if has_m else f"{method}() NOT found",
                )

        # Check write path wiring: memory imports from ingestion and grammar
        import inspect
        mem_src = inspect.getsource(memory)
        _check(
            "memory.imports_ingestion",
            "from app.engines.ingestion" in mem_src,
            "memory.py imports from ingestion"
            if "from app.engines.ingestion" in mem_src
            else "memory.py does NOT import from ingestion",
        )
        _check(
            "memory.imports_grammar",
            "from app.engines.grammar_engine" in mem_src,
            "memory.py imports from grammar_engine"
            if "from app.engines.grammar_engine" in mem_src
            else "memory.py does NOT import from grammar_engine",
        )
        _check(
            "memory.writes_edges",
            "INSERT INTO edges" in mem_src,
            "memory.py writes to edges table"
            if "INSERT INTO edges" in mem_src
            else "memory.py does NOT write to edges table",
        )
        _check(
            "memory.writes_edge_extraction",
            "INTO edge_extraction" in mem_src,
            "memory.py writes to edge_extraction table"
            if "INTO edge_extraction" in mem_src
            else "memory.py does NOT write to edge_extraction table",
        )

    except Exception as e:
        _check("memory.import", False, f"Cannot import memory: {e}")

    # =================================================================
    # ENGINE 5: RECONSTRUCTION (reconstruction.py) — THE READER
    # =================================================================
    try:
        from app.engines import reconstruction

        for fn_name in ["reconstruct", "resolve_query_entities"]:
            has_fn = hasattr(reconstruction, fn_name) and callable(
                getattr(reconstruction, fn_name)
            )
            _check(
                f"reconstruction.{fn_name}",
                has_fn,
                f"{fn_name}() exists" if has_fn else f"{fn_name}() NOT found",
            )

        # Check read path wiring
        recon_src = inspect.getsource(reconstruction)
        _check(
            "reconstruction.reads_edges",
            "FROM edges" in recon_src,
            "reconstruction.py reads from edges table"
            if "FROM edges" in recon_src
            else "reconstruction.py does NOT read from edges table",
        )
        _check(
            "reconstruction.uses_edges_fts",
            "edges_fts" in recon_src,
            "reconstruction.py uses edges_fts"
            if "edges_fts" in recon_src
            else "reconstruction.py does NOT use edges_fts",
        )

    except Exception as e:
        _check("reconstruction.import", False, f"Cannot import reconstruction: {e}")

    # =================================================================
    # DB SCHEMA VERIFICATION
    # =================================================================
    if db_path:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            def _table_exists(name: str) -> bool:
                r = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (name,),
                ).fetchone()
                return r is not None

            def _table_has_column(table: str, col: str) -> bool:
                try:
                    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                    return col in cols
                except Exception:
                    return False

            # Core tables
            for table in ["edges", "edge_extraction", "entities", "arcs", "facts"]:
                exists = _table_exists(table)
                _check(
                    f"db.{table}",
                    exists,
                    f"{table} table exists" if exists else f"{table} table MISSING",
                )

            # FTS5
            fts_exists = _table_exists("edges_fts")
            _check(
                "db.edges_fts",
                fts_exists,
                "edges_fts exists" if fts_exists else "edges_fts MISSING",
            )

            # edges columns — all 38
            edges_cols = [
                # Core Edge
                "id", "user_id", "subject", "predicate", "object",
                "source_text", "source_text_hash", "created_at",
                # Five Traces
                "edge_schematic_category", "edge_temporal_context",
                "edge_relational_type", "edge_episodic_significance",
                "edge_emotional_valence", "edge_emotional_label",
                # Temporal Truth State
                "source_timestamp", "temporal_expression", "resolved_event_date",
                "is_historical", "is_current", "superseded_at", "superseded_by",
                # Lifecycle
                "tombstoned_at", "first_learned_at", "last_confirmed_at",
                # Structure
                "cluster_id", "arc_id", "sequence_number", "relational_entities",
                # Retrieval Aids
                "edge_embedding", "predicate_embedding", "subject_type", "object_type",
                # Predicted Queries
                "pq_1", "pq_2", "pq_3", "pq_4",
                # Verified Queries
                "vq_1", "vq_2",
            ]
            if _table_exists("edges"):
                for col in edges_cols:
                    has = _table_has_column("edges", col)
                    _check(
                        f"db.edges.{col}",
                        has,
                        f"edges.{col} exists" if has else f"edges.{col} MISSING",
                    )

            # edge_extraction columns — all 16
            extraction_cols = [
                "edge_id",
                "edge_negated", "edge_mood", "episodic_fact",
                "emotional_target", "extraction_rule", "canonical_fields",
                "edge_affiliation",
                "provenance_memory_id", "tombstone_reason", "tombstone_op_id", "source_tag",
                "confidence", "utterance_type_id",
                "subject_type_confidence", "object_type_confidence",
            ]
            if _table_exists("edge_extraction"):
                for col in extraction_cols:
                    has = _table_has_column("edge_extraction", col)
                    _check(
                        f"db.edge_extraction.{col}",
                        has,
                        f"edge_extraction.{col} exists"
                        if has else f"edge_extraction.{col} MISSING",
                    )

            # Verify NO old tables
            for old_table in ["relationships", "predicted_queries", "relationships_fts"]:
                exists = _table_exists(old_table)
                _check(
                    f"db.no_{old_table}",
                    not exists,
                    f"{old_table} correctly absent"
                    if not exists
                    else f"WARNING: legacy {old_table} table still exists",
                )

            conn.close()

        except Exception as e:
            _check("db.connection", False, f"Cannot connect to DB: {e}")

    # =================================================================
    # SUMMARY
    # =================================================================
    passes = sum(1 for r in results if r["status"] == "PASS")
    fails = sum(1 for r in results if r["status"] == "FAIL")
    _check(
        "summary",
        fails == 0,
        f"{passes} passed, {fails} failed out of {len(results) - 1} checks",
    )

    return results


def print_report(results: List[Dict[str, str]]) -> None:
    """Pretty-print verification results."""
    for r in results:
        status = r["status"]
        marker = "  PASS" if status == "PASS" else "  FAIL"
        print(f"{marker}: {r['check']} -- {r['detail']}")


# CLI entry point
if __name__ == "__main__":
    results = verify_kenotic_architecture_v1()
    print_report(results)
