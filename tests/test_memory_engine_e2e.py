"""
Memory Engine End-to-End Test
=============================
20 sentences through full pipeline: ingestion -> cleanup -> grammar engine -> memory engine -> SQLite.

Root cause context (from source reading):
- MemoryEngine.store() uses 3-phase write: PREPARE -> WRITE -> SIDE-EFFECTS
- Facts dedup on VerbClass key (schema::VerbClass::subject) via WordNet hypernyms
- Correction detection: grammar engine sets extraction_rule containing "correction"
- Reversal detection: grammar engine sets negated=True on TraceDecomposition
- Clustering: _assign_cluster() uses cosine similarity of edge_embeddings
- FTS5: synced on every INSERT/UPDATE in _write_row()
- Entities: populated from relational_entities (NER-filtered, not raw S/P/O)

59 checks across 10 verification categories.
"""
import json
import os
import sys
import sqlite3
import traceback
from pathlib import Path
from datetime import datetime

# ── Setup paths ──
PROJECT_ROOT = Path(r"D:\Nura\Code\nura_living_memory_code")
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Test DB: fresh every run ──
TEST_DB = PROJECT_ROOT / "Memory Storage" / "test_e2e.db"

def setup_test_db():
    """Delete existing test DB and create fresh one with migrations."""
    if TEST_DB.exists():
        TEST_DB.unlink()
    for suffix in (".db-wal", ".db-shm"):
        p = TEST_DB.with_suffix(suffix)
        if p.exists():
            p.unlink()

    os.environ["NURA_SQLITE_PATH"] = str(TEST_DB)

    from config.settings import Settings
    import config.settings as settings_mod
    settings_mod.settings = Settings(sqlite_path=str(TEST_DB))

    from app.db.session import init_db
    init_db(str(TEST_DB))
    print(f"[SETUP] Fresh DB at {TEST_DB}")


SENTENCES = [
    "I work at Google as a software engineer.",
    "I moved to Portland last year.",
    "My mom's name is Linda.",
    "I got married in June 2024.",
    "I graduated from Stanford.",
    "I had coffee with Jake this morning.",
    "I'm really nervous about the interview tomorrow.",
    "I just started at Apple.",
    "My cat Whiskers is 3 years old.",
    "I've been training for a marathon.",
    "Actually I graduated in 2020 not 2019.",
    "I quit my job.",
    "Jake and I went hiking last Saturday.",
    "The weather was nice.",
    "I'm allergic to peanuts.",
    "My sister Sarah just had a baby.",
    "I used to live in Boston.",
    "I love playing guitar.",
    "I feel guilty about not visiting my grandmother.",
    "We met at a conference in 2019, started dating, got engaged, and now we're getting married.",
]


class CheckResult:
    def __init__(self, check_id: str, category: str, description: str):
        self.check_id = check_id
        self.category = category
        self.description = description
        self.passed = False
        self.detail = ""

    def mark(self, passed: bool, detail: str = ""):
        self.passed = passed
        self.detail = detail
        return self


results: list[CheckResult] = []


def check(check_id: str, category: str, description: str) -> CheckResult:
    c = CheckResult(check_id, category, description)
    results.append(c)
    return c


def run():
    setup_test_db()

    from app.engines.memory import MemoryEngine
    engine = MemoryEngine()

    print("\n[INGEST] Processing 20 sentences...")
    ingest_counts = []
    for i, sentence in enumerate(SENTENCES):
        try:
            count = engine.ingest_text(user_id=1, text=sentence, speaker="Sam")
            ingest_counts.append(count)
            print(f"  [{i:2d}] {count} rows <- {sentence[:60]}")
        except Exception as e:
            ingest_counts.append(0)
            print(f"  [{i:2d}] ERROR: {e}")

    conn = sqlite3.connect(str(TEST_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # ====================================================================
    # 1. RELATIONSHIPS TABLE
    # ====================================================================
    print("\n" + "="*70)
    print("1. RELATIONSHIPS TABLE")
    print("="*70)

    rels = conn.execute("SELECT * FROM relationships WHERE user_id = 1 ORDER BY id").fetchall()
    rel_count = len(rels)
    print(f"  Total rows: {rel_count}")

    col_name_set = set(c[1] for c in conn.execute("PRAGMA table_info(relationships)").fetchall())

    for r in rels:
        r_dict = dict(r)
        has_edge_emb = r_dict.get("edge_embedding") is not None
        has_pred_emb = r_dict.get("predicate_embedding") is not None
        print(f"  id={r_dict['id']:3d} | S={str(r_dict.get('subject',''))[:20]:20s} "
              f"| P={str(r_dict.get('predicate',''))[:20]:20s} "
              f"| O={str(r_dict.get('object',''))[:20]:20s} "
              f"| cat={str(r_dict.get('edge_schematic_category',''))[:15]:15s} "
              f"| sig={str(r_dict.get('edge_episodic_significance',''))[:10]:10s} "
              f"| mood={str(r_dict.get('edge_mood',''))[:10]:10s} "
              f"| hist={r_dict.get('is_historical', 0)} "
              f"| rule={str(r_dict.get('extraction_rule',''))[:15]:15s} "
              f"| cluster={str(r_dict.get('cluster_id',''))[:8]:8s} "
              f"| arc={str(r_dict.get('arc_id',''))[:8]:8s} "
              f"| aff={r_dict.get('edge_affiliation', '')} "
              f"| e_emb={'Y' if has_edge_emb else 'N'} "
              f"| p_emb={'Y' if has_pred_emb else 'N'}")

    check("R01", "relationships", "At least 20 relationship rows created").mark(
        rel_count >= 20, f"Got {rel_count} rows")
    has_subjects = sum(1 for r in rels if r["subject"])
    check("R02", "relationships", "At least one row has a non-null subject").mark(
        has_subjects > 0, f"{has_subjects} rows have subjects")
    has_predicates = sum(1 for r in rels if r["predicate"])
    check("R03", "relationships", "At least one row has a non-null predicate").mark(
        has_predicates > 0, f"{has_predicates} rows have predicates")
    has_objects = sum(1 for r in rels if r["object"])
    check("R04", "relationships", "At least one row has a non-null object").mark(
        has_objects > 0, f"{has_objects} rows have objects")
    has_emb = sum(1 for r in rels if r["edge_embedding"] is not None)
    check("R05", "relationships", "At least one row has edge_embedding").mark(
        has_emb > 0, f"{has_emb} rows have edge_embedding")
    has_pred_emb = sum(1 for r in rels if "predicate_embedding" in col_name_set and r["predicate_embedding"] is not None)
    check("R06", "relationships", "At least one row has predicate_embedding").mark(
        has_pred_emb > 0, f"{has_pred_emb} rows have predicate_embedding")

    google_rows = [r for r in rels if "Google" in (r["subject"] or "") or "Google" in (r["object"] or "") or "Google" in (r["source_text"] or "")]
    check("R07", "relationships", "Google appears in at least one row").mark(
        len(google_rows) > 0, f"{len(google_rows)} rows mention Google")
    jake_rows = [r for r in rels if "Jake" in (r["subject"] or "") or "Jake" in (r["object"] or "") or "Jake" in (r["source_text"] or "")]
    check("R08", "relationships", "Jake appears in at least one row").mark(
        len(jake_rows) > 0, f"{len(jake_rows)} rows mention Jake")
    stanford_rows = [r for r in rels if "Stanford" in (r["subject"] or "") or "Stanford" in (r["object"] or "") or "Stanford" in (r["source_text"] or "")]
    check("R09", "relationships", "Stanford appears in at least one row").mark(
        len(stanford_rows) > 0, f"{len(stanford_rows)} rows mention Stanford")
    apple_rows = [r for r in rels if "Apple" in (r["subject"] or "") or "Apple" in (r["object"] or "") or "Apple" in (r["source_text"] or "")]
    check("R10", "relationships", "Apple appears in at least one row").mark(
        len(apple_rows) > 0, f"{len(apple_rows)} rows mention Apple")

    non_default_cat = [r for r in rels if r["edge_schematic_category"] not in (None, "uncategorized")]
    check("R11", "relationships", "At least one row has non-default schematic_category").mark(
        len(non_default_cat) > 0, f"{len(non_default_cat)} rows have non-default category")
    non_routine = [r for r in rels if r["edge_episodic_significance"] not in (None, "routine")]
    check("R12", "relationships", "At least one row has non-routine significance").mark(
        len(non_routine) > 0, f"{len(non_routine)} rows have non-routine significance")
    # Root cause: sqlite3.Row lacks .get(); convert to plain dict for safe access
    rels_d = [dict(r) for r in rels]

    non_default_mood_count = sum(1 for r in rels_d if r.get("edge_mood") not in (None, "indicative"))
    check("R13", "relationships", "At least one row has non-indicative mood").mark(
        non_default_mood_count > 0, f"{non_default_mood_count} rows have non-indicative mood")

    historical = [r for r in rels_d if r.get("is_historical") == 1]
    check("R14", "relationships", "At least one row is_historical=1 (used to live in Boston)").mark(
        len(historical) > 0, f"{len(historical)} historical rows")
    has_rule = [r for r in rels_d if r.get("extraction_rule")]
    check("R15", "relationships", "At least one row has extraction_rule").mark(
        len(has_rule) > 0, f"{len(has_rule)} rows have extraction_rule")
    has_cluster = [r for r in rels_d if r.get("cluster_id")]
    check("R16", "relationships", "At least one row has cluster_id").mark(
        len(has_cluster) > 0, f"{len(has_cluster)} rows have cluster_id")
    has_arc = [r for r in rels_d if r.get("arc_id")]
    check("R17", "relationships", "At least one row has arc_id").mark(
        len(has_arc) > 0, f"{len(has_arc)} rows have arc_id")
    has_aff = [r for r in rels_d if r.get("edge_affiliation") is not None]
    check("R18", "relationships", "At least one row has edge_affiliation").mark(
        len(has_aff) > 0, f"{len(has_aff)} rows have edge_affiliation")
    # Verify grammar engine produced varied emotional valences (not all identical)
    distinct_valences = set(r.get("edge_emotional_valence") for r in rels_d if r.get("edge_emotional_valence") is not None)
    check("R19", "relationships", "At least one row has non-default emotional_valence").mark(
        len(distinct_valences) > 1, f"Distinct valences: {distinct_valences}")
    is_current_rows = [r for r in rels_d if r.get("is_current") == 1]
    check("R20", "relationships", "Most rows have is_current=1").mark(
        len(is_current_rows) >= rel_count - 5, f"{len(is_current_rows)}/{rel_count} rows are is_current=1")

    # ====================================================================
    # 2. FACTS TABLE
    # ====================================================================
    print("\n" + "="*70)
    print("2. FACTS TABLE")
    print("="*70)

    facts_raw = conn.execute("SELECT * FROM facts WHERE user_id = 1 ORDER BY id").fetchall()
    facts = [dict(f) for f in facts_raw]
    fact_count = len(facts)
    print(f"  Total rows: {fact_count}")
    for f in facts:
        f_dict = dict(f)
        print(f"  key={f_dict['key'][:50]:50s} | value={str(f_dict['value'])[:40]:40s} | history={f_dict.get('history', 'NULL')}")

    check("F01", "facts", "At least 1 fact row created").mark(fact_count >= 1, f"Got {fact_count} fact rows")
    check("F02", "facts", "At least 3 fact rows created").mark(fact_count >= 3, f"Got {fact_count} fact rows")
    check("F03", "facts", "At least one fact has non-null history").mark(
        any(f["history"] and f["history"] != "[]" for f in facts),
        "History chain exists" if any(f["history"] and f["history"] != "[]" for f in facts) else "No history entries")

    career_facts = [f for f in facts if "Google" in (f["value"] or "") or "Apple" in (f["value"] or "") or "Google" in (f["key"] or "")]
    check("F04", "facts", "Career-related fact exists (Google or Apple)").mark(
        len(career_facts) > 0, f"{len(career_facts)} career facts found")
    allergy_facts = [f for f in facts if "allerg" in (f["key"] or "").lower() or "peanut" in (f["value"] or "").lower() or "allerg" in (f["value"] or "").lower()]
    check("F05", "facts", "Allergy fact exists (peanuts)").mark(
        len(allergy_facts) > 0, f"{len(allergy_facts)} allergy facts found")

    # ====================================================================
    # 3. MILESTONES TABLE
    # ====================================================================
    print("\n" + "="*70)
    print("3. MILESTONES TABLE")
    print("="*70)

    milestones_raw = conn.execute("SELECT * FROM milestones WHERE user_id = 1 ORDER BY id").fetchall()
    milestones = [dict(m) for m in milestones_raw]
    ms_count = len(milestones)
    print(f"  Total rows: {ms_count}")
    for m in milestones:
        m_dict = dict(m)
        print(f"  event_type={str(m_dict.get('event_type',''))[:25]:25s} | desc={str(m_dict.get('description',''))[:60]}")

    check("M01", "milestones", "At least 1 milestone row created").mark(ms_count >= 1, f"Got {ms_count} milestone rows")
    check("M02", "milestones", "At least 2 milestone rows created").mark(ms_count >= 2, f"Got {ms_count} milestone rows")

    marriage_ms = [m for m in milestones if "marri" in (m["description"] or "").lower()]
    check("M03", "milestones", "Marriage milestone exists").mark(len(marriage_ms) > 0, f"{len(marriage_ms)} marriage milestones")
    grad_ms = [m for m in milestones if "graduat" in (m["description"] or "").lower() or "stanford" in (m["description"] or "").lower()]
    check("M04", "milestones", "Graduation milestone exists").mark(len(grad_ms) > 0, f"{len(grad_ms)} graduation milestones")
    baby_ms = [m for m in milestones if "baby" in (m["description"] or "").lower() or "Sarah" in (m["description"] or "")]
    check("M05", "milestones", "Baby milestone exists (sister Sarah)").mark(len(baby_ms) > 0, f"{len(baby_ms)} baby milestones")

    # ====================================================================
    # 4. ENTITIES TABLE
    # ====================================================================
    print("\n" + "="*70)
    print("4. ENTITIES TABLE")
    print("="*70)

    entities_raw = conn.execute("SELECT * FROM entities WHERE user_id = 1 ORDER BY name").fetchall()
    entities = [dict(e) for e in entities_raw]
    ent_count = len(entities)
    print(f"  Total rows: {ent_count}")
    for e in entities:
        e_dict = dict(e)
        print(f"  name={str(e_dict['name']):20s} | type={str(e_dict.get('entity_type',''))[:15]:15s} | mentions={e_dict.get('mention_count', 0)}")

    check("E01", "entities", "At least 1 entity row created").mark(ent_count >= 1, f"Got {ent_count} entity rows")
    check("E02", "entities", "At least 3 entity rows created").mark(ent_count >= 3, f"Got {ent_count} entity rows")

    jake_ents = [e for e in entities if "jake" in e["name"].lower()]
    check("E03", "entities", "Jake entity exists").mark(len(jake_ents) > 0, f"Found: {[e['name'] for e in jake_ents]}")
    jake_mentions = jake_ents[0]["mention_count"] if jake_ents else 0
    check("E04", "entities", "Jake mention_count >= 2").mark(jake_mentions >= 2, f"Jake mention_count={jake_mentions}")
    family_ents = [e for e in entities if e["name"].lower() in ("linda", "sarah")]
    check("E05", "entities", "Linda or Sarah entity exists").mark(len(family_ents) > 0, f"Found: {[e['name'] for e in family_ents]}")

    # ====================================================================
    # 5. PREDICTED QUERIES TABLE
    # ====================================================================
    print("\n" + "="*70)
    print("5. PREDICTED QUERIES TABLE")
    print("="*70)

    pqs_raw = conn.execute("SELECT * FROM predicted_queries WHERE user_id = 1 ORDER BY id").fetchall()
    pqs = [dict(pq) for pq in pqs_raw]
    pq_count = len(pqs)
    print(f"  Total rows: {pq_count}")
    for i, pq in enumerate(pqs[:5]):
        pq_dict = dict(pq)
        print(f"  [{i}] Q={str(pq_dict['predicted_question'])[:50]:50s} | A={str(pq_dict['answer_text'])[:40]}")

    check("PQ01", "predicted_queries", "At least 1 predicted query created").mark(pq_count >= 1, f"Got {pq_count} predicted queries")
    check("PQ02", "predicted_queries", "At least 10 predicted queries created").mark(pq_count >= 10, f"Got {pq_count} predicted queries")
    has_q_emb = sum(1 for pq in pqs if pq["question_embedding"] is not None)
    check("PQ03", "predicted_queries", "All predicted queries have question_embedding").mark(
        has_q_emb == pq_count and pq_count > 0, f"{has_q_emb}/{pq_count} have embeddings")

    # ====================================================================
    # 6. FTS5 TEST
    # ====================================================================
    print("\n" + "="*70)
    print("6. FTS5 TEST")
    print("="*70)

    try:
        fts_google = [dict(r) for r in conn.execute("SELECT rowid FROM relationships_fts WHERE relationships_fts MATCH 'Google'").fetchall()]
        print(f"  FTS5 MATCH 'Google': {len(fts_google)} rows, rowids={[r['rowid'] for r in fts_google]}")
        check("FTS01", "fts5", "FTS5 table exists and is queryable").mark(True, "FTS5 works")
        check("FTS02", "fts5", "FTS5 MATCH 'Google' returns results").mark(len(fts_google) > 0, f"{len(fts_google)} rows")
    except Exception as e:
        print(f"  FTS5 ERROR: {e}")
        check("FTS01", "fts5", "FTS5 table exists and is queryable").mark(False, str(e))
        check("FTS02", "fts5", "FTS5 MATCH 'Google' returns results").mark(False, str(e))

    try:
        fts_jake = [dict(r) for r in conn.execute("SELECT rowid FROM relationships_fts WHERE relationships_fts MATCH 'Jake'").fetchall()]
        print(f"  FTS5 MATCH 'Jake': {len(fts_jake)} rows, rowids={[r['rowid'] for r in fts_jake]}")
        check("FTS03", "fts5", "FTS5 MATCH 'Jake' returns results").mark(len(fts_jake) > 0, f"{len(fts_jake)} rows")
    except Exception as e:
        check("FTS03", "fts5", "FTS5 MATCH 'Jake' returns results").mark(False, str(e))

    try:
        fts_total = dict(conn.execute("SELECT COUNT(*) as c FROM relationships_fts").fetchone())["c"]
        check("FTS04", "fts5", "FTS5 row count matches relationships count").mark(
            fts_total == rel_count, f"FTS5={fts_total}, relationships={rel_count}")
    except Exception as e:
        check("FTS04", "fts5", "FTS5 row count matches relationships count").mark(False, str(e))

    # ====================================================================
    # 7. CLUSTER TEST
    # ====================================================================
    print("\n" + "="*70)
    print("7. CLUSTER TEST")
    print("="*70)

    jake_cluster_ids = set()
    for r in rels_d:
        src = r["source_text"] or ""
        if "Jake" in (r["subject"] or "") or "Jake" in (r["object"] or "") or "Jake" in src:
            cid = r.get("cluster_id")
            jake_cluster_ids.add(cid)
            print(f"  Jake row id={r['id']}, cluster_id={cid}")

    check("CL01", "cluster", "Jake rows found for cluster check").mark(
        len(jake_cluster_ids) > 0, f"{len(jake_cluster_ids)} distinct cluster_ids for Jake rows")
    jake_non_null_clusters = jake_cluster_ids - {None}
    check("CL02", "cluster", "At least one Jake row has a non-null cluster_id").mark(
        len(jake_non_null_clusters) > 0, f"Non-null cluster_ids: {jake_non_null_clusters}")

    if len(jake_non_null_clusters) >= 2:
        check("CL03", "cluster", "Jake rows share the same cluster_id").mark(
            False, f"Jake rows have {len(jake_non_null_clusters)} different cluster_ids: {jake_non_null_clusters}")
    elif len(jake_non_null_clusters) == 1:
        check("CL03", "cluster", "Jake rows share the same cluster_id").mark(
            True, f"All Jake rows share cluster_id={list(jake_non_null_clusters)[0]}")
    else:
        check("CL03", "cluster", "Jake rows share the same cluster_id").mark(
            False, "No Jake rows have cluster_id set")

    # ====================================================================
    # 8. CONTRADICTION TEST (Google -> Apple)
    # ====================================================================
    print("\n" + "="*70)
    print("8. CONTRADICTION TEST (Google -> Apple)")
    print("="*70)

    contradiction_found = False
    contradiction_type = None
    for f in facts:
        hist = f.get("history")
        val = f.get("value") or ""
        if not hist:
            continue
        try:
            h = json.loads(hist)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(h, list):
            continue
        for entry in h:
            old_val = entry.get("value", "")
            if "Google" in old_val or "Google" in val:
                contradiction_found = True
                contradiction_type = entry.get("type")
                print(f"  FOUND: key={f['key']}, current={val}, old={old_val}, type={contradiction_type}")

    check("CON01", "contradiction", "A fact row has Google in its history (superseded)").mark(
        contradiction_found, "Google was superseded" if contradiction_found else "No Google supersession found in facts history")

    superseded_rels = [dict(r) for r in conn.execute("SELECT * FROM relationships WHERE user_id = 1 AND is_current = 0").fetchall()]
    print(f"  Superseded relationships (is_current=0): {len(superseded_rels)}")
    for sr in superseded_rels:
        print(f"    id={sr['id']} S={sr['subject']} P={sr['predicate']} O={sr['object']} superseded_by={sr.get('superseded_by')}")

    check("CON02", "contradiction", "At least one relationship is superseded (is_current=0)").mark(
        len(superseded_rels) > 0, f"{len(superseded_rels)} superseded rows")
    google_superseded = [r for r in superseded_rels if "Google" in (r["source_text"] or "") or "Google" in (r["object"] or "")]
    check("CON03", "contradiction", "Google row specifically is superseded").mark(
        len(google_superseded) > 0, f"{len(google_superseded)} Google rows superseded")
    apple_current = [r for r in rels_d if r.get("is_current") == 1 and ("Apple" in (r["object"] or "") or "Apple" in (r["source_text"] or ""))]
    check("CON04", "contradiction", "Apple row is current (is_current=1)").mark(
        len(apple_current) > 0, f"{len(apple_current)} Apple rows are current")
    check("CON05", "contradiction", "Contradiction type is recorded").mark(
        contradiction_type is not None, f"type={contradiction_type}")

    # ====================================================================
    # 9. CORRECTION TEST (graduated 2020 not 2019)
    # ====================================================================
    print("\n" + "="*70)
    print("9. CORRECTION TEST (graduated 2020 not 2019)")
    print("="*70)

    correction_found = False
    correction_type = None
    for f in facts:
        hist = f.get("history")
        val = f.get("value") or ""
        key = f.get("key") or ""
        if not hist:
            continue
        try:
            h = json.loads(hist)
        except (json.JSONDecodeError, TypeError):
            continue
        for entry in h:
            if ("2020" in val or "Stanford" in val or "graduat" in key.lower()) and entry.get("type") == "correction":
                correction_found = True
                correction_type = entry.get("type")
                print(f"  FOUND: key={key}, current={val}, old={entry.get('value')}, type={correction_type}")

    check("CR01", "correction", "Graduation correction recorded in facts").mark(
        correction_found, "Correction found" if correction_found else "No correction entry found")

    any_correction = False
    for f in facts:
        hist = f.get("history")
        if not hist:
            continue
        try:
            h = json.loads(hist)
        except:
            continue
        for entry in h:
            if entry.get("type") == "correction":
                any_correction = True
                print(f"  ANY CORRECTION: key={f['key']}, old={entry.get('value')}, type=correction")

    check("CR02", "correction", "Any fact has type=correction in history").mark(
        any_correction, "correction type found" if any_correction else "No correction type in any fact history")

    correction_rels = [r for r in rels_d if "2020" in (r["source_text"] or "") and "2019" in (r["source_text"] or "")]
    check("CR03", "correction", "Correction sentence stored as relationship").mark(
        len(correction_rels) > 0, f"{len(correction_rels)} correction rows")

    has_correction_rule = any(
        r.get("extraction_rule") and "correction" in (r.get("extraction_rule") or "").lower()
        for r in correction_rels
    )
    check("CR04", "correction", "extraction_rule indicates correction").mark(
        has_correction_rule, "Rule found" if has_correction_rule else "No correction in extraction_rule")

    # ====================================================================
    # 10. REVERSAL TEST (I quit my job)
    # ====================================================================
    print("\n" + "="*70)
    print("10. REVERSAL TEST (I quit my job)")
    print("="*70)

    reversal_found = False
    for f in facts:
        hist = f.get("history")
        if not hist:
            continue
        try:
            h = json.loads(hist)
        except:
            continue
        for entry in h:
            if entry.get("type") == "reversal":
                reversal_found = True
                print(f"  FOUND reversal: key={f['key']}, old={entry.get('value')}, type=reversal")

    check("REV01", "reversal", "Any fact has type=reversal in history").mark(
        reversal_found, "reversal found" if reversal_found else "No reversal type in any fact history")

    quit_rels = [r for r in rels_d if "quit" in (r["source_text"] or "").lower() or "quit" in (r["predicate"] or "").lower()]
    check("REV02", "reversal", "Quit sentence stored as relationship").mark(
        len(quit_rels) > 0, f"{len(quit_rels)} quit rows")
    quit_negated = any(r.get("edge_negated") == 1 for r in quit_rels)
    check("REV03", "reversal", "Quit row has edge_negated=1").mark(
        quit_negated, "negated=1" if quit_negated else "Not negated")

    quit_supersession = any(
        r.get("superseded_by") is not None
        for r in superseded_rels
        if "work" in (r.get("predicate") or "").lower() or "job" in (r.get("source_text") or "").lower()
    )
    check("REV04", "reversal", "Quit caused a work-related supersession").mark(
        quit_supersession, "Supersession found" if quit_supersession else "No work supersession from quit")

    quit_sig = [r.get("edge_episodic_significance") for r in quit_rels]
    check("REV05", "reversal", "Quit has non-routine significance").mark(
        any(s not in (None, "routine") for s in quit_sig), f"Significance values: {quit_sig}")

    # ====================================================================
    # FINAL REPORT
    # ====================================================================
    conn.close()

    print("\n" + "="*70)
    print("FINAL REPORT: Memory Engine E2E Test")
    print("="*70)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    for cat, checks_in_cat in categories.items():
        cat_pass = sum(1 for c in checks_in_cat if c.passed)
        cat_total = len(checks_in_cat)
        print(f"\n  [{cat.upper()}] {cat_pass}/{cat_total}")
        for c in checks_in_cat:
            status = "PASS" if c.passed else "FAIL"
            print(f"    [{status}] {c.check_id}: {c.description}")
            if c.detail:
                print(f"           {c.detail}")

    print(f"\n{'='*70}")
    print(f"TOTAL: {passed}/{total} passed, {failed}/{total} failed")
    pct = (passed / total * 100) if total > 0 else 0
    print(f"PASS RATE: {pct:.1f}%")
    print(f"{'='*70}")

    return passed, total


if __name__ == "__main__":
    try:
        passed, total = run()
    except Exception as e:
        print(f"\n[FATAL] Test run failed: {e}")
        traceback.print_exc()
        sys.exit(1)
