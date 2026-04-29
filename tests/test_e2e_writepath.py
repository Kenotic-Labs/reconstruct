#!/usr/bin/env python
"""
End-to-end write path test: ingest 20 sentences through MemoryEngine.ingest_text(),
then query SQLite directly to verify every thread was laid.

Run: cd /d/Nura/Code/nura_living_memory_code && py -3.10 tests/test_e2e_writepath.py
"""
import json
import os
import sqlite3
import sys
import tempfile
import time
import traceback
from pathlib import Path

# -- Path setup --
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -- Create a fresh temp DB and monkeypatch settings before any app imports --
_tmpdir = tempfile.mkdtemp(prefix="nura_e2e_")
DB_PATH = os.path.join(_tmpdir, "test_e2e.db")

# Patch settings before any app code imports it
from config.settings import settings
settings.sqlite_path = DB_PATH

# Now set up the database
from app.db.models import MIGRATIONS, run_schema_upgrades

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row
conn.executescript(MIGRATIONS)
conn.commit()
run_schema_upgrades(conn)
conn.close()

# -- Now import the engine (it will use patched settings.sqlite_path) --
from app.engines.memory import MemoryEngine

engine = MemoryEngine()

# =============================================================================
# TEST DATA
# =============================================================================
sentences = [
    "I work at Google as a software engineer.",           # 1  fact (career)
    "I moved to Portland last year.",                     # 2  fact (location), temporal
    "My mom's name is Linda.",                            # 3  fact (family)
    "I got married in June 2024.",                        # 4  milestone
    "I graduated from Stanford.",                         # 5  milestone (education)
    "I had coffee with Jake this morning.",               # 6  episodic, temporal
    "I'm really nervous about the interview tomorrow.",   # 7  emotional, temporal
    "I just started at Apple.",                           # 8  fact (career) - contradicts #1
    "My cat Whiskers is 3 years old.",                    # 9  fact (pet/family)
    "I've been training for a marathon.",                 # 10 episodic
    "Actually I graduated in 2020 not 2019.",             # 11 correction
    "I quit my job.",                                     # 12 milestone/career
    "Jake and I went hiking last Saturday.",              # 13 episodic, temporal
    "The weather was nice.",                              # 14 episodic (noise)
    "I'm allergic to peanuts.",                           # 15 fact (health)
    "My sister Sarah just had a baby.",                   # 16 milestone (family)
    "I used to live in Boston.",                          # 17 fact (historical), temporal
    "I love playing guitar.",                             # 18 fact (hobby)
    "I feel guilty about not visiting my grandmother.",   # 19 emotional
    "We met at a conference in 2019, started dating, got engaged, and now we're getting married.",  # 20 milestone (compound)
]

# =============================================================================
# INGEST
# =============================================================================
print("=" * 80)
print("E2E WRITE PATH TEST -- 20 Sentences")
print("=" * 80)
print(f"DB: {DB_PATH}")
print()

total_rows = 0
start = time.time()
for i, sentence in enumerate(sentences, 1):
    t0 = time.time()
    try:
        count = engine.ingest_text(user_id=1, text=sentence, speaker="Sam")
        elapsed = time.time() - t0
        print(f"  [{i:2d}] {count} edge(s) in {elapsed:.2f}s: {sentence[:60]}")
        total_rows += count
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{i:2d}] ERROR in {elapsed:.2f}s: {e}")
        traceback.print_exc()

total_time = time.time() - start
print(f"\nIngested {total_rows} total edges in {total_time:.1f}s\n")

# =============================================================================
# QUERY AND PRINT
# =============================================================================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row


def rget(row, col, default=None):
    """Safe get from sqlite3.Row -- returns default if column missing or value is None."""
    try:
        val = row[col]
        return val if val is not None else default
    except (IndexError, KeyError):
        return default


# -- RELATIONSHIPS --
rows = conn.execute("SELECT * FROM relationships WHERE user_id = 1 ORDER BY id").fetchall()
print(f"\n=== RELATIONSHIPS: {len(rows)} rows ===")
for r in rows:
    print(f"  #{r['id']}: S={r['subject']} P={r['predicate']} O={r['object']}")
    print(f"    schema={r['edge_schematic_category']} sig={r['edge_episodic_significance']} mood={rget(r, 'edge_mood')}")
    print(f"    emotion={r['edge_emotional_label']} target={rget(r, 'emotional_target')} temporal={rget(r, 'temporal_expression')}")
    print(f"    temporal_ctx={r['edge_temporal_context']} historical={rget(r, 'is_historical')} rule={rget(r, 'extraction_rule')}")
    print(f"    edge_emb={'BLOB' if rget(r, 'edge_embedding') else 'NULL'} pred_emb={'BLOB' if rget(r, 'predicate_embedding') else 'NULL'}")
    print(f"    affiliation={rget(r, 'edge_affiliation')} cluster={rget(r, 'cluster_id')} arc={rget(r, 'arc_id')}")
    print(f"    subj_type={rget(r, 'subject_type')} obj_type={rget(r, 'object_type')}")
    print(f"    rel_entities={rget(r, 'relational_entities')} is_current={rget(r, 'is_current')}")
    print(f"    source_text={r['source_text'][:80] if r['source_text'] else 'NULL'}")
    print()

# -- FACTS --
facts = conn.execute("SELECT * FROM facts WHERE user_id = 1 ORDER BY id").fetchall()
print(f"\n=== FACTS: {len(facts)} rows ===")
for f in facts:
    print(f"  key={f['key']} value={f['value']} history={rget(f, 'history')}")

# -- MILESTONES --
miles = conn.execute("SELECT * FROM milestones WHERE user_id = 1 ORDER BY id").fetchall()
print(f"\n=== MILESTONES: {len(miles)} rows ===")
for m in miles:
    print(f"  type={m['event_type']} desc={m['description'][:80]}")

# -- PREDICTED QUERIES --
pqs = conn.execute("SELECT * FROM predicted_queries WHERE user_id = 1 ORDER BY id").fetchall()
print(f"\n=== PREDICTED QUERIES: {len(pqs)} rows ===")
for p in pqs[:5]:
    print(f"  rel_id={p['relationship_id']} q={p['predicted_question'][:60]} emb={'BLOB' if p['question_embedding'] else 'NULL'}")
if len(pqs) > 5:
    print(f"  ... and {len(pqs) - 5} more")

# -- FTS5 --
print("\n=== FTS5 TESTS ===")
try:
    fts = conn.execute("SELECT rowid FROM relationships_fts WHERE relationships_fts MATCH 'Google'").fetchall()
    print(f"  FTS5 MATCH 'Google': {len(fts)} results, rowids={[r['rowid'] for r in fts]}")
except Exception as e:
    print(f"  FTS5 MATCH 'Google': ERROR {e}")

try:
    fts2 = conn.execute("SELECT rowid FROM relationships_fts WHERE relationships_fts MATCH 'Jake'").fetchall()
    print(f"  FTS5 MATCH 'Jake': {len(fts2)} results, rowids={[r['rowid'] for r in fts2]}")
except Exception as e:
    print(f"  FTS5 MATCH 'Jake': ERROR {e}")

# -- CLUSTER VERIFICATION --
print("\n=== CLUSTER VERIFICATION ===")
jake_rows = conn.execute(
    "SELECT id, source_text, cluster_id FROM relationships WHERE user_id = 1 AND (subject LIKE '%Jake%' OR object LIKE '%Jake%' OR relational_entities LIKE '%Jake%')"
).fetchall()
clusters = set()
for jr in jake_rows:
    print(f"  Jake edge #{jr['id']}: cluster={jr['cluster_id']} text={(jr['source_text'] or '')[:50]}")
    if jr['cluster_id']:
        clusters.add(jr['cluster_id'])
print(f"  Jake edges share cluster: {'PASS' if len(clusters) == 1 and len(jake_rows) >= 2 else 'FAIL'} (clusters={clusters})")

weather = conn.execute(
    "SELECT id, cluster_id FROM relationships WHERE user_id = 1 AND source_text LIKE '%weather%'"
).fetchall()
for w in weather:
    weather_cluster = w['cluster_id']
    print(f"  Weather edge #{w['id']}: cluster={weather_cluster}")
    print(f"  Weather has own cluster: {'PASS' if weather_cluster and weather_cluster not in clusters else 'FAIL'}")

# =============================================================================
# PASS/FAIL GRID
# =============================================================================
print("\n" + "=" * 80)
print("PASS / FAIL GRID")
print("=" * 80)

# Build lookup: sentence index -> list of relationship rows
# We match by checking if the source_text of the relationship contains key words
# from each sentence. Since grammar engine may split/transform, we use a broader match.
all_rels = conn.execute("SELECT * FROM relationships WHERE user_id = 1 ORDER BY id").fetchall()
all_facts = conn.execute("SELECT * FROM facts WHERE user_id = 1").fetchall()
all_miles = conn.execute("SELECT * FROM milestones WHERE user_id = 1").fetchall()

# Sentence keywords for matching to edges
sentence_keywords = [
    ["Google", "software engineer"],       # 1
    ["Portland", "moved"],                 # 2
    ["Linda", "mom"],                      # 3
    ["married", "June 2024"],              # 4
    ["graduated", "Stanford"],             # 5
    ["coffee", "Jake"],                    # 6
    ["nervous", "interview"],              # 7
    ["Apple", "started"],                  # 8
    ["Whiskers", "cat"],                   # 9
    ["marathon", "training"],              # 10
    ["graduated", "2020"],                 # 11
    ["quit", "job"],                       # 12
    ["hiking", "Jake"],                    # 13
    ["weather", "nice"],                   # 14
    ["allergic", "peanuts"],               # 15
    ["Sarah", "baby"],                     # 16
    ["Boston", "used to"],                 # 17
    ["guitar", "love"],                    # 18
    ["guilty", "grandmother"],             # 19
    ["conference", "dating", "engaged", "married"],  # 20
]


def find_edges_for_sentence(idx):
    """Find relationship rows that match sentence keywords."""
    kws = sentence_keywords[idx]
    matched = []
    for r in all_rels:
        src = (r['source_text'] or '').lower()
        subj = (r['subject'] or '').lower()
        pred = (r['predicate'] or '').lower()
        obj = (r['object'] or '').lower()
        combined = f"{src} {subj} {pred} {obj}"
        if any(kw.lower() in combined for kw in kws):
            matched.append(r)
    return matched


# Expected checks per sentence
fact_sentences = {0, 1, 2, 8, 14, 16, 17}       # 1,2,3,9,15,17,18 (0-indexed)
milestone_sentences = {3, 4, 15, 19}              # 4,5,16,20
emotional_sentences = {6, 18}                      # 7,19
temporal_sentences = {1, 5, 6, 16}                 # 2,6,7,17 (0-indexed)
historical_sentences = {16}                         # 17

results = []
for idx in range(20):
    edges = find_edges_for_sentence(idx)
    checks = {}

    # 1. Has at least one edge?
    checks["has_edge"] = len(edges) > 0

    # 2. Trace columns non-default?
    if edges:
        has_non_default_schema = any(
            (r['edge_schematic_category'] or 'uncategorized') != 'uncategorized'
            for r in edges
        )
        checks["schema_set"] = has_non_default_schema
    else:
        checks["schema_set"] = False

    # 3. Fact check
    if idx in fact_sentences:
        # Check if a facts row exists with matching content
        found_fact = False
        for f in all_facts:
            fkey = (f['key'] or '').lower()
            fval = (f['value'] or '').lower()
            combined = f"{fkey} {fval}"
            kws = sentence_keywords[idx]
            if any(kw.lower() in combined for kw in kws):
                found_fact = True
                break
        checks["fact_created"] = found_fact

    # 4. Milestone check
    if idx in milestone_sentences:
        found_milestone = False
        for m in all_miles:
            desc = (m['description'] or '').lower()
            kws = sentence_keywords[idx]
            if any(kw.lower() in desc for kw in kws):
                found_milestone = True
                break
        checks["milestone_created"] = found_milestone

    # 5. Emotional check
    if idx in emotional_sentences:
        has_emotion = any(
            r['edge_emotional_label'] is not None and r['edge_emotional_label'] != ''
            for r in edges
        )
        checks["emotional_label"] = has_emotion

    # 6. Temporal check
    if idx in temporal_sentences:
        has_temporal = any(
            rget(r, 'temporal_expression') is not None and rget(r, 'temporal_expression') != ''
            for r in edges
        )
        checks["temporal_expr"] = has_temporal

    # 7. Historical check
    if idx in historical_sentences:
        has_historical = any(
            rget(r, 'is_historical') == 1
            for r in edges
        )
        checks["is_historical"] = has_historical

    # 8. Contradiction check (sentence 8 vs sentence 1)
    if idx == 7:  # sentence 8 (0-indexed=7)
        # Check if facts history shows old career value
        found_history = False
        for f in all_facts:
            fkey = (f['key'] or '').lower()
            history = rget(f, 'history') or ''
            if 'career' in fkey and history and 'Google' in history:
                found_history = True
                break
        checks["contradiction_history"] = found_history

    results.append((idx, edges, checks))

# Print grid
total_pass = 0
total_fail = 0
total_checks = 0
for idx, edges, checks in results:
    sentence_num = idx + 1
    status_parts = []
    for check_name, passed in checks.items():
        total_checks += 1
        if passed:
            total_pass += 1
            status_parts.append(f"{check_name}=PASS")
        else:
            total_fail += 1
            status_parts.append(f"{check_name}=FAIL")
    all_pass = all(checks.values()) if checks else False
    icon = "PASS" if all_pass else "FAIL"
    n_edges = len(edges)
    print(f"  [{sentence_num:2d}] {icon} ({n_edges} edges) {' | '.join(status_parts)}")
    print(f"       {sentences[idx][:70]}")

print(f"\n{'=' * 80}")
print(f"SUMMARY: {total_pass}/{total_checks} checks passed, {total_fail} failed")
print(f"Total relationship edges: {len(all_rels)}")
print(f"Total facts: {len(all_facts)}")
print(f"Total milestones: {len(all_miles)}")
print(f"Total predicted queries: {len(pqs)}")
print(f"{'=' * 80}")

# -- ENTITIES table --
entities = conn.execute("SELECT * FROM entities WHERE user_id = 1 ORDER BY id").fetchall()
print(f"\n=== ENTITIES: {len(entities)} rows ===")
for e in entities:
    print(f"  name={e['name']} type={e['entity_type']} mentions={e['mention_count']}")

conn.close()

# Cleanup
print(f"\nDB preserved at: {DB_PATH}")
print("Done.")
