"""Strict scale test: full write path + strict validation."""
from __future__ import annotations
import os, sys, time, sqlite3, statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Override DB path BEFORE imports
TEST_DB_PATH = str(PROJECT_ROOT / "Memory Storage" / "test_strict_scale.db")
os.environ["NURA_SQLITE_PATH"] = TEST_DB_PATH

from config.settings import settings
settings.sqlite_path = TEST_DB_PATH

from app.db.models import MIGRATIONS, run_schema_upgrades
from app.db.session import get_db_context
from app.engines.memory import MemoryEngine
from app.engines.retrieval import RetrievalEngine, Answer, StructuralRefusal

# ── Step 1: Create fresh DB ──
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
print(f"[SETUP] Fresh DB: {db_path}")

# ── Step 2: Ingest via full write path ──
engine = MemoryEngine()

sam_facts = [
    "I work at Google as a software engineer.",
    "I live in Portland, Oregon.",
    "My mom's name is Linda and she lives in Boston.",
    "I got married to Sarah in June 2024.",
    "I graduated from Stanford with a degree in computer science.",
    "I have a cat named Whiskers.",
    "I love playing guitar in my free time.",
    "I'm allergic to peanuts.",
    "I've been training for a marathon since January.",
    "I feel pretty optimistic about this year.",
    "I had coffee with Jake this morning at the usual place.",
    "Jake and I went hiking last Saturday in Forest Park.",
    "I'm really nervous about my interview at Apple next Tuesday.",
    "My sister Emily just had a baby girl named Lily.",
    "I used to live in Boston before moving to Portland.",
    "The weather was nice today so I went for a run.",
    "I signed up for a pottery class that starts next month.",
    "My dad retired last year after 30 years at Boeing.",
    "I've been reading a lot of science fiction lately.",
    "I quit smoking two years ago.",
]

alex_facts = [
    "I work as an art director at a design agency.",
    "I live in Seattle with my partner Jordan.",
    "My best friend Bella and I meet every Thursday for dinner.",
    "I adopted a rescue dog named Biscuit last month.",
    "I'm taking an online photography course.",
    "I moved to Seattle from Chicago three years ago.",
    "I feel stressed about the project deadline this Friday.",
    "I love cooking Thai food, especially pad thai.",
    "My brother Marcus lives in Denver and works in finance.",
    "I've been doing yoga every morning for the past six months.",
]

print("[INGEST] Ingesting Sam's 20 facts via full write path...")
for i, text in enumerate(sam_facts):
    try:
        engine.ingest_text(user_id=1, text=text, speaker="Sam")
        print(f"  Sam [{i+1}/20]: OK")
    except Exception as e:
        print(f"  Sam [{i+1}/20]: ERROR - {e}")

print("[INGEST] Ingesting Alex's 10 facts via full write path...")
for i, text in enumerate(alex_facts):
    try:
        engine.ingest_text(user_id=2, text=text, speaker="Alex")
        print(f"  Alex [{i+1}/10]: OK")
    except Exception as e:
        print(f"  Alex [{i+1}/10]: ERROR - {e}")

# ── Step 3: Verify write path populated everything ──
print("\n[VERIFY] Checking DB state...")
with get_db_context() as conn:
    rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
    rel_u1 = conn.execute("SELECT COUNT(*) FROM relationships WHERE user_id=1").fetchone()[0]
    rel_u2 = conn.execute("SELECT COUNT(*) FROM relationships WHERE user_id=2").fetchone()[0]

    try:
        facts_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    except:
        facts_count = "N/A"

    try:
        pq_count = conn.execute("SELECT COUNT(*) FROM predicted_queries").fetchone()[0]
    except:
        pq_count = "N/A"

    null_emb = conn.execute("SELECT COUNT(*) FROM relationships WHERE edge_embedding IS NULL").fetchone()[0]

    print(f"  Relationships: {rel_count} (Sam={rel_u1}, Alex={rel_u2})")
    print(f"  Facts: {facts_count}")
    print(f"  Predicted queries: {pq_count}")
    print(f"  Null embeddings: {null_emb}")

# Show some facts
print("\n[VERIFY] Facts table contents:")
with get_db_context() as conn:
    try:
        for row in conn.execute("SELECT key, value FROM facts ORDER BY key").fetchall():
            print(f"  {row[0]} = {row[1]}")
    except:
        print("  (no facts table)")

# Show some predicted queries
print("\n[VERIFY] Predicted queries sample:")
with get_db_context() as conn:
    try:
        for row in conn.execute("SELECT predicted_question, answer_text FROM predicted_queries LIMIT 10").fetchall():
            print(f"  Q: {row[0]}")
            print(f"  A: {row[1]}")
            print()
    except:
        print("  (no predicted_queries table)")

# ── Step 4: Run queries with strict validation ──
retrieval = RetrievalEngine(memory_engine=engine)

queries = [
    # Cat 4 — single-hop factual
    ("What does Sam do for work?", 1, ["Google", "software engineer"], "career fact"),
    ("Where does Sam live?", 1, ["Portland"], "location fact"),
    ("What is Sam's hobby?", 1, ["guitar"], "hobby semantic gap"),
    ("Who is Sam's mom?", 1, ["Linda"], "family fact"),
    ("What is Sam allergic to?", 1, ["peanuts"], "health fact"),
    ("What is Sam's cat's name?", 1, ["Whiskers"], "pet fact"),
    ("Where does Alex work?", 2, ["design agency", "art director"], "Alex career"),
    ("Who is Alex's best friend?", 2, ["Bella"], "Alex social"),
    ("What kind of dog does Alex have?", 2, ["Biscuit"], "Alex pet"),

    # Cat 2 — temporal
    ("When did Sam get married?", 1, ["June 2024", "June", "2024"], "temporal"),
    ("When is Sam's interview?", 1, ["Tuesday"], "future temporal"),
    ("When did Alex move to Seattle?", 2, ["three years ago", "three years", "Chicago"], "relative temporal"),

    # Cat 5 — adversarial
    ("What does Alex do for work?", 1, ["REFUSE"], "wrong user_id"),
    ("Where does Sam live?", 2, ["REFUSE"], "wrong user_id"),
    ("Who is Sam's best friend?", 2, ["REFUSE"], "wrong user_id"),

    # Emotional
    ("How does Sam feel about the interview?", 1, ["nervous"], "emotional + target"),
    ("How does Alex feel?", 2, ["stressed"], "emotional state"),

    # Multi-hop
    ("Where is Sam's mom?", 1, ["Boston"], "multi-hop"),
    ("What does Alex's brother do?", 2, ["finance"], "multi-hop"),

    # Semantic gap
    ("What does Sam do for fun?", 1, ["guitar", "pottery", "reading", "hiking", "run"], "synonym for hobby"),
    ("What exercise does Sam do?", 1, ["marathon", "run", "training"], "health activity"),
    ("What is Alex studying?", 2, ["photography"], "education"),

    # Historical
    ("Where did Sam used to live?", 1, ["Boston"], "is_historical"),

    # Vague
    ("What's going on in Sam's life?", 1, ["NONEMPTY"], "reconstruction"),

    # NEW: Strict bonus
    ("Where did Sam graduate from?", 1, ["Stanford"], "education fact"),
]

def validate(answer, expected_keywords, query):
    if answer is None:
        return "REFUSE" in expected_keywords

    is_refusal = hasattr(answer, 'reason')
    text = ""
    if hasattr(answer, 'text'):
        text = answer.text or ""

    if "REFUSE" in expected_keywords:
        return is_refusal

    if "NONEMPTY" in expected_keywords:
        return not is_refusal and text and len(text.strip()) > 3

    if is_refusal:
        return False

    text_lower = text.lower()
    for kw in expected_keywords:
        if kw.lower() in text_lower:
            return True
    return False

# Warmup query
print("\n[WARMUP] Running warmup query...")
try:
    retrieval.retrieve(1, "warmup query for model loading")
except:
    pass
print("[WARMUP] Done.\n")

# Run all queries
print("=" * 100)
print("STRICT SCALE TEST: 25 queries, full write path, strict keyword validation")
print("=" * 100)

results = []
for query_text, user_id, expected_kws, description in queries:
    t0 = time.perf_counter()
    try:
        result = retrieval.retrieve(user_id, query_text)
        error = None
    except Exception as e:
        result = None
        error = str(e)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    passed = False
    if error is None:
        try:
            passed = validate(result, expected_kws, query_text)
        except:
            passed = False

    # Extract details
    is_refusal = hasattr(result, 'reason') if result else True
    answer_text = ""
    source = ""
    pool_size = 0

    if result and hasattr(result, 'text'):
        answer_text = result.text or ""
    if result and hasattr(result, 'source'):
        source = result.source or ""
    if result and hasattr(result, 'convergence_details'):
        cd = result.convergence_details or {}
        pool_size = cd.get("pool_size", 0)
    if result and hasattr(result, 'survivors'):
        pool_size = pool_size or (result.survivors if hasattr(result, 'survivors') else 0)

    results.append({
        "query": query_text,
        "user_id": user_id,
        "expected": expected_kws,
        "description": description,
        "answer": answer_text[:80],
        "source": source,
        "pool_size": pool_size,
        "latency_ms": round(elapsed_ms, 1),
        "passed": passed,
        "is_refusal": is_refusal,
        "error": error,
    })

# Print results table
print(f"\n{'#':>2} | {'Pass':>4} | {'ms':>7} | {'Pool':>4} | {'Path':>20} | {'Query':<45} | Answer (first 60)")
print("-" * 160)
for i, r in enumerate(results, 1):
    status = "PASS" if r["passed"] else "FAIL"
    ans = "REFUSAL" if r["is_refusal"] else r["answer"][:60]
    print(f"{i:>2} | {status:>4} | {r['latency_ms']:>7.1f} | {r['pool_size']:>4} | {r['source']:>20} | {r['query']:<45} | {ans}")

# Summary
pass_count = sum(1 for r in results if r["passed"])
fail_count = len(results) - pass_count

print(f"\n{'=' * 80}")
print(f"RESULT: {pass_count}/{len(results)} ({100*pass_count//len(results)}%)")
print(f"{'=' * 80}")

# Latency stats (excluding first query which has cold-start)
latencies = [r["latency_ms"] for r in results[1:]]
if latencies:
    lat_sorted = sorted(latencies)
    p95_idx = int(len(lat_sorted) * 0.95)
    print(f"\n[LATENCY] (excluding Q1 cold-start)")
    print(f"  Min:    {min(latencies):.1f}ms")
    print(f"  Median: {statistics.median(latencies):.1f}ms")
    print(f"  P95:    {lat_sorted[min(p95_idx, len(lat_sorted)-1)]:.1f}ms")
    print(f"  Max:    {max(latencies):.1f}ms")

# Failure analysis
failures = [r for r in results if not r["passed"]]
if failures:
    print(f"\n[FAILURES] {len(failures)} queries failed:")
    for r in failures:
        print(f"  Q: {r['query']}")
        print(f"  Expected: {r['expected']}")
        ans = "REFUSAL" if r["is_refusal"] else r["answer"]
        print(f"  Got: {ans}")
        if r["error"]:
            print(f"  Error: {r['error']}")
        print()

# Path analysis
print(f"\n[PATH ANALYSIS]")
paths = {}
for r in results:
    p = r["source"] or "refusal"
    if p not in paths:
        paths[p] = 0
    paths[p] += 1
for p, c in sorted(paths.items()):
    print(f"  {p}: {c} queries")

# Facts fast-path check
print(f"\n[FACTS FAST-PATH]")
for r in results:
    if r["source"] == "fact_fast_path":
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['query']} -> {r['answer'][:50]}")

# False refusals
print(f"\n[FALSE REFUSALS] (expected content, got refusal)")
for r in results:
    if r["is_refusal"] and "REFUSE" not in r["expected"]:
        print(f"  {r['query']} (expected {r['expected']})")

# False answers
print(f"\n[FALSE ANSWERS] (expected refusal, got content)")
for r in results:
    if not r["is_refusal"] and "REFUSE" in r["expected"]:
        print(f"  {r['query']} -> {r['answer'][:50]}")
