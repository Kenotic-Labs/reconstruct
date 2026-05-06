"""Query-only LOCOMO runner — reuses an already-ingested DB.
Skips the 30-min ingestion. Runs 199 queries in ~20s."""
from __future__ import annotations
import json, sys, time, glob, os, tempfile
from pathlib import Path
from collections import defaultdict

_PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "locomo_bench" / "locomo"))

from task_eval.evaluation import eval_question_answering
from sdk import KenoticV1
from sdk.types import Answer, Situation

DATA_PATH = _PROJECT / "locomo_bench" / "locomo" / "data" / "locomo10.json"
CATEGORY_NAMES = {1:"multi-hop",2:"temporal",3:"open-domain",4:"narrative",5:"adversarial"}

# Use fixed DB for consistent comparison across runs
DB_PATH = str(_PROJECT / "locomo_conv0_fixed.db")
import sqlite3 as _sq
_c = _sq.connect(DB_PATH)
_cnt = _c.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
_pq = _c.execute("SELECT COUNT(*) FROM predicted_queries").fetchone()[0]
_c.close()
print(f"Using DB: {DB_PATH} ({_cnt} edges, {_pq} PQs)")

if not DB_PATH:
    print("No ingested DB found. Run run_locomo.py --sample 0 first.")
    sys.exit(1)

def _extract(result) -> str:
    if result is None: return ""
    if isinstance(result, Answer):
        if result.refusal: return "This information is not mentioned in the conversation."
        return result.text or ""
    if isinstance(result, Situation): return result.narrative or ""
    return str(result) if result else ""

data = json.load(open(DATA_PATH, encoding="utf-8"))
qa_list = data[0]["qa"]

t0 = time.time()
scored = []
for qa in qa_list:
    q = qa["question"]
    result = KenoticV1(q, db_path=DB_PATH)
    pred = _extract(result.result)
    scored.append({
        "question": q, "category": qa["category"],
        "evidence": qa.get("evidence", []),
        "prediction": pred,
        "answer": str(qa["answer"]) if "answer" in qa else qa.get("adversarial_answer", ""),
    })

elapsed = time.time() - t0
print(f"Queried {len(scored)} questions in {elapsed:.1f}s")

f1s, _, _ = eval_question_answering(scored, eval_key="prediction")
for i, s in enumerate(scored): s["f1"] = float(f1s[i])

cat_scores = defaultdict(list)
for s in scored: cat_scores[s["category"]].append(s["f1"])

for cat in sorted(cat_scores):
    scores = cat_scores[cat]
    mean = sum(scores)/len(scores)
    print(f"  Cat {cat} ({CATEGORY_NAMES.get(cat,'?'):>12s}): F1 = {mean:.4f}  (n={len(scores)})")

overall = sum(f1s)/len(f1s)
print(f"  Overall F1: {overall:.4f}")
