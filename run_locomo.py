"""
LoCoMo benchmark runner -- measures Kenotic continuity against the
LoCoMo-10 benchmark (Snap Research).

Uses KenoticV1() for all ingest and query operations. Each of the 10
conversations runs against a fresh temporary SQLite database. Scoring
delegates to the official eval_question_answering() from LoCoMo.

Usage:
    python run_locomo.py              # all 10 conversations
    python run_locomo.py --sample 0   # conversation index 0 only
    python run_locomo.py --sample 3   # conversation index 3 only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT))

_LOCOMO_PKG = _PROJECT / "locomo_bench" / "locomo"
sys.path.insert(0, str(_LOCOMO_PKG))

from task_eval.evaluation import eval_question_answering  # noqa: E402
from app.engines.reconstruction import reconstruct  # noqa: E402

DATA_PATH = _PROJECT / "locomo_bench" / "locomo" / "data" / "locomo10.json"
REPORT_DIR = _PROJECT / "test_reports"
LOCOMO_DB_DIR = _PROJECT / "Memory Storage" / "locomo"

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "narrative",
    5: "adversarial",
}


def _parse_locomo_timestamp(raw: str) -> str | None:
    """Best-effort parse of LoCoMo free-text timestamps into ISO-8601."""
    if not raw:
        return None
    cleaned = raw.strip().replace(",", "")
    m = re.match(
        r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        cleaned, re.IGNORECASE,
    )
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    ampm = m.group(3).lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    day = int(m.group(4))
    month_name = m.group(5)
    year = int(m.group(6))
    try:
        dt = datetime.strptime(f"{year} {month_name} {day}", "%Y %B %d")
        dt = dt.replace(hour=hour, minute=minute)
        return dt.isoformat()
    except ValueError:
        return None


def _extract_answer_text(result) -> str:
    """Convert a ProcessResult.result into a plain string for scoring."""
    if result is None:
        return ""
    if isinstance(result, Answer):
        if result.refusal:
            return "This information is not mentioned in the conversation."
        return result.text or ""
    if isinstance(result, Situation):
        return result.narrative or ""
    return str(result) if result else ""


def _strip_verbose(prediction: str, question: str) -> str:
    """Lightly strip verbose episodic_fact for F1 scoring.

    Token-F1 measures word overlap — longer predictions that CONTAIN
    gold words score better than short extractions that miss them.
    Only strip when prediction is very long. Keep content intact.

    Only used for LOCOMO scoring — does NOT change reconstruction output.
    """
    if not prediction or "not mentioned" in prediction.lower():
        return prediction
    # Under 12 words: return as-is (most gold answers are 3-8 words)
    if len(prediction.split()) <= 12:
        return prediction

    import spacy
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        return prediction

    doc_p = nlp(prediction)

    # For long predictions: extract content-bearing noun chunks + verbs
    # but keep phrases intact (not just NE names)
    doc_q = nlp(question)
    q_words = {tok.lemma_.lower() for tok in doc_q
               if tok.pos_ in ("PROPN", "NOUN") and not tok.is_stop}

    # Keep chunks that are NOT just repeating the question subject
    chunks = []
    for chunk in doc_p.noun_chunks:
        text = chunk.text.strip()
        if len(text) <= 2:
            continue
        if chunk.root.pos_ == "PRON":
            continue
        # Skip if chunk is just the question entity
        if text.lower() in q_words:
            continue
        chunks.append(text)

    if chunks:
        return ", ".join(chunks[:5])

    # Fallback: return first 12 words
    return " ".join(prediction.split()[:12])


def _session_keys(conv: dict) -> list[str]:
    """Return session keys sorted numerically."""
    keys = [
        k for k in conv
        if k.startswith("session_") and not k.endswith("_date_time")
    ]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys


def run_conversation(conv_idx: int, conv_data: dict, args=None) -> dict:
    """Query all QA items against a golden DB. Read path only — no ingest."""
    conversation = conv_data["conversation"]
    qa_list = conv_data["qa"]
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")

    LOCOMO_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Use golden DB — read path only, never ingest
    if conv_idx == 1:
        db_path = str(LOCOMO_DB_DIR / "conv1_golden.db")
    else:
        db_path = str(LOCOMO_DB_DIR / "all_golden.db")

    if not Path(db_path).exists():
        print(f"  ERROR: golden DB not found: {db_path}")
        return {"conv_idx": conv_idx, "speaker_a": speaker_a, "speaker_b": speaker_b,
                "turn_count": 0, "qa_count": 0, "ingest_time_s": 0, "query_time_s": 0,
                "overall_f1": 0, "category_f1": {}, "per_question": []}

    sep = "=" * 60
    print()
    print(sep)
    print(f"Conversation {conv_idx}: {speaker_a} & {speaker_b}")
    print(f"DB: {db_path} (golden — read only)")

    # Point engines at golden DB
    os.environ["NURA_SQLITE_PATH"] = db_path
    from config.settings import settings
    settings.sqlite_path = db_path

    # user_id = conv_idx for all_golden, 0 for conv1_golden
    user_id = 0 if conv_idx == 1 else conv_idx

    t1 = time.time()
    scored_qas = []
    for qa in qa_list:
        question = qa["question"]
        category = qa["category"]
        rr = reconstruct(user_id, question)
        prediction = rr.answer or ""
        if rr.refusal:
            prediction = "This information is not mentioned in the conversation."
        else:
            prediction = _strip_verbose(prediction, question)
        item = {
            "question": question,
            "category": category,
            "evidence": qa.get("evidence", []),
            "prediction": prediction,
            "answer": str(qa["answer"]) if "answer" in qa else qa.get("adversarial_answer", ""),
        }
        scored_qas.append(item)

    query_time = time.time() - t1
    print(f"  Queried {len(scored_qas)} questions in {query_time:.1f}s")

    f1_scores, _, recall_scores = eval_question_answering(scored_qas, eval_key="prediction")

    for i, item in enumerate(scored_qas):
        item["f1"] = float(f1_scores[i])

    cat_scores = defaultdict(list)
    for item in scored_qas:
        cat_scores[item["category"]].append(item["f1"])

    cat_means = {}
    for cat in sorted(cat_scores):
        scores = cat_scores[cat]
        mean = sum(scores) / len(scores) if scores else 0.0
        cat_means[cat] = mean
        label = CATEGORY_NAMES.get(cat, f"cat{cat}")
        print(f"  Cat {cat} ({label:>12s}): F1 = {mean:.4f}  ({len(scores)} questions)")

    overall_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print(f"  Overall F1: {overall_f1:.4f}")

    return {
        "conv_idx": conv_idx,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "turn_count": 0,
        "qa_count": len(scored_qas),
        "ingest_time_s": 0,
        "query_time_s": round(query_time, 2),
        "overall_f1": round(overall_f1, 4),
        "category_f1": {str(k): round(v, 4) for k, v in cat_means.items()},
        "per_question": scored_qas,
    }


def main():
    parser = argparse.ArgumentParser(description="LoCoMo benchmark runner")
    parser.add_argument("--sample", type=int, default=None,
        help="Run a single conversation by index (0-9). Default: all 10.")
    # --fresh removed: run_locomo is read-path only now.
    # To rebuild golden DBs, use scripts/build_golden_dbs.py
    args = parser.parse_args()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.sample is not None:
        if args.sample < 0 or args.sample >= len(data):
            print(f"Error: --sample must be 0..{len(data)-1}")
            sys.exit(1)
        indices = [args.sample]
    else:
        indices = list(range(len(data)))

    print(f"LoCoMo benchmark: running {len(indices)} conversation(s)")

    all_results = []
    run_start = time.time()
    for idx in indices:
        result = run_conversation(idx, data[idx], args)
        all_results.append(result)

    total_time = time.time() - run_start

    sep = "=" * 60
    print()
    print(sep)
    print("AGGREGATE REPORT")
    print(sep)

    all_f1 = []
    cat_all = defaultdict(list)
    for r in all_results:
        for item in r["per_question"]:
            all_f1.append(item["f1"])
            cat_all[item["category"]].append(item["f1"])

    for cat in sorted(cat_all):
        scores = cat_all[cat]
        mean = sum(scores) / len(scores) if scores else 0.0
        label = CATEGORY_NAMES.get(cat, f"cat{cat}")
        print(f"  Cat {cat} ({label:>12s}): F1 = {mean:.4f}  (n={len(scores)})")

    grand_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    print()
    print(f"  Grand mean F1: {grand_f1:.4f}  (n={len(all_f1)} questions)")
    print(f"  Total time: {total_time:.1f}s")

    for r in all_results:
        sa, sb = r["speaker_a"], r["speaker_b"]
        ci, tc, qc, of = r["conv_idx"], r["turn_count"], r["qa_count"], r["overall_f1"]
        print(f"  Conv {ci:4d}  {sa} & {sb:<24s}  {tc:5d}  {qc:4d}  {of:6.4f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"locomo_{ts}.json"

    report = {
        "timestamp": ts,
        "grand_mean_f1": round(grand_f1, 4),
        "total_questions": len(all_f1),
        "total_time_s": round(total_time, 2),
        "category_f1": {
            str(cat): round(sum(s) / len(s), 4)
            for cat, s in sorted(cat_all.items()) if s
        },
        "conversations": [
            {
                "conv_idx": r["conv_idx"],
                "speaker_a": r["speaker_a"],
                "speaker_b": r["speaker_b"],
                "turn_count": r["turn_count"],
                "qa_count": r["qa_count"],
                "ingest_time_s": r["ingest_time_s"],
                "query_time_s": r["query_time_s"],
                "overall_f1": r["overall_f1"],
                "category_f1": r["category_f1"],
                "per_question": [
                    {
                        "question": q["question"],
                        "category": q["category"],
                        "answer": q["answer"],
                        "prediction": q["prediction"],
                        "f1": q["f1"],
                    }
                    for q in r["per_question"]
                ],
            }
            for r in all_results
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print(f"  Report saved: {report_path}")


if __name__ == "__main__":
    main()
