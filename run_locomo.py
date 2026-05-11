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
from sdk import KenoticV1  # noqa: E402
from sdk.types import Answer, Situation  # noqa: E402

DATA_PATH = _PROJECT / "locomo_bench" / "locomo" / "data" / "locomo10.json"
REPORT_DIR = _PROJECT / "test_reports"

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


def _session_keys(conv: dict) -> list[str]:
    """Return session keys sorted numerically."""
    keys = [
        k for k in conv
        if k.startswith("session_") and not k.endswith("_date_time")
    ]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys


def run_conversation(conv_idx: int, conv_data: dict) -> dict:
    """Ingest all turns, query all QA items, return per-question results."""
    conversation = conv_data["conversation"]
    qa_list = conv_data["qa"]
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")

    tmp = tempfile.NamedTemporaryFile(
        suffix=".db", prefix=f"locomo_conv{conv_idx}_", delete=False,
    )
    db_path = tmp.name
    tmp.close()

    sep = "=" * 60
    print()
    print(sep)
    print(f"Conversation {conv_idx}: {speaker_a} & {speaker_b}")
    print(f"DB: {db_path}")

    t0 = time.time()
    turn_count = 0
    for session_key in _session_keys(conversation):
        date_key = f"{session_key}_date_time"
        raw_ts = conversation.get(date_key, "")
        iso_ts = _parse_locomo_timestamp(raw_ts)
        turns = conversation[session_key]
        if not isinstance(turns, list):
            continue
        for turn in turns:
            text = turn.get("text", "")
            if not text:
                continue
            speaker = turn.get("speaker", "unknown")
            # listener = the other speaker in the conversation
            _listener = speaker_b if speaker == speaker_a else speaker_a
            KenoticV1(
                text,
                speaker=speaker,
                listener=_listener,
                speaker_is_user=(speaker == speaker_a),
                source_timestamp=iso_ts,
                db_path=db_path,
            )
            turn_count += 1

    ingest_time = time.time() - t0
    print(f"  Ingested {turn_count} turns in {ingest_time:.1f}s")

    t1 = time.time()
    scored_qas = []
    for qa in qa_list:
        question = qa["question"]
        category = qa["category"]
        result = KenoticV1(question, db_path=db_path)
        prediction = _extract_answer_text(result.result)
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

    try:
        os.unlink(db_path)
        for ext in ("-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.unlink(p)
    except OSError:
        pass

    return {
        "conv_idx": conv_idx,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "turn_count": turn_count,
        "qa_count": len(scored_qas),
        "ingest_time_s": round(ingest_time, 2),
        "query_time_s": round(query_time, 2),
        "overall_f1": round(overall_f1, 4),
        "category_f1": {str(k): round(v, 4) for k, v in cat_means.items()},
        "per_question": scored_qas,
    }


def main():
    parser = argparse.ArgumentParser(description="LoCoMo benchmark runner")
    parser.add_argument("--sample", type=int, default=None,
        help="Run a single conversation by index (0-9). Default: all 10.")
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
        result = run_conversation(idx, data[idx])
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
