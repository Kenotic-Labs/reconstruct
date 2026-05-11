"""
Audit Grammar Engine Extraction Performance across LoCoMo conversations.

Root cause for this script: The ingest pipeline (memory.py ingest_text ->
grammar_engine.process -> _validate_triple -> store) silently drops triples
at three stages (classification, validation, deduplication) with zero
aggregated visibility.  This instruments each stage.

Measures:
  1. Ingest stats per conversation (turns, triples pre/post validation, stored)
  2. Rejection breakdown by validation gate
  3. QA coverage: how many expected answers have token overlap with stored triples

Usage:
  python scripts/audit_grammar_engine.py                    # all 10 conversations
  python scripts/audit_grammar_engine.py --sample 0         # conversation 0 only
  python scripts/audit_grammar_engine.py --sample 0,1,2     # conversations 0,1,2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RAYA_EMBED_DEVICE", "cuda")
# Enable verbose validation logging
os.environ["RAYA_VALIDATE_VERBOSE"] = "1"


# ---------------------------------------------------------------------------
# Monkey-patch MemoryEngine._validate_triple to intercept rejection reasons
# ---------------------------------------------------------------------------

_rejection_log: List[Tuple[str, str, str, str]] = []  # (subject, predicate, object, reason)
_pre_validate_count = 0


def _patched_validate_triple(self, subject, predicate, obj, extraction_rule=""):
    global _pre_validate_count
    _pre_validate_count += 1
    ok, normalized, reason = _original_validate_triple(self, subject, predicate, obj, extraction_rule=extraction_rule)
    if not ok:
        _rejection_log.append((subject, predicate, obj, reason))
    return ok, normalized, reason


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

LOCOMO_PATH = PROJECT_ROOT / "locomo_bench" / "locomo" / "data" / "locomo10.json"

STOPWORDS = frozenset({
    "a", "an", "the", "is", "was", "were", "are", "am", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "don", "t", "s", "ll", "ve",
    "re", "d", "m", "and", "but", "or", "if", "while", "because",
    "about", "up", "down", "i", "me", "my", "myself", "we", "our",
    "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "that", "also", "still", "well", "really", "quite",
})


def load_locomo() -> list:
    with open(LOCOMO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_turns(conv_data: dict) -> List[Dict[str, Any]]:
    """Extract all turns from a conversation in order."""
    conv = conv_data["conversation"]
    speaker_a = conv.get("speaker_a", "SpeakerA")
    speaker_b = conv.get("speaker_b", "SpeakerB")

    session_keys = sorted(
        [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    turns = []
    for sk in session_keys:
        date_key = sk + "_date_time"
        session_date = conv.get(date_key, "")
        for turn in conv[sk]:
            turns.append({
                "speaker": turn["speaker"],
                "text": turn["text"],
                "dia_id": turn.get("dia_id", ""),
                "session_date": session_date,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
            })
    return turns


def tokenize_for_overlap(text: str) -> set:
    """Tokenize text into lowercased content words for overlap check."""
    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return tokens - STOPWORDS


# ---------------------------------------------------------------------------
# Main audit logic
# ---------------------------------------------------------------------------

def audit_conversation(conv_idx: int, conv_data: dict) -> dict:
    """Run full audit on one conversation. Returns stats dict."""
    global _rejection_log, _pre_validate_count

    _rejection_log = []
    _pre_validate_count = 0

    turns = get_turns(conv_data)
    conv = conv_data["conversation"]
    speaker_a = conv.get("speaker_a", "SpeakerA")
    speaker_b = conv.get("speaker_b", "SpeakerB")
    qa_pairs = conv_data.get("qa", [])

    # Create temp DB
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        # Initialize Kenotic SDK with temp DB
        from sdk.client import Kenotic
        k = Kenotic(user_id=1, db_path=tmp_path, embed_device="cuda")

        # Intercept grammar engine classification for counting
        from app.engines import grammar_engine as ge

        original_process = ge.process

        grammar_triples_before_validate = 0
        grammar_no_triple_turns = 0
        grammar_question_turns = 0
        grammar_backchannel_turns = 0
        grammar_emotion_turns = 0
        grammar_command_turns = 0
        grammar_statement_turns = 0

        def counting_process(text, speaker=None):
            nonlocal grammar_triples_before_validate, grammar_no_triple_turns
            nonlocal grammar_question_turns, grammar_backchannel_turns
            nonlocal grammar_emotion_turns, grammar_command_turns, grammar_statement_turns
            result = original_process(text, speaker=speaker)
            cat = result.classification.category
            if cat == "QUESTION":
                grammar_question_turns += 1
            elif cat == "BACKCHANNEL":
                grammar_backchannel_turns += 1
            elif cat == "EMOTION":
                grammar_emotion_turns += 1
            elif cat == "COMMAND":
                grammar_command_turns += 1
            else:
                grammar_statement_turns += 1

            grammar_triples_before_validate += len(result.triples)
            if not result.triples:
                grammar_no_triple_turns += 1
            return result

        ge.process = counting_process

        # Ingest all turns
        total_stored = 0
        t0 = time.time()
        for turn in turns:
            speaker = turn["speaker"]
            text = turn["text"]
            session_date = turn["session_date"]

            count = k.ingest(
                text,
                speaker=speaker,
                source_timestamp=session_date,
            )
            total_stored += count

        ingest_time = time.time() - t0

        # Restore original process
        ge.process = original_process

        # Query DB for all stored triples
        conn = sqlite3.connect(tmp_path)
        conn.row_factory = sqlite3.Row
        stored_rows = conn.execute(
            "SELECT id, subject, predicate, object, source_text FROM relationships WHERE user_id = 1"
        ).fetchall()
        conn.close()

        # Build token set from stored triples for coverage check
        store_tokens = set()
        for row in stored_rows:
            for field in ("subject", "predicate", "object", "source_text"):
                val = row[field] or ""
                # For predicate, replace underscores
                if field == "predicate":
                    val = val.replace("_", " ")
                store_tokens |= tokenize_for_overlap(val)

        # QA coverage check
        qa_no_overlap = 0
        qa_has_overlap = 0
        qa_details = []
        for qa in qa_pairs:
            answer = str(qa.get("answer", ""))
            question = qa.get("question", "")
            answer_tokens = tokenize_for_overlap(answer)
            if not answer_tokens:
                # Empty or all-stopword answer
                qa_no_overlap += 1
                qa_details.append({
                    "question": question,
                    "answer": answer,
                    "overlap": [],
                    "category": qa.get("category", 0),
                })
                continue

            overlap = answer_tokens & store_tokens
            if overlap:
                qa_has_overlap += 1
            else:
                qa_no_overlap += 1
            qa_details.append({
                "question": question,
                "answer": answer,
                "overlap": sorted(overlap),
                "category": qa.get("category", 0),
            })

        # Rejection breakdown
        rejection_breakdown = Counter()
        for s, p, o, reason in _rejection_log:
            # Normalize reason (strip trailing number from length reasons)
            base_reason = re.sub(r"_\d+$", "", reason)
            rejection_breakdown[base_reason] += 1

        # Stats
        stats = {
            "conv_idx": conv_idx,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "total_turns": len(turns),
            "ingest_time_s": round(ingest_time, 1),

            # Classification breakdown
            "question_turns": grammar_question_turns,
            "backchannel_turns": grammar_backchannel_turns,
            "emotion_turns": grammar_emotion_turns,
            "command_turns": grammar_command_turns,
            "statement_turns": grammar_statement_turns,

            # Triple counts
            "triples_before_validate": grammar_triples_before_validate,
            "triples_rejected": len(_rejection_log),
            "triples_passed_validate": grammar_triples_before_validate - len(_rejection_log),
            "triples_stored_in_db": len(stored_rows),
            "total_stored_count": total_stored,

            # Rates
            "extraction_rate": round(grammar_triples_before_validate / max(len(turns), 1), 2),
            "survival_rate": round(len(stored_rows) / max(len(turns), 1), 2),
            "no_triple_turns": grammar_no_triple_turns,
            "no_triple_pct": round(100 * grammar_no_triple_turns / max(len(turns), 1), 1),

            # Rejection breakdown
            "rejection_breakdown": dict(rejection_breakdown.most_common()),

            # QA coverage
            "qa_total": len(qa_pairs),
            "qa_with_overlap": qa_has_overlap,
            "qa_no_overlap": qa_no_overlap,
            "qa_coverage_pct": round(100 * qa_has_overlap / max(len(qa_pairs), 1), 1),

            # Category breakdown for no-overlap QAs
            "qa_no_overlap_by_cat": dict(Counter(
                d["category"] for d in qa_details if not d["overlap"]
            ).most_common()),

            # Total QA by category for this conversation (denominator
            # must come from the SAME population as the no-overlap counts;
            # loading all 10 convs for cat_totals when only N were audited
            # creates a denominator mismatch -- diagnosed from the first
            # conv-0 run output where Category 1 showed 277/282 but conv 0
            # only has 199 QA pairs total)
            "qa_total_by_cat": dict(Counter(
                d["category"] for d in qa_details
            ).most_common()),
        }

        return stats

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def print_summary(all_stats: List[dict]):
    """Print summary table."""
    print("\n" + "=" * 120)
    print("GRAMMAR ENGINE AUDIT -- LOCOMO 10 CONVERSATIONS")
    print("=" * 120)

    # Header
    print(f"\n{'Conv':>4}  {'Speakers':<25} {'Turns':>5}  "
          f"{'Quest':>5} {'Back':>5} {'Emot':>5} {'Cmd':>4} {'Stmt':>5}  "
          f"{'Pre-V':>5} {'Rej':>4} {'Store':>5}  "
          f"{'ExtR':>5} {'SurvR':>5}  "
          f"{'QA':>4} {'Cov%':>5}  "
          f"{'Time':>5}")
    print("-" * 120)

    totals = defaultdict(int)
    for s in all_stats:
        speakers = f"{s['speaker_a']}/{s['speaker_b']}"
        print(f"{s['conv_idx']:>4}  {speakers:<25} {s['total_turns']:>5}  "
              f"{s['question_turns']:>5} {s['backchannel_turns']:>5} "
              f"{s['emotion_turns']:>5} {s['command_turns']:>4} {s['statement_turns']:>5}  "
              f"{s['triples_before_validate']:>5} {s['triples_rejected']:>4} {s['triples_stored_in_db']:>5}  "
              f"{s['extraction_rate']:>5} {s['survival_rate']:>5}  "
              f"{s['qa_total']:>4} {s['qa_coverage_pct']:>5.1f}  "
              f"{s['ingest_time_s']:>5.1f}s")

        for k in ("total_turns", "question_turns", "backchannel_turns", "emotion_turns",
                   "command_turns", "statement_turns", "triples_before_validate",
                   "triples_rejected", "triples_stored_in_db", "no_triple_turns",
                   "qa_total", "qa_with_overlap", "qa_no_overlap"):
            totals[k] += s[k]

    print("-" * 120)
    total_turns = totals["total_turns"]
    total_pre = totals["triples_before_validate"]
    total_store = totals["triples_stored_in_db"]
    total_qa = totals["qa_total"]
    total_qa_cov = totals["qa_with_overlap"]
    print(f"{'TOT':>4}  {'ALL':<25} {total_turns:>5}  "
          f"{totals['question_turns']:>5} {totals['backchannel_turns']:>5} "
          f"{totals['emotion_turns']:>5} {totals['command_turns']:>4} {totals['statement_turns']:>5}  "
          f"{total_pre:>5} {totals['triples_rejected']:>4} {total_store:>5}  "
          f"{total_pre/max(total_turns,1):>5.2f} {total_store/max(total_turns,1):>5.2f}  "
          f"{total_qa:>4} {100*total_qa_cov/max(total_qa,1):>5.1f}")

    # Rejection breakdown
    print("\n" + "=" * 80)
    print("REJECTION BREAKDOWN (aggregated across all conversations)")
    print("=" * 80)
    agg_rejections = Counter()
    for s in all_stats:
        for reason, count in s["rejection_breakdown"].items():
            agg_rejections[reason] += count
    for reason, count in agg_rejections.most_common():
        print(f"  {reason:<45} {count:>5}")
    print(f"  {'TOTAL':<45} {sum(agg_rejections.values()):>5}")

    # Classification vs extraction
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    total_q = totals["question_turns"]
    total_b = totals["backchannel_turns"]
    total_e = totals["emotion_turns"]
    total_c = totals["command_turns"]
    total_s = totals["statement_turns"]
    print(f"  Questions (no triples extracted):     {total_q:>5}  ({100*total_q/max(total_turns,1):>5.1f}%)")
    print(f"  Backchannels (no triples extracted):  {total_b:>5}  ({100*total_b/max(total_turns,1):>5.1f}%)")
    print(f"  Emotions (triples extracted):         {total_e:>5}  ({100*total_e/max(total_turns,1):>5.1f}%)")
    print(f"  Commands (no triples extracted):      {total_c:>5}  ({100*total_c/max(total_turns,1):>5.1f}%)")
    print(f"  Statements (triples extracted):       {total_s:>5}  ({100*total_s/max(total_turns,1):>5.1f}%)")
    print(f"  Turns producing 0 triples:            {totals['no_triple_turns']:>5}  ({100*totals['no_triple_turns']/max(total_turns,1):>5.1f}%)")

    # QA coverage by category
    print("\n" + "=" * 80)
    print("QA COVERAGE BREAKDOWN BY CATEGORY (no-overlap counts)")
    print("=" * 80)
    agg_qa_nocover = Counter()
    for s in all_stats:
        for cat, count in s.get("qa_no_overlap_by_cat", {}).items():
            agg_qa_nocover[cat] += count
    # Total QA by category from the data
    data = load_locomo()
    cat_totals = Counter()
    for conv_data in data:
        for qa in conv_data.get("qa", []):
            cat_totals[qa.get("category", 0)] += 1
    for cat in sorted(cat_totals.keys()):
        total_cat = cat_totals[cat]
        no_cover = agg_qa_nocover.get(cat, 0)
        covered = total_cat - no_cover
        print(f"  Category {cat}: {covered:>4}/{total_cat:>4} covered ({100*covered/max(total_cat,1):>5.1f}%),  "
              f"{no_cover:>4} no overlap")


def main():
    parser = argparse.ArgumentParser(description="Audit grammar engine extraction on LoCoMo")
    parser.add_argument("--sample", type=str, default=None,
                        help="Comma-separated conversation indices (0-9). Default: all.")
    args = parser.parse_args()

    data = load_locomo()
    print(f"Loaded {len(data)} conversations from LoCoMo")

    if args.sample is not None:
        indices = [int(x.strip()) for x in args.sample.split(",")]
    else:
        indices = list(range(len(data)))

    print(f"Will audit conversations: {indices}")

    # Patch validate_triple
    from app.engines.memory import MemoryEngine
    global _original_validate_triple
    _original_validate_triple = MemoryEngine._validate_triple
    MemoryEngine._validate_triple = _patched_validate_triple

    all_stats = []
    for idx in indices:
        print(f"\n--- Auditing conversation {idx} ---")
        stats = audit_conversation(idx, data[idx])
        all_stats.append(stats)
        print(f"  Turns: {stats['total_turns']}, "
              f"Triples pre-validate: {stats['triples_before_validate']}, "
              f"Rejected: {stats['triples_rejected']}, "
              f"Stored: {stats['triples_stored_in_db']}, "
              f"QA coverage: {stats['qa_coverage_pct']}%")

    # Restore
    MemoryEngine._validate_triple = _original_validate_triple

    print_summary(all_stats)


if __name__ == "__main__":
    main()
