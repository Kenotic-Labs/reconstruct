"""
Pre-filter raw candidates for silver pass.
Mechanical filtering only — removes obvious junk so labeler agents
can focus on candidates that MIGHT contain personal facts.

Root cause: 1.27M raw pool is ~90% non-personal-fact content (math,
code, assistant turns, spam, academic questions). Labeler agents
process one row at a time. Without pre-filtering, they waste time
on ~1.1M rows of junk. This script removes rows that definitively
cannot contain personal facts using mechanical checks only.

This is NOT extraction. It's garbage removal.
"""
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "extractor_v7" / "raw_web"
FILTERED_DIR = ROOT / "data" / "extractor_v7" / "filtered_for_silver"
FILTERED_DIR.mkdir(parents=True, exist_ok=True)

# Personal-fact signal words (first-person pronouns + possessives)
# If an utterance doesn't contain any of these, it's almost certainly not a personal fact
FIRST_PERSON = {"i ", "i'm", "i've", "i'd", "i'll", "my ", "me ", "mine", "myself",
                "we ", "we're", "we've", "our ", "ours"}

# Obvious non-personal prefixes (mechanical, not semantic)
JUNK_STARTS = [
    "every day, a tree",
    "in analytical chemistry",
    "write a ",
    "create a ",
    "generate a ",
    "explain ",
    "what is ",
    "what are ",
    "how do ",
    "how does ",
    "how can ",
    "can you ",
    "could you ",
    "please provide ",
    "list ",
    "describe the ",
    "compare ",
    "calculate ",
    "solve ",
    "def ",
    "function ",
    "class ",
    "import ",
    "```",
    "here's ",
    "here is ",
    "sure, ",
    "certainly",
    "of course",
    "as an ai",
    "as a language model",
]


def has_first_person(text: str) -> bool:
    lower = text.lower()
    return any(fp in lower for fp in FIRST_PERSON)


def is_junk(text: str) -> bool:
    lower = text.lower().strip()
    if len(lower.split()) < 5:
        return True
    if len(lower) > 2000:
        return True
    for prefix in JUNK_STARTS:
        if lower.startswith(prefix):
            return True
    return False


def is_assistant_turn(row: dict) -> bool:
    st = row.get("source_type", "")
    notes = row.get("notes", "")
    return "assistant" in st or "role_hint=gpt" in notes or "role_hint=assistant" in notes


def prefilter_file(input_path: Path, output_path: Path) -> dict:
    stats = {"total": 0, "kept": 0, "skipped_assistant": 0,
             "skipped_junk": 0, "skipped_no_first_person": 0}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = row.get("source_text_raw", "")

            if is_assistant_turn(row):
                stats["skipped_assistant"] += 1
                continue

            if is_junk(text):
                stats["skipped_junk"] += 1
                continue

            # PersonaChat persona lines are all personal facts — keep without first-person check
            source_type = row.get("source_type", "")
            if "persona" not in source_type and not has_first_person(text):
                stats["skipped_no_first_person"] += 1
                continue

            stats["kept"] += 1
            fout.write(line + "\n")

    return stats


def main():
    raw_files = sorted(RAW_DIR.glob("*.jsonl"))
    total_stats = {"total": 0, "kept": 0, "skipped_assistant": 0,
                   "skipped_junk": 0, "skipped_no_first_person": 0}

    for raw_file in raw_files:
        if raw_file.stat().st_size == 0:
            continue
        out_name = raw_file.name.replace("raw_candidates_", "filtered_")
        out_path = FILTERED_DIR / out_name
        print(f"Processing {raw_file.name}...")
        stats = prefilter_file(raw_file, out_path)
        for k in total_stats:
            total_stats[k] += stats[k]
        yield_pct = (stats["kept"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {stats['total']:>8,} total -> {stats['kept']:>8,} kept ({yield_pct:.1f}%)")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_stats['total']:>10,} scanned")
    print(f"       {total_stats['kept']:>10,} kept ({total_stats['kept']/total_stats['total']*100:.1f}%)")
    print(f"       {total_stats['skipped_assistant']:>10,} skipped (assistant turns)")
    print(f"       {total_stats['skipped_junk']:>10,} skipped (junk)")
    print(f"       {total_stats['skipped_no_first_person']:>10,} skipped (no first person)")


if __name__ == "__main__":
    main()
