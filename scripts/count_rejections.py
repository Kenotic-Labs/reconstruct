"""
Instrumentation script: count how many triples are produced vs rejected
vs stored during LoCoMo conv 0 ingest, broken down by rejection reason.

Root cause being investigated:
- 419 LoCoMo turns produce ~60 stored facts (14% survival).
- Two filtering stages: grammar_engine pronoun filter, then _validate_triple.
- _filter_unresolved_pronoun_subjects (line 2934 grammar_engine.py) drops
  single-token PRON/DET subjects because _resolve_clause_subject fails to
  resolve third-person pronouns to entity names.
- _validate_triple (line 368 memory.py) rejects: object > 8 tokens,
  object starts with pronoun (PRP), object starts with verb (VB*),
  predicate not verb-shaped, subject wrong length.
- This script counts each rejection reason to identify the biggest killer.

Usage:
    python scripts/count_rejections.py
"""
from __future__ import annotations
import json, os, sys, tempfile, re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

_LOCOMO_PKG = _PROJECT / "locomo_bench" / "locomo"
sys.path.insert(0, str(_LOCOMO_PKG))

DATA_PATH = _PROJECT / "locomo_bench" / "locomo" / "data" / "locomo10.json"

# ── Monkey-patch _validate_triple to count rejections ─────────────────────
from app.engines.memory import MemoryEngine

_orig_validate = MemoryEngine._validate_triple

rejection_counts = Counter()
validation_total = 0
rejected_examples = defaultdict(list)


def _counting_validate(self, subject, predicate, object, extraction_rule=""):
    global validation_total
    validation_total += 1
    ok, normalized, reason = _orig_validate(self, subject, predicate, object, extraction_rule=extraction_rule)
    if not ok:
        rejection_counts[reason] += 1
        rejected_examples[reason].append((subject, predicate, object))
    else:
        rejection_counts["__passed__"] += 1
    return ok, normalized, reason


MemoryEngine._validate_triple = _counting_validate

# ── Monkey-patch grammar_engine.process to count classification ──────────
from app.engines import grammar_engine as _ge

_orig_process = _ge.process

classification_counts = Counter()
sentence_no_triples = 0
grammar_triple_count = 0
grammar_pronoun_filtered = 0

_orig_filter = _ge._filter_unresolved_pronoun_subjects

# Collect examples of what the pronoun filter drops
pronoun_filter_examples = []


def _counting_filter(triples, speaker=None):
    global grammar_pronoun_filtered
    before = len(triples)
    result = _orig_filter(triples, speaker)
    dropped = before - len(result)
    grammar_pronoun_filtered += dropped
    kept_ids = set(id(t) for t in result)
    for t in triples:
        if id(t) not in kept_ids:
            pronoun_filter_examples.append((t.subject, t.predicate, t.object))
    return result


_ge._filter_unresolved_pronoun_subjects = _counting_filter


def _counting_process(text, speaker=None):
    global grammar_triple_count, sentence_no_triples
    result = _orig_process(text, speaker=speaker)
    grammar_triple_count += len(result.triples)
    if result.classification:
        classification_counts[result.classification.category] += 1
    if not result.triples:
        sentence_no_triples += 1
    return result


_ge.process = _counting_process

# ── LoCoMo timestamp parser ─────────────────────────────────────────────
def _parse_ts(raw):
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


def _session_keys(conv):
    keys = [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    conv_data = data[0]
    conversation = conv_data["conversation"]
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")

    tmp = tempfile.NamedTemporaryFile(suffix=".db", prefix="rejection_count_", delete=False)
    db_path = tmp.name
    tmp.close()

    os.environ["RAYA_DB_PATH"] = db_path
    os.environ["RAYA_EMBED_DEVICE"] = "cpu"

    from sdk import KenoticV1

    turn_count = 0

    for session_key in _session_keys(conversation):
        date_key = f"{session_key}_date_time"
        raw_ts = conversation.get(date_key, "")
        iso_ts = _parse_ts(raw_ts)
        turns = conversation[session_key]
        if not isinstance(turns, list):
            continue
        for turn in turns:
            text = turn.get("text", "")
            if not text:
                continue
            speaker = turn.get("speaker", "unknown")
            KenoticV1(text, speaker=speaker, source_timestamp=iso_ts, db_path=db_path)
            turn_count += 1

    print()
    print("=" * 60)
    print(f"LoCoMo Conv 0: {speaker_a} & {speaker_b}")
    print(f"Total turns ingested: {turn_count}")
    print()

    print("--- Grammar Engine Classification ---")
    for cat, cnt in classification_counts.most_common():
        print(f"  {cat:>15s}: {cnt}")
    print(f"  Turns with 0 triples: {sentence_no_triples}")
    print(f"  Triples produced (post grammar pronoun filter): {grammar_triple_count}")
    print(f"  Triples killed by pronoun filter: {grammar_pronoun_filtered}")
    print()

    print("--- Pronoun Filter Examples (first 15) ---")
    for s, p, o in pronoun_filter_examples[:15]:
        print(f"  ({s!r}, {p!r}, {o!r})")
    print()

    print("--- Validation Gate (_validate_triple) ---")
    print(f"  Total triples entering validation: {validation_total}")
    for reason, cnt in rejection_counts.most_common():
        pct = cnt / validation_total * 100 if validation_total else 0
        tag = "PASS" if reason == "__passed__" else "REJECT"
        print(f"  [{tag}] {reason:>35s}: {cnt:>4d}  ({pct:.1f}%)")
    print()

    passed = rejection_counts.get("__passed__", 0)
    rejected = validation_total - passed
    print(f"  Passed: {passed}/{validation_total} ({passed/validation_total*100:.1f}%)")
    print(f"  Rejected: {rejected}/{validation_total} ({rejected/validation_total*100:.1f}%)")
    print()

    survival = passed / turn_count * 100 if turn_count else 0
    print(f"  Survival rate (stored/turns): {passed}/{turn_count} = {survival:.1f}%")
    print()

    print("--- Rejection Examples (first 3 each) ---")
    for reason in sorted(rejected_examples):
        print(f"  {reason}:")
        for s, p, o in rejected_examples[reason][:3]:
            print(f"    ({s!r}, {p!r}, {o!r})")

    try:
        os.unlink(db_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
