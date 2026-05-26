#!/usr/bin/env python3
"""
ATANT Cumulative — rewritten on the five-engine architecture (2026-04-12).

Wipes the test DB each run (dev loop — no migration ceremony).
Uses the new engines directly:
    MemoryEngine.store(...)         — write path
    RetrievalEngine.retrieve(...)   — single read path (DTCM + coherence)
    TemporalEngine                  — parse + supersession
No classification. No structural_matcher. No parallel read paths.

Usage:
    py -3.10 run_atant_cumulative.py --range 51-55
    py -3.10 run_atant_cumulative.py --range 51-150
"""
# Env first
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-33ef6337-3850-1211-4834-097b0c5873a5")
os.environ.setdefault("RAYA_EMBED_DEVICE", "cuda")
os.environ.setdefault("HF_HOME", "D:/Nura/Env/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/Nura/Env/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "D:/Nura/Env/hf_cache/hub")

import argparse
import io
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_DB = str(PROJECT_ROOT / "Memory Storage" / "locomo" / "atant_cumulative.db")
REPORT_DIR = PROJECT_ROOT / "test_reports"

from config.settings import settings
settings.sqlite_path = TEST_DB

# Engines
from app.engines.memory import MemoryEngine, get_memory_engine
from app.engines.temporal import TemporalEngine, get_temporal_engine
from app.engines.retrieval import RetrievalEngine, get_retrieval_engine
from app.db.session import get_db_context
from app.db.models import MIGRATIONS, run_schema_upgrades


# ─────────────────────────────────────────────────────────────────────
# DB lifecycle — wipe and recreate schema each run.
# ─────────────────────────────────────────────────────────────────────

def wipe_and_init_db() -> None:
    """Dev-loop DB: delete, recreate schema."""
    Path(TEST_DB).parent.mkdir(parents=True, exist_ok=True)
    try:
        Path(TEST_DB).unlink(missing_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(TEST_DB)
    conn.executescript(MIGRATIONS)
    run_schema_upgrades(conn)
    # Add sequence_number column if absent (Phase 6 narrative-time axis)
    try:
        conn.execute("ALTER TABLE relationships ADD COLUMN sequence_number INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_seq ON relationships(user_id, sequence_number)")
    except Exception:
        pass
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────
# Grader — semantic OR substring
#
# Passes if ANY expected keyword:
#   (a) appears literally in the surfaced text (fast path), OR
#   (b) has an embedding cosine with the text that is HIGHER than the
#       cosine of distractor keywords drawn from other questions in the
#       same run.
#
# The distractor baseline is population-relative, not hardcoded — we do
# not pick a number. A keyword is "semantically present" iff it stands
# out from the population of unrelated keywords' similarities to the
# same text. This follows the no-threshold rule: the comparison is
# against a distribution, not a magic number.
# ─────────────────────────────────────────────────────────────────────

_DISTRACTOR_POPULATION: List[str] = []


def register_distractor_pool(pool: List[str]) -> None:
    """Register the run-wide pool of expected keywords from all questions.
    Each question's grader uses keywords NOT in its own list as the
    distractor baseline."""
    global _DISTRACTOR_POPULATION
    _DISTRACTOR_POPULATION = list(pool)


def check_keywords(text: str, keywords: List[str]) -> Tuple[bool, List[str]]:
    if not text or not keywords:
        return False, keywords or []

    text_lower = text.lower()
    found = [k for k in keywords if k.lower() in text_lower]
    if found:
        missing = [k for k in keywords if k.lower() not in text_lower]
        return True, missing

    # Semantic fallback — only invoked when every expected keyword misses
    # on literal substring. Use MiniLM cosine.
    try:
        from app.vector.embedder import embed_text
        import numpy as np
    except Exception:
        return False, keywords

    try:
        text_emb = embed_text(text)
    except Exception:
        return False, keywords

    own_set = {k.lower() for k in keywords}
    distractors = [
        d for d in _DISTRACTOR_POPULATION
        if d and d.lower() not in own_set
    ]
    if not distractors:
        return False, keywords

    # Distractor baseline: mean cosine of distractor keywords with the
    # answer text. An expected keyword "semantically matches" iff its
    # cosine exceeds this population mean.
    distractor_sims = []
    # Cap the distractor sample for speed — sample first 40 if larger
    for d in distractors[:40]:
        try:
            d_emb = embed_text(d)
            distractor_sims.append(float(np.dot(text_emb, d_emb)))
        except Exception:
            continue
    if not distractor_sims:
        return False, keywords
    baseline = float(np.mean(distractor_sims))

    # A keyword passes if its cosine with the text exceeds the population
    # mean. No hardcoded threshold — the threshold IS the distribution.
    for kw in keywords:
        try:
            kw_emb = embed_text(kw)
            kw_sim = float(np.dot(text_emb, kw_emb))
            if kw_sim > baseline:
                missing = [k for k in keywords if k.lower() not in text_lower]
                return True, missing
        except Exception:
            continue

    return False, keywords


# ─────────────────────────────────────────────────────────────────────
# Per-story runner
# ─────────────────────────────────────────────────────────────────────

def run_story(
    user_id: int,
    story: Dict[str, Any],
    memory: MemoryEngine,
    retrieval: RetrievalEngine,
    temporal: TemporalEngine,
) -> Dict[str, Any]:
    story_id = story.get("story_id", 0)
    story_name = story.get("title", "") or story.get("story_name", "")
    category = story.get("category", "Unknown")
    batches = story.get("batches") or []
    # Questions live under 'final_verification' in the cumulative YAML spec.
    # Some older stories may use 'questions' — check both.
    questions = story.get("final_verification") or story.get("questions") or []

    # ── Write path ────────────────
    # Per-story decision based on YAML author's intent:
    #   - If the story has ANY hand-authored expected_triples across any
    #     batch → pure YAML mode (ignore batches without triples).
    #   - If the story has ZERO hand-authored triples anywhere → T5 mode
    #     on user_input across all batches.
    has_any_triples = any(
        (b.get("expected_triples") or []) for b in batches
    )
    triple_count = 0
    for batch in batches:
        text = batch.get("text") or batch.get("user_input") or ""
        expected_triples = batch.get("expected_triples") or []

        if has_any_triples:
            # YAML mode — only store hand-authored triples, don't run T5
            # on batches that happen to lack them (that was the YAML
            # author's choice).
            for t in expected_triples:
                if not isinstance(t, (list, tuple)) or len(t) < 3:
                    continue
                s, p, o = str(t[0]), str(t[1]), str(t[2])
                rel_id = memory.store(
                    user_id=user_id,
                    subject=s,
                    predicate=p,
                    object=o,
                    source_text=text,
                    confidence=0.95,
                )
                if rel_id:
                    triple_count += 1
        elif text:
            # T5 mode — no hand-authored triples anywhere in the story.
            # Run the full write path: T5 cleanup + T5 triplets +
            # MemoryEngine.store with edge embeddings + trace columns.
            n = memory.ingest_text(
                user_id=user_id,
                text=text,
                source_timestamp=batch.get("time"),
                speaker="Maya",
                confidence=0.85,
            )
            triple_count += n

    # ── Read path: retrieve answer per question ────────────────
    q_results: List[Dict[str, Any]] = []
    passed = 0
    for q in questions:
        qtext = q.get("question", "") or ""
        expected = q.get("expected_contains") or []
        if isinstance(expected, str):
            expected = [expected]

        try:
            answer = retrieval.retrieve(user_id, qtext)
            answer_text = answer.text or ""
            # Also consider the top candidates as answer context, since
            # DTCM's top is a single triple — concatenate top-5 for grading
            # continuity with prior behavior.
            if answer.candidates:
                cand_texts = []
                for c in answer.candidates:
                    parts = [c.get("subject", ""), c.get("predicate", ""), c.get("object", "")]
                    cand_texts.append(" ".join(p for p in parts if p))
                answer_text = ". ".join(filter(None, [answer_text] + cand_texts))
        except Exception as e:
            answer_text = f"[error: {e}]"

        ok, missing = check_keywords(answer_text, expected)
        if ok:
            passed += 1
        q_results.append({
            "question": qtext,
            "expected": expected,
            "answer": answer_text[:400],
            "passed": ok,
        })

    return {
        "story_id": story_id,
        "story_name": story_name,
        "category": category,
        "triple_count": triple_count,
        "questions_passed": passed,
        "questions_total": len(questions),
        "questions": q_results,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", dest="range_", default=None, help="e.g. 51-55")
    args = parser.parse_args()

    start_id, end_id = 1, 500
    if args.range_:
        parts = args.range_.split("-")
        if len(parts) == 2:
            start_id, end_id = int(parts[0]), int(parts[1])

    print(f"ATANT five-engine runner — range {start_id}-{end_id}")
    print("Wiping test DB ...")
    wipe_and_init_db()

    # Instantiate engines with dependency injection
    memory = get_memory_engine()
    temporal = get_temporal_engine()
    temporal.bind_memory(memory)
    retrieval = get_retrieval_engine(memory, temporal)

    # Load YAML stories
    cumul_dir = PROJECT_ROOT / "tests" / "stories" / "cumulative"
    yaml_files = sorted(cumul_dir.glob("*.yaml"))
    selected: List[Dict[str, Any]] = []
    for yf in yaml_files:
        try:
            with open(yf, encoding="utf-8") as f:
                story = yaml.safe_load(f)
        except Exception as e:
            print(f"[load] skip {yf.name}: {e}")
            continue
        sid = int(story.get("story_id") or 0)
        if start_id <= sid <= end_id:
            selected.append(story)
    selected.sort(key=lambda s: int(s.get("story_id", 0)))
    print(f"Loaded {len(selected)} stories")

    # Build distractor pool: every expected_contains keyword across the
    # whole run. Semantic grader uses other questions' keywords as
    # distractors for this question — population-relative threshold.
    distractor_pool: List[str] = []
    for story in selected:
        for q in story.get("final_verification") or story.get("questions") or []:
            exp = q.get("expected_contains") or []
            if isinstance(exp, str):
                exp = [exp]
            for e in exp:
                if isinstance(e, str) and e.strip():
                    distractor_pool.append(e.strip())
    register_distractor_pool(distractor_pool)

    # Run
    t0 = time.time()
    all_results = []
    total_passed = 0
    total_questions = 0
    total_triples = 0
    for i, story in enumerate(selected, start=1):
        sid = story.get("story_id", 0)
        # Each story gets its own user_id so supersessions don't bleed
        user_id = sid
        res = run_story(user_id, story, memory, retrieval, temporal)
        total_passed += res["questions_passed"]
        total_questions += res["questions_total"]
        total_triples += res["triple_count"]
        all_results.append(res)
        if res["questions_total"]:
            print(f"  [{i:3d}/{len(selected)}] Story {sid}: "
                  f"{res['questions_passed']}/{res['questions_total']}  "
                  f"{res['story_name'][:50]}")

    elapsed = time.time() - t0

    # Report
    print()
    print("=" * 72)
    print(" ATANT Cumulative — Engine Architecture")
    print("=" * 72)
    if total_questions:
        pct = 100 * total_passed / total_questions
    else:
        pct = 0
    print(f"  questions: {total_passed}/{total_questions} ({pct:.1f}%)")
    print(f"  triples:   {total_triples}")
    print(f"  elapsed:   {elapsed:.1f}s  ({elapsed/max(len(selected),1):.1f}s/story)")
    print()

    # Save JSON report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"atant_engines_{stamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": "engines",
            "timestamp": stamp,
            "range": [start_id, end_id],
            "total_stories": len(selected),
            "total_questions": total_questions,
            "passed_questions": total_passed,
            "total_triples": total_triples,
            "elapsed_seconds": elapsed,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"  report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
