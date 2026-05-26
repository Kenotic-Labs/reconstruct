#!/usr/bin/env python3
"""
Demo: Hard story reconstruction via YAML-bypass ingest.

Runs 3 harder demo stories that stress the reconstruction engine on
continuity — temporal arcs, entity webs, and supersession/corrections.

Stories:
  1. job_search_arc.yaml   — career arc over 5 sessions
  2. relationship_web.yaml — complex social dynamics, secrets
  3. contradiction_update.yaml — facts that change, must supersede

Usage:
    py -3.10 tools/demo_hard_stories.py
    py -3.10 tools/demo_hard_stories.py --story job_search_arc
    py -3.10 tools/demo_hard_stories.py --story contradiction_update
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-33ef6337-3850-1211-4834-097b0c5873a5")
os.environ.setdefault("RAYA_EMBED_DEVICE", "cuda")
os.environ.setdefault("HF_HOME", "D:/Nura/Env/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/Nura/Env/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "D:/Nura/Env/hf_cache/hub")
os.environ["RAYA_SENTENCE_POLISH"] = "0"

import argparse
import io
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STORY_DIR = PROJECT_ROOT / "tests" / "stories" / "demo"

STORIES = [
    {
        "file": "job_search_arc.yaml",
        "db": "demo_job_search.db",
        "user_id": 901,
        "label": "Job Search Arc",
    },
    {
        "file": "relationship_web.yaml",
        "db": "demo_relationship_web.db",
        "user_id": 902,
        "label": "Relationship Web",
    },
    {
        "file": "contradiction_update.yaml",
        "db": "demo_contradiction.db",
        "user_id": 903,
        "label": "Contradiction Update",
    },
]


def wipe_and_init(db_path: str):
    """Create a fresh SQLite database with full schema."""
    from app.db.models import MIGRATIONS, run_schema_upgrades

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        Path(db_path).unlink(missing_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(db_path)
    conn.executescript(MIGRATIONS)
    run_schema_upgrades(conn)
    try:
        conn.execute("ALTER TABLE relationships ADD COLUMN sequence_number INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_seq ON relationships(user_id, sequence_number)")
    except Exception:
        pass
    conn.commit()
    conn.close()


def apply_overrides(rel_id: int, overrides: dict):
    """Apply trace overrides to a stored edge."""
    from app.db.session import get_db_context
    sets = []
    params = []
    for col, val in overrides.items():
        sets.append(f"{col} = ?")
        params.append(val)
    if not sets:
        return
    params.append(rel_id)
    sql = f"UPDATE relationships SET {', '.join(sets)} WHERE id = ?"
    try:
        with get_db_context() as conn:
            conn.execute(sql, params)
            conn.commit()
    except Exception as e:
        print(f"    (override warning: {e})")


def ingest_story(story_data: dict, memory, user_id: int) -> int:
    """Ingest all batches from a story YAML. Returns count of stored triples.

    For batches with 'supersedes_triples', looks up the old triple by
    (subject, predicate, old_object) and explicitly calls memory.supersede().
    """
    from app.db.session import get_db_context

    n_stored = 0
    # Track stored triple IDs for supersession lookup
    # Key: (lower(subject), lower(predicate), lower(object)) -> rel_id
    triple_index: Dict[tuple, int] = {}

    for batch in story_data.get("batches", []):
        text = batch.get("user_input") or batch.get("text") or ""
        overrides = batch.get("trace_overrides", {})
        batch_num = batch.get("batch", "?")
        batch_time = batch.get("time", "")
        triples = batch.get("expected_triples") or []
        supersedes = batch.get("supersedes_triples") or []

        print(f"  Batch {batch_num} ({batch_time}): {len(triples)} triples", end="")
        if supersedes:
            print(f" + {len(supersedes)} supersessions", end="")
        print()

        batch_new_ids = []
        for t in triples:
            if not isinstance(t, (list, tuple)) or len(t) < 3:
                continue
            s, p, o = str(t[0]), str(t[1]), str(t[2])

            rel_id = memory.store(
                user_id=user_id,
                subject=s,
                predicate=p,
                object=o,
                source_text=text.strip()[:200],
                confidence=0.95,
            )
            if rel_id:
                n_stored += 1
                triple_index[(s.lower(), p.lower(), o.lower())] = rel_id
                batch_new_ids.append((s, p, o, rel_id))
                if overrides:
                    apply_overrides(rel_id, overrides)

        # Handle explicit supersession for contradictions
        for sup in supersedes:
            old_key = (
                sup["subject"].lower(),
                sup["predicate"].lower(),
                sup["old_object"].lower(),
            )
            old_id = triple_index.get(old_key)
            if old_id is None:
                print(f"    (supersession warning: could not find old triple {old_key})")
                continue
            # Find the new triple that supersedes the old one.
            # First try: match on same subject + predicate from this batch.
            # Fallback: use the last triple in the batch — the supersession
            # is structurally about invalidation, not about pairing, so any
            # new triple from the same batch can serve as the superseding ID.
            new_id = None
            for s, p, o, rid in batch_new_ids:
                if (s.lower() == sup["subject"].lower()
                        and p.lower() == sup["predicate"].lower()):
                    new_id = rid
                    break
            if new_id is None and batch_new_ids:
                new_id = batch_new_ids[-1][3]
            if new_id:
                memory.supersede(old_id, new_id)
                print(f"    superseded: [{sup['subject']}, {sup['predicate']}, {sup['old_object']}] -> id={new_id}")
            else:
                print(f"    (supersession warning: no new triple found for {sup['subject']}/{sup['predicate']})")

    return n_stored


def run_verification(story_data: dict, retrieval, user_id: int) -> List[Dict[str, Any]]:
    """Run final_verification queries and check results."""
    from app.engines.retrieval import Answer, Situation

    results = []
    verifications = story_data.get("final_verification", [])

    for v in verifications:
        question = v["question"]
        mode = v.get("type", "reconstruct")
        expected = v.get("expected_contains", [])
        must_not = v.get("must_not_contain", [])

        print(f"  Q: {question}")
        print(f"  Mode: {mode}")

        try:
            result = retrieval.answer(user_id, question)
        except Exception as e:
            print(f"  *** ERROR: {e}")
            results.append({
                "question": question,
                "passed": False,
                "failure_layer": "engine_error",
                "detail": str(e),
            })
            print("-" * 50)
            print()
            continue

        # Extract text from result
        output_text = ""
        if isinstance(result, Situation):
            output_text = result.text or ""
            print(f"  Type: Situation")
            print(f"  Edges: {len(result.edge_ids)}")
            print()
            # Wrap narrative
            words = output_text.split()
            lines = []
            current = "  "
            for w in words:
                if len(current) + len(w) + 1 > 78:
                    lines.append(current)
                    current = "  " + w
                else:
                    current += (" " if len(current) > 2 else "") + w
            if current.strip():
                lines.append(current)
            print("  --- NARRATIVE ---")
            print("\n".join(lines))
            print("  --- END NARRATIVE ---")
        elif isinstance(result, Answer):
            output_text = result.text or ""
            print(f"  Type: Answer")
            print(f"  Text: {output_text}")
            print(f"  Grounding: {result.grounding[:1]}")
        else:
            print(f"  Type: {type(result).__name__} (unexpected)")

        # Check expected_contains
        text_lower = output_text.lower()
        found = []
        missing = []
        for kw in expected:
            if kw.lower() in text_lower:
                found.append(kw)
            else:
                missing.append(kw)

        # Check must_not_contain (supersession test)
        leaked = []
        for kw in must_not:
            if kw.lower() in text_lower:
                leaked.append(kw)

        passed = (len(missing) == 0 and len(leaked) == 0 and len(output_text.strip()) > 0)

        if not output_text.strip():
            failure_layer = "empty_output"
        elif missing:
            failure_layer = "missing_expected_content"
        elif leaked:
            failure_layer = "supersession_leak"
        else:
            failure_layer = None

        print()
        if found:
            print(f"  FOUND: {found}")
        if missing:
            print(f"  *** MISSING: {missing}")
        if leaked:
            print(f"  *** LEAKED (superseded content appeared): {leaked}")
        print(f"  {'PASS' if passed else 'FAIL'}", end="")
        if failure_layer:
            print(f" [{failure_layer}]", end="")
        print()

        results.append({
            "question": question,
            "passed": passed,
            "failure_layer": failure_layer,
            "found": found,
            "missing": missing,
            "leaked": leaked,
            "output_text": output_text[:300],
        })

        print("-" * 50)
        print()

    return results


def run_story(story_info: dict) -> Dict[str, Any]:
    """Run a single story end-to-end: wipe DB, ingest, verify."""
    import yaml
    from config.settings import settings
    from app.engines.memory import get_memory_engine
    from app.engines.temporal import get_temporal_engine
    from app.engines.retrieval import get_retrieval_engine

    SEP = "=" * 70
    db_path = str(PROJECT_ROOT / "Memory Storage" / story_info["db"])
    user_id = story_info["user_id"]
    label = story_info["label"]

    print(SEP)
    print(f"  KENOTIC DEMO: {label}")
    print(f"  YAML-bypass ingest -> structural reconstruction -> grammar engine")
    print(f"  No LLM at read time. Deterministic.")
    print(SEP)
    print()

    wipe_and_init(db_path)
    settings.sqlite_path = db_path

    # Force fresh engine instances for the new DB
    memory = get_memory_engine()
    temporal = get_temporal_engine()
    temporal.bind_memory(memory)
    retrieval = get_retrieval_engine(memory, temporal)

    story_path = STORY_DIR / story_info["file"]
    with open(story_path, "r", encoding="utf-8") as f:
        story = yaml.safe_load(f)

    print("-- WRITE PATH (YAML-bypass: hand-authored triples) --")
    print()
    n_stored = ingest_story(story, memory, user_id)
    print()
    print(f"  Total triples stored: {n_stored}")
    print()

    print(SEP)
    print("  READ PATH: Structural Reconstruction")
    print(SEP)
    print()

    results = run_verification(story, retrieval, user_id)

    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])

    print(SEP)
    print(f"  {label}: {passed}/{len(results)} passed, {failed} failed")
    if failed > 0:
        failure_layers = {}
        for r in results:
            if not r["passed"] and r.get("failure_layer"):
                fl = r["failure_layer"]
                failure_layers[fl] = failure_layers.get(fl, 0) + 1
        print(f"  Failure layers: {failure_layers}")
    print(SEP)
    print()

    return {
        "story": label,
        "file": story_info["file"],
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hard demo stories")
    parser.add_argument(
        "--story",
        type=str,
        default=None,
        help="Run a single story by filename stem (e.g. 'job_search_arc')",
    )
    args = parser.parse_args()

    if args.story:
        filtered = [s for s in STORIES if args.story in s["file"]]
        if not filtered:
            print(f"No story matching '{args.story}'. Available:")
            for s in STORIES:
                print(f"  {s['file']}")
            return 1
        stories_to_run = filtered
    else:
        stories_to_run = STORIES

    SEP = "=" * 70
    all_reports = []

    for story_info in stories_to_run:
        report = run_story(story_info)
        all_reports.append(report)

    # Summary
    print()
    print(SEP)
    print("  OVERALL SUMMARY")
    print(SEP)
    total_pass = 0
    total_fail = 0
    for rpt in all_reports:
        status = "PASS" if rpt["failed"] == 0 else "FAIL"
        print(f"  [{status}] {rpt['story']}: {rpt['passed']}/{rpt['total']}")
        total_pass += rpt["passed"]
        total_fail += rpt["failed"]
    print()
    print(f"  Grand total: {total_pass}/{total_pass + total_fail} passed")
    print(SEP)

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
