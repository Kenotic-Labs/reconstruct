import argparse
import json
import random
from pathlib import Path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample a gold-judge queue from silver master rows (rows with >=1 fact).")
    ap.add_argument("--in", dest="src", required=True, help="Silver master JSONL")
    ap.add_argument("--out", required=True, help="Output queue JSONL (raw-candidate shape)")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = []
    for row in _iter_jsonl(Path(args.src)):
        facts = row.get("facts") or []
        if not isinstance(facts, list) or len(facts) == 0:
            continue
        pool.append(row)

    if args.count > len(pool):
        raise SystemExit(f"Requested {args.count}, available {len(pool)} rows with facts.")

    chosen = rng.sample(pool, args.count)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fo:
        for row in chosen:
            # Convert to raw-candidate shape for judges.
            fo.write(
                json.dumps(
                    {
                        "candidate_id": row.get("id"),
                        "source_text_raw": row.get("source_text"),
                        "source_url": row.get("source_url"),
                        "source_domain": row.get("source_domain"),
                        "date_accessed": row.get("date_accessed"),
                        "source_type": row.get("source_type"),
                        "license_or_usage_note": row.get("license_or_usage_note"),
                        "domain": row.get("domain"),
                        "predicate_family_guess": "from_silver",
                        "style_band": row.get("style_band"),
                        "notes": "sampled_from_silver",
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    print(f"wrote_queue {out} {len(chosen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

