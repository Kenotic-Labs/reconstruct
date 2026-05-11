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
    ap = argparse.ArgumentParser(description="Sample a deterministic gold-labeling queue from raw candidates.")
    ap.add_argument("--in", dest="inputs", action="append", required=True, help="Input raw candidate JSONL (repeatable)")
    ap.add_argument("--out", required=True, help="Output queue JSONL")
    ap.add_argument("--count", type=int, required=True, help="Number of rows to sample")
    ap.add_argument("--seed", type=int, required=True, help="Deterministic RNG seed")
    ap.add_argument(
        "--user-only",
        action="store_true",
        help="Only include rows with source_type indicating user/human turns",
    )
    ap.add_argument(
        "--first-person-only",
        action="store_true",
        help="Only include rows that contain closed-class first-person pronouns (I/me/my/we/us/our)",
    )
    ap.add_argument(
        "--no-questionmark",
        action="store_true",
        help="Exclude rows containing '?' to bias toward declarative statements",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen_ids: set[str] = set()
    pool: list[dict] = []

    for in_path in args.inputs:
        p = Path(in_path)
        for obj in _iter_jsonl(p):
            cid = obj.get("candidate_id")
            if not cid or not isinstance(cid, str):
                continue
            if cid in seen_ids:
                continue
            if args.user_only:
                st = (obj.get("source_type") or "").lower()
                if "user" not in st and "human" not in st and "prompter" not in st:
                    continue
            if args.first_person_only:
                text = (obj.get("source_text_raw") or "")
                if not isinstance(text, str):
                    continue
                t = text.lower()
                # Closed-class pronouns only (allowed); avoids open-ended word lists.
                if (
                    " i " not in f" {t} "
                    and " i'm" not in t
                    and " i've" not in t
                    and " i'd" not in t
                    and " me " not in f" {t} "
                    and " my " not in f" {t} "
                    and " we " not in f" {t} "
                    and " we're" not in t
                    and " we've" not in t
                    and " us " not in f" {t} "
                    and " our " not in f" {t} "
                ):
                    continue
                if args.no_questionmark and "?" in t:
                    continue
            seen_ids.add(cid)
            pool.append(obj)

    if not pool:
        raise SystemExit("No candidates available after filtering.")

    if args.count > len(pool):
        raise SystemExit(f"Requested {args.count} but only {len(pool)} available.")

    # Deterministic sample without thresholds or heuristic ranking.
    chosen = rng.sample(pool, args.count)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in chosen:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")

    print(f"wrote_queue {out_path} {len(chosen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
