import argparse
import json
from pathlib import Path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _is_user_like_source_type(obj: dict) -> bool:
    st = (obj.get("source_type") or "")
    if not isinstance(st, str):
        return False
    st = st.lower()
    return ("user" in st) or ("human" in st) or ("prompter" in st)


def _has_first_person_pronoun(text: str) -> bool:
    # Closed-class pronouns only (allowed): no open-ended wordlists.
    t = text.lower()
    padded = f" {t} "
    return (
        (" i " in padded)
        or (" me " in padded)
        or (" my " in padded)
        or (" we " in padded)
        or (" us " in padded)
        or (" our " in padded)
        or (" i'm" in t)
        or (" i've" in t)
        or (" i'd" in t)
        or (" i'll" in t)
        or (" we're" in t)
        or (" we've" in t)
        or (" we'd" in t)
        or (" we'll" in t)
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a gold-labeling queue by streaming raw candidates and applying only structural filters."
    )
    ap.add_argument("--in", dest="inputs", action="append", required=True, help="Input raw candidate JSONL (repeatable)")
    ap.add_argument("--out", required=True, help="Output queue JSONL")
    ap.add_argument("--count", type=int, required=True, help="Max rows to write")
    ap.add_argument("--user-only", action="store_true", help="Only include rows that appear to be user/human turns")
    ap.add_argument(
        "--first-person-only",
        action="store_true",
        help="Only include rows that contain closed-class first-person pronouns (I/me/my/we/us/our)",
    )
    ap.add_argument("--no-questionmark", action="store_true", help="Exclude rows containing '?' to bias toward statements")
    ap.add_argument("--dedupe-by-id", action="store_true", help="Drop duplicate candidate_id values across inputs")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0

    with out_path.open("w", encoding="utf-8") as fo:
        for in_path in args.inputs:
            p = Path(in_path)
            for obj in _iter_jsonl(p):
                if written >= args.count:
                    break
                cid = obj.get("candidate_id")
                if args.dedupe_by_id:
                    if not isinstance(cid, str) or not cid:
                        continue
                    if cid in seen:
                        continue
                    seen.add(cid)

                if args.user_only and not _is_user_like_source_type(obj):
                    continue

                text = obj.get("source_text_raw")
                if not isinstance(text, str) or not text.strip():
                    continue

                if args.first_person_only and not _has_first_person_pronoun(text):
                    continue

                if args.no_questionmark and ("?" in text):
                    continue

                fo.write(json.dumps(obj, ensure_ascii=True) + "\n")
                written += 1

            if written >= args.count:
                break

    print(f"wrote_queue {out_path} {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

