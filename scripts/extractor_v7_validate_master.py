import argparse
import json
from pathlib import Path


REQUIRED_MASTER_FIELDS = [
    "id",
    "source_text",
    "clean_text",
    "facts",
    "roles",
    "affect",
    "domain",
    "style_band",
    "difficulty",
    "source_type",
    "source_url",
    "source_domain",
    "date_accessed",
    "license_or_usage_note",
]


def _load_allowed_predicates(predicate_inventory_path: Path) -> set[str]:
    inv = json.loads(predicate_inventory_path.read_text(encoding="utf-8"))
    preds = inv.get("predicates")
    if not isinstance(preds, list) or not preds:
        raise SystemExit(f"Invalid predicate inventory: missing non-empty 'predicates' list in {predicate_inventory_path}")
    return set(preds)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield lineno, json.loads(line)


def validate_master_file(path: Path, allowed_predicates: set[str]) -> list[str]:
    errors: list[str] = []
    for lineno, obj in _iter_jsonl(path):
        for field in REQUIRED_MASTER_FIELDS:
            if field not in obj:
                errors.append(f"{path}:{lineno}: missing_field:{field}")

        facts = obj.get("facts") or []
        if not isinstance(facts, list):
            errors.append(f"{path}:{lineno}: facts_not_list")
            continue

        bad_predicates = []
        for fact in facts:
            if not isinstance(fact, dict):
                errors.append(f"{path}:{lineno}: fact_not_object")
                continue
            pred = fact.get("predicate")
            if pred is None:
                continue
            if pred not in allowed_predicates:
                bad_predicates.append(pred)

        if bad_predicates:
            uniq = sorted(set(bad_predicates))
            errors.append(f"{path}:{lineno}: invalid_predicates:{','.join(uniq)}")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate extractor_v7 master JSONL files against predicate inventory and required fields.")
    ap.add_argument("--master-dir", default="data/extractor_v7/master", help="Master JSONL directory (default: data/extractor_v7/master)")
    ap.add_argument(
        "--predicate-inventory",
        default="data/extractor_v7/predicate_inventory.json",
        help="Predicate inventory JSON (default: data/extractor_v7/predicate_inventory.json)",
    )
    args = ap.parse_args()

    master_dir = Path(args.master_dir)
    inv_path = Path(args.predicate_inventory)

    allowed = _load_allowed_predicates(inv_path)
    master_files = sorted(master_dir.glob("*.jsonl"))
    if not master_files:
        print(f"No master files found in {master_dir}")
        return 1

    all_errors: list[str] = []
    for fp in master_files:
        try:
            all_errors.extend(validate_master_file(fp, allowed))
        except json.JSONDecodeError as e:
            all_errors.append(f"{fp}:{e.lineno}: json_decode_error:{e.msg}")

    if all_errors:
        for e in all_errors:
            print(e)
        print(f"FAIL {len(all_errors)} errors")
        return 2

    print(f"PASS {len(master_files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

