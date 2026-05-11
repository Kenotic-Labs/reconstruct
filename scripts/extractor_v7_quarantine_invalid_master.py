import argparse
import json
import shutil
from pathlib import Path

from extractor_v7_validate_master import _load_allowed_predicates, _iter_jsonl, REQUIRED_MASTER_FIELDS


def _is_master_row_valid(obj: dict, allowed_predicates: set[str]) -> bool:
    for field in REQUIRED_MASTER_FIELDS:
        if field not in obj:
            return False

    facts = obj.get("facts") or []
    if not isinstance(facts, list):
        return False

    for fact in facts:
        if not isinstance(fact, dict):
            return False
        pred = fact.get("predicate")
        if pred is None:
            continue
        if pred not in allowed_predicates:
            return False

    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Quarantine invalid master JSONL files instead of deleting them.")
    ap.add_argument("--master-dir", default="data/extractor_v7/master")
    ap.add_argument("--quarantine-dir", default="data/extractor_v7/quarantine/master")
    ap.add_argument("--predicate-inventory", default="data/extractor_v7/predicate_inventory.json")
    args = ap.parse_args()

    master_dir = Path(args.master_dir)
    quarantine_dir = Path(args.quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    allowed = _load_allowed_predicates(Path(args.predicate_inventory))

    moved = 0
    for fp in sorted(master_dir.glob("*.jsonl")):
        invalid = False
        try:
            for _, obj in _iter_jsonl(fp):
                if not _is_master_row_valid(obj, allowed):
                    invalid = True
                    break
        except json.JSONDecodeError:
            invalid = True

        if invalid:
            dest = quarantine_dir / fp.name
            shutil.move(str(fp), str(dest))
            moved += 1

    print(f"quarantined_files {moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

