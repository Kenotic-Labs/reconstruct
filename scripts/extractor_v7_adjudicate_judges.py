import argparse
import json
from collections import defaultdict
from pathlib import Path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _fact_sig(f: dict) -> tuple:
    # Keep this minimal: subject/predicate/object/historical + temporal.surface if present.
    temporal = f.get("temporal") or {}
    temporal_surface = None
    if isinstance(temporal, dict):
        temporal_surface = temporal.get("surface")
    return (
        f.get("subject"),
        f.get("predicate"),
        f.get("object"),
        bool(f.get("is_historical", False)),
        temporal_surface,
    )


def _row_sig(row: dict) -> tuple:
    facts = row.get("facts") or []
    roles = row.get("roles") or []
    affect = row.get("affect") or []

    fact_sigs = tuple(sorted(_fact_sig(f) for f in facts if isinstance(f, dict)))
    role_sigs = tuple(sorted((r.get("label"), r.get("text")) for r in roles if isinstance(r, dict)))
    affect_sigs = tuple(sorted((a.get("subject"), a.get("predicate"), a.get("object")) for a in affect if isinstance(a, dict)))
    clean_text = row.get("clean_text")
    return (clean_text, fact_sigs, role_sigs, affect_sigs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Adjudicate multiple judge master-schema files by exact agreement.")
    ap.add_argument("--judge", action="append", required=True, help="Judge master-schema JSONL file (repeatable)")
    ap.add_argument("--out", required=True, help="Adjudicated master JSONL output path")
    ap.add_argument("--conflicts-out", required=True, help="Conflict JSONL output path")
    args = ap.parse_args()

    by_key: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for jp in args.judge:
        p = Path(jp)
        for row in _iter_jsonl(p):
            key = row.get("source_url") or row.get("id")
            if not key:
                continue
            by_key[str(key)].append((p.name, row))

    out_path = Path(args.out)
    conflicts_path = Path(args.conflicts_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conflicts_path.parent.mkdir(parents=True, exist_ok=True)

    agreed = 0
    conflicts = 0

    with out_path.open("w", encoding="utf-8") as fo, conflicts_path.open("w", encoding="utf-8") as fc:
        for key, entries in by_key.items():
            if len(entries) < 2:
                # Not enough independent judges; keep it as conflict for later.
                conflicts += 1
                fc.write(json.dumps({"key": key, "reason": "single_judge", "entries": entries}, ensure_ascii=True) + "\n")
                continue
            sigs = [(_row_sig(r), jname, r) for jname, r in entries]
            unique = {}
            for sig, jname, r in sigs:
                unique.setdefault(sig, []).append((jname, r))

            if len(unique) == 1:
                # Exact agreement across all judge versions.
                agreed += 1
                any_row = sigs[0][2]
                fo.write(json.dumps(any_row, ensure_ascii=True) + "\n")
            else:
                conflicts += 1
                fc.write(json.dumps({"key": key, "reason": "disagreement", "variants": {str(i): v for i, v in enumerate(unique.values())}}, ensure_ascii=True) + "\n")

    print(f"adjudicated_agreed {agreed}")
    print(f"adjudicated_conflicts {conflicts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

