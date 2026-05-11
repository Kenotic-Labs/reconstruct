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


def _triplet_string(facts: list[dict]) -> str:
    parts: list[str] = []
    for fact in facts:
        s = fact.get("subject")
        p = fact.get("predicate")
        o = fact.get("object")
        if s is None or p is None or o is None:
            continue
        parts.append(f"({s}, {p}, {o})")
    return " | ".join(parts)


def _roles_string(roles: list[dict]) -> str:
    parts: list[str] = []
    for r in roles:
        label = r.get("label")
        text = r.get("text")
        if not label or not text:
            continue
        parts.append(f"{label}:{text}")
    return " | ".join(parts)


def _affect_string(affect: list[dict]) -> str:
    parts: list[str] = []
    for a in affect:
        s = a.get("subject")
        p = a.get("predicate")
        o = a.get("object")
        if s is None or p is None or o is None:
            continue
        parts.append(f"({s}, {p}, {o})")
    return " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive cleanup/triplets/roles/affect task files from master JSONL.")
    ap.add_argument("--master", required=True, help="Master JSONL input")
    ap.add_argument("--out-dir", required=True, help="Output directory (creates cleanup/triplets/roles/affect subfolders)")
    args = ap.parse_args()

    master_path = Path(args.master)
    out_dir = Path(args.out_dir)
    cleanup_dir = out_dir / "cleanup"
    triplets_dir = out_dir / "triplets"
    roles_dir = out_dir / "roles"
    affect_dir = out_dir / "affect"
    for d in (cleanup_dir, triplets_dir, roles_dir, affect_dir):
        d.mkdir(parents=True, exist_ok=True)

    cleanup_out = cleanup_dir / (master_path.stem + ".jsonl")
    triplets_out = triplets_dir / (master_path.stem + ".jsonl")
    roles_out = roles_dir / (master_path.stem + ".jsonl")
    affect_out = affect_dir / (master_path.stem + ".jsonl")

    with cleanup_out.open("w", encoding="utf-8") as fc, triplets_out.open("w", encoding="utf-8") as ft, roles_out.open("w", encoding="utf-8") as fr, affect_out.open("w", encoding="utf-8") as fa:
        n = 0
        for obj in _iter_jsonl(master_path):
            rid = obj.get("id")
            src = obj.get("source_text") or ""
            clean = obj.get("clean_text") or ""
            facts = obj.get("facts") or []
            roles = obj.get("roles") or []
            affect = obj.get("affect") or []

            fc.write(json.dumps({"id": rid, "task": "cleanup", "input": f"<cleanup> {src}", "output": clean}, ensure_ascii=True) + "\n")
            ft.write(json.dumps({"id": rid, "task": "triplets", "input": f"<triplets> {clean}", "output": _triplet_string(facts)}, ensure_ascii=True) + "\n")
            fr.write(json.dumps({"id": rid, "task": "roles", "input": f"<roles> {clean}", "output": _roles_string(roles)}, ensure_ascii=True) + "\n")
            fa.write(json.dumps({"id": rid, "task": "affect", "input": f"<affect> {clean}", "output": _affect_string(affect)}, ensure_ascii=True) + "\n")
            n += 1

    print(f"derived {n} rows into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

