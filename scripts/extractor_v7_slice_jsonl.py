import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Slice a JSONL file by 1-based line numbers (non-empty lines only).")
    ap.add_argument("--in", dest="src", required=True, help="Input JSONL")
    ap.add_argument("--out", required=True, help="Output JSONL")
    ap.add_argument("--start", type=int, required=True, help="Start line (1-based, inclusive)")
    ap.add_argument("--end", type=int, required=True, help="End line (1-based, inclusive)")
    args = ap.parse_args()

    if args.start < 1 or args.end < args.start:
        raise SystemExit("Invalid start/end.")

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    i = 0
    wrote = 0
    with src.open("r", encoding="utf-8") as fi, out.open("w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            i += 1
            if i < args.start:
                continue
            if i > args.end:
                break
            fo.write(line if line.endswith("\n") else line + "\n")
            wrote += 1

    print(f"wrote_slice {out} {wrote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

