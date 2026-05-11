import argparse
import json
from pathlib import Path
from typing import Any, Iterable


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


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield lineno, json.loads(line)


def _load_allowed_predicates(predicate_inventory_path: Path) -> set[str]:
    inv = json.loads(predicate_inventory_path.read_text(encoding="utf-8"))
    preds = inv.get("predicates")
    if not isinstance(preds, list) or not preds:
        raise SystemExit(f"Invalid predicate inventory: missing non-empty 'predicates' list in {predicate_inventory_path}")
    return set(preds)


def _clean_english(text: str) -> str:
    t = " ".join(text.strip().split())
    if not t:
        return t
    # Minimal denoise without semantic rewriting.
    repl = {
        " i ": " I ",
        " i'm ": " I'm ",
        " i've ": " I've ",
        " i'd ": " I'd ",
        " i'll ": " I'll ",
    }
    padded = f" {t} "
    low = padded.lower()
    for k, v in repl.items():
        low = low.replace(k, v)
    t = low.strip()
    if t.startswith("i "):
        t = "I " + t[2:]
    if t.endswith(" ."):
        t = t[:-2] + "."
    if t and t[-1] not in ".!?":
        t += "."
    return t


def _strip_trailing_punct(s: str) -> str:
    return s.strip().rstrip(".,;:!?").strip()


def _take_after(text: str, marker: str) -> str | None:
    idx = text.find(marker)
    if idx < 0:
        return None
    return text[idx + len(marker) :].strip()


def _first_token_name(s: str) -> str | None:
    s = s.strip()
    if not s:
        return None
    tok = s.split()[0].strip(".,;:!?()[]{}\"'")
    if not tok:
        return None
    if not tok[0].isalpha():
        return None
    return tok[:1].upper() + tok[1:]


def _domain_for_predicates(preds: set[str]) -> str:
    # Stable buckets; keep coarse.
    if {"works_at", "occupation", "manager_name", "worked_for_duration"} & preds:
        return "career"
    if {"lives_in", "lives_since", "rent_per_month"} & preds:
        return "housing"
    if {"graduated_from", "degree", "thesis_topic"} & preds:
        return "education"
    if {"allergic_to"} & preds:
        return "health"
    if {"partner_name", "relationship_duration", "best_friend"} & preds:
        return "relationships"
    if {"has_pet", "pet_name", "pet_breed"} & preds:
        return "pets"
    if {"favorite_food"} & preds:
        return "preferences"
    if {"has_emotion", "emotional_state", "feels_about"} & preds:
        return "affect"
    if {"training_for", "runs_per_week", "birthday_date"} & preds:
        return "mixed"
    return "mixed"


def _mk_fact(
    subject: str,
    predicate: str,
    obj: str,
    *,
    is_historical: bool = False,
    object_type_hint: str | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "is_historical": bool(is_historical),
        "temporal": None,
        "emotion": None,
    }
    if object_type_hint:
        d["object_type_hint"] = object_type_hint
    return d


def _map_text_to_outputs(
    text_raw: str,
    allowed_predicates: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """
    Conservative, inventory-locked mapping from English user/profile lines to:
    - facts: canonical (subject, predicate, object) rows
    - roles: temporal role tags (minimal)
    - affect: explicit affect facts (minimal)
    - notes: reasons / mapping notes
    """
    t = _strip_trailing_punct(" ".join(text_raw.strip().split()))
    low = t.lower()
    facts: list[dict[str, Any]] = []
    roles: list[dict[str, Any]] = []
    affect: list[dict[str, Any]] = []
    notes: list[str] = []

    def add_fact(subj: str, pred: str, obj: str, **kw):
        if pred not in allowed_predicates:
            return
        obj2 = _strip_trailing_punct(obj)
        if not obj2:
            return
        facts.append(_mk_fact(subj, pred, obj2, **kw))

    # Employment / occupation
    if "works_at" in allowed_predicates:
        for marker in ["i work at ", "i work for ", "i work in "]:
            rest = _take_after(low, marker)
            if rest:
                org = _first_token_name(rest)
                if org:
                    add_fact("user", "works_at", org, object_type_hint="ORG")
                    notes.append(f"works_at:{marker.strip()}")
                    break

    if "occupation" in allowed_predicates:
        for marker in ["i am a ", "i'm a ", "i'm an ", "i am an "]:
            rest = _take_after(low, marker)
            if rest:
                job = _strip_trailing_punct(rest)
                if job:
                    add_fact("user", "occupation", job, object_type_hint="ROLE")
                    notes.append(f"occupation:{marker.strip()}")
                    break

    # Manager
    if "manager_name" in allowed_predicates:
        for marker in ["my manager is ", "my boss is "]:
            rest = _take_after(low, marker)
            if rest:
                nm = _first_token_name(rest)
                if nm:
                    # Keep as surface token; schema doesn't require full name parsing here.
                    add_fact("user", "manager_name", nm, object_type_hint="PERSON")
                    notes.append(f"manager_name:{marker.strip()}")
                    break

    # Location / housing
    if "lives_in" in allowed_predicates:
        for marker in ["i live in ", "i'm in ", "i am in "]:
            rest = _take_after(low, marker)
            if rest:
                place = _strip_trailing_punct(rest)
                if place:
                    add_fact("user", "lives_in", place, object_type_hint="LOCATION")
                    notes.append(f"lives_in:{marker.strip()}")
                    break

    # Education
    if "graduated_from" in allowed_predicates:
        for marker in ["i graduated from ", "i went to "]:
            rest = _take_after(low, marker)
            if rest:
                inst = _first_token_name(rest)
                if inst:
                    add_fact("user", "graduated_from", inst, object_type_hint="ORG")
                    notes.append(f"graduated_from:{marker.strip()}")
                    break

    if "degree" in allowed_predicates:
        for marker in ["my degree is in ", "i studied ", "i majored in "]:
            rest = _take_after(low, marker)
            if rest:
                field = _strip_trailing_punct(rest)
                if field:
                    add_fact("user", "degree", field)
                    notes.append(f"degree:{marker.strip()}")
                    break

    # Pets
    if "has_pet" in allowed_predicates or "pet_name" in allowed_predicates:
        for marker in ["my dog is named ", "my dog named ", "i have a dog named ", "i have a cat named ", "my cat is named "]:
            rest = _take_after(low, marker)
            if rest:
                nm = _first_token_name(rest)
                if nm:
                    if "has_pet" in allowed_predicates:
                        add_fact("user", "has_pet", "dog" if "dog" in marker else "cat")
                    if "pet_name" in allowed_predicates:
                        add_fact("user", "pet_name", nm, object_type_hint="ANIMAL")
                    notes.append(f"pet_name:{marker.strip()}")
                    break

    # Preferences
    if "favorite_food" in allowed_predicates:
        for marker in ["my favorite food is ", "my favourite food is "]:
            rest = _take_after(low, marker)
            if rest:
                food = _strip_trailing_punct(rest)
                if food:
                    add_fact("user", "favorite_food", food, object_type_hint="FOOD")
                    notes.append(f"favorite_food:{marker.strip()}")
                    break

    # Health
    if "allergic_to" in allowed_predicates:
        for marker in ["i'm allergic to ", "i am allergic to "]:
            rest = _take_after(low, marker)
            if rest:
                a = _strip_trailing_punct(rest)
                if a:
                    add_fact("user", "allergic_to", a)
                    notes.append(f"allergic_to:{marker.strip()}")
                    break

    # Relationships
    if "partner_name" in allowed_predicates:
        for marker in ["my girlfriend is ", "my boyfriend is ", "my partner is ", "my wife is ", "my husband is "]:
            rest = _take_after(low, marker)
            if rest:
                nm = _first_token_name(rest)
                if nm:
                    add_fact("user", "partner_name", nm, object_type_hint="PERSON")
                    notes.append(f"partner_name:{marker.strip()}")
                    break

    if "best_friend" in allowed_predicates:
        for marker in ["my best friend is "]:
            rest = _take_after(low, marker)
            if rest:
                nm = _first_token_name(rest)
                if nm:
                    add_fact("user", "best_friend", nm, object_type_hint="PERSON")
                    notes.append("best_friend")
                    break

    if "relationship_duration" in allowed_predicates:
        for marker in ["we've been together for ", "we have been together for "]:
            rest = _take_after(low, marker)
            if rest:
                dur = _strip_trailing_punct(rest)
                if dur:
                    add_fact("user", "relationship_duration", dur)
                    roles.append({"label": "TEMPORAL", "text": dur})
                    notes.append("relationship_duration")
                    break

    # Explicit affect
    if "has_emotion" in allowed_predicates:
        for marker in ["i feel ", "i'm feeling ", "i am feeling "]:
            rest = _take_after(low, marker)
            if rest:
                emo = _strip_trailing_punct(rest)
                if emo:
                    add_fact("user", "has_emotion", emo)
                    affect.append({"subject": "user", "emotion": emo})
                    notes.append("has_emotion")
                    break

    return facts, roles, affect, notes


def _validate_master_row(row: dict, allowed_predicates: set[str]) -> list[str]:
    errs: list[str] = []
    for k in REQUIRED_MASTER_FIELDS:
        if k not in row:
            errs.append(f"missing_field:{k}")
    if not isinstance(row.get("facts"), list):
        errs.append("facts_not_list")
    else:
        for i, f in enumerate(row["facts"]):
            if f.get("predicate") not in allowed_predicates:
                errs.append(f"invalid_predicate:facts[{i}]")
    return errs


def main():
    ap = argparse.ArgumentParser(description="Gold-judge an entire queue JSONL into master-schema JSONL (inventory-locked).")
    ap.add_argument("--in", dest="input_path", required=True, help="Input queue JSONL")
    ap.add_argument("--out", required=True, help="Output master-schema JSONL")
    ap.add_argument("--report", required=True, help="Output report markdown")
    ap.add_argument(
        "--predicate-inventory",
        default="data/extractor_v7/predicate_inventory.json",
        help="Predicate inventory JSON",
    )
    args = ap.parse_args()

    input_path = Path(args.input_path)
    out_path = Path(args.out)
    report_path = Path(args.report)
    allowed_predicates = _load_allowed_predicates(Path(args.predicate_inventory))

    kept = 0
    rejected = 0
    invalid = 0
    by_domain: dict[str, int] = {}
    by_pred: dict[str, int] = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for _, row in _iter_jsonl(input_path):
            cand_id = row.get("candidate_id") or row.get("id") or ""
            src_raw = row.get("source_text_raw") or row.get("source_text") or ""
            if not isinstance(src_raw, str):
                rejected += 1
                continue
            src_raw = src_raw.strip()
            if not src_raw:
                rejected += 1
                continue

            facts, roles, affect, notes = _map_text_to_outputs(src_raw, allowed_predicates)
            if not facts:
                rejected += 1
                continue

            preds = {f["predicate"] for f in facts if isinstance(f, dict) and "predicate" in f}
            dom = _domain_for_predicates(preds)
            by_domain[dom] = by_domain.get(dom, 0) + 1
            for p in preds:
                by_pred[p] = by_pred.get(p, 0) + 1

            master = {
                "id": f"{cand_id}",
                "source_text": src_raw,
                "clean_text": _clean_english(src_raw),
                "facts": facts,
                "roles": roles,
                "affect": affect,
                "domain": row.get("domain") or dom,
                "style_band": row.get("style_band") or "direct",
                "difficulty": "medium",
                "source_type": row.get("source_type") or "public_dataset_queue",
                "source_url": row.get("source_url") or "",
                "source_domain": row.get("source_domain") or "",
                "date_accessed": row.get("date_accessed") or "",
                "license_or_usage_note": row.get("license_or_usage_note") or "",
                "notes": " ; ".join(notes) if notes else "",
            }
            errs = _validate_master_row(master, allowed_predicates)
            if errs:
                invalid += 1
                continue

            out_f.write(json.dumps(master, ensure_ascii=True) + "\n")
            kept += 1

    lines = [
        "# Gold Judge Report",
        "",
        f"- input: `{input_path.as_posix()}`",
        f"- output: `{out_path.as_posix()}`",
        f"- kept: `{kept}`",
        f"- rejected: `{rejected}`",
        f"- invalid_master_rows: `{invalid}`",
        "",
        "## Kept By Domain",
    ]
    for k in sorted(by_domain.keys()):
        lines.append(f"- {k}: {by_domain[k]}")
    lines += ["", "## Predicates (Kept Rows)"]
    for k in sorted(by_pred.keys()):
        lines.append(f"- {k}: {by_pred[k]}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

