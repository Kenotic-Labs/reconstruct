import argparse
import json
from pathlib import Path
from typing import Any


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


def _iter_jsonl(path: Path):
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
    # Ensure initial "i" is capitalized if present.
    if t.startswith("i "):
        t = "I " + t[2:]
    if t.startswith("i'm"):
        t = "I'm" + t[3:]
    if t.startswith("i've"):
        t = "I've" + t[3:]
    if t.startswith("i'd"):
        t = "I'd" + t[2:]
    if t.startswith("i'll"):
        t = "I'll" + t[3:]
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    if t[-1] not in ".!?":
        t = t + "."
    return t


def _strip_trailing_punct(s: str) -> str:
    return s.strip().strip(" \t\r\n\"'.,!?;:()[]{}")


def _take_after(text: str, marker: str) -> str | None:
    idx = text.lower().find(marker.lower())
    if idx < 0:
        return None
    return text[idx + len(marker) :].strip()


def _first_token_name(s: str) -> str | None:
    # Conservative: keep the first token if it looks like a proper name.
    s = _strip_trailing_punct(s)
    if not s:
        return None
    tok = s.split()[0]
    tok = _strip_trailing_punct(tok)
    if not tok:
        return None
    if tok[0].isupper() and tok.isalpha():
        return tok
    return None


def _domain_for_predicates(preds: set[str]) -> str:
    if preds & {"has_pet", "pet_name", "pet_breed"}:
        return "pets"
    if preds & {"favorite_food"}:
        return "preferences"
    if preds & {"works_at", "occupation", "manager_name", "worked_for_duration"}:
        return "career"
    if preds & {"lives_in", "rent_per_month", "lives_since"}:
        return "housing"
    if preds & {"graduated_from", "degree", "thesis_topic"}:
        return "education"
    if preds & {"allergic_to"}:
        return "health"
    if preds & {"partner_name", "relationship_duration", "best_friend"}:
        return "relationships"
    if preds & {"has_emotion", "emotional_state", "feels_about"}:
        return "affect"
    return "mixed"


def _mk_fact(subject: str, predicate: str, obj: str, *, is_historical: bool = False, object_type_hint: str | None = None) -> dict[str, Any]:
    fact: dict[str, Any] = {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "is_historical": bool(is_historical),
        "temporal": None,
        "emotion": None,
        "subject_type_hint": "PERSON",
        "object_type_hint": object_type_hint,
    }
    return fact


def _map_personachat_line_to_facts(text_raw: str, allowed_predicates: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """
    Conservative, high-precision mapper from PersonaChat-like persona lines to the closed predicate inventory.
    Returns (facts, roles, affect, reasons).
    """
    facts: list[dict[str, Any]] = []
    roles: list[dict[str, Any]] = []
    affect: list[dict[str, Any]] = []
    reasons: list[str] = []

    t = " ".join(text_raw.strip().split())
    tl = t.lower()

    # lives_in
    if "lives_in" in allowed_predicates:
        after = _take_after(t, "I live in ")
        if after:
            loc = _strip_trailing_punct(after)
            if loc:
                facts.append(_mk_fact("user", "lives_in", loc, object_type_hint="LOCATION"))

    # works_at
    if "works_at" in allowed_predicates:
        after = _take_after(t, "I work at ")
        if after:
            org = _strip_trailing_punct(after)
            if org:
                facts.append(_mk_fact("user", "works_at", org, object_type_hint="ORG"))

    # occupation
    if "occupation" in allowed_predicates:
        for marker in ("I work as a ", "I work as an ", "My job is ", "I am a ", "I am an "):
            after = _take_after(t, marker)
            if after:
                role = _strip_trailing_punct(after)
                if role:
                    facts.append(_mk_fact("user", "occupation", role, object_type_hint="ROLE"))
                    break

    # graduated_from
    if "graduated_from" in allowed_predicates:
        after = _take_after(t, "I graduated from ")
        if after:
            school = _strip_trailing_punct(after)
            if school:
                facts.append(_mk_fact("user", "graduated_from", school, object_type_hint="ORG"))

    # degree
    if "degree" in allowed_predicates:
        after = _take_after(t, "I have a degree in ")
        if after:
            field = _strip_trailing_punct(after)
            if field:
                facts.append(_mk_fact("user", "degree", field, object_type_hint="GENERIC"))

    # birthday_date
    if "birthday_date" in allowed_predicates:
        after = _take_after(t, "My birthday is ")
        if after:
            bd = _strip_trailing_punct(after)
            if bd:
                facts.append(_mk_fact("user", "birthday_date", bd, object_type_hint="TIME"))

    # allergic_to
    if "allergic_to" in allowed_predicates:
        after = _take_after(t, "I am allergic to ")
        if after:
            allergen = _strip_trailing_punct(after)
            if allergen:
                facts.append(_mk_fact("user", "allergic_to", allergen, object_type_hint="GENERIC"))

    # has_pet / pet_name
    if "has_pet" in allowed_predicates:
        for species in ("dog", "cat", "bird", "fish", "rabbit", "hamster"):
            if f"i have a {species}" in tl or f"i have an {species}" in tl or f"i own a {species}" in tl or f"i own an {species}" in tl:
                facts.append(_mk_fact("user", "has_pet", species, object_type_hint="ANIMAL"))
                break
    if "pet_name" in allowed_predicates:
        for marker in ("My dog's name is ", "My dog is named ", "My cat's name is ", "My cat is named "):
            after = _take_after(t, marker)
            if after:
                name = _first_token_name(after)
                if name:
                    facts.append(_mk_fact("user", "pet_name", name, object_type_hint="PERSON"))
                break

    # favorite_food
    if "favorite_food" in allowed_predicates:
        after = _take_after(t, "My favorite food is ")
        if after:
            food = _strip_trailing_punct(after)
            if food:
                facts.append(_mk_fact("user", "favorite_food", food, object_type_hint="FOOD"))

    # best_friend / partner_name
    if "best_friend" in allowed_predicates:
        after = _take_after(t, "My best friend is ")
        if after:
            name = _first_token_name(after)
            if name:
                facts.append(_mk_fact("user", "best_friend", name, object_type_hint="PERSON"))
    if "partner_name" in allowed_predicates:
        for marker in ("My boyfriend is ", "My girlfriend is ", "My wife is ", "My husband is ", "My partner is "):
            after = _take_after(t, marker)
            if after:
                name = _first_token_name(after)
                if name:
                    facts.append(_mk_fact("user", "partner_name", name, object_type_hint="PERSON"))
                break

    # has_emotion
    if "has_emotion" in allowed_predicates and "i feel " in tl:
        after = _take_after(t, "I feel ")
        if after:
            emo = _strip_trailing_punct(after)
            if emo:
                facts.append(_mk_fact("user", "has_emotion", emo, object_type_hint="GENERIC"))
                affect.append({"subject": "user", "predicate": "has_emotion", "object": emo})

    if not facts:
        reasons.append("no_inventory_mappable_user_fact")

    # Enforce predicate inventory.
    for f in facts:
        pred = f.get("predicate")
        if pred not in allowed_predicates:
            reasons.append(f"invalid_predicate_emitted:{pred}")

    return facts, roles, affect, reasons


def _validate_master_row(row: dict, allowed_predicates: set[str]) -> list[str]:
    errs: list[str] = []
    for field in REQUIRED_MASTER_FIELDS:
        if field not in row:
            errs.append(f"missing_field:{field}")
    facts = row.get("facts") or []
    if not isinstance(facts, list):
        errs.append("facts_not_list")
        return errs
    for f in facts:
        if not isinstance(f, dict):
            errs.append("fact_not_object")
            continue
        pred = f.get("predicate")
        if pred is None:
            continue
        if pred not in allowed_predicates:
            errs.append(f"invalid_predicate:{pred}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description="Gold-judge a slice of PersonaChat queue rows into master-schema JSONL.")
    ap.add_argument("--in", dest="input_path", required=True, help="Input queue JSONL")
    ap.add_argument("--start", type=int, required=True, help="1-based start row (inclusive)")
    ap.add_argument("--end", type=int, required=True, help="1-based end row (inclusive)")
    ap.add_argument("--out", required=True, help="Output master-schema JSONL")
    ap.add_argument("--report", required=True, help="Output report markdown")
    ap.add_argument(
        "--predicate-inventory",
        default="data/extractor_v7/predicate_inventory.json",
        help="Predicate inventory JSON (default: data/extractor_v7/predicate_inventory.json)",
    )
    args = ap.parse_args()

    in_path = Path(args.input_path)
    out_path = Path(args.out)
    report_path = Path(args.report)
    allowed = _load_allowed_predicates(Path(args.predicate_inventory))

    kept = 0
    rejected = 0
    reject_reasons: dict[str, int] = {}
    kept_by_pred: dict[str, int] = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fo:
        for rowno, obj in _iter_jsonl(in_path):
            if rowno < args.start or rowno > args.end:
                continue
            text_raw = obj.get("source_text_raw") or ""
            if not isinstance(text_raw, str) or not text_raw.strip():
                rejected += 1
                reject_reasons["empty_text"] = reject_reasons.get("empty_text", 0) + 1
                continue

            facts, roles, affect, reasons = _map_personachat_line_to_facts(text_raw, allowed)
            if not facts:
                rejected += 1
                for r in reasons:
                    reject_reasons[r] = reject_reasons.get(r, 0) + 1
                continue

            preds = {f.get("predicate") for f in facts if isinstance(f, dict)}
            preds = {p for p in preds if isinstance(p, str)}
            domain = _domain_for_predicates(preds)
            style_band = obj.get("style_band") if isinstance(obj.get("style_band"), str) else "direct"
            difficulty = "easy" if len(facts) <= 1 else "medium"

            master_id = f"gold_pc_w1_j3_{obj.get('candidate_id')}"
            row = {
                "id": master_id,
                "source_text": text_raw.strip(),
                "clean_text": _clean_english(text_raw),
                "facts": facts,
                "roles": roles,
                "affect": affect,
                "domain": domain,
                "style_band": style_band,
                "difficulty": difficulty,
                "source_type": obj.get("source_type"),
                "source_url": obj.get("source_url"),
                "source_domain": obj.get("source_domain"),
                "date_accessed": obj.get("date_accessed"),
                "license_or_usage_note": obj.get("license_or_usage_note"),
            }
            errs = _validate_master_row(row, allowed)
            if errs:
                rejected += 1
                for e in errs:
                    reject_reasons[e] = reject_reasons.get(e, 0) + 1
                continue

            kept += 1
            for p in preds:
                kept_by_pred[p] = kept_by_pred.get(p, 0) + 1
            fo.write(json.dumps(row, ensure_ascii=True) + "\n")

    total = kept + rejected
    with report_path.open("w", encoding="utf-8") as fr:
        fr.write("# Gold Judge Report (PersonaChat slice)\n\n")
        fr.write(f"- input: `{in_path}`\n")
        fr.write(f"- slice: rows {args.start} to {args.end}\n")
        fr.write(f"- output: `{out_path}`\n")
        fr.write(f"- total_seen: {total}\n")
        fr.write(f"- kept: {kept}\n")
        fr.write(f"- rejected: {rejected}\n\n")
        fr.write("## Kept By Predicate\n\n")
        for k in sorted(kept_by_pred.keys()):
            fr.write(f"- {k}: {kept_by_pred[k]}\n")
        fr.write("\n## Reject Reasons\n\n")
        for k in sorted(reject_reasons.keys()):
            fr.write(f"- {k}: {reject_reasons[k]}\n")

    print(f"kept {kept}")
    print(f"rejected {rejected}")
    print(f"wrote {out_path}")
    print(f"report {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

