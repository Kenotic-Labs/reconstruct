import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import spacy


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


@dataclass(frozen=True)
class SilverOptions:
    spacy_model: str
    max_rows: int | None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _denoise_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def _span_text(span) -> str:
    return _denoise_whitespace(span.text)


def _token_text(tok) -> str:
    return _denoise_whitespace(tok.text)


def _extract_svo_facts(doc) -> list[dict[str, Any]]:
    """
    Structural SVO-ish extraction:
    - pick verbs
    - pair nominal subjects with direct objects / pobj through prepositions
    This is intentionally simple and deterministic.
    """
    facts: list[dict[str, Any]] = []

    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue

        subjects = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass") and c.pos_ in ("NOUN", "PROPN", "PRON")]
        if not subjects:
            continue

        # Direct objects
        dobjs = [c for c in tok.children if c.dep_ in ("dobj", "obj") and c.pos_ in ("NOUN", "PROPN", "PRON")]

        # Prepositional objects: verb -> prep -> pobj
        prep_pobjs = []
        for prep in (c for c in tok.children if c.dep_ == "prep"):
            for pobj in (c for c in prep.children if c.dep_ == "pobj" and c.pos_ in ("NOUN", "PROPN", "PRON", "NUM")):
                prep_pobjs.append((prep, pobj))

        # Build facts. Predicate is verb lemma, optionally with prep lemma.
        pred_base = tok.lemma_.lower() if tok.lemma_ else tok.text.lower()

        for subj in subjects:
            subj_txt = _token_text(subj)

            for obj in dobjs:
                obj_txt = _span_text(obj.subtree)
                facts.append(
                    {
                        "subject": subj_txt,
                        "predicate": pred_base,
                        "object": obj_txt,
                        "is_historical": False,
                        "temporal": None,
                        "emotion": None,
                    }
                )

            for prep, pobj in prep_pobjs:
                pobj_txt = _span_text(pobj.subtree)
                pred = f"{pred_base}_{prep.lemma_.lower() if prep.lemma_ else prep.text.lower()}"
                facts.append(
                    {
                        "subject": subj_txt,
                        "predicate": pred,
                        "object": pobj_txt,
                        "is_historical": False,
                        "temporal": None,
                        "emotion": None,
                    }
                )

    return facts


def _extract_temporal_roles(doc) -> list[dict[str, Any]]:
    roles: list[dict[str, Any]] = []
    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            roles.append({"label": "TEMPORAL", "text": _span_text(ent)})
    return roles


def _extract_affect_facts(doc) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Minimal affect: keep explicit emotion-labeled entities if the model provides them.
    Many spaCy models don't emit emotion labels; this is additive when present.
    """
    facts: list[dict[str, Any]] = []
    affect: list[dict[str, Any]] = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC"):
            continue
        # No open-world affect inference here.
    return facts, affect


def _make_master_row(raw: dict[str, Any], doc, facts: list[dict[str, Any]], roles: list[dict[str, Any]]) -> dict[str, Any]:
    # Preserve provenance fields from raw candidate format.
    source_text = _denoise_whitespace(raw.get("source_text_raw", "") or "")
    candidate_id = raw.get("candidate_id") or ""

    row = {
        "id": f"silver_{candidate_id}",
        "source_text": source_text,
        "clean_text": source_text,
        "facts": facts,
        "roles": roles,
        "affect": [],
        "domain": raw.get("domain") or "mixed",
        "style_band": raw.get("style_band") or "casual",
        "difficulty": "silver",
        "source_type": raw.get("source_type") or "unknown",
        "source_url": raw.get("source_url"),
        "source_domain": raw.get("source_domain"),
        "date_accessed": raw.get("date_accessed"),
        "license_or_usage_note": raw.get("license_or_usage_note"),
    }

    for k in REQUIRED_MASTER_FIELDS:
        if k not in row:
            raise ValueError(f"missing required field {k}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Silver extraction via spaCy structural parsing (SVO + NER temporal).")
    ap.add_argument("--in", dest="inputs", action="append", required=True, help="Raw candidate JSONL (repeatable)")
    ap.add_argument("--out", required=True, help="Output master-schema JSONL")
    ap.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name (default: en_core_web_sm)")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional max output rows")
    args = ap.parse_args()

    opts = SilverOptions(spacy_model=args.spacy_model, max_rows=args.max_rows)

    nlp = spacy.load(opts.spacy_model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    with out.open("w", encoding="utf-8") as fo:
        for in_path in args.inputs:
            for raw in _iter_jsonl(Path(in_path)):
                if opts.max_rows is not None and wrote >= opts.max_rows:
                    break
                text = raw.get("source_text_raw")
                if not isinstance(text, str) or not text.strip():
                    continue
                doc = nlp(text)
                facts = _extract_svo_facts(doc)
                roles = _extract_temporal_roles(doc)
                row = _make_master_row(raw, doc, facts, roles)
                fo.write(json.dumps(row, ensure_ascii=True) + "\n")
                wrote += 1
            if opts.max_rows is not None and wrote >= opts.max_rows:
                break

    print(f"wrote_silver {out} {wrote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

