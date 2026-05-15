#!/usr/bin/env python3
"""Verify reconstructed candidate facts against stored edge triples.

This script is intentionally stricter than retrieval overlap:

1. Build an implied fact from a query and a candidate answer.
2. Search the DB for that complete subject-predicate-object claim.
3. Reject candidates whose full implied fact is not present.
4. Remember rejected facts so a loop does not retry the same candidate.

Example:
    python scripts/verify_implied_fact.py ^
        --db "Memory Storage/locomo/conv0.db" ^
        --user-id 1 ^
        --query "Who researched adoption agencies?" ^
        --candidate Melanie ^
        --candidate Caroline
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional


LOCOMO_REFUSAL = "This information is not mentioned in the conversation."


@dataclass(frozen=True)
class ImpliedFact:
    subject: str
    predicate: str
    object: str
    statement: str


@dataclass
class CandidateDecision:
    candidate: str
    implied_fact: ImpliedFact
    verified: bool
    reason: str
    edge_id: Optional[int] = None


@dataclass
class VerificationResult:
    answer: str
    refusal: bool
    refusal_reason: str = ""
    verified_edge_id: Optional[int] = None
    implied_fact: Optional[ImpliedFact] = None
    rejected_memory: list[str] = field(default_factory=list)
    decisions: list[CandidateDecision] = field(default_factory=list)


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("_", " ").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _simple_verb_lemma(surface: str) -> str:
    verb = normalize_text(surface).split(" ")[0] if surface else ""
    try:
        from nltk.corpus import wordnet as wn

        lemma = wn.morphy(verb, wn.VERB)
        if lemma:
            return lemma
    except Exception:
        pass
    if verb.endswith("ied") and len(verb) > 4:
        return verb[:-3] + "y"
    if verb.endswith("ed") and len(verb) > 3:
        if verb.endswith("eed"):
            return verb[:-1]
        stem = verb[:-2]
        if stem.endswith("ch") or stem.endswith("sh") or stem.endswith("ss"):
            return stem
        if len(verb) > 4 and verb[-3] == "e":
            return verb[:-1]
        return stem.rstrip("e")
    if verb.endswith("es") and len(verb) > 3:
        return verb[:-2]
    if verb.endswith("s") and len(verb) > 3:
        return verb[:-1]
    return verb


def _object_text_for_token(token) -> str:
    for child in token.children:
        if child.dep_ in ("dobj", "obj", "attr", "oprd"):
            return " ".join(t.text for t in child.subtree).strip()
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return " ".join(t.text for t in grandchild.subtree).strip()
    return ""


def _extract_predicate_object_with_spacy(query: str) -> Optional[tuple[str, str, str]]:
    try:
        from app.engines.grammar_engine import _get_nlp

        doc = _get_nlp()(query)
    except Exception:
        return None

    candidates: list[tuple[int, str, str, str]] = []
    for token in doc:
        if token.pos_ != "VERB":
            continue
        lemma = token.lemma_.lower()
        if lemma in {"be", "do", "have"}:
            continue
        obj = _object_text_for_token(token)
        if obj:
            candidates.append((token.i, lemma, token.text, obj))

    if not candidates:
        return None

    _, lemma, surface, obj = candidates[-1]
    return lemma, surface, obj


def _extract_predicate_object_fallback(query: str) -> Optional[tuple[str, str, str]]:
    q = re.sub(r"[?!.]+$", "", query.strip())
    q = re.sub(r"\s+", " ", q)
    if not q:
        return None

    # If the query has multiple clauses, verify the final lexical claim.
    # Example: "who did the work and researched adoption agencies"
    segment = re.split(r"\band\b", q, flags=re.IGNORECASE)[-1].strip()
    segment = re.sub(
        r"^(who|what|which person|which people)\s+", "", segment,
        flags=re.IGNORECASE,
    )
    segment = re.sub(
        r"^(did|does|do|was|were|is|are|has|have|had)\s+", "", segment,
        flags=re.IGNORECASE,
    )
    words = segment.split()
    if len(words) < 2:
        return None

    surface = words[0]
    obj = " ".join(words[1:])
    return _simple_verb_lemma(surface), surface, obj


def extract_predicate_object(query: str) -> tuple[str, str, str]:
    extracted = _extract_predicate_object_with_spacy(query)
    if extracted is None:
        extracted = _extract_predicate_object_fallback(query)
    if extracted is None:
        raise ValueError(f"Could not derive predicate/object from query: {query!r}")
    return extracted


def build_implied_fact(
    query: str,
    candidate: str,
    *,
    predicate: Optional[str] = None,
    object_text: Optional[str] = None,
) -> ImpliedFact:
    if predicate and object_text:
        predicate_lemma = normalize_text(predicate).split(" ")[0]
        predicate_surface = predicate
        obj = object_text
    else:
        predicate_lemma, predicate_surface, obj = extract_predicate_object(query)
        if predicate:
            predicate_lemma = normalize_text(predicate).split(" ")[0]
            predicate_surface = predicate
        if object_text:
            obj = object_text

    statement = f"{candidate.strip()} {predicate_surface.strip()} {obj.strip()}".strip()
    return ImpliedFact(
        subject=candidate.strip(),
        predicate=predicate_lemma,
        object=obj.strip(),
        statement=statement,
    )


def rejection_key(fact: ImpliedFact) -> str:
    return "|".join(
        (
            normalize_text(fact.subject),
            normalize_text(fact.predicate),
            normalize_text(fact.object),
        )
    )


def load_rejection_memory(path: Optional[Path]) -> set[str]:
    if path is None or not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if isinstance(raw, list):
        return {str(item) for item in raw}
    if isinstance(raw, dict):
        return {str(item) for item in raw.get("rejected", [])}
    return set()


def save_rejection_memory(path: Optional[Path], rejected: Iterable[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"rejected": sorted(set(rejected))}, indent=2),
        encoding="utf-8",
    )


def _row_value(row: sqlite3.Row, column: str) -> str:
    try:
        return row[column] or ""
    except (IndexError, KeyError):
        return ""


def _predicate_matches(stored_predicate: str, wanted_predicate: str) -> bool:
    stored = normalize_text(stored_predicate)
    wanted = normalize_text(wanted_predicate)
    if not stored or not wanted:
        return False
    stored_head = stored.split(" ")[0]
    wanted_head = wanted.split(" ")[0]
    return stored_head == wanted_head


def _object_matches(row: sqlite3.Row, wanted_object: str) -> bool:
    wanted = normalize_text(wanted_object)
    if not wanted:
        return False
    for column in ("object", "object_full"):
        stored = normalize_text(_row_value(row, column))
        if stored and (stored == wanted or stored.endswith(f" {wanted}")):
            return True
    return False


def _split_compound_subject(subject: str) -> list[str]:
    """Split compound subjects into individual entities via spaCy conj dep.
    'Jon and Gina' → ['jon and gina', 'jon', 'gina']
    'Caroline' → ['caroline']
    Always includes the full compound as first candidate (exact match first).
    """
    subjects = [normalize_text(subject)]
    try:
        from app.engines.grammar_engine import _get_nlp
        doc = _get_nlp()(subject)
        for tok in doc:
            if tok.dep_ == "conj" and tok.pos_ in ("PROPN", "NOUN"):
                subjects.append(normalize_text(tok.text))
            if tok.dep_ in ("nsubj", "ROOT", "compound") and tok.pos_ in ("PROPN", "NOUN"):
                # The head entity (before "and")
                if tok.dep_ != "conj":
                    name = normalize_text(tok.text)
                    if name and name != subjects[0]:
                        subjects.append(name)
    except Exception:
        pass
    return subjects


def find_matching_edge(
    conn: sqlite3.Connection,
    user_id: int,
    fact: ImpliedFact,
) -> Optional[sqlite3.Row]:
    subject = normalize_text(fact.subject)
    if not subject:
        return None

    # Split compound subjects: "Jon and Gina" → try each individually
    subject_candidates = _split_compound_subject(fact.subject)

    for subj in subject_candidates:
        if not subj:
            continue

        rows = conn.execute(
            """SELECT id, subject, predicate, object, object_full, source_text,
                      pq_1, pq_2, pq_3, pq_4, vq_1, vq_2
               FROM edges
               WHERE user_id = ?
                 AND tombstoned_at IS NULL
                 AND COALESCE(is_current, 1) = 1
                 AND lower(COALESCE(subject, '')) = ?
               ORDER BY COALESCE(sequence_number, id) DESC, id DESC""",
            (user_id, subj),
        ).fetchall()

        for row in rows:
            if not _predicate_matches(_row_value(row, "predicate"), fact.predicate):
                continue
            if not _object_matches(row, fact.object):
                continue
            return row

    return None


def _set_verified_query(conn: sqlite3.Connection, edge_id: int, query: str) -> None:
    row = conn.execute(
        """SELECT id, subject, predicate, object, source_text,
                  pq_1, pq_2, pq_3, pq_4, vq_1, vq_2
           FROM edges
           WHERE id = ?""",
        (edge_id,),
    ).fetchone()
    if row is None:
        return

    existing = {
        normalize_text(_row_value(row, "vq_1")),
        normalize_text(_row_value(row, "vq_2")),
    }
    if normalize_text(query) in existing:
        return

    if not _row_value(row, "vq_1"):
        conn.execute("UPDATE edges SET vq_1 = ? WHERE id = ?", (query, edge_id))
    elif not _row_value(row, "vq_2"):
        conn.execute("UPDATE edges SET vq_2 = ? WHERE id = ?", (query, edge_id))
    else:
        return

    try:
        refreshed = conn.execute(
            """SELECT subject, predicate, object, source_text,
                      pq_1, pq_2, pq_3, pq_4, vq_1, vq_2
               FROM edges
               WHERE id = ?""",
            (edge_id,),
        ).fetchone()
        conn.execute(
            """INSERT INTO edges_fts(edges_fts, rowid, subject, predicate,
                                     object, source_text, pq_1, pq_2, pq_3,
                                     pq_4, vq_1, vq_2)
               VALUES('delete', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                edge_id,
                _row_value(row, "subject"),
                _row_value(row, "predicate").replace("_", " "),
                _row_value(row, "object"),
                _row_value(row, "source_text"),
                _row_value(row, "pq_1"),
                _row_value(row, "pq_2"),
                _row_value(row, "pq_3"),
                _row_value(row, "pq_4"),
                _row_value(row, "vq_1"),
                _row_value(row, "vq_2"),
            ),
        )
        conn.execute(
            """INSERT INTO edges_fts(rowid, subject, predicate, object,
                                     source_text, pq_1, pq_2, pq_3, pq_4,
                                     vq_1, vq_2)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                edge_id,
                _row_value(refreshed, "subject"),
                _row_value(refreshed, "predicate").replace("_", " "),
                _row_value(refreshed, "object"),
                _row_value(refreshed, "source_text"),
                _row_value(refreshed, "pq_1"),
                _row_value(refreshed, "pq_2"),
                _row_value(refreshed, "pq_3"),
                _row_value(refreshed, "pq_4"),
                _row_value(refreshed, "vq_1"),
                _row_value(refreshed, "vq_2"),
            ),
        )
    except sqlite3.Error:
        # Some test or old DBs do not have the 10-column FTS table. The edge
        # row is still the source of truth, so verification remains valid.
        pass


def verify_candidates(
    conn: sqlite3.Connection,
    user_id: int,
    query: str,
    candidates: Iterable[str],
    *,
    rejected: Optional[set[str]] = None,
    predicate: Optional[str] = None,
    object_text: Optional[str] = None,
    write_verified_query: bool = False,
) -> VerificationResult:
    rejected = rejected if rejected is not None else set()
    decisions: list[CandidateDecision] = []

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue

        fact = build_implied_fact(
            query,
            candidate,
            predicate=predicate,
            object_text=object_text,
        )
        key = rejection_key(fact)
        if key in rejected:
            decisions.append(
                CandidateDecision(candidate, fact, False, "previously_rejected")
            )
            continue

        row = find_matching_edge(conn, user_id, fact)
        if row is None:
            rejected.add(key)
            decisions.append(CandidateDecision(candidate, fact, False, "not_in_db"))
            continue

        edge_id = int(row["id"])
        if write_verified_query:
            _set_verified_query(conn, edge_id, query)
            conn.commit()

        decisions.append(CandidateDecision(candidate, fact, True, "verified", edge_id))
        return VerificationResult(
            answer=candidate,
            refusal=False,
            verified_edge_id=edge_id,
            implied_fact=fact,
            rejected_memory=sorted(rejected),
            decisions=decisions,
        )

    return VerificationResult(
        answer=LOCOMO_REFUSAL,
        refusal=True,
        refusal_reason="no_verified_candidate",
        rejected_memory=sorted(rejected),
        decisions=decisions,
    )


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _json_default(value):
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate answer to verify. Repeat for ranked candidates.",
    )
    parser.add_argument("--predicate", help="Optional predicate override.")
    parser.add_argument("--object", dest="object_text", help="Optional object override.")
    parser.add_argument("--rejections", type=Path, help="JSON rejection memory path.")
    parser.add_argument(
        "--write-verified-query",
        action="store_true",
        help="Write a passed query to vq_1/vq_2 on the verified edge.",
    )
    args = parser.parse_args(argv)

    if not args.candidate:
        parser.error("at least one --candidate is required")

    rejected = load_rejection_memory(args.rejections)
    with _connect(args.db) as conn:
        result = verify_candidates(
            conn,
            args.user_id,
            args.query,
            args.candidate,
            rejected=rejected,
            predicate=args.predicate,
            object_text=args.object_text,
            write_verified_query=args.write_verified_query,
        )
    save_rejection_memory(args.rejections, set(result.rejected_memory))

    print(json.dumps(result, default=_json_default, indent=2))
    return 2 if result.refusal else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
