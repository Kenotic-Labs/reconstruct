"""
Reverse PQ — reverse-engineered verification from PQ generator logic.

The PQ generator turns statements into questions by:
  1. Identifying a target constituent (subject, object, location, time, modifier)
  2. Replacing it with a WH-word
  3. Applying subject-aux inversion

This module reverses that process:
  1. Identify the WH-word → know which slot was removed
  2. Find the remaining constituents (subject, verb, object)
  3. Place the candidate answer back into the removed slot
  4. Reconstruct the original statement

PQ forward:  "Jon loves contemporary dance" → "What does Jon love?"
PQ reverse:  "What does Jon love?" + "contemporary dance" → "Jon loves contemporary dance"

The WH-word IS the slot label. The dep tags around it confirm the structure.
No hardcoding — same structural rules as the PQ generator, reversed.

Usage:
    python reverse_pq.py
"""

import spacy
from dataclasses import dataclass
from typing import Optional
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")


@dataclass(frozen=True)
class ReversedFact:
    subject: str
    predicate: str
    object: str
    statement: str
    slot: str          # which slot the candidate filled
    confidence: str    # "structural" or "fallback"


# ---------------------------------------------------------------------------
# WH-slot detection — mirrors PQ generator's WH-word selection in reverse
# ---------------------------------------------------------------------------

# PQ generator forward mapping (from pq_lab.py):
#   dobj with DATE/TIME NER      → "When"
#   pobj with locative prep      → "Where"
#   pobj with "for" + NUM        → "How long"
#   acomp ADJ                    → "How"
#   agent                        → "By whom"
#   PERSON NER                   → "Who"
#   dobj (default)               → "What"
#   Modifier dep tags:
#     poss                       → "Whose"
#     nummod                     → "How many"
#     amod (ordinal via WordNet) → "Which"
#     amod (quality)             → "What [noun]"
#     compound                   → "What kind of"
#
# Reverse: WH-word → slot that was removed

_WH_TO_SLOT = {
    "who":      "subject",
    "what":     "object",
    "where":    "location",
    "when":     "time",
    "how":      "manner",
    "whose":    "possessor",
    "which":    "selector",
}


def _detect_wh_and_slot(doc):
    """Detect WH-word and what slot it fills.

    Returns (slot, wh_indices, wh_head_noun).
    wh_head_noun is the content noun in the WH-phrase: "What book" → "book".
    """
    first = None
    for tok in doc:
        if tok.pos_ not in ("PUNCT", "INTJ", "SPACE"):
            first = tok
            break
    if not first:
        return "unknown", set(), ""

    text = first.text.lower()
    wh_indices = {first.i}

    # "How many/much" — special compound WH
    if text == "how":
        nxt = doc[first.i + 1] if first.i + 1 < len(doc) else None
        if nxt and nxt.text.lower() in ("many", "much"):
            wh_indices.add(nxt.i)
            # The noun after "how many" is the unit — part of WH-phrase
            if nxt.i + 1 < len(doc) and doc[nxt.i + 1].pos_ in ("NOUN", "PROPN"):
                unit_tok = doc[nxt.i + 1]
                wh_indices.add(unit_tok.i)
                return "count", wh_indices, unit_tok.text
            return "count", wh_indices, ""
        return "manner", wh_indices, ""

    # WH + noun phrase: "What book", "Which team", "Whose car"
    head_noun = ""
    if text in ("what", "which", "whose"):
        # Check if WH-word modifies a noun (det/poss of a head noun)
        if first.dep_ in ("det", "poss", "amod", "attr", "advmod"):
            head = first.head
            if head.pos_ in ("NOUN", "PROPN"):
                wh_indices.update(t.i for t in head.subtree
                                  if t.dep_ in ("det", "poss", "case", "amod", "compound")
                                  and t.i != head.i)
                wh_indices.add(head.i)
                head_noun = head.text
        # Also check children of WH-word
        for child in first.children:
            if child.dep_ in ("amod", "compound", "det", "case"):
                wh_indices.add(child.i)

    # Map WH-word to slot
    slot = _WH_TO_SLOT.get(text, "unknown")

    # "What [noun]" → modifier slot if head_noun exists
    # PQ generator: amod → "What [noun]", so reverse: "What [noun]" → object with type hint
    # "Whose [noun]" → possessor slot, head_noun is the possessed thing

    # Subject-WH check: if the WH-phrase head is nsubj, candidate IS the subject
    # Mirrors PQ generator's Rule 3 (subject WH)
    for i in wh_indices:
        if doc[i].dep_ in ("nsubj", "nsubjpass"):
            # But only if there's no other subject outside WH
            other_subj = False
            for tok in doc:
                if tok.i in wh_indices:
                    continue
                if tok.dep_ in ("nsubj", "nsubjpass"):
                    other_subj = True
                    break
                # Check for misparsed compound (WordNet gerund recovery)
                if tok.dep_ == "compound" and tok.head.pos_ == "NOUN":
                    vl = wn.morphy(tok.head.text.lower(), wn.VERB)
                    if vl and (tok.pos_ == "PROPN" or
                               len(wn.synsets(tok.text.lower(), wn.NOUN)) >=
                               len(wn.synsets(tok.text.lower(), wn.VERB))):
                        other_subj = True
                        break
            if not other_subj:
                slot = "subject"
            break

    # Yes/No: starts with aux/modal, no WH-word
    if text in ("is", "are", "was", "were", "do", "does", "did",
                "has", "have", "had", "can", "could", "will", "would",
                "should", "shall", "may", "might"):
        return "confirmation", set(), ""

    return slot, wh_indices, head_noun


# ---------------------------------------------------------------------------
# Constituent extraction — mirrors PQ generator's decomposition in reverse
# ---------------------------------------------------------------------------

def _find_subject(doc, wh_indices) -> str:
    """Find subject outside WH-phrase. Handles misparsed compounds via WordNet."""
    # Direct nsubj
    for tok in doc:
        if tok.i in wh_indices:
            continue
        if tok.dep_ in ("nsubj", "nsubjpass"):
            skip = set(wh_indices)
            for child in tok.children:
                if child.dep_ in ("npadvmod", "advmod", "advcl"):
                    skip.update(t.i for t in child.subtree)
            toks = [t for t in tok.subtree if t.i not in skip]
            if toks:
                return " ".join(t.text for t in toks).strip()

    # WordNet gerund recovery: PROPN/NOUN compound of misparsed verb-as-NOUN
    for tok in doc:
        if tok.i in wh_indices or tok.pos_ != "NOUN":
            continue
        vl = wn.morphy(tok.text.lower(), wn.VERB)
        if not vl:
            continue
        # Structural: acts as verb (attr of copula, has aux, or ROOT)
        acts = any(ch.dep_ == "aux" for ch in tok.children)
        if not acts and tok.dep_ in ("attr", "ROOT", "nsubj"):
            if tok.head.lemma_ == "be" and tok.head.pos_ == "AUX":
                acts = True
            if not acts:
                for sib in tok.head.children:
                    if sib.dep_ in ("aux", "auxpass") and sib.pos_ == "AUX":
                        acts = True
                        break
        if not acts:
            continue
        for child in tok.children:
            if child.dep_ == "compound" and child.i not in wh_indices:
                if child.pos_ == "PROPN":
                    return child.text
                if child.pos_ == "NOUN":
                    nc = len(wn.synsets(child.text.lower(), wn.NOUN))
                    vc = len(wn.synsets(child.text.lower(), wn.VERB))
                    if nc >= vc:
                        return child.text
    return ""


def _find_verb(doc, wh_indices):
    """Find main content verb. WordNet overrides spaCy gerund mislabeling."""
    # Real VERB token
    for tok in doc:
        if tok.i in wh_indices:
            continue
        if tok.pos_ == "VERB" and tok.lemma_.lower() not in ("be", "do", "have"):
            return tok

    # WordNet recovery: NOUN that's a verb form + has compound child
    for tok in doc:
        if tok.i in wh_indices or tok.pos_ != "NOUN":
            continue
        vl = wn.morphy(tok.text.lower(), wn.VERB)
        if vl:
            acts = any(ch.dep_ == "aux" for ch in tok.children)
            if not acts and tok.head.lemma_ == "be" and tok.head.pos_ == "AUX":
                acts = True
            if acts:
                return tok

    # Copula: "be" as ROOT is a real verb for "What is X?" questions
    for tok in doc:
        if tok.dep_ == "ROOT" and tok.lemma_ == "be":
            return tok

    # Last fallback: ROOT
    for tok in doc:
        if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX"):
            return tok
    return None


def _find_object(verb_tok, wh_indices, doc) -> str:
    """Find object/complement outside WH-phrase."""
    if not verb_tok:
        return ""
    for child in verb_tok.children:
        if child.i in wh_indices:
            continue
        if child.dep_ in ("dobj", "obj", "attr", "oprd", "acomp"):
            toks = [t for t in child.subtree if t.i not in wh_indices]
            if toks:
                return " ".join(t.text for t in toks).strip()
    for child in verb_tok.children:
        if child.i in wh_indices:
            continue
        if child.dep_ == "prep":
            for gc in child.children:
                if gc.dep_ == "pobj" and gc.i not in wh_indices:
                    prep = child.text
                    pobj_toks = [t for t in gc.subtree if t.i not in wh_indices]
                    pobj = " ".join(t.text for t in pobj_toks).strip()
                    return f"{prep} {pobj}"
    return ""


def _get_verb_lemma(verb_tok):
    """Get verb lemma. For misparsed gerunds, use WordNet."""
    if not verb_tok:
        return ""
    if verb_tok.pos_ == "NOUN":
        vl = wn.morphy(verb_tok.text.lower(), wn.VERB)
        if vl:
            return vl
    return verb_tok.lemma_


def _get_verb_surface(verb_tok, doc) -> str:
    """Recover tensed verb surface form from question structure."""
    if not verb_tok:
        return ""

    # Misparsed gerund: "is reading" → reconstruct progressive
    if verb_tok.pos_ == "NOUN" and wn.morphy(verb_tok.text.lower(), wn.VERB):
        for tok in doc:
            if tok.lemma_ == "be" and tok.pos_ == "AUX":
                return f"{tok.text} {verb_tok.text}"
        return verb_tok.text

    # Do-support: "did love" → "loved", "does love" → "loves"
    for child in verb_tok.children:
        if child.dep_ == "aux" and child.lemma_ == "do":
            lemma = verb_tok.lemma_
            if child.tag_ == "VBD":
                return lemma + "d" if lemma.endswith("e") else lemma + "ed"
            if child.tag_ == "VBZ":
                return lemma + "s"

    # No do-support — verb already carries tense
    return verb_tok.text


# ---------------------------------------------------------------------------
# Reverse PQ: question + candidate → implied fact
# ---------------------------------------------------------------------------

def reverse_pq(query: str, candidate: str) -> Optional[ReversedFact]:
    """Reverse the PQ generator's transformation.

    Takes a question and a candidate answer. Detects which slot the
    PQ generator removed (via WH-word), places the candidate back
    into that slot, and reconstructs the implied statement.

    Same structural rules as the PQ generator, reversed direction.
    """
    doc = nlp(query)
    slot, wh_indices, wh_head_noun = _detect_wh_and_slot(doc)
    verb_tok = _find_verb(doc, wh_indices)
    verb_lemma = _get_verb_lemma(verb_tok)
    verb_surface = _get_verb_surface(verb_tok, doc)
    subject = _find_subject(doc, wh_indices)
    obj = _find_object(verb_tok, wh_indices, doc)

    if slot == "subject":
        # PQ Rule 3 reversed: candidate was the subject
        # "Who loves contemporary dance?" + "Jon" → "Jon loves contemporary dance"
        statement = f"{candidate} {verb_surface} {obj}".strip()
        return ReversedFact(
            subject=candidate, predicate=verb_lemma, object=obj,
            statement=statement, slot="subject", confidence="structural",
        )

    elif slot == "object":
        # PQ Rule 1/4 reversed: candidate was the object
        # "What does Jon love?" + "contemporary dance" → "Jon loves contemporary dance"
        # If wh_head_noun exists ("What book"), candidate is the modifier
        if wh_head_noun:
            full_obj = f"{candidate} {wh_head_noun}"
        else:
            full_obj = candidate
        statement = f"{subject} {verb_surface} {full_obj}".strip()
        return ReversedFact(
            subject=subject, predicate=verb_lemma, object=full_obj,
            statement=statement, slot="object", confidence="structural",
        )

    elif slot == "location":
        # PQ generator only removed the location — S, V, O are still in the query.
        # "Where did Jon host a dance competition?" → obj = "a dance competition"
        # The candidate (location) was what was removed — not part of the S-P-O fact.
        # For verification: check S+P+O from query. Candidate is metadata.
        real_obj = obj if obj else candidate
        statement = f"{subject} {verb_surface} {real_obj}".strip()
        return ReversedFact(
            subject=subject, predicate=verb_lemma, object=real_obj,
            statement=statement, slot="location", confidence="structural",
        )

    elif slot == "time":
        # PQ generator only removed the time — S, V, O are still in the query.
        # "When did Jon host a dance competition?" → obj = "a dance competition"
        # The candidate (date) was what was removed — not part of the S-P-O fact.
        # For verification: check S+P+O from query. Candidate is metadata.
        real_obj = obj if obj else candidate
        statement = f"{subject} {verb_surface} {real_obj}".strip()
        return ReversedFact(
            subject=subject, predicate=verb_lemma, object=real_obj,
            statement=statement, slot="time", confidence="structural",
        )

    elif slot == "count":
        # PQ Rule 4 reversed: nummod → "How many X"
        # "How many dogs does Karen have?" + "three" → "Karen has three dogs"
        full_obj = f"{candidate} {wh_head_noun}".strip() if wh_head_noun else candidate
        statement = f"{subject} {verb_surface} {full_obj}".strip()
        return ReversedFact(
            subject=subject, predicate=verb_lemma, object=full_obj,
            statement=statement, slot="count", confidence="structural",
        )

    elif slot == "possessor":
        # PQ Rule 4 reversed: poss → "Whose X"
        # "Whose car did Jon drive?" + "Sarah" → "Sarah's car"
        possessed = wh_head_noun or obj
        statement = f"{candidate}'s {possessed}".strip()
        return ReversedFact(
            subject=candidate, predicate="own", object=possessed,
            statement=statement, slot="possessor", confidence="structural",
        )

    elif slot == "selector":
        # PQ Rule 4 reversed: ordinal amod → "Which X"
        # "Which novel is Sophia writing?" + "her first" → "Sophia is writing her first"
        statement = f"{subject} {verb_surface} {candidate}".strip()
        return ReversedFact(
            subject=subject, predicate=verb_lemma, object=candidate,
            statement=statement, slot="selector", confidence="structural",
        )

    elif slot == "manner":
        # "How" questions — candidate is manner/degree
        statement = f"{subject} {verb_surface} {candidate}".strip()
        return ReversedFact(
            subject=subject, predicate=verb_lemma, object=candidate,
            statement=statement, slot="manner", confidence="structural",
        )

    elif slot == "confirmation":
        # Yes/No reversed: the fact is already in the question
        # "Does Mike work at Google?" + "yes" → "Mike works at Google"
        statement = f"{subject} {verb_surface} {obj}".strip()
        return ReversedFact(
            subject=subject, predicate=verb_lemma, object=obj,
            statement=statement, slot="confirmation", confidence="structural",
        )

    else:
        # Can't determine slot — fallback: candidate is object
        if subject and verb_surface:
            statement = f"{subject} {verb_surface} {candidate}".strip()
            return ReversedFact(
                subject=subject, predicate=verb_lemma, object=candidate,
                statement=statement, slot="unknown", confidence="fallback",
            )
        return None


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(query, candidate):
    doc = nlp(query)
    slot, wh_indices, wh_head_noun = _detect_wh_and_slot(doc)
    result = reverse_pq(query, candidate)

    print(f"\n{'='*70}")
    print(f"QUERY:     \"{query}\"")
    print(f"CANDIDATE: \"{candidate}\"")
    print(f"SLOT:      {slot}  (wh_head_noun=\"{wh_head_noun}\")")

    if result:
        print(f"REVERSED:  subject={result.subject}, pred={result.predicate}, obj={result.object}")
        print(f"           statement=\"{result.statement}\"")
        print(f"           confidence={result.confidence}")
    else:
        print(f"REVERSED:  (failed)")


if __name__ == "__main__":
    print("REVERSE PQ — interactive mode")
    print("Format: query | candidate")
    print("Type 'q' to quit.\n")

    while True:
        line = input(">>> ").strip()
        if not line or line.lower() == "q":
            break
        if "|" not in line:
            print("  Format: query | candidate")
            continue
        query, candidate = line.split("|", 1)
        run_test(query.strip(), candidate.strip())
