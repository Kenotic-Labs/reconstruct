"""
Verify Lab — standalone test harness for implied fact verification.

Tests whether build_implied_fact correctly handles ALL English question types,
not just "Who" questions.

The problem: current verifier always puts the candidate in the subject slot.
That only works for Who-questions. For What/Where/When/Yes-No, the candidate
fills a different grammatical role.

Usage:
    python verify_lab.py
"""

import spacy
from dataclasses import dataclass
from typing import Optional

nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# ORIGINAL — copy of build_implied_fact from verify_implied_fact.py
# Always puts candidate in subject slot.
# ---------------------------------------------------------------------------

def _simple_verb_lemma(surface: str) -> str:
    verb = surface.strip().lower().split(" ")[0] if surface else ""
    try:
        from nltk.corpus import wordnet as wn
        lemma = wn.morphy(verb, wn.VERB)
        if lemma:
            return lemma
    except Exception:
        pass
    if verb.endswith("ed") and len(verb) > 3:
        return verb[:-2].rstrip("e") or verb
    if verb.endswith("s") and len(verb) > 3:
        return verb[:-1]
    return verb


def _extract_pred_obj_spacy(query: str):
    doc = nlp(query)
    candidates = []
    for token in doc:
        if token.pos_ != "VERB":
            continue
        lemma = token.lemma_.lower()
        if lemma in {"be", "do", "have"}:
            continue
        obj = ""
        for child in token.children:
            if child.dep_ in ("dobj", "obj", "attr", "oprd"):
                obj = " ".join(t.text for t in child.subtree).strip()
                break
            if child.dep_ == "prep":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        obj = " ".join(t.text for t in gc.subtree).strip()
                        break
                if obj:
                    break
        if obj:
            candidates.append((token.i, lemma, token.text, obj))
    if not candidates:
        return None
    _, lemma, surface, obj = candidates[-1]
    return lemma, surface, obj


@dataclass(frozen=True)
class ImpliedFact:
    subject: str
    predicate: str
    object: str
    statement: str


def build_implied_fact_ORIGINAL(query: str, candidate: str) -> Optional[ImpliedFact]:
    extracted = _extract_pred_obj_spacy(query)
    if not extracted:
        return None
    pred_lemma, pred_surface, obj = extracted
    statement = f"{candidate} {pred_surface} {obj}"
    return ImpliedFact(
        subject=candidate,
        predicate=pred_lemma,
        object=obj,
        statement=statement,
    )


# ---------------------------------------------------------------------------
# NEW — WH-type-aware implied fact builder
# ---------------------------------------------------------------------------

def _classify_wh_type(doc):
    """Classify question type from spaCy parse.
    Returns: (wh_type, wh_span_indices)
    wh_span_indices = set of token indices that belong to the WH-phrase.
    """
    first_real = None
    for tok in doc:
        if tok.pos_ not in ("PUNCT", "INTJ", "SPACE"):
            first_real = tok
            break
    if not first_real:
        return "unknown", set()

    text = first_real.text.lower()
    # WH-phrase span: the WH-word + any nouns it modifies
    # "What book" = 2 tokens, "How many dogs" = 3 tokens, "Whose car" = 2 tokens
    wh_indices = {first_real.i}

    if text == "who":
        return "who", wh_indices
    if text in ("what", "which", "whose"):
        # Check if next token is a noun (part of WH-phrase): "What book", "Which novel"
        # Collect the full subtree of the WH-word if it's a determiner/modifier
        if first_real.dep_ in ("det", "poss", "amod", "attr", "nsubj", "dobj", "advmod"):
            # WH-word modifies a noun — collect head + subtree
            head = first_real.head
            if head.pos_ in ("NOUN", "PROPN"):
                wh_indices.update(t.i for t in head.subtree)
        # Also check children of WH-word (if WH is the head)
        for child in first_real.children:
            if child.dep_ in ("amod", "compound", "det", "case"):
                wh_indices.add(child.i)
        if text == "what":
            return "what", wh_indices
        if text == "which":
            return "which", wh_indices
        if text == "whose":
            return "whose", wh_indices
    if text == "where":
        return "where", wh_indices
    if text == "when":
        return "when", wh_indices
    if text == "how":
        next_tok = doc[first_real.i + 1] if first_real.i + 1 < len(doc) else None
        if next_tok and next_tok.text.lower() in ("many", "much"):
            wh_indices.add(next_tok.i)
            # "How many dogs" — the noun after "how many" is part of the WH-phrase
            if next_tok.i + 1 < len(doc) and doc[next_tok.i + 1].pos_ in ("NOUN", "PROPN"):
                wh_indices.add(doc[next_tok.i + 1].i)
            return "how_many", wh_indices
        return "how", wh_indices

    # Yes/No: starts with aux/modal
    if first_real.pos_ in ("AUX", "VERB") and first_real.dep_ in ("ROOT", "aux", "auxpass"):
        return "yes_no", set()
    if text in ("is", "are", "was", "were", "do", "does", "did",
                "has", "have", "had", "can", "could", "will", "would",
                "should", "shall", "may", "might"):
        return "yes_no", set()

    return "unknown", set()


def _is_verb_form(word: str) -> Optional[str]:
    """WordNet check: is this word a verb form? Returns verb lemma or None."""
    from nltk.corpus import wordnet as wn
    return wn.morphy(word.lower(), wn.VERB)


def _recover_misparsed_compound(doc, wh_indices):
    """Fix spaCy's mislabeling of subject+verb as compound noun.

    spaCy parses "What book is Jon reading?" as:
        Jon = compound, reading = NOUN (attr)
    But WordNet knows "reading" is a verb form (morphy → "read"),
    and structurally: the token has an aux ("is") or is ROOT,
    meaning it's functioning as a verb.

    Two signals combined:
        1. WordNet says the token is a verb form
        2. Structure says it's acting as a verb (has aux, or is ROOT/attr with aux sibling)

    Returns (subject_tok, verb_tok, verb_lemma) or (None, None, None).
    """
    for tok in doc:
        if tok.i in wh_indices:
            continue
        if tok.pos_ != "NOUN":
            continue
        verb_lemma = _is_verb_form(tok.text)
        if not verb_lemma:
            continue
        # Structural check: is this "noun" acting as a verb?
        # - has an aux child, OR
        # - is attr of a copula (be) — "is Jon reading" → reading is attr of "is"
        # - has a sibling aux on the same head
        acts_as_verb = any(ch.dep_ == "aux" for ch in tok.children)
        if not acts_as_verb and tok.dep_ in ("attr", "ROOT", "nsubj"):
            # Head is a copula — attr of "be" with a gerund = progressive aspect
            if tok.head.lemma_ == "be" and tok.head.pos_ == "AUX":
                acts_as_verb = True
            # Or a sibling aux exists
            if not acts_as_verb:
                for sibling in tok.head.children:
                    if sibling.dep_ in ("aux", "auxpass") and sibling.pos_ == "AUX":
                        acts_as_verb = True
                        break
        if not acts_as_verb:
            continue
        # Find the compound child — that's the misparsed subject.
        # Disambiguation: the child is the subject if it has
        # MORE noun synsets than verb synsets in WordNet,
        # OR if spaCy tagged it as PROPN (proper noun = name).
        from nltk.corpus import wordnet as wn
        for child in tok.children:
            if child.dep_ != "compound":
                continue
            if child.pos_ == "PROPN":
                return child, tok, verb_lemma
            if child.pos_ == "NOUN":
                v_count = len(wn.synsets(child.text.lower(), pos=wn.VERB))
                n_count = len(wn.synsets(child.text.lower(), pos=wn.NOUN))
                if n_count >= v_count:
                    return child, tok, verb_lemma
    return None, None, None


def _wh_head_noun(doc, wh_indices) -> str:
    """Extract the content noun from the WH-phrase.
    'Whose car' → 'car'. 'How many dogs' → 'dogs'. 'What book' → 'book'.
    The WH-word is the question marker. The noun is content — keep it.
    """
    from nltk.corpus import wordnet as wn
    for i in sorted(wh_indices):
        tok = doc[i]
        if tok.pos_ in ("NOUN", "PROPN"):
            # Confirm it's a real noun, not a verb form
            if not wn.morphy(tok.text.lower(), wn.VERB) or wn.morphy(tok.text.lower(), wn.NOUN):
                return tok.text
    return ""


def _wh_is_subject(doc, wh_indices) -> bool:
    """Check if the WH-phrase IS the only subject of the question.
    'Which team won?' — 'team' is nsubj, no other subject → True.
    'Which novel is Sophia writing?' — 'novel' is nsubj but Sophia is
      the real subject (recovered from compound) → False.
    """
    wh_has_nsubj = any(doc[i].dep_ in ("nsubj", "nsubjpass") for i in wh_indices)
    if not wh_has_nsubj:
        return False
    # Check if there's ANOTHER subject or a recoverable subject outside WH
    for tok in doc:
        if tok.i in wh_indices:
            continue
        if tok.dep_ in ("nsubj", "nsubjpass"):
            return False  # Real subject exists elsewhere
        # Check for misparsed compound that would be recovered as subject
        if tok.dep_ == "compound" and tok.pos_ in ("PROPN", "NOUN"):
            head_is_verb = _is_verb_form(tok.head.text) if tok.head.pos_ == "NOUN" else False
            if head_is_verb:
                return False  # Recoverable subject exists
    return True


def _find_subject(doc, wh_indices) -> str:
    """Find the subject of the question, skipping WH-phrase tokens.
    Falls back to WordNet compound recovery if spaCy mislabeled."""
    # First: look for a real subject outside the WH-phrase
    for tok in doc:
        if tok.i in wh_indices:
            continue
        if tok.dep_ in ("nsubj", "nsubjpass"):
            # Collect indices to skip: WH-phrase + temporal/adverbial subtrees
            skip = set(wh_indices)
            for child in tok.children:
                if child.dep_ in ("npadvmod", "advmod", "advcl"):
                    skip.update(t.i for t in child.subtree)
            subtree_tokens = [t for t in tok.subtree if t.i not in skip]
            if subtree_tokens:
                return " ".join(t.text for t in subtree_tokens).strip()
    # Fallback: recover misparsed compound (Jon reading → Jon is subject)
    subj_tok, _, _ = _recover_misparsed_compound(doc, wh_indices)
    if subj_tok:
        return subj_tok.text
    return ""


def _find_verb(doc, wh_indices):
    """Find the main content verb, skipping WH-phrase.
    Uses WordNet to override spaCy when it mislabels a gerund as NOUN."""
    for tok in doc:
        if tok.i in wh_indices:
            continue
        if tok.pos_ == "VERB" and tok.lemma_.lower() not in ("be", "do", "have"):
            return tok
    # Fallback: WordNet recovery — is any "NOUN" actually a verb form?
    _, verb_tok, _ = _recover_misparsed_compound(doc, wh_indices)
    if verb_tok:
        return verb_tok
    # Last fallback: ROOT
    for tok in doc:
        if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX"):
            return tok
    return None


def _find_object(verb_tok, wh_indices, doc) -> str:
    """Find the object/complement of a verb.
    Excludes WH-question words but keeps content nouns from WH-phrase."""
    if not verb_tok:
        return ""
    for child in verb_tok.children:
        if child.i in wh_indices:
            continue
        if child.dep_ in ("dobj", "obj", "attr", "oprd", "acomp"):
            toks = [t for t in child.subtree if t.i not in wh_indices]
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
    # Nothing found outside WH-phrase — return the head noun FROM the WH-phrase
    return _wh_head_noun(doc, wh_indices)


def _get_verb_surface(verb_tok, doc) -> str:
    """Get the right surface form of the verb for the statement.

    Uses spaCy morphology to recover tense from the auxiliary.
    For misparsed gerunds (spaCy says NOUN, WordNet says verb),
    recover the lemma from WordNet.
    """
    if not verb_tok:
        return ""

    # If spaCy tagged this as NOUN but WordNet says it's a verb (misparsed gerund),
    # use the gerund form directly — it carries progressive aspect
    if verb_tok.pos_ == "NOUN":
        verb_lemma = _is_verb_form(verb_tok.text)
        if verb_lemma:
            # Check for aux in the doc to determine tense
            for tok in doc:
                if tok.dep_ == "aux" and tok.head.i == verb_tok.i:
                    if tok.tag_ == "VBD":  # "was reading"
                        return f"was {verb_tok.text}"
                    return f"is {verb_tok.text}"
            # Find aux anywhere (misparsed tree might not have direct link)
            for tok in doc:
                if tok.lemma_ == "be" and tok.pos_ == "AUX":
                    return f"{tok.text} {verb_tok.text}"
            return verb_tok.text

    # Check if there's do-support
    for child in verb_tok.children:
        if child.dep_ == "aux" and child.lemma_ == "do":
            if child.tag_ == "VBD":
                # did + base → past: use lemma + "ed" (approximate)
                lemma = verb_tok.lemma_
                if lemma.endswith("e"):
                    return lemma + "d"
                return lemma + "ed"
            if child.tag_ == "VBZ":
                return verb_tok.lemma_ + "s"
    # No do-support — verb already carries tense
    return verb_tok.text


def build_implied_fact_NEW(query: str, candidate: str) -> Optional[ImpliedFact]:
    """WH-type-aware implied fact builder.

    Detects what role the candidate fills based on the question type:
      Who  → candidate is subject:   "{candidate} {verb} {object}"
      What → candidate is object:    "{subject} {verb} {candidate}"
      Where → candidate is location: "{subject} {verb} in {candidate}"
      When → candidate is time:      "{subject} {verb} {candidate}"
      Yes/No → candidate confirms:   fact is already in the query
      How many → candidate is count: "{subject} {verb} {candidate} {object}"
      Which → candidate is selector: "{subject} {verb} {candidate}"
      Whose → candidate is possessor: "{candidate}'s {object}"
    """
    doc = nlp(query)
    wh_type, wh_indices = _classify_wh_type(doc)
    verb_tok = _find_verb(doc, wh_indices)
    # For misparsed gerunds, WordNet gives us the real lemma
    if verb_tok and verb_tok.pos_ == "NOUN":
        verb_lemma = _is_verb_form(verb_tok.text) or verb_tok.text
    else:
        verb_lemma = verb_tok.lemma_ if verb_tok else ""
    verb_surface = _get_verb_surface(verb_tok, doc)
    subject = _find_subject(doc, wh_indices)
    obj = _find_object(verb_tok, wh_indices, doc) if verb_tok else _wh_head_noun(doc, wh_indices)

    # Subject-WH detection: if the WH-phrase IS the subject,
    # the candidate fills the subject slot regardless of WH-type.
    # "Which team won?" — candidate is the team (subject).
    # "What color is Maria's dress?" — WH is attr, not nsubj — stays as "what".
    if wh_type in ("what", "which") and _wh_is_subject(doc, wh_indices):
        wh_type = "who"  # redirect to subject path

    if wh_type == "who":
        # Candidate IS the subject
        statement = f"{candidate} {verb_surface} {obj}".strip()
        return ImpliedFact(
            subject=candidate,
            predicate=verb_lemma,
            object=obj,
            statement=statement,
        )

    elif wh_type == "what":
        # Candidate IS the object
        statement = f"{subject} {verb_surface} {candidate}".strip()
        return ImpliedFact(
            subject=subject,
            predicate=verb_lemma,
            object=candidate,
            statement=statement,
        )

    elif wh_type == "where":
        # Candidate IS the location
        statement = f"{subject} {verb_surface} in {candidate}".strip()
        return ImpliedFact(
            subject=subject,
            predicate=verb_lemma,
            object=candidate,
            statement=statement,
        )

    elif wh_type == "when":
        # Candidate IS the time
        statement = f"{subject} {verb_surface} {candidate}".strip()
        return ImpliedFact(
            subject=subject,
            predicate=verb_lemma,
            object=candidate,
            statement=statement,
        )

    elif wh_type == "yes_no":
        # Candidate is "yes"/"no" — the fact is already in the query
        statement = f"{subject} {verb_surface} {obj}".strip()
        return ImpliedFact(
            subject=subject,
            predicate=verb_lemma,
            object=obj,
            statement=statement,
        )

    elif wh_type == "how_many":
        # Candidate is a count. The WH head noun is the unit (dogs, dollars, books).
        unit = _wh_head_noun(doc, wh_indices)
        full_obj = f"{candidate} {unit}".strip() if unit else candidate
        statement = f"{subject} {verb_surface} {full_obj}".strip()
        return ImpliedFact(
            subject=subject,
            predicate=verb_lemma,
            object=full_obj,
            statement=statement,
        )

    elif wh_type == "which":
        # Candidate selects — treat like What
        statement = f"{subject} {verb_surface} {candidate}".strip()
        return ImpliedFact(
            subject=subject,
            predicate=verb_lemma,
            object=candidate,
            statement=statement,
        )

    elif wh_type == "whose":
        # Candidate is the possessor — obj already has "whose" stripped
        statement = f"{candidate}'s {obj}".strip()
        return ImpliedFact(
            subject=candidate,
            predicate="own",
            object=obj,
            statement=statement,
        )

    else:
        return build_implied_fact_ORIGINAL(query, candidate)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(query, candidate):
    doc = nlp(query)
    wh_type, wh_indices = _classify_wh_type(doc)

    orig = build_implied_fact_ORIGINAL(query, candidate)
    new = build_implied_fact_NEW(query, candidate)

    print(f"\n{'='*70}")
    print(f"QUERY:     \"{query}\"")
    print(f"CANDIDATE: \"{candidate}\"")
    print(f"WH-TYPE:   {wh_type}")

    if orig:
        print(f"\nORIGINAL:  subject={orig.subject}, pred={orig.predicate}, obj={orig.object}")
        print(f"           statement=\"{orig.statement}\"")
    else:
        print(f"\nORIGINAL:  (failed to extract)")

    if new:
        print(f"\nNEW:       subject={new.subject}, pred={new.predicate}, obj={new.object}")
        print(f"           statement=\"{new.statement}\"")
    else:
        print(f"\nNEW:       (failed to extract)")


if __name__ == "__main__":
    print("VERIFY LAB — interactive mode")
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
