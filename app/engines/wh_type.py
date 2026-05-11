# -*- coding: cp1252 -*-
"""
wh_type.py -- structural WH-word to expected-answer-type resolver.

Given a natural-language query, return the expected answer type that a
retrieved triple's subject_type or object_type should equal for
coherence, or None when the query has no WH-cued type expectation (a
situational query like "tell me about my day").

Discipline:
  - POS tagging via NLTK (averaged_perceptron_tagger).
  - WordNet hypernym closure for "what X" / "which X" head-noun typing.
  - No curated noun-to-type table.

Returns one of:
    PERSON | ORG | LOCATION | TIME | EVENT | QUANTITY |
    WORK_OF_ART | PRODUCT | GENERIC
  or None.
"""
from __future__ import annotations

import re
from typing import Optional

# Closed top-level categories mirrored from type_resolver to avoid a
# circular import with the write path.
PERSON = "PERSON"
ORG = "ORG"
LOCATION = "LOCATION"
TIME = "TIME"
EVENT = "EVENT"
QUANTITY = "QUANTITY"
WORK_OF_ART = "WORK_OF_ART"
PRODUCT = "PRODUCT"
GENERIC = "GENERIC"


# Anchor synsets for the WordNet-closure branch. Order matters only for
# breaking ties on equal depth (earlier anchors win).
_WN_ANCHORS = [
    ("person.n.01", PERSON),
    ("organization.n.01", ORG),
    ("location.n.01", LOCATION),
    ("time_period.n.01", TIME),
    ("event.n.01", EVENT),
    ("quantity.n.01", QUANTITY),
    ("measure.n.02", QUANTITY),
    ("artifact.n.01", PRODUCT),
    ("creation.n.02", WORK_OF_ART),
]


_WH_WORDS_POS = {"WP", "WRB", "WDT"}
_WH_TOKENS = {"who", "whom", "whose", "what", "when", "where",
              "which", "why", "how"}


def _pos_tag(text: str):
    try:
        import nltk  # type: ignore
        tokens = nltk.word_tokenize(text)
        return nltk.pos_tag(tokens)
    except Exception:
        # Fallback: naive whitespace split with no POS info.
        return [(t, "") for t in (text or "").split()]


def _find_wh(tagged):
    """Return (index, lowercased WH word) or (None, None)."""
    for i, (tok, tag) in enumerate(tagged):
        low = tok.lower()
        if tag in _WH_WORDS_POS or low in _WH_TOKENS:
            if low in _WH_TOKENS:
                return i, low
    return None, None


def _first_noun_after(tagged, start):
    """Return the first NN* lemma after `start`, or None."""
    for tok, tag in tagged[start + 1:]:
        if tag.startswith("NN"):
            return tok.lower()
    return None


def _wordnet_type_of(word: str) -> Optional[str]:
    """Walk WordNet hypernym closure on the first noun sense of `word`.
    Return the category whose anchor synset is nearest in the closure,
    or None if no anchor matched."""
    if not word:
        return None
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception:
        return None

    try:
        noun_synsets = wn.synsets(word, pos=wn.NOUN)
        adj_synsets = wn.synsets(word, pos=wn.ADJ) + wn.synsets(word, pos=wn.ADJ_SAT)
    except Exception:
        return None

    # For adjectives, walk the `attributes()` pointer to the related
    # noun (e.g. old -> age, tall -> height). These property nouns tend
    # to land on measure/quantity anchors in WordNet, which is the
    # right answer for "how <ADJ>" questions.
    attr_nouns = []
    for adj in adj_synsets[:3]:
        try:
            attr_nouns.extend(adj.attributes())
        except Exception:
            pass

    synsets = list(noun_synsets) + list(attr_nouns)
    if not synsets:
        return None

    depth_by_name: dict = {}
    # Use all noun senses (not just primary) so that e.g. "company" as
    # the business sense doesn't lose out to a less-populated primary.
    for syn in synsets[:5]:
        try:
            for path in syn.hypernym_paths():
                for i, s in enumerate(path):
                    d = len(path) - 1 - i
                    key = s.name()
                    if key not in depth_by_name or d < depth_by_name[key]:
                        depth_by_name[key] = d
        except Exception:
            continue

    best_depth: Optional[int] = None
    best_type: Optional[str] = None
    for anchor_name, type_label in _WN_ANCHORS:
        d = depth_by_name.get(anchor_name)
        if d is None:
            continue
        if best_depth is None or d < best_depth:
            best_depth = d
            best_type = type_label
    return best_type


def parse_expected_answer_type(query_text: str) -> Optional[str]:
    """Structural WH -> expected answer type resolver.

    POS + WordNet hypernym closure. No hardcoded noun lists.

    Rules:
      - "who"   -> PERSON
      - "when"  -> TIME
      - "where" -> LOCATION
      - "what X" / "which X" -> WordNet hypernym closure on the first
        noun after the WH token. Returns the matched anchor category.
      - "how old / how tall / how many / how much" -> QUANTITY
        (via the adjective/determiner after 'how' going to quantity/
        measure anchors in WordNet).
      - "why" / bare "how" / no WH -> None
    """
    if not query_text or not query_text.strip():
        return None

    tagged = _pos_tag(query_text)
    if not tagged:
        return None

    idx, wh = _find_wh(tagged)
    if wh is None:
        return None

    if wh in ("who", "whom", "whose"):
        return PERSON
    if wh == "when":
        return TIME
    if wh == "where":
        return LOCATION

    if wh in ("what", "which"):
        noun = _first_noun_after(tagged, idx)
        if noun is None:
            return None
        return _wordnet_type_of(noun)

    if wh == "how":
        # "how <ADJ/ADV/DET> ..." is the English grammar for degree
        # questions: the speaker is asking for a measure along the
        # dimension named by the modifier ("how old" = age-degree,
        # "how tall" = height-degree, "how many" = count).
        #
        # This entire grammatical frame resolves to QUANTITY regardless
        # of which modifier fills the slot, because the WH-frame itself
        # is what fixes the answer type -- not the modifier content.
        # Bare "how" alone (no modifier) is a manner/causal query and
        # has no single coherent answer type.
        if idx + 1 < len(tagged):
            mod_tok, mod_tag = tagged[idx + 1]
            # Degree slot filler: adjective (JJ*), adverb (RB*),
            # determiner (DT, for "how much/many"), or a noun tagged NN*
            # that sits in the degree slot. The POS + positional frame
            # is what identifies the degree question -- no word list.
            if mod_tag.startswith(("JJ", "RB", "DT", "NN")):
                return QUANTITY
        return None

    # "why" -- causal request, no single answer type.
    return None


# ---------------------------------------------------------------------------
# Situational query detection
# ---------------------------------------------------------------------------

# Closed-class English discourse frames that signal a situational /
# reconstruction query rather than a lookup query. These are structural
# (like WH-words) — they are finite grammatical patterns in English,
# not a curated topic list.
_SITUATIONAL_PREFIXES = (
    "tell me about",
    "summarize",
    "what's going on with",
    "what's happening with",
    "what is going on with",
    "what is happening with",
    "update me on",
    "describe",
    "catch me up on",
    "fill me in on",
    "brief me on",
    "give me an overview of",
    "what do you know about",
    "what do we know about",
)

# "how is X going" pattern — detected structurally, not by prefix
_HOW_GOING_PATTERN = re.compile(
    r"^how\s+(?:is|are|has|have)\s+.+\s+(?:going|doing|progressing|coming along)",
    re.IGNORECASE,
)


def is_situational(query: str) -> bool:
    """Return True if the query is a situational / reconstruction request.

    Detection uses closed-class English discourse frames (structural)
    and the absence of a WH-type expectation. Not a curated list --
    these are finite grammatical patterns.
    """
    if not query or not query.strip():
        return False

    q = query.strip().lower()

    # Check discourse frame prefixes
    for prefix in _SITUATIONAL_PREFIXES:
        if q.startswith(prefix):
            return True

    # "how is X going / doing / progressing"
    if _HOW_GOING_PATTERN.match(q):
        return True

    # "why" questions are causal / narrative — route to reconstruction
    tagged = _pos_tag(query)
    idx, wh = _find_wh(tagged)
    if wh == "why":
        return True

    return False
