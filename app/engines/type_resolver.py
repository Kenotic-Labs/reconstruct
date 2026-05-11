# -*- coding: cp1252 -*-
"""
type_resolver.py -- entity-type resolution for the write-path foundation.

Resolves a surface name (string) to one of the closed top-level categories
used by the coherence gate:

    PERSON | ORG | LOCATION | TIME | EVENT | QUANTITY |
    WORK_OF_ART | PRODUCT | GENERIC

Two structural paths, in order:

1. spaCy NER (en_core_web_sm). Argmax over entity labels found in the
   string; map to the closed vocabulary. If the model is not installed or
   no label is found, fall through.

2. WordNet hypernym closure on the head noun. Walk all hypernym paths;
   argmax over which top-level anchor synset (person.n.01,
   organization.n.01, location.n.01, time_period.n.01, event.n.01,
   quantity.n.01, artifact.n.01) the closure contains.

Returns GENERIC if no path yields a match.

No curated string lists. The vocabulary lives in (a) the standard spaCy
NER label set and (b) the WordNet synset identifiers. Both are external,
shared resources — not project-enumerated.
"""
from __future__ import annotations

from typing import Optional

# Closed top-level categories.
PERSON = "PERSON"
ORG = "ORG"
LOCATION = "LOCATION"
TIME = "TIME"
EVENT = "EVENT"
QUANTITY = "QUANTITY"
WORK_OF_ART = "WORK_OF_ART"
PRODUCT = "PRODUCT"
GENERIC = "GENERIC"

VALID_TYPES = (
    PERSON, ORG, LOCATION, TIME, EVENT,
    QUANTITY, WORK_OF_ART, PRODUCT, GENERIC,
)

# ---- spaCy NER path ----------------------------------------------------
# Standard spaCy OntoNotes label set -> our top-level vocabulary.
# This mapping is not a curated word list; it is a label->label
# dimensionality reduction from the spaCy schema to our schema.
_NER_LABEL_MAP = {
    "PERSON": PERSON,
    "NORP": ORG,            # nationalities/religious/political groups -> org-like
    "ORG": ORG,
    "FAC": LOCATION,        # facility (buildings, airports)
    "GPE": LOCATION,        # geopolitical
    "LOC": LOCATION,
    "DATE": TIME,
    "TIME": TIME,
    "EVENT": EVENT,
    "QUANTITY": QUANTITY,
    "MONEY": QUANTITY,
    "PERCENT": QUANTITY,
    "CARDINAL": QUANTITY,
    "ORDINAL": QUANTITY,
    "WORK_OF_ART": WORK_OF_ART,
    "PRODUCT": PRODUCT,
    "LAW": WORK_OF_ART,
    "LANGUAGE": GENERIC,
}

_spacy_nlp = None
_spacy_tried = False


def _get_spacy():
    """Lazy-load en_core_web_sm. Returns None if unavailable."""
    global _spacy_nlp, _spacy_tried
    if _spacy_tried:
        return _spacy_nlp
    _spacy_tried = True
    try:
        import spacy  # type: ignore
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            _spacy_nlp = None
    except Exception:
        _spacy_nlp = None
    return _spacy_nlp


def _resolve_via_ner(name: str) -> Optional[str]:
    nlp = _get_spacy()
    if nlp is None:
        return None

    # Frame the bare name in a neutral carrier sentence so NER has the
    # surrounding tokens it was trained on. Then only consider entities
    # whose span overlaps the target token window. This is a structural
    # wrapping, not a curated prompt list.
    carrier = f"I visited {name} yesterday."
    offset = carrier.index(name)
    end = offset + len(name)

    try:
        doc = nlp(carrier)
    except Exception:
        return None

    candidates = []
    for ent in doc.ents:
        if ent.start_char >= offset and ent.end_char <= end:
            candidates.append(ent)
    if not candidates:
        # Also try the bare form — short inputs sometimes NER-tag fine
        # without framing.
        try:
            doc2 = nlp(name)
        except Exception:
            doc2 = None
        if doc2 is not None:
            candidates = list(doc2.ents)

    if not candidates:
        return None

    # Argmax by span length (longest entity wins).
    best = max(candidates, key=lambda e: (e.end_char - e.start_char))
    return _NER_LABEL_MAP.get(best.label_)


# ---- WordNet hypernym path --------------------------------------------
# Anchor synsets for each top-level category. Priority order matters only
# for breaking ties when a head noun's closure reaches multiple anchors
# (argmax by depth-to-anchor — closest anchor wins).
_WN_ANCHORS = [
    ("person.n.01", PERSON),
    ("organization.n.01", ORG),
    ("location.n.01", LOCATION),
    ("time_period.n.01", TIME),
    ("event.n.01", EVENT),
    ("quantity.n.01", QUANTITY),
    ("artifact.n.01", PRODUCT),
]

# ---- NER location-override path ---------------------------------------
# Structural geo-class ancestors used to detect when a head noun names a
# *type* of place (inn, mountain, lake) rather than an entity that happens
# to be located somewhere.  The set is grounded in WordNet synset
# identifiers — not a curated word list.
#
# Coverage rationale (each anchor and the class words it reaches):
#   building.n.01          → inn, hotel, lodge, hostel, mansion, cottage
#   facility.n.01          → museum, airport, library, stadium
#   geological_formation.n.01 → mountain, valley, canyon, glacier, ridge, beach
#   body_of_water.n.01     → lake, river, bay, falls, harbor, gulf
#   geographical_area.n.01 → park, tract, territory
#   natural_elevation.n.01 → mountain, hill, peak, ridge (subset of geological)
#   location.n.01          → resort, island, harbor, peak (direct sense)
_GEO_CLASS_ANCESTORS: frozenset = frozenset({
    "building.n.01",
    "facility.n.01",
    "geological_formation.n.01",
    "body_of_water.n.01",
    "geographical_area.n.01",
    "natural_elevation.n.01",
    "location.n.01",
})


def _head_noun(name: str) -> str:
    """Return the rightmost noun in the phrase via POS tag. Falls back to
    the last whitespace token if POS tagging is unavailable."""
    tokens = (name or "").strip().split()
    if not tokens:
        return ""
    try:
        import nltk  # type: ignore
        tagged = nltk.pos_tag(tokens)
        # Rightmost noun-ish tag (NN, NNS, NNP, NNPS).
        for tok, tag in reversed(tagged):
            if tag.startswith("NN"):
                return tok
    except Exception:
        pass
    return tokens[-1]


def _is_geo_class_head(head_word: str) -> bool:
    """Return True if *head_word* is a generic geo-class noun (mountain,
    inn, lake, museum …) — not a proper-name instance like Canada or Toronto.

    Two structural conditions must both hold:

    1. The primary sense must NOT be a WordNet instance synset.  Instance
       synsets (countries, cities, named rivers) have instance_hypernyms()
       populated and no regular hypernyms().  Generic class words (mountain,
       inn) have regular hypernyms and no instance hypernyms.  This guard
       prevents overriding NER for 'Bell Canada' where the head 'Canada' is
       itself an instance/country name.

    2. At least one of the top-five noun senses (skipping instances) must
       reach a synset in _GEO_CLASS_ANCESTORS through its hypernym path.

    No curated word lists.  The vocabulary is entirely WordNet synset
    identifiers.
    """
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception:
        return False

    try:
        synsets = wn.synsets(head_word.lower(), pos=wn.NOUN)
    except Exception:
        return False
    if not synsets:
        return False

    # Condition 1: primary sense must be a generic class word (not an instance).
    primary = synsets[0]
    try:
        if primary.instance_hypernyms():
            return False
    except Exception:
        return False

    # Condition 2: any non-instance sense among the top-5 must reach a
    # geo-class ancestor.
    for syn in synsets[:5]:
        try:
            if syn.instance_hypernyms():
                continue  # skip proper-name instance senses
            for path in syn.hypernym_paths():
                for ancestor in path:
                    if ancestor.name() in _GEO_CLASS_ANCESTORS:
                        return True
        except Exception:
            continue
    return False


def _ner_location_override(name: str, ner_type: str) -> Optional[str]:
    """Return LOCATION if NER may have misclassified a multi-word place name.

    NER (en_core_web_sm) reliably tags unambiguous single-token proper names
    (Amazon -> ORG, Toronto -> GPE).  It is unreliable for multi-word place
    names whose surface form resembles a person or organisation ('Moraine Inn'
    -> PERSON, 'Sulphur Mountain' -> ORG).

    Structural rule: when NER returns PERSON or ORG for a *multi-word* name,
    check whether the head noun is a generic geo-class word via
    _is_geo_class_head().  If it is, the entity is a named place and the
    correct type is LOCATION.

    Additive: this function is only called when NER returned PERSON or ORG.
    It does not touch correctly classified entities.
    """
    if ner_type not in (PERSON, ORG):
        return None  # only override misclassifications
    tokens = (name or "").strip().split()
    if len(tokens) < 2:
        return None  # single-token: trust NER
    head = _head_noun(name)
    if not head:
        return None
    if _is_geo_class_head(head):
        return LOCATION
    return None


def _resolve_via_wordnet(name: str) -> Optional[str]:
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception:
        return None

    head = _head_noun(name).lower()
    if not head:
        return None

    try:
        synsets = wn.synsets(head, pos=wn.NOUN)
    except Exception:
        return None
    if not synsets:
        return None

    # Argmax: pick the top-level whose anchor synset is nearest in the
    # hypernym closure of the first sense. No thresholds — smallest depth
    # wins; ties broken by anchor order.
    best_depth: Optional[int] = None
    best_type: Optional[str] = None

    primary = synsets[0]
    try:
        hyp_paths = primary.hypernym_paths()
    except Exception:
        return None

    # Flatten all hypernyms with their depth along the shortest path.
    depth_by_name = {}
    for path in hyp_paths:
        # path goes root -> ... -> primary. Depth from primary = len-1-i.
        for i, syn in enumerate(path):
            d = len(path) - 1 - i
            key = syn.name()
            if key not in depth_by_name or d < depth_by_name[key]:
                depth_by_name[key] = d

    for anchor_name, type_label in _WN_ANCHORS:
        d = depth_by_name.get(anchor_name)
        if d is None:
            continue
        if best_depth is None or d < best_depth:
            best_depth = d
            best_type = type_label

    return best_type


# ---- public API --------------------------------------------------------

# Confidence levels for type_confidence field.
CONFIDENCE_HIGH = "high"
CONFIDENCE_LOW = "low"


def resolve_entity_type_with_confidence(name: str) -> tuple:
    """Return (type_label, confidence) for the given surface name.

    Confidence is HIGH when:
      - NER and WordNet agree on the same top-level category, OR
      - Only one source produced a result (no disagreement signal).

    Confidence is LOW when:
      - NER and WordNet disagree (e.g. NER says LOCATION, WordNet
        says PERSON for "Luminary"). This is the structural signal
        that the type is uncertain.

    When confidence is LOW and both sources returned a type, the
    WordNet result is preferred over NER. Rationale: spaCy's
    en_core_web_sm NER defaults to GPE/PERSON for capitalized
    unknown tokens (training bias), while WordNet's hypernym closure
    is structurally grounded in lexical semantics.
    """
    if not name or not name.strip():
        return GENERIC, CONFIDENCE_LOW

    cleaned = name.strip()

    # Bare pronouns / canonical user token -> GENERIC (the coherence gate
    # treats "user" as a role, not a typed entity).
    if cleaned.lower() in ("user", "i", "me", "myself", "you"):
        return GENERIC, CONFIDENCE_HIGH

    via_ner = _resolve_via_ner(cleaned)
    if via_ner:
        # Structural guard: NER misclassifies multi-word place names whose
        # surface form resembles a person or org (e.g. "Moraine Inn" ->
        # PERSON, "Sulphur Mountain" -> ORG).  Check whether the head noun
        # is a generic geo-class word; if so, the correct type is LOCATION.
        override = _ner_location_override(cleaned, via_ner)
        if override:
            via_ner = override

    via_wn = _resolve_via_wordnet(cleaned)

    # Both sources returned a result — compare for agreement.
    if via_ner and via_wn:
        if via_ner == via_wn:
            return via_ner, CONFIDENCE_HIGH
        else:
            # Disagreement: prefer WordNet — NER's en_core_web_sm has
            # a known bias toward GPE/PERSON for capitalized unknowns.
            # WordNet hypernym closure is structurally grounded.
            return via_wn, CONFIDENCE_LOW

    # Only one source returned — no disagreement signal but also no
    # corroboration. Return with HIGH confidence (single-source is the
    # norm for most entities; LOW should only flag actual disagreement).
    if via_ner:
        return via_ner, CONFIDENCE_HIGH
    if via_wn:
        return via_wn, CONFIDENCE_HIGH

    return GENERIC, CONFIDENCE_LOW


def resolve_entity_type(name: str) -> str:
    """Return one of VALID_TYPES for the given surface name.

    Backward-compatible wrapper around resolve_entity_type_with_confidence.
    """
    type_label, _confidence = resolve_entity_type_with_confidence(name)
    return type_label
