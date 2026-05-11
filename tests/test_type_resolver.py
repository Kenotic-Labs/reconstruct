# -*- coding: cp1252 -*-
"""
test_type_resolver.py -- unit tests for app.engines.type_resolver.

Tests exercise both NER (when spaCy is installed) and WordNet-hypernym
paths. Accepts plausible alternatives (e.g. 'interview' may land as
EVENT or GENERIC depending on whether NER tags it).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _r(name):
    from app.engines.type_resolver import resolve_entity_type
    return resolve_entity_type(name)


def test_person_maya():
    assert _r("Maya") == "PERSON"


def test_org_google():
    assert _r("Google") == "ORG"


def test_location_ann_arbor():
    assert _r("Ann Arbor") == "LOCATION"


def test_time_tuesday():
    assert _r("Tuesday") == "TIME"


def test_quantity_money():
    assert _r("five dollars") == "QUANTITY"


def test_event_world_war_ii():
    assert _r("World War II") == "EVENT"


def test_generic_dog():
    # No NER entity, WordNet hypernyms of 'dog' land on animal, not on
    # any of our top-level anchors, so GENERIC is the correct structural
    # result.
    assert _r("dog") == "GENERIC"


def test_user_pronoun_generic():
    assert _r("user") == "GENERIC"
    assert _r("") == "GENERIC"


def test_interview_event_or_generic():
    # 'interview' sits under event.n.01 in WordNet, but only if the head
    # noun sense is picked; accept both honest results.
    got = _r("interview")
    assert got in ("EVENT", "GENERIC")


def test_honda_civic_product_or_org():
    # Compound product names are an honest NER edge case; accept either
    # the spaCy label argmax (ORG for 'Honda') or the intended PRODUCT.
    got = _r("Honda Civic")
    assert got in ("PRODUCT", "ORG")


# ---- NER location-override regression tests (Banff diagnosis 2026-04-21) --
# spaCy en_core_web_sm misclassifies multi-word place names whose surface
# form resembles a person or org.  The _ner_location_override() path checks
# whether the head noun is a generic geo-class word via WordNet and corrects
# the type to LOCATION.

def test_moraine_inn_location():
    # NER tags 'Moraine Inn' as PERSON; head 'Inn' -> building.n.01 -> LOCATION.
    assert _r("Moraine Inn") == "LOCATION"


def test_sulphur_mountain_location():
    # NER tags 'Sulphur Mountain' as ORG; head 'Mountain' ->
    # geological_formation.n.01 -> LOCATION.
    assert _r("Sulphur Mountain") == "LOCATION"


def test_amazon_org_not_overridden():
    # Single-token name: NER returns ORG; override does not fire for
    # single-token names, so ORG is preserved.
    assert _r("Amazon") == "ORG"


def test_arjun_person_not_overridden():
    # Single-token person name: no override fires.
    assert _r("Arjun") == "PERSON"


def test_toronto_location_unchanged():
    # NER correctly tags 'Toronto' as GPE -> LOCATION; no override needed.
    assert _r("Toronto") == "LOCATION"
