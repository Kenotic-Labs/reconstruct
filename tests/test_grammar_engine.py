"""
Tests for the structural grammar engine underlying
RetrievalEngine._edge_to_sentence.

Covers:
  1. predicate_shape.parse_predicate — structural decomposition
  2. predicate_shape.inflect_verb    — morphological inflection
  3. RetrievalEngine._edge_to_sentence — surface rendering

Rule: every test asserts the EXACT surface string, not a substring.
These tests pin down the known renderer failures reported against the
old "feel"-as-fallback renderer:

  (user, went_to, Banff)    -> "You went to Banff."   (was "You feel Banff")
  (user, corrected, Kobe)   -> "You corrected Kobe."  (was "Sam feels Kobe")
  (user, birth_date, X)     -> "Your birth date is X." (was unparseable)

No curated word lists. No verb-lookup maps. POS + WordNet + morphology.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Disable the optional seq2seq polish so tests pin down pure grammar
# output, not a model-dependent surface form.
os.environ["RAYA_SENTENCE_POLISH"] = "0"


# ── predicate_shape: parse_predicate ──────────────────────────────

from app.engines.predicate_shape import (
    parse_predicate,
    inflect_verb,
    is_verb_token,
)


def test_parse_went_to():
    r = parse_predicate("went_to")
    assert r.ok is True
    assert r.verb_lemma == "go"
    assert r.verb_surface == "went"
    assert r.preposition == "to"
    assert r.particle is None
    assert r.embedded_noun is None
    assert r.middle == ["to"]


def test_parse_went_home_to():
    r = parse_predicate("went_home_to")
    assert r.ok is True
    assert r.verb_lemma == "go"
    assert r.embedded_noun == "home"
    assert r.preposition == "to"
    assert r.middle == ["home", "to"]


def test_parse_hiked_up():
    r = parse_predicate("hiked_up")
    assert r.ok is True
    assert r.verb_lemma == "hike"
    assert r.particle == "up"
    assert r.middle == ["up"]


def test_parse_twisted_ankle():
    r = parse_predicate("twisted_ankle")
    assert r.ok is True
    assert r.verb_lemma == "twist"
    assert r.embedded_noun == "ankle"
    assert r.preposition is None


def test_parse_made_by():
    r = parse_predicate("made_by")
    assert r.ok is True
    assert r.verb_lemma == "make"
    assert r.preposition == "by"


def test_parse_interviewed_at():
    r = parse_predicate("interviewed_at")
    assert r.ok is True
    assert r.verb_lemma == "interview"
    assert r.preposition == "at"


def test_parse_birth_date_is_not_a_verb():
    # Noun-headed compound — parser must refuse verb-hood so the
    # renderer falls back to a copular construction ("Your birth date
    # is February 29") rather than inflecting the noun as a verb.
    r = parse_predicate("birth_date")
    assert r.ok is False
    assert "birth" in r.failure_reason


# ── predicate_shape: inflect_verb ─────────────────────────────────

@pytest.mark.parametrize("lemma,tense,person,expected", [
    ("go", "past", "2s", "went"),
    ("make", "past", "2s", "made"),
    ("say", "past", "3s", "said"),
    ("interview", "past", "2s", "interviewed"),
    ("try", "past", "2s", "tried"),       # -y -> -ied
    ("stop", "past", "2s", "stopped"),    # CVC doubling
    ("hike", "past", "2s", "hiked"),      # silent-e drop
    ("go", "present", "3s", "goes"),      # -o + -es
    ("go", "present", "2s", "go"),        # base for 2s/1s/plural
    ("watch", "present", "3s", "watches"),
])
def test_inflect_verb(lemma, tense, person, expected):
    assert inflect_verb(lemma, tense, person) == expected


# ── End-to-end surface rendering via RetrievalEngine ──────────────

@pytest.fixture
def renderer():
    from app.engines.retrieval import RetrievalEngine
    # Bypass __init__ — _edge_to_sentence needs no DB state.
    return RetrievalEngine.__new__(RetrievalEngine)


def _edge(subject, predicate, object_, tense="past", eid=1, **extra):
    base = {
        "id": eid,
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "edge_temporal_context": tense,
        "edge_emotional_label": None,
        "edge_episodic_significance": None,
    }
    base.update(extra)
    return base


# ── The 7 target renderings from the reconstruction-grammar brief ──

def test_render_went_to_banff(renderer):
    out = renderer._edge_to_sentence(
        _edge("user", "went_to", "Banff"), None
    )
    assert out == "You went to Banff."


def test_render_went_home_to_michigan(renderer):
    out = renderer._edge_to_sentence(
        _edge("user", "went_home_to", "Michigan"), None
    )
    assert out == "You went home to Michigan."


def test_render_hiked_up_sulphur(renderer):
    out = renderer._edge_to_sentence(
        _edge("user", "hiked_up", "Sulphur Mountain"), None
    )
    assert out == "You hiked up Sulphur Mountain."


def test_render_twisted_ankle(renderer):
    out = renderer._edge_to_sentence(
        _edge("Arjun", "twisted_ankle", "coming down"), None
    )
    assert out == "Arjun twisted ankle coming down."


def test_render_made_by_priya(renderer):
    # Passive construction — "by" preposition + past tense.
    out = renderer._edge_to_sentence(
        _edge("user", "made_by", "Priya"), None
    )
    assert out == "You were made by Priya."


def test_render_priya_said(renderer):
    out = renderer._edge_to_sentence(
        _edge("Priya", "said", "better than anything in Banff"), None
    )
    assert out == "Priya said better than anything in Banff."


def test_render_interviewed_at_google(renderer):
    out = renderer._edge_to_sentence(
        _edge("user", "interviewed_at", "Google"), None
    )
    assert out == "You interviewed at Google."


# ── Regression pins for the reported failures ─────────────────────

def test_corrected_is_not_feels(renderer):
    # Was: "Sam feels Kobe" (feel fallback). Must be "corrected".
    out = renderer._edge_to_sentence(
        _edge("user", "corrected", "Kobe"), None
    )
    assert out == "You corrected Kobe."


def test_birth_date_copular_fallback(renderer):
    # Noun-headed predicate — parser returns ok=False, renderer
    # falls back to "Your {pred} is {obj}".
    out = renderer._edge_to_sentence(
        _edge("user", "birth_date", "February 29", tense="present"), None
    )
    assert out == "Your birth date is February 29."


# ── Additional edge cases: present / past / irregular ─────────────

def test_present_you_base_form(renderer):
    # Present + 2s -> base form. "loves" stored -> 2s rendering
    # drops the -s via morphy.
    out = renderer._edge_to_sentence(
        _edge("user", "loves", "ramen", tense="present"), None
    )
    assert out == "You love ramen."


def test_present_third_person_singular(renderer):
    # Present + 3s -> -s on base form.
    out = renderer._edge_to_sentence(
        _edge("Priya", "cook", "dinner", tense="present"), None
    )
    assert out == "Priya cooks dinner."


def test_future_prepends_will(renderer):
    out = renderer._edge_to_sentence(
        _edge("user", "visit", "Tokyo", tense="future"), None
    )
    assert out == "You will visit Tokyo."


def test_irregular_past_say(renderer):
    out = renderer._edge_to_sentence(
        _edge("Kobe", "say", "hello", tense="past"), None
    )
    assert out == "Kobe said hello."


def test_phrasal_verb_preserves_particle_order(renderer):
    # Particle must stay between verb and object.
    out = renderer._edge_to_sentence(
        _edge("user", "picked_up", "the keys"), None
    )
    assert out == "You picked up the keys."


# ── Verb-token recognition ────────────────────────────────────────

@pytest.mark.parametrize("tok,expected", [
    ("went", True),
    ("hiked", True),
    ("interviewed", True),  # NN-tagged in isolation; morphy reduces
    ("said", True),
    ("twisted", True),
    ("ankle", False),        # no verb synset at all
])
def test_is_verb_token(tok, expected):
    assert is_verb_token(tok) is expected


# Compound-noun predicates ("birth_date", "phone_number") must refuse
# parsing — the structural tell is two NN-tagged segments in sequence,
# not the nounhood of any single segment.
@pytest.mark.parametrize("pred", [
    "birth_date",
    "phone_number",
])
def test_noun_compound_predicate_refused(pred):
    r = parse_predicate(pred)
    assert r.ok is False
    assert "noun_compound_head" in r.failure_reason
