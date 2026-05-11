"""Tests for Task 1 (emotional_valence) and Task 2 (broadened schematic_category).

Root cause 1 (emotional_valence):
    TraceDecomposition.emotional_valence defaults to None (L130).
    _build_trace_decomposition (L3451) never sets it.
    _enrich_defaults_from_doc (L3253-3257) fills emotional_state (the string,
    e.g. "nervous") by scanning for predicate ADJ tokens, but never derives
    the float valence from it.  memory.py L898 always falls back to 0.5.
    Fix: after emotional_state is filled, locate the emotion ADJ token in the
    doc, check for UD dep label "neg" on it or its head, encode as float.

Root cause 2 (schematic_category):
    NER refinement (L3426-3431) only maps ORG->"career" and GPE/FAC->"housing".
    The remaining OntoNotes NER categories (EVENT, MONEY, NORP, LAW,
    WORK_OF_ART, PRODUCT, QUANTITY) are available in doc.ents but never checked.
    Fix: extend the elif chain with additional OntoNotes category mappings.
    Mirror the extension in classify_query (L3991-3996).

Both fixes use closed OntoNotes annotation categories and UD dep labels --
the same structural feature classes already used throughout grammar_engine.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["RAYA_SENTENCE_POLISH"] = "0"

from app.engines.grammar_engine import (
    process,
    classify_query,
    TraceDecomposition,
    _build_trace_decomposition,
    classify_verb_class,
    VerbClass,
    TenseAspect,
    Triple,
)


def _make_triple(subj="user", pred="feels", obj="nervous", **kw):
    defaults = dict(
        subject=subj, predicate=pred, object=obj,
        is_historical=False, utterance_type=36,
        negated=False, mood="indicative", extraction_rule="E_EMOTION",
    )
    defaults.update(kw)
    return Triple(**defaults)


def _make_doc(text):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    return nlp(text)


# =========================================================================
# Task 1: emotional_valence — derived from UD "neg" dep label
# =========================================================================


class TestEmotionalValence:
    """Verify emotional_valence is structurally filled from neg dep label.

    Root cause: _enrich_defaults_from_doc fills emotional_state but never
    derives emotional_valence.  After fix, the function locates the emotion
    ADJ token in the doc, checks for "neg" dep on it or its head verb, and
    sets emotional_valence = float(not has_negation).
    """

    def test_positive_emotion_valence(self):
        """'I feel nervous' -> emotional_valence = 1.0 (no negation)."""
        doc = _make_doc("I feel nervous")
        triple = _make_triple()
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("present", "simple"),
            VerbClass.BE,
            emotion="nervous",
        )
        assert td.emotional_state == "nervous"
        assert td.emotional_valence == 1.0

    def test_negated_emotion_valence(self):
        """'I don't feel happy' -> emotional_valence = 0.0 (negation present)."""
        doc = _make_doc("I don't feel happy")
        triple = _make_triple(pred="feels", obj="happy", negated=True)
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("present", "simple"),
            VerbClass.BE,
            emotion="happy",
        )
        assert td.emotional_state == "happy"
        assert td.emotional_valence == 0.0

    def test_no_emotion_no_valence(self):
        """'I started a new job' -> emotional_valence stays None.

        Root cause: no predicate ADJ (acomp/attr/oprd) exists in the parse,
        so emotional_state stays None, and emotional_valence is never derived.
        """
        doc = _make_doc("I started a new job")
        triple = _make_triple(pred="start", obj="a new job")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("past", "simple"),
            VerbClass.EXPERIENCE,
        )
        assert td.emotional_state is None
        assert td.emotional_valence is None

    def test_valence_via_process_positive(self):
        """Integration: process('I feel nervous') -> valence = 1.0."""
        r = process("I feel nervous", speaker="user")
        assert len(r.trace_decompositions) >= 1
        td = r.trace_decompositions[0]
        assert td.emotional_state is not None
        assert td.emotional_valence == 1.0

    def test_valence_via_process_negated(self):
        """Integration: process('I do not feel happy') -> valence = 0.0."""
        r = process("I do not feel happy", speaker="user")
        assert len(r.trace_decompositions) >= 1
        td = r.trace_decompositions[0]
        if td.emotional_state is not None:
            assert td.emotional_valence == 0.0


# =========================================================================
# Task 2: broadened schematic_category — OntoNotes NER chain extension
# =========================================================================


class TestBroadenedSchema:
    """Verify additional OntoNotes NER categories map to schema values.

    Root cause: NER refinement (L3426-3431) only mapped ORG->"career" and
    GPE/FAC->"housing".  After fix, EVENT, MONEY, NORP, LAW, WORK_OF_ART,
    PRODUCT, QUANTITY are also mapped.
    """

    def test_money_to_finance(self):
        """MONEY entity -> schematic_category='finance'."""
        doc = _make_doc("I spent $500 on groceries")
        triple = _make_triple(pred="spend", obj="$500 on groceries")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("past", "simple"),
            VerbClass.UNKNOWN,
        )
        money_ents = [e for e in doc.ents if e.label_ == "MONEY"]
        if money_ents:
            assert td.schematic_category == "finance"

    def test_org_still_career(self):
        """ORG entity still maps to 'career' (existing behavior preserved)."""
        doc = _make_doc("I have an interview at Google")
        triple = _make_triple(pred="have", obj="an interview")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("present", "simple"),
            VerbClass.HAVE,
        )
        assert td.schematic_category == "career"

    def test_gpe_still_housing(self):
        """GPE entity still maps to 'housing' (existing behavior preserved)."""
        doc = _make_doc("I have a house in Portland")
        triple = _make_triple(pred="have", obj="a house")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("present", "simple"),
            VerbClass.HAVE,
        )
        gpe_ents = [e for e in doc.ents if e.label_ == "GPE"]
        if gpe_ents:
            assert td.schematic_category == "housing"

    def test_classify_query_org_career(self):
        """classify_query with ORG entity -> match_schema='career'."""
        qd = classify_query("Where does Maya work at Google?")
        assert qd.match_schema == "career"

    def test_family_schema_preserved(self):
        """Possessive-subject family detection still works."""
        doc = _make_doc("My daughter's birthday is next week")
        triple = _make_triple(pred="be", obj="next week")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("present", "simple"),
            VerbClass.BE,
        )
        assert td.schematic_category == "family"

    def test_norp_to_social(self):
        """NORP entity -> schematic_category='social'."""
        doc = _make_doc("I joined a Christian community group")
        triple = _make_triple(pred="join", obj="a Christian community group")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("past", "simple"),
            VerbClass.UNKNOWN,
        )
        norp_ents = [e for e in doc.ents if e.label_ == "NORP"]
        if norp_ents:
            assert td.schematic_category == "social"
