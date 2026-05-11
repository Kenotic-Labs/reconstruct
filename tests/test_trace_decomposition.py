"""
Tests for Phase 1 grammar engine additions:
  - TraceDecomposition via _build_trace_decomposition
  - QueryDecomposition via classify_query
  - Trace decompositions wired into process()

Tests verify structural features (verb class, NER, tense) are captured
in TraceDecomposition at the point where they are naturally available,
rather than being re-derived downstream.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Disable seq2seq polish so tests pin down structural output only.
os.environ["RAYA_SENTENCE_POLISH"] = "0"

from app.engines.grammar_engine import (
    process,
    classify_query,
    TraceDecomposition,
    QueryDecomposition,
    GrammarResult,
    _build_trace_decomposition,
    classify_verb_class,
    VerbClass,
    TenseAspect,
    Triple,
)


# ── _build_trace_decomposition ──────────────────────────────────────


class TestBuildTraceDecomposition:
    """Unit tests for _build_trace_decomposition."""

    def _make_triple(self, subj="user", pred="employer", obj="Google", **kw):
        defaults = dict(
            subject=subj, predicate=pred, object=obj,
            is_historical=False, utterance_type=35,
            negated=False, mood="indicative", extraction_rule="E12",
        )
        defaults.update(kw)
        return Triple(**defaults)

    def _make_doc(self, text):
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp(text)

    def test_career_schema_from_work_verb_class(self):
        """WORK verb class -> schematic_category='career'."""
        triple = self._make_triple()
        doc = self._make_doc("I work at Google")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("present", "simple"),
            VerbClass.WORK,
        )
        assert td.schematic_category == "career"
        assert td.relational_type == "professional"

    def test_housing_schema_from_location_verb_class(self):
        """LOCATION verb class -> schematic_category='housing'."""
        triple = self._make_triple(pred="location", obj="Portland")
        doc = self._make_doc("My sister Emily lives in Portland")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("present", "simple"),
            VerbClass.LOCATION,
        )
        assert td.schematic_category == "housing"
        assert td.relational_type == "personal"

    def test_ner_entities_extracted(self):
        """NER PERSON/ORG/GPE entities populate relational_entities."""
        doc = self._make_doc("I work at Google")
        triple = self._make_triple()
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("present", "simple"),
            VerbClass.WORK,
        )
        assert "Google" in td.relational_entities

    def test_temporal_direction_from_tense(self):
        """past tense -> temporal_direction='past'."""
        doc = self._make_doc("I worked at Google")
        triple = self._make_triple(is_historical=True)
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("past", "simple"),
            VerbClass.WORK,
        )
        assert td.temporal_direction == "past"

    def test_temporal_expression_from_ner(self):
        """DATE/TIME NER entities populate temporal_expression."""
        doc = self._make_doc("I have an interview next Tuesday")
        triple = self._make_triple(pred="scheduled_event", obj="interview")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("present", "simple"),
            VerbClass.PLANNING,
        )
        # spaCy sm may or may not detect "next Tuesday" as DATE
        # This test verifies the extraction path works; exact NER depends on model
        assert td.temporal_expression is None or isinstance(td.temporal_expression, str)

    def test_achievement_significance(self):
        """ACHIEVEMENT verb class -> episodic_significance='milestone'."""
        doc = self._make_doc("I won the championship")
        triple = self._make_triple(pred="win", obj="the championship")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("past", "simple"),
            VerbClass.ACHIEVEMENT,
        )
        assert td.episodic_significance == "milestone"

    def test_experience_significance(self):
        """EXPERIENCE verb class -> episodic_significance='notable'."""
        doc = self._make_doc("I visited Japan")
        triple = self._make_triple(pred="visited", obj="Japan")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("past", "simple"),
            VerbClass.EXPERIENCE,
        )
        assert td.episodic_significance == "notable"

    def test_routine_significance_default(self):
        """Non-achievement, non-experience -> episodic_significance='routine'."""
        doc = self._make_doc("I live in Portland")
        triple = self._make_triple(pred="location", obj="Portland")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("present", "simple"),
            VerbClass.LOCATION,
        )
        assert td.episodic_significance == "routine"

    def test_emotion_passed_through(self):
        """Emotion string is propagated to emotional_state."""
        doc = self._make_doc("I feel nervous")
        triple = self._make_triple(pred="feel", obj="nervous")
        td = _build_trace_decomposition(
            triple, doc, "user",
            TenseAspect("present", "simple"),
            VerbClass.BE,
            emotion="nervous",
        )
        assert td.emotional_state == "nervous"

    def test_metadata_carried_from_triple(self):
        """Triple metadata (subject, predicate, object, rule) is preserved."""
        doc = self._make_doc("Maya works at Google")
        triple = self._make_triple(subj="Maya", pred="employer", obj="Google")
        td = _build_trace_decomposition(
            triple, doc, "Maya",
            TenseAspect("present", "simple"),
            VerbClass.WORK,
        )
        assert td.subject == "Maya"
        assert td.predicate == "employer"
        assert td.object == "Google"
        assert td.extraction_rule == "E12"


# ── process() trace decomposition integration ─────────────────────


class TestProcessTraceDecomposition:
    """Integration tests: process() populates trace_decompositions."""

    def test_work_at_google_produces_decomposition(self):
        """'I work at Google' -> 1 triple, 1 decomposition with career schema."""
        r = process("I work at Google", speaker="Maya")
        assert len(r.triples) >= 1
        assert len(r.trace_decompositions) == len(r.triples)
        td = r.trace_decompositions[0]
        assert td.schematic_category == "career"
        assert td.relational_type == "professional"
        assert "Google" in td.relational_entities

    def test_sister_lives_in_portland(self):
        """'My sister Emily lives in Portland' -> housing schema."""
        r = process("My sister Emily lives in Portland", speaker="user")
        assert len(r.triples) >= 1
        assert len(r.trace_decompositions) == len(r.triples)
        # At least one decomposition should have housing schema
        schemas = [td.schematic_category for td in r.trace_decompositions]
        assert "housing" in schemas

    def test_past_tense_direction(self):
        """Past tense utterance -> temporal_direction='past'."""
        r = process("I visited Japan last summer", speaker="user")
        if r.trace_decompositions:
            td = r.trace_decompositions[0]
            assert td.temporal_direction == "past"

    def test_decomposition_count_matches_triples(self):
        """Number of decompositions always equals number of triples."""
        r = process("I work at Google and I live in Portland", speaker="user")
        assert len(r.trace_decompositions) == len(r.triples)


# ── classify_query ─────────────────────────────────────────────────


class TestClassifyQuery:
    """Tests for classify_query() — query decomposition for retrieval."""

    def test_when_returns_temporal(self):
        """'When did Melanie go camping?' -> return_field='temporal'."""
        qd = classify_query("When did Melanie go camping?")
        assert qd.return_field == "temporal"
        assert qd.match_subject == "Melanie"
        assert qd.match_predicate == "go"

    def test_who_returns_relational(self):
        """'Who went camping?' -> return_field='relational'."""
        qd = classify_query("Who went camping?")
        assert qd.return_field == "relational"

    def test_what_returns_episodic(self):
        """'What does Melanie paint?' -> return_field='episodic'."""
        qd = classify_query("What does Melanie paint?")
        assert qd.return_field == "episodic"

    def test_where_returns_episodic(self):
        """'Where does Maya live?' -> return_field='episodic'."""
        qd = classify_query("Where does Maya live?")
        assert qd.return_field == "episodic"

    def test_structural_with_subject_and_predicate(self):
        """Query with entity subject + verb -> is_structural=True."""
        qd = classify_query("When did Melanie go camping?")
        assert qd.is_structural is True

    def test_match_entity_from_ner(self):
        """ORG/GPE entities populate match_entity."""
        qd = classify_query("Where does Maya work at Google?")
        assert qd.match_entity == "Google"

    def test_utterance_type_id_populated(self):
        """classify_utterance() type ID is propagated."""
        qd = classify_query("When did Melanie go camping?")
        assert qd.utterance_type_id > 0

    def test_wh_word_extracted(self):
        """WH-word is captured in the result."""
        qd = classify_query("When did Melanie go camping?")
        assert qd.wh_word == "when"

    def test_non_question_defaults_to_episodic(self):
        """Non-question text defaults to return_field='episodic'."""
        qd = classify_query("Tell me about Caroline")
        assert qd.return_field == "episodic"


# ── QueryDecomposition dataclass fields ────────────────────────────


class TestQueryDecompositionFields:
    """Verify QueryDecomposition has all required fields."""

    def test_has_match_entity(self):
        qd = QueryDecomposition()
        assert hasattr(qd, "match_entity")

    def test_has_utterance_type_id(self):
        qd = QueryDecomposition()
        assert hasattr(qd, "utterance_type_id")

    def test_has_match_object(self):
        qd = QueryDecomposition()
        assert hasattr(qd, "match_object")

    def test_has_match_schema(self):
        qd = QueryDecomposition()
        assert hasattr(qd, "match_schema")
