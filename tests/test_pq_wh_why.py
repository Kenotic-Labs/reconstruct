# -*- coding: utf-8 -*-
"""
test_pq_wh_why.py — Unit tests for WH_WHY structural path in predicted_queries.

The WH_WHY path fires when a predicate's head verb encodes agent causation,
detected via two structural tiers:

    Tier 1: VerbNet class membership  (engender-27, force-59, force-59-1)
    Tier 2: WordNet hypernym closure  (any path reaching cause.v.01)

Laws: RESEARCH FIRST | STRUCTURAL | NO WORD LISTS | ADDITIVE | NO SCORES |
      NO RANGES | NO PATCHWORK

TC-01  Causal predicate (VerbNet)         — 'caused' fires WHY via engender-27
TC-02  Causal predicate (WordNet fallback) — 'triggered' fires WHY via cause.v.01
TC-03  Non-causal predicate excluded      — 'went' does NOT fire WHY
TC-04  Consequence predicate excluded     — 'resulted' does NOT fire WHY
TC-05  ADP predicate with causation       — 'led_to' fires both WHERE and WHY
TC-06  Full regression (27/27 ADP tests)  — test_pq_adp_where.py must still pass

Run:
    py -3.10 -m pytest tests/test_pq_wh_why.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engines.predicted_queries import (
    _predicate_is_causal,
    _applicable_wh_types,
    _PREDICATE_CAUSAL_CACHE,
    generate_predicted_queries,
    WH_WHY,
    WH_WHERE,
    WH_WHAT,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _wh_types_set(subject_type: str, object_type: str, predicate: str) -> set:
    """Convenience: return _applicable_wh_types result as a set."""
    return set(_applicable_wh_types(subject_type, object_type, predicate))


def _pq_questions(subject: str, predicate: str, obj: str,
                  subject_type: str = "GENERIC",
                  object_type: str = "GENERIC") -> list[str]:
    """Return plain question strings from generate_predicted_queries."""
    pairs = generate_predicted_queries(subject, predicate, obj, subject_type, object_type)
    return [q for q, _ in pairs]


def _any_why(questions: list[str]) -> bool:
    """True if at least one question starts with 'Why' (case-insensitive)."""
    return any(q.strip().lower().startswith("why") for q in questions)


def _any_where(questions: list[str]) -> bool:
    """True if at least one question starts with 'Where' (case-insensitive)."""
    return any(q.strip().lower().startswith("where") for q in questions)


# ── TC-01: Causal predicate (VerbNet) fires WHY ──────────────────────────────

class TestTC01CausalVerbNet:
    """'caused' is in VerbNet class engender-27 — WH_WHY must fire.

    _predicate_is_causal('caused') -> True (Tier 1 hit)
    _applicable_wh_types must include WH_WHY
    generate_predicted_queries must produce at least one WHY-framed question
    """

    def test_predicate_is_causal_caused(self):
        """_predicate_is_causal('caused') must return True via VerbNet engender-27."""
        assert _predicate_is_causal("caused"), (
            "_predicate_is_causal('caused') returned False; "
            "expected True via VerbNet engender-27"
        )

    def test_applicable_wh_types_includes_why(self):
        """_applicable_wh_types must include WH_WHY for 'caused'."""
        whs = _wh_types_set("GENERIC", "GENERIC", "caused")
        assert WH_WHY in whs, (
            f"WH_WHY must fire for predicate 'caused'. Got: {whs}"
        )

    def test_what_still_fires(self):
        """WHAT is always present — additive regression guard."""
        whs = _wh_types_set("GENERIC", "GENERIC", "caused")
        assert WH_WHAT in whs, f"WH_WHAT must always fire. Got: {whs}"

    def test_generate_pq_produces_why_question(self):
        """generate_predicted_queries must emit at least one WHY-framed question."""
        questions = _pq_questions("the_drought", "caused", "the_famine")
        assert _any_why(questions), (
            f"Expected a WHY question for (the_drought, caused, the_famine). "
            f"Got questions: {questions}"
        )

    def test_result_is_cached(self):
        """Cache must be populated after the call (per-predicate cost amortised)."""
        _PREDICATE_CAUSAL_CACHE.pop("caused", None)
        _predicate_is_causal("caused")
        assert "caused" in _PREDICATE_CAUSAL_CACHE, (
            "_PREDICATE_CAUSAL_CACHE not populated after _predicate_is_causal('caused')"
        )
        assert _PREDICATE_CAUSAL_CACHE["caused"] is True


# ── TC-02: Causal predicate (WordNet fallback) fires WHY ─────────────────────

class TestTC02CausalWordNetFallback:
    """'triggered' is absent from VerbNet; WordNet fallback must fire.

    WordNet path: trigger -> trip.v.04 -> initiate.v.02 -> cause.v.01
    _predicate_is_causal('triggered') -> True (Tier 2 hit)
    """

    def test_predicate_is_causal_triggered(self):
        """_predicate_is_causal('triggered') must return True via WordNet fallback."""
        assert _predicate_is_causal("triggered"), (
            "_predicate_is_causal('triggered') returned False; "
            "expected True via WordNet hypernym path trigger -> cause.v.01"
        )

    def test_applicable_wh_types_includes_why(self):
        """_applicable_wh_types must include WH_WHY for 'triggered'."""
        whs = _wh_types_set("GENERIC", "GENERIC", "triggered")
        assert WH_WHY in whs, (
            f"WH_WHY must fire for predicate 'triggered'. Got: {whs}"
        )

    def test_generate_pq_produces_why_question(self):
        """generate_predicted_queries must emit at least one WHY-framed question."""
        questions = _pq_questions("the_spark", "triggered", "the_explosion")
        assert _any_why(questions), (
            f"Expected a WHY question for (the_spark, triggered, the_explosion). "
            f"Got questions: {questions}"
        )

    def test_cached_after_lookup(self):
        """Result must be cached after the WordNet fallback fires."""
        _PREDICATE_CAUSAL_CACHE.pop("triggered", None)
        result = _predicate_is_causal("triggered")
        assert "triggered" in _PREDICATE_CAUSAL_CACHE, (
            "_PREDICATE_CAUSAL_CACHE not populated after WordNet fallback"
        )
        assert _PREDICATE_CAUSAL_CACHE["triggered"] == result


# ── TC-03: Non-causal predicate does NOT fire WHY ─────────────────────────────

class TestTC03NonCausalExcluded:
    """'went' encodes motion, not causation — WH_WHY must NOT fire.

    VerbNet: go-51.1 (not in _CAUSAL_VN_CLASSES)
    WordNet: hypernym path of 'go' does not reach cause.v.01
    """

    def test_predicate_is_causal_went_false(self):
        """_predicate_is_causal('went') must return False."""
        assert not _predicate_is_causal("went"), (
            "_predicate_is_causal('went') returned True; "
            "expected False ('went' is motion, not causation)"
        )

    def test_applicable_wh_types_no_why(self):
        """_applicable_wh_types must NOT include WH_WHY for 'went'."""
        whs = _wh_types_set("GENERIC", "GENERIC", "went")
        assert WH_WHY not in whs, (
            f"WH_WHY must NOT fire for non-causal predicate 'went'. Got: {whs}"
        )

    def test_generate_pq_no_why_question(self):
        """generate_predicted_queries must produce zero WHY-framed questions."""
        questions = _pq_questions("Maya", "went", "school")
        assert not _any_why(questions), (
            f"WHY question must not be generated for (Maya, went, school). "
            f"Got questions: {questions}"
        )

    def test_what_still_fires(self):
        """WHAT is always present — confirms output is non-empty, just lacks WHY."""
        whs = _wh_types_set("GENERIC", "GENERIC", "went")
        assert WH_WHAT in whs, f"WH_WHAT must always fire. Got: {whs}"


# ── TC-04: Consequence predicate 'resulted' correctly excluded ────────────────

class TestTC04ConsequenceExcluded:
    """'resulted' encodes the patient/consequence perspective (not agent causation).

    The engineer's semantic decision: resulted -> False (consequence side).
    VerbNet: appear-48.1.1 (not in _CAUSAL_VN_CLASSES)
    WordNet: 'result' hypernym path does not reach cause.v.01
    This is the critical asymmetry: cause(X,Y) fires WHY, result(Y,X) does not.
    """

    def test_predicate_is_causal_resulted_false(self):
        """_predicate_is_causal('resulted') must return False."""
        assert not _predicate_is_causal("resulted"), (
            "_predicate_is_causal('resulted') returned True; "
            "expected False ('resulted' is consequence side, not causal agent)"
        )

    def test_applicable_wh_types_no_why(self):
        """_applicable_wh_types must NOT include WH_WHY for 'resulted'."""
        whs = _wh_types_set("GENERIC", "GENERIC", "resulted")
        assert WH_WHY not in whs, (
            f"WH_WHY must NOT fire for consequence predicate 'resulted'. Got: {whs}"
        )

    def test_generate_pq_no_why_question(self):
        """generate_predicted_queries must produce zero WHY-framed questions."""
        questions = _pq_questions("the_famine", "resulted", "drought")
        assert not _any_why(questions), (
            f"WHY question must not be generated for (the_famine, resulted, drought). "
            f"Got questions: {questions}"
        )

    def test_asymmetry_cause_vs_result(self):
        """Structural asymmetry: 'caused' fires WHY, 'resulted' does not.

        This encodes the agent-causation semantic: only the causing entity's
        perspective generates a WHY question, not the consequence entity's.
        """
        caused_whs = _wh_types_set("GENERIC", "GENERIC", "caused")
        resulted_whs = _wh_types_set("GENERIC", "GENERIC", "resulted")
        assert WH_WHY in caused_whs, (
            f"'caused' must include WHY: {caused_whs}"
        )
        assert WH_WHY not in resulted_whs, (
            f"'resulted' must NOT include WHY: {resulted_whs}"
        )


# ── TC-05: ADP predicate with causation fires both WHERE and WHY ──────────────

class TestTC05AdpCausalBothFire:
    """'led_to' carries two structural signals simultaneously:

    1. ADP tail ('to' is a preposition) -> WH_WHERE fires via _predicate_tail_is_adp
    2. VerbNet force-59 ('lead') -> WH_WHY fires via _predicate_is_causal

    Both paths are additive; neither suppresses the other.
    This validates that the WH_WHY addition is strictly additive with no
    regression to the existing WH_WHERE ADP path.
    """

    def test_predicate_is_causal_led_to(self):
        """_predicate_is_causal('led_to') must return True via VerbNet force-59."""
        assert _predicate_is_causal("led_to"), (
            "_predicate_is_causal('led_to') returned False; "
            "expected True via VerbNet force-59 (lead is in force-59)"
        )

    def test_why_fires_for_led_to_generic(self):
        """WH_WHY must fire for 'led_to' with GENERIC object."""
        whs = _wh_types_set("GENERIC", "GENERIC", "led_to")
        assert WH_WHY in whs, (
            f"WH_WHY must fire for 'led_to'. Got: {whs}"
        )

    def test_where_fires_for_led_to_generic(self):
        """WH_WHERE must fire for 'led_to' via ADP tail even with GENERIC object."""
        whs = _wh_types_set("GENERIC", "GENERIC", "led_to")
        assert WH_WHERE in whs, (
            f"WH_WHERE must fire for 'led_to' via ADP tail. Got: {whs}"
        )

    def test_both_where_and_why_fire_with_location_object(self):
        """With LOCATION object, both WH_WHERE (type path + ADP path) and WH_WHY fire."""
        whs = _wh_types_set("GENERIC", "LOCATION", "led_to")
        assert WH_WHERE in whs, (
            f"WH_WHERE must fire for LOCATION object. Got: {whs}"
        )
        assert WH_WHY in whs, (
            f"WH_WHY must fire for causal 'led_to' with LOCATION object. Got: {whs}"
        )

    def test_no_duplicate_where(self):
        """Even with LOCATION object + ADP tail, WH_WHERE appears exactly once."""
        wh_list = _applicable_wh_types("GENERIC", "LOCATION", "led_to")
        count = wh_list.count(WH_WHERE)
        assert count == 1, (
            f"WH_WHERE must appear exactly once even when multiple paths fire. "
            f"Got count={count}, wh_list={wh_list}"
        )

    def test_no_duplicate_why(self):
        """WH_WHY appears exactly once in the output list."""
        wh_list = _applicable_wh_types("GENERIC", "LOCATION", "led_to")
        count = wh_list.count(WH_WHY)
        assert count == 1, (
            f"WH_WHY must appear exactly once. Got count={count}, wh_list={wh_list}"
        )

    def test_generate_pq_produces_both_why_and_where_questions(self):
        """generate_predicted_queries must emit both WHY and WHERE-framed questions."""
        questions = _pq_questions(
            "the_investigation", "led_to", "the_arrest",
            subject_type="GENERIC", object_type="LOCATION"
        )
        has_why = _any_why(questions)
        has_where = _any_where(questions)
        assert has_why, (
            f"Expected a WHY question for (the_investigation, led_to, the_arrest). "
            f"Got: {questions}"
        )
        assert has_where, (
            f"Expected a WHERE question for (the_investigation, led_to, the_arrest). "
            f"Got: {questions}"
        )


# ── TC-06: Full regression — test_pq_adp_where.py must still pass 27/27 ──────

class TestTC06AdpWhereRegression:
    """Verify that the WH_WHY addition did not break any existing ADP-WHERE tests.

    Runs test_pq_adp_where.py as a subprocess and asserts 27 passed / 0 failed.
    This is the canonical regression gate for the additive WHY implementation.
    """

    def test_adp_where_suite_still_27_of_27(self):
        """py -3.10 tests/test_pq_adp_where.py must still pass 27/27."""
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/test_pq_adp_where.py",
                "-v", "--tb=short",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        output = result.stdout + result.stderr
        print("\n[TC-06 regression output]\n" + output)

        # Assert the process exited with 0 (all tests passed)
        assert result.returncode == 0, (
            f"TC-06 FAIL: test_pq_adp_where.py did not exit cleanly.\n"
            f"Return code: {result.returncode}\n"
            f"Output:\n{output}"
        )

        # Assert exactly 27 passed and 0 failed (explicit count verification)
        assert "27 passed" in output, (
            f"TC-06 FAIL: expected '27 passed' in pytest output.\n"
            f"Output:\n{output}"
        )
        assert "failed" not in output.lower() or "0 failed" in output, (
            f"TC-06 FAIL: unexpected failure in test_pq_adp_where.py.\n"
            f"Output:\n{output}"
        )
