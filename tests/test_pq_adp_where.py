# -*- coding: utf-8 -*-
"""
test_pq_adp_where.py — Unit tests for the ADP-tail WH_WHERE structural path.

Task 1: _predicate_tail_is_adp fires WH_WHERE for locative predicates
regardless of what object_type was assigned by type_resolver.

Laws: STRUCTURAL | NO WORD LISTS | ADDITIVE | NO SCORES | NO RANGES

Run:
    py -3.10 -m pytest tests/test_pq_adp_where.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engines.predicted_queries import (
    _predicate_tail_is_adp,
    _applicable_wh_types,
    WH_WHERE,
    WH_WHAT,
)


# ── Unit: _predicate_tail_is_adp ────────────────────────────────────────────

class TestPredicateTailIsAdp:
    """_predicate_tail_is_adp must return True iff the predicate's last token
    is tagged ADP by spaCy. No word lists — pure POS structural detection."""

    @pytest.mark.parametrize("predicate", [
        "stayed_at",
        "based_in",
        "works_at",
        "located_near",
        "led_to",
        "resulted_in",
        "lives_in",
        "moved_to",
        "came_from",
    ])
    def test_adp_tail_predicates_return_true(self, predicate):
        assert _predicate_tail_is_adp(predicate), (
            f"Expected _predicate_tail_is_adp({predicate!r}) = True "
            f"(last token is ADP), got False"
        )

    @pytest.mark.parametrize("predicate", [
        "visited",
        "is",
        "married",
        "twisted",
        "owns",
        "runs",
        "completed",
    ])
    def test_non_adp_tail_predicates_return_false(self, predicate):
        assert not _predicate_tail_is_adp(predicate), (
            f"Expected _predicate_tail_is_adp({predicate!r}) = False "
            f"(last token is not ADP), got True"
        )

    def test_empty_predicate_returns_false(self):
        assert not _predicate_tail_is_adp("")
        assert not _predicate_tail_is_adp("  ")

    def test_result_is_cached(self):
        """Second call on same predicate must return same result (cache hit)."""
        from app.engines.predicted_queries import _PREDICATE_LOCATIVE_CACHE
        pred = "stayed_at"
        _PREDICATE_LOCATIVE_CACHE.pop(pred.lower(), None)
        first = _predicate_tail_is_adp(pred)
        assert pred.lower() in _PREDICATE_LOCATIVE_CACHE
        second = _predicate_tail_is_adp(pred)
        assert first == second


# ── Integration: _applicable_wh_types fires WH_WHERE via ADP path ───────────

class TestApplicableWhTypesAdpPath:
    """WH_WHERE must fire for ADP-tail predicates regardless of object_type.

    The bug: (user, stayed_at, Moraine Inn) with object_type=PERSON never
    produced a WHERE question. The fix: add ADP-tail as a second structural
    path to WH_WHERE eligibility.
    """

    def test_stayed_at_with_person_object_fires_where(self):
        """Core regression: stayed_at + PERSON object -> WHERE must fire."""
        wh_types = _applicable_wh_types("PERSON", "PERSON", "stayed_at")
        assert WH_WHERE in wh_types, (
            f"WH_WHERE must fire for 'stayed_at' with PERSON object. "
            f"Got: {wh_types}"
        )

    def test_stayed_at_with_generic_object_fires_where(self):
        wh_types = _applicable_wh_types("PERSON", "GENERIC", "stayed_at")
        assert WH_WHERE in wh_types, (
            f"WH_WHERE must fire for 'stayed_at' with GENERIC object. "
            f"Got: {wh_types}"
        )

    def test_lives_in_with_person_object_fires_where(self):
        wh_types = _applicable_wh_types("PERSON", "PERSON", "lives_in")
        assert WH_WHERE in wh_types, f"Got: {wh_types}"

    def test_works_at_with_person_object_fires_where(self):
        wh_types = _applicable_wh_types("PERSON", "PERSON", "works_at")
        assert WH_WHERE in wh_types, f"Got: {wh_types}"

    def test_existing_location_path_still_fires(self):
        """Additive: LOCATION object still fires WHERE (original path unbroken)."""
        wh_types = _applicable_wh_types("PERSON", "LOCATION", "visited")
        assert WH_WHERE in wh_types, (
            f"Original LOCATION path must still fire WHERE. Got: {wh_types}"
        )

    def test_existing_org_path_still_fires(self):
        """Additive: ORG object still fires WHERE (original path unbroken)."""
        wh_types = _applicable_wh_types("PERSON", "ORG", "works_at")
        assert WH_WHERE in wh_types, (
            f"Original ORG path must still fire WHERE. Got: {wh_types}"
        )

    def test_non_locative_verb_with_generic_object_does_not_fire_where(self):
        """Non-ADP predicate + non-locative object type -> WHERE must NOT fire."""
        wh_types = _applicable_wh_types("PERSON", "GENERIC", "twisted")
        assert WH_WHERE not in wh_types, (
            f"WHERE must not fire for non-locative predicate 'twisted' "
            f"with GENERIC object. Got: {wh_types}"
        )

    def test_what_always_fires(self):
        """WHAT is always present regardless of ADP path. Regression guard."""
        wh_types = _applicable_wh_types("PERSON", "PERSON", "stayed_at")
        assert WH_WHAT in wh_types, f"WHAT must always fire. Got: {wh_types}"

    def test_no_duplicates_in_output(self):
        """works_at fires WHERE via both ORG path and ADP path — must dedup."""
        wh_types = _applicable_wh_types("PERSON", "ORG", "works_at")
        assert wh_types.count(WH_WHERE) == 1, (
            f"WH_WHERE must appear exactly once even when both paths fire. "
            f"Got: {wh_types}"
        )
