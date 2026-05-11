"""
PHASE 1, STEP 6: Adaptation Engine - 50 Isolated Tests

Tests user style adaptation, profile management, and signal detection.
Must pass 50/50 before proceeding to Step 7.

Run: python -m pytest tests/test_adaptation_engine.py -v
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import pytest

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adaptation.adaptation_engine import (
    AdaptationEngineV2,
    get_adaptation_engine,
    AdaptationProfile,
    AdaptationContext,
    AdaptationResult,
    FAST_PATH_SIGNALS,
    TIME_OF_DAY_MODIFIERS,
    DAY_MODIFIERS,
    DEFAULT_PROFILE,
)
from app.adaptation.adaptation_rules import AdaptationDelta


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def engine():
    """Fresh adaptation engine for each test."""
    return AdaptationEngineV2()


@pytest.fixture
def now():
    """Current datetime for consistent tests."""
    return datetime.now(timezone.utc)


# =============================================================================
# PROFILE TESTS (1-10)
# =============================================================================

def test_01_default_profile_values():
    """Default profile has expected values."""
    assert DEFAULT_PROFILE["warmth"] == 0.5
    assert DEFAULT_PROFILE["formality"] == 0.5
    assert DEFAULT_PROFILE["initiative"] == 0.5
    assert DEFAULT_PROFILE["check_in_frequency"] == 0.5


def test_02_adaptation_profile_defaults():
    """AdaptationProfile has expected defaults."""
    profile = AdaptationProfile(user_id=1)
    assert profile.warmth == 0.5
    assert profile.formality == 0.5
    assert profile.initiative == 0.5


def test_03_adaptation_profile_custom():
    """AdaptationProfile accepts custom values."""
    profile = AdaptationProfile(user_id=1, warmth=0.8, formality=0.3)
    assert profile.warmth == 0.8
    assert profile.formality == 0.3


def test_04_profile_to_dict():
    """Profile to_dict includes all fields."""
    profile = AdaptationProfile(user_id=1, warmth=0.7)
    d = profile.to_dict()
    assert "warmth" in d
    assert "formality" in d
    assert "initiative" in d
    assert "check_in_frequency" in d


def test_05_profile_extended_fields():
    """Extended profile has style fields."""
    profile = AdaptationProfile(
        user_id=1,
        preferred_style="direct",
        engagement_trend="increasing"
    )
    assert profile.preferred_style == "direct"
    assert profile.engagement_trend == "increasing"


def test_06_profile_emotional_baseline():
    """Profile tracks emotional baseline."""
    profile = AdaptationProfile(user_id=1, emotional_baseline="positive")
    assert profile.emotional_baseline == "positive"


def test_07_profile_updated_at():
    """Profile tracks update timestamp."""
    now = datetime.now(timezone.utc)
    profile = AdaptationProfile(user_id=1, updated_at=now)
    assert profile.updated_at == now


def test_08_adaptation_context_basic():
    """AdaptationContext stores basic info."""
    now = datetime.now(timezone.utc)
    ctx = AdaptationContext(user_id=1, now=now, user_text="hello")
    assert ctx.user_id == 1
    assert ctx.user_text == "hello"


def test_09_adaptation_context_temporal():
    """AdaptationContext stores temporal context."""
    now = datetime.now(timezone.utc)
    ctx = AdaptationContext(
        user_id=1,
        now=now,
        user_text="hello",
        temporal_context={"time_of_day": "morning"}
    )
    assert ctx.temporal_context["time_of_day"] == "morning"


def test_10_adaptation_context_memory():
    """AdaptationContext stores memory context."""
    now = datetime.now(timezone.utc)
    ctx = AdaptationContext(
        user_id=1,
        now=now,
        user_text="hello",
        memory_context={"memories": 5}
    )
    assert ctx.memory_context["memories"] == 5


# =============================================================================
# FAST PATH TESTS (11-20)
# =============================================================================

def test_11_fast_path_vulnerability_im_struggling(engine):
    """'i'm struggling' triggers warmth increase."""
    deltas = engine._check_fast_path("i'm struggling")
    assert deltas is not None
    assert deltas["warmth"] > 0


def test_12_fast_path_vulnerability_im_scared(engine):
    """'i'm scared' triggers warmth increase."""
    deltas = engine._check_fast_path("i'm scared")
    assert deltas is not None
    assert deltas["warmth"] > 0


def test_13_fast_path_gratitude_thank_you(engine):
    """'thank you' triggers initiative increase."""
    deltas = engine._check_fast_path("thank you")
    assert deltas is not None
    assert deltas["initiative"] > 0


def test_14_fast_path_style_get_to_point(engine):
    """'get to the point' decreases formality."""
    deltas = engine._check_fast_path("get to the point")
    assert deltas is not None
    assert deltas["formality"] < 0


def test_15_fast_path_style_be_professional(engine):
    """'be professional' increases formality."""
    deltas = engine._check_fast_path("be professional")
    assert deltas is not None
    assert deltas["formality"] > 0


def test_16_fast_path_engagement_tell_me_more(engine):
    """'tell me more' increases initiative."""
    deltas = engine._check_fast_path("tell me more")
    assert deltas is not None
    assert deltas["initiative"] > 0


def test_17_fast_path_low_engagement_ok(engine):
    """'ok' decreases initiative."""
    deltas = engine._check_fast_path("ok")
    assert deltas is not None
    assert deltas["initiative"] < 0


def test_18_fast_path_no_match(engine):
    """Random text returns None."""
    deltas = engine._check_fast_path("what is the weather")
    assert deltas is None


def test_19_fast_path_substring_match(engine):
    """Substring matching works."""
    deltas = engine._check_fast_path("I just want to say thank you so much for helping")
    assert deltas is not None
    assert deltas["initiative"] > 0


def test_20_fast_path_hopeless(engine):
    """'i feel hopeless' triggers high warmth."""
    deltas = engine._check_fast_path("i feel hopeless")
    assert deltas is not None
    assert deltas["warmth"] >= 0.15


# =============================================================================
# TEMPORAL MODIFIER TESTS (21-28)
# =============================================================================

def test_21_morning_modifier_exists():
    """Morning modifier exists."""
    assert "morning" in TIME_OF_DAY_MODIFIERS


def test_22_morning_increases_warmth():
    """Morning increases warmth."""
    mods = TIME_OF_DAY_MODIFIERS["morning"]
    assert mods.get("warmth", 0) > 0


def test_23_evening_modifier_exists():
    """Evening modifier exists."""
    assert "evening" in TIME_OF_DAY_MODIFIERS


def test_24_evening_decreases_initiative():
    """Evening decreases initiative."""
    mods = TIME_OF_DAY_MODIFIERS["evening"]
    assert mods.get("initiative", 0) < 0


def test_25_night_modifier_exists():
    """Night modifier exists."""
    assert "night" in TIME_OF_DAY_MODIFIERS


def test_26_weekend_modifier_exists():
    """Weekend modifier exists."""
    assert "weekend" in DAY_MODIFIERS


def test_27_weekend_less_formal():
    """Weekend decreases formality."""
    mods = DAY_MODIFIERS["weekend"]
    assert mods.get("formality", 0) < 0


def test_28_weekday_modifier_exists():
    """Weekday modifier exists."""
    assert "weekday" in DAY_MODIFIERS


# =============================================================================
# ADAPTATION DELTA TESTS (29-35)
# =============================================================================

def test_29_delta_defaults_to_zero():
    """AdaptationDelta defaults to zero."""
    delta = AdaptationDelta()
    assert delta.warmth == 0.0
    assert delta.formality == 0.0
    assert delta.initiative == 0.0


def test_30_delta_custom_values():
    """AdaptationDelta accepts custom values."""
    delta = AdaptationDelta(warmth=0.1, formality=-0.05)
    assert delta.warmth == 0.1
    assert delta.formality == -0.05


def test_31_delta_check_in_frequency():
    """AdaptationDelta has check_in_frequency."""
    delta = AdaptationDelta(check_in_frequency=0.1)
    assert delta.check_in_frequency == 0.1


def test_32_result_has_profile():
    """AdaptationResult has profile."""
    profile = AdaptationProfile(user_id=1)
    delta = AdaptationDelta()
    result = AdaptationResult(profile=profile, delta_applied=delta)
    assert result.profile.user_id == 1


def test_33_result_has_delta():
    """AdaptationResult has delta."""
    profile = AdaptationProfile(user_id=1)
    delta = AdaptationDelta(warmth=0.1)
    result = AdaptationResult(profile=profile, delta_applied=delta)
    assert result.delta_applied.warmth == 0.1


def test_34_result_signals_detected():
    """AdaptationResult has signals_detected dict."""
    profile = AdaptationProfile(user_id=1)
    delta = AdaptationDelta()
    result = AdaptationResult(
        profile=profile,
        delta_applied=delta,
        signals_detected={"vulnerability": True}
    )
    assert result.signals_detected["vulnerability"] == True


def test_35_result_temporal_modifiers():
    """AdaptationResult has temporal_modifiers dict."""
    profile = AdaptationProfile(user_id=1)
    delta = AdaptationDelta()
    result = AdaptationResult(
        profile=profile,
        delta_applied=delta,
        temporal_modifiers={"warmth": 0.02}
    )
    assert result.temporal_modifiers["warmth"] == 0.02


# =============================================================================
# TEXT HEURISTICS TESTS (36-42)
# =============================================================================

def test_36_long_text_increases_warmth(engine):
    """Long text (>200 chars) increases warmth."""
    long_text = "a" * 201
    deltas = engine._analyze_text_heuristics(long_text)
    assert deltas.get("warmth", 0) > 0


def test_37_short_text_decreases_initiative(engine):
    """Short text (<20 chars) decreases initiative."""
    deltas = engine._analyze_text_heuristics("hi")
    assert deltas.get("initiative", 0) < 0


def test_38_multiple_questions_increase_initiative(engine):
    """Multiple questions increase initiative."""
    deltas = engine._analyze_text_heuristics("What is this? How does it work? Why?")
    assert deltas.get("initiative", 0) > 0


def test_39_first_person_pronouns_increase_warmth(engine):
    """Many first-person pronouns increase warmth."""
    deltas = engine._analyze_text_heuristics("I think that I am doing well and my dog is happy with me")
    assert deltas.get("warmth", 0) > 0


def test_40_medium_text_no_change(engine):
    """Medium length text has minimal effect."""
    medium_text = "a" * 100
    deltas = engine._analyze_text_heuristics(medium_text)
    # Should have no length-based deltas
    assert abs(deltas.get("warmth", 0)) <= 0.05


def test_41_single_question_minimal_effect(engine):
    """Single question has minimal initiative effect."""
    deltas = engine._analyze_text_heuristics("What is this?")
    # Single question doesn't meet threshold
    assert deltas.get("initiative", 0) <= 0.03


def test_42_no_pronouns_no_warmth_boost(engine):
    """Text without first-person pronouns has no warmth boost."""
    deltas = engine._analyze_text_heuristics("The weather is nice today")
    # No first-person pronouns
    assert deltas.get("warmth", 0) <= 0.05


# =============================================================================
# FAST PATH SIGNALS TESTS (43-50)
# =============================================================================

def test_43_fast_path_signals_has_vulnerability():
    """FAST_PATH_SIGNALS has vulnerability patterns."""
    assert "i'm struggling" in FAST_PATH_SIGNALS
    assert "i feel hopeless" in FAST_PATH_SIGNALS


def test_44_fast_path_signals_has_gratitude():
    """FAST_PATH_SIGNALS has gratitude patterns."""
    assert "thank you" in FAST_PATH_SIGNALS
    assert "i appreciate it" in FAST_PATH_SIGNALS


def test_45_fast_path_signals_has_style():
    """FAST_PATH_SIGNALS has style patterns."""
    assert "get to the point" in FAST_PATH_SIGNALS
    assert "be professional" in FAST_PATH_SIGNALS


def test_46_fast_path_signals_has_engagement():
    """FAST_PATH_SIGNALS has engagement patterns."""
    assert "tell me more" in FAST_PATH_SIGNALS
    assert "ok" in FAST_PATH_SIGNALS


def test_47_youre_amazing_boosts_warmth(engine):
    """'you're amazing' boosts warmth."""
    deltas = engine._check_fast_path("you're amazing")
    assert deltas is not None
    assert deltas.get("warmth", 0) > 0


def test_48_no_need_formal_decreases_formality(engine):
    """'no need to be formal' decreases formality."""
    deltas = engine._check_fast_path("no need to be formal")
    assert deltas is not None
    assert deltas["formality"] < 0


def test_49_i_need_help_increases_initiative(engine):
    """'i need help' increases initiative."""
    deltas = engine._check_fast_path("i need help")
    assert deltas is not None
    assert deltas.get("initiative", 0) > 0


def test_50_whatever_decreases_initiative(engine):
    """'whatever' decreases initiative."""
    deltas = engine._check_fast_path("whatever")
    assert deltas is not None
    assert deltas["initiative"] < 0


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
