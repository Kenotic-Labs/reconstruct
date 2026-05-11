"""
PHASE 1, STEP 4: Temporal Engine - 50 Isolated Tests

Tests temporal parsing, tag generation, and retrieval integration.
Must pass 50/50 before proceeding to Step 5.

Run: python -m pytest tests/test_temporal_engine.py -v
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.temporal.temporal_engine import (
    get_temporal_engine,
    TemporalEngineV2,
    TemporalResult,
    TemporalGranularity,
    temporal_tags_from_dt,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def engine():
    """Get temporal engine instance."""
    return get_temporal_engine()


@pytest.fixture
def now():
    """Fixed datetime for consistent tests."""
    return datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)  # Saturday afternoon


# =============================================================================
# FAST PATH TESTS (1-10) - Common phrases
# =============================================================================

def test_01_today_parses_correctly(engine, now):
    """'today' returns day granularity with direction=0."""
    result = engine.parse("today", now)
    assert result.granularity == TemporalGranularity.DAY
    assert result.direction == 0
    assert result.confidence >= 0.8


def test_02_yesterday_parses_correctly(engine, now):
    """'yesterday' returns past direction."""
    result = engine.parse("yesterday", now)
    assert result.granularity == TemporalGranularity.DAY
    assert result.direction == -1
    assert result.confidence >= 0.8


def test_03_tomorrow_parses_correctly(engine, now):
    """'tomorrow' returns future direction."""
    result = engine.parse("tomorrow", now)
    assert result.granularity == TemporalGranularity.DAY
    assert result.direction == 1
    assert result.confidence >= 0.8


def test_04_this_week_parses_correctly(engine, now):
    """'this week' returns week granularity."""
    result = engine.parse("this week", now)
    assert result.granularity == TemporalGranularity.WEEK
    assert result.direction == 0
    assert result.retrieval_window_days == 7


def test_05_last_week_parses_correctly(engine, now):
    """'last week' returns past week."""
    result = engine.parse("last week", now)
    assert result.granularity == TemporalGranularity.WEEK
    assert result.direction == -1
    assert result.retrieval_window_days == 14


def test_06_next_week_parses_correctly(engine, now):
    """'next week' returns future week."""
    result = engine.parse("next week", now)
    assert result.granularity == TemporalGranularity.WEEK
    assert result.direction == 1


def test_07_this_month_parses_correctly(engine, now):
    """'this month' returns month granularity."""
    result = engine.parse("this month", now)
    assert result.granularity == TemporalGranularity.MONTH
    assert result.retrieval_window_days == 30


def test_08_last_month_parses_correctly(engine, now):
    """'last month' returns past month."""
    result = engine.parse("last month", now)
    assert result.granularity == TemporalGranularity.MONTH
    assert result.direction == -1


def test_09_now_parses_correctly(engine, now):
    """'now' returns second granularity."""
    result = engine.parse("now", now)
    assert result.granularity == TemporalGranularity.SECOND
    assert result.direction == 0


def test_10_a_while_ago_disables_recency(engine, now):
    """'a while ago' sets disable_recency=True."""
    result = engine.parse("a while ago", now)
    assert result.disable_recency == True
    assert result.direction == -1


# =============================================================================
# TEMPORAL PARSING TESTS (11-20) - Direction and granularity
# =============================================================================

def test_11_2_days_ago_parses_correctly(engine, now):
    """'2 days ago' returns correct past direction."""
    result = engine.parse("2 days ago", now)
    assert result.direction == -1
    assert "days" in result.concept_matched or result.concept_matched == "recently"
    assert result.disable_recency == True  # Past references disable recency


def test_12_3_weeks_ago_parses_correctly(engine, now):
    """'3 weeks ago' returns correct past direction."""
    result = engine.parse("3 weeks ago", now)
    assert result.direction == -1
    assert "weeks" in result.concept_matched or result.disable_recency == True


def test_13_6_months_ago_parses_correctly(engine, now):
    """'6 months ago' returns correct past direction."""
    result = engine.parse("6 months ago", now)
    assert result.direction == -1
    assert "months" in result.concept_matched or result.disable_recency == True


def test_14_years_ago_disables_recency(engine, now):
    """'years ago' should disable recency penalty."""
    result = engine.parse("years ago", now)
    assert result.disable_recency == True


def test_15_last_year_parses_correctly(engine, now):
    """'last year' returns past year."""
    result = engine.parse("last year", now)
    assert result.granularity == TemporalGranularity.YEAR
    assert result.direction == -1


def test_16_text_with_tomorrow_in_sentence(engine, now):
    """Sentence containing 'tomorrow' is detected."""
    result = engine.parse("I have a meeting tomorrow", now)
    assert result.direction == 1
    assert result.granularity == TemporalGranularity.DAY


def test_17_text_with_yesterday_in_sentence(engine, now):
    """Sentence containing 'yesterday' is detected."""
    result = engine.parse("I talked to John yesterday", now)
    assert result.direction == -1
    assert result.granularity == TemporalGranularity.DAY


def test_18_text_with_this_week_in_sentence(engine, now):
    """Sentence containing 'this week' is detected."""
    result = engine.parse("What happened this week?", now)
    assert result.granularity == TemporalGranularity.WEEK


def test_19_text_with_long_time_ago(engine, now):
    """'long time ago' sets disable_recency."""
    result = engine.parse("That was a long time ago", now)
    assert result.disable_recency == True


def test_20_back_then_disables_recency(engine, now):
    """'back then' sets disable_recency."""
    result = engine.parse("Back then, things were different", now)
    assert result.disable_recency == True


# =============================================================================
# TEMPORAL TAGS TESTS (21-28) - Tag generation
# =============================================================================

def test_21_tags_include_day_of_week(engine, now):
    """Temporal tags include day_of_week."""
    tags = engine.generate_temporal_tags(now)
    assert "day_of_week" in tags
    assert tags["day_of_week"] == 5  # Saturday


def test_22_tags_include_hour_of_day(engine, now):
    """Temporal tags include hour_of_day."""
    tags = engine.generate_temporal_tags(now)
    assert "hour_of_day" in tags
    assert tags["hour_of_day"] == 14


def test_23_tags_include_season(engine, now):
    """Temporal tags include season."""
    tags = engine.generate_temporal_tags(now)
    assert "season" in tags
    assert tags["season"] == "summer"  # June


def test_24_tags_include_date(engine, now):
    """Temporal tags include ISO date."""
    tags = engine.generate_temporal_tags(now)
    assert "date" in tags
    assert tags["date"] == "2024-06-15"


def test_25_tags_include_is_weekend(engine, now):
    """Temporal tags include is_weekend."""
    tags = engine.generate_temporal_tags(now)
    assert "is_weekend" in tags
    assert tags["is_weekend"] == True  # Saturday


def test_26_tags_include_time_of_day(engine, now):
    """Temporal tags include time_of_day."""
    tags = engine.generate_temporal_tags(now)
    assert "time_of_day" in tags
    assert tags["time_of_day"] == "afternoon"  # 14:30


def test_27_tags_winter_season():
    """December is winter."""
    engine = get_temporal_engine()
    dt = datetime(2024, 12, 25, 10, 0, 0, tzinfo=timezone.utc)
    tags = engine.generate_temporal_tags(dt)
    assert tags["season"] == "winter"


def test_28_tags_morning_time_of_day():
    """9am is morning."""
    engine = get_temporal_engine()
    dt = datetime(2024, 6, 15, 9, 0, 0, tzinfo=timezone.utc)
    tags = engine.generate_temporal_tags(dt)
    assert tags["time_of_day"] == "morning"


# =============================================================================
# MILESTONE DATE EXTRACTION TESTS (29-35)
# =============================================================================

def test_29_extract_iso_date(engine, now):
    """Extract ISO format date."""
    result = engine.extract_milestone_date("It happened on 2020-01-15", now)
    assert result == "2020-01-15"


def test_30_extract_month_day_year_date(engine, now):
    """Extract 'January 15, 2020' format."""
    result = engine.extract_milestone_date("My dad passed on January 15, 2020", now)
    assert result == "2020-01-15"


def test_31_extract_day_month_year_date(engine, now):
    """Extract '15 January 2020' format."""
    result = engine.extract_milestone_date("The event was 15 January 2020", now)
    assert result == "2020-01-15"


def test_32_no_date_returns_none(engine, now):
    """No date in text returns None."""
    result = engine.extract_milestone_date("I like pizza", now)
    assert result is None


def test_33_extract_march_date(engine, now):
    """Extract March date."""
    result = engine.extract_milestone_date("I got married March 20, 2019", now)
    assert result == "2019-03-20"


def test_34_extract_december_date(engine, now):
    """Extract December date."""
    result = engine.extract_milestone_date("Born December 25, 1990", now)
    assert result == "1990-12-25"


def test_35_extract_date_with_on_prefix(engine, now):
    """Extract date with 'on' prefix."""
    result = engine.extract_milestone_date("He died on August 5, 2021", now)
    assert result == "2021-08-05"


# =============================================================================
# RETRIEVAL WINDOW TESTS (36-43)
# =============================================================================

def test_36_today_window_is_1_day(engine, now):
    """'today' has 1 day retrieval window."""
    result = engine.parse("What happened today", now)
    assert result.retrieval_window_days == 1


def test_37_this_week_window_is_7_days(engine, now):
    """'this week' has 7 day retrieval window."""
    result = engine.parse("Summary of this week", now)
    assert result.retrieval_window_days == 7


def test_38_last_month_window_is_60_days(engine, now):
    """'last month' has 60 day retrieval window."""
    result = engine.parse("What happened last month", now)
    assert result.retrieval_window_days == 60


def test_39_get_retrieval_window_api(engine, now):
    """get_retrieval_window API works."""
    days = engine.get_retrieval_window("this week", now)
    assert days == 7


def test_40_future_has_no_retrieval_window(engine, now):
    """Future reference has no retrieval window."""
    result = engine.parse("next week", now)
    assert result.retrieval_window_days is None


def test_41_yesterday_window_is_2_days(engine, now):
    """'yesterday' has 2 day retrieval window."""
    result = engine.parse("What we talked about yesterday", now)
    assert result.retrieval_window_days == 2


def test_42_should_disable_recency_api(engine):
    """should_disable_recency API works."""
    assert engine.should_disable_recency("years ago") == True
    assert engine.should_disable_recency("a while ago") == True


def test_43_recent_reference_keeps_recency(engine):
    """Recent reference keeps recency enabled."""
    result = engine.parse("today", datetime.now(timezone.utc))
    # Today shouldn't disable recency
    assert result.disable_recency == False or result.direction >= 0


# =============================================================================
# GRANULARITY TESTS (44-50)
# =============================================================================

def test_44_second_granularity(engine, now):
    """'now' has second granularity."""
    result = engine.parse("right now", now)
    assert result.granularity == TemporalGranularity.SECOND


def test_45_day_granularity(engine, now):
    """'today' has day granularity."""
    result = engine.parse("today", now)
    assert result.granularity == TemporalGranularity.DAY


def test_46_week_granularity(engine, now):
    """'this week' has week granularity."""
    result = engine.parse("this week", now)
    assert result.granularity == TemporalGranularity.WEEK


def test_47_month_granularity(engine, now):
    """'this month' has month granularity."""
    result = engine.parse("this month", now)
    assert result.granularity == TemporalGranularity.MONTH


def test_48_year_granularity(engine, now):
    """'this year' has year granularity."""
    result = engine.parse("this year", now)
    assert result.granularity == TemporalGranularity.YEAR


def test_49_temporal_tags_from_dt_function():
    """temporal_tags_from_dt function works."""
    dt = datetime(2024, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
    tags = temporal_tags_from_dt(dt)
    assert "day_of_week" in tags
    assert "season" in tags
    assert tags["season"] == "spring"


def test_50_result_has_timestamps(engine, now):
    """TemporalResult has start_ts and end_ts."""
    result = engine.parse("today", now)
    assert result.start_ts is not None
    assert result.end_ts is not None


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
