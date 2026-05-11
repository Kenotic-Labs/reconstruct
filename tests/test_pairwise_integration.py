"""
PHASE 2: Pairwise Integration Tests - 50 Tests

Tests pairs of engines working together:
1. Intent Gate → Memory Engine (1-10)
2. Intent Gate → Retrieval Engine (11-20)
3. Memory Engine ↔ Retrieval Engine (21-30)
4. Temporal Engine → Memory Engine (31-40)
5. Intent Gate → Orchestrator Routing (41-50)

Run: python -m pytest tests/test_pairwise_integration.py -v
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestrator.intent_gate import (
    get_intent_gate,
    IntentResult,
    INTENT_GENERAL_KNOWLEDGE,
    INTENT_PERSONAL_STATE,
    INTENT_PAST_REFERENCE,
    INTENT_EXPLICIT_MEMORY,
)
from app.temporal.temporal_engine import get_temporal_engine
from app.orchestrator.orchestrator import OrchestratorV2, OrchestratorContext


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def intent_gate():
    """Get intent gate instance."""
    return get_intent_gate()


@pytest.fixture(scope="module")
def temporal_engine():
    """Get temporal engine instance."""
    return get_temporal_engine()


@pytest.fixture
def orchestrator():
    """Fresh orchestrator for each test."""
    return OrchestratorV2()


@pytest.fixture
def now():
    """Current datetime for consistent tests."""
    return datetime.now(timezone.utc)


# =============================================================================
# INTENT GATE → MEMORY ENGINE ROUTING (1-10)
# =============================================================================

def test_01_personal_state_requires_memory_write(intent_gate):
    """PERSONAL_STATE intent requires MemoryEngine.ingest_event."""
    result = intent_gate.classify("I like Ferrari")
    assert result.primary_intent == INTENT_PERSONAL_STATE
    assert result.should_call_engine("MemoryEngine.ingest_event")


def test_02_i_love_pizza_routes_to_memory(intent_gate):
    """'I love pizza' routes to memory write."""
    result = intent_gate.classify("I love pizza")
    assert result.primary_intent == INTENT_PERSONAL_STATE
    assert "MemoryEngine.ingest_event" in result.engines_required


def test_03_my_name_is_routes_to_memory(intent_gate):
    """'My name is X' routes to memory write."""
    result = intent_gate.classify("My name is Sam")
    assert result.primary_intent == INTENT_PERSONAL_STATE
    assert result.should_call_engine("MemoryEngine.ingest_event")


def test_04_i_work_as_routes_to_memory(intent_gate):
    """'I work as X' routes to memory write."""
    result = intent_gate.classify("I work as a developer")
    assert result.primary_intent == INTENT_PERSONAL_STATE
    assert result.should_call_engine("MemoryEngine.ingest_event")


def test_05_explicit_memory_requires_memory(intent_gate):
    """EXPLICIT_MEMORY_COMMAND requires MemoryEngine."""
    result = intent_gate.classify("Remember this: I have a meeting tomorrow")
    assert result.primary_intent == INTENT_EXPLICIT_MEMORY
    assert result.should_call_engine("MemoryEngine.ingest_event")


def test_06_general_knowledge_forbids_memory(intent_gate):
    """GENERAL_KNOWLEDGE forbids memory write."""
    result = intent_gate.classify("What is the capital of France?")
    assert result.primary_intent == INTENT_GENERAL_KNOWLEDGE
    assert not result.should_call_engine("MemoryEngine.ingest_event")


def test_07_i_feel_sad_routes_to_memory(intent_gate):
    """Emotional state routes to memory."""
    result = intent_gate.classify("I feel sad today")
    assert result.primary_intent == INTENT_PERSONAL_STATE
    assert result.should_call_engine("MemoryEngine.ingest_event")


def test_08_i_have_a_dog_routes_to_memory(intent_gate):
    """Personal fact routes to memory."""
    result = intent_gate.classify("I have a dog named Max")
    assert result.primary_intent == INTENT_PERSONAL_STATE
    assert result.should_call_engine("MemoryEngine.ingest_event")


def test_09_my_favorite_color_routes_to_memory(intent_gate):
    """Preference statement routes to memory."""
    result = intent_gate.classify("My favorite color is blue")
    assert result.primary_intent == INTENT_PERSONAL_STATE
    assert result.should_call_engine("MemoryEngine.ingest_event")


def test_10_i_prefer_coffee_routes_to_memory(intent_gate):
    """Preference comparison routes to memory."""
    result = intent_gate.classify("I prefer coffee over tea")
    assert result.primary_intent == INTENT_PERSONAL_STATE
    assert result.should_call_engine("MemoryEngine.ingest_event")


# =============================================================================
# INTENT GATE → RETRIEVAL ENGINE ROUTING (11-20)
# =============================================================================

def test_11_past_reference_requires_retrieval(intent_gate):
    """PAST_SELF_REFERENCE requires RetrievalEngine."""
    result = intent_gate.classify("What car do I like?")
    assert result.primary_intent == INTENT_PAST_REFERENCE
    assert result.should_call_engine("RetrievalEngine.retrieve")


def test_12_what_is_my_name_routes_to_retrieval(intent_gate):
    """'What is my name?' routes to retrieval."""
    result = intent_gate.classify("What is my name?")
    assert result.primary_intent == INTENT_PAST_REFERENCE
    assert result.should_call_engine("RetrievalEngine.retrieve")


def test_13_what_do_i_like_routes_to_retrieval(intent_gate):
    """'What do I like?' routes to retrieval."""
    result = intent_gate.classify("What do I like?")
    assert result.primary_intent == INTENT_PAST_REFERENCE
    assert result.should_call_engine("RetrievalEngine.retrieve")


def test_14_where_do_i_live_routes_to_retrieval(intent_gate):
    """'Where do I live?' routes to retrieval."""
    result = intent_gate.classify("Where do I live?")
    assert result.primary_intent == INTENT_PAST_REFERENCE
    assert result.should_call_engine("RetrievalEngine.retrieve")


def test_15_what_did_i_say_routes_to_retrieval(intent_gate):
    """'What did I say?' routes to retrieval."""
    result = intent_gate.classify("What did I say about that?")
    assert result.primary_intent == INTENT_PAST_REFERENCE
    assert result.should_call_engine("RetrievalEngine.retrieve")


def test_16_general_knowledge_forbids_retrieval(intent_gate):
    """GENERAL_KNOWLEDGE forbids retrieval."""
    result = intent_gate.classify("How does photosynthesis work?")
    assert result.primary_intent == INTENT_GENERAL_KNOWLEDGE
    assert not result.should_call_engine("RetrievalEngine.retrieve")


def test_17_past_reference_forbids_memory_write(intent_gate):
    """PAST_SELF_REFERENCE forbids memory write."""
    result = intent_gate.classify("What is my job?")
    assert result.primary_intent == INTENT_PAST_REFERENCE
    assert not result.should_call_engine("MemoryEngine.ingest_event")


def test_18_when_is_my_birthday_routes_to_retrieval(intent_gate):
    """'When is my birthday?' routes to retrieval."""
    result = intent_gate.classify("When is my birthday?")
    assert result.primary_intent == INTENT_PAST_REFERENCE
    assert result.should_call_engine("RetrievalEngine.retrieve")


def test_19_what_are_my_hobbies_routes_to_retrieval(intent_gate):
    """'What are my hobbies?' routes to retrieval."""
    result = intent_gate.classify("What are my hobbies?")
    assert result.primary_intent == INTENT_PAST_REFERENCE
    assert result.should_call_engine("RetrievalEngine.retrieve")


def test_20_remind_me_routes_to_retrieval(intent_gate):
    """'Remind me what I said' routes to retrieval."""
    result = intent_gate.classify("Remind me what I said about the project")
    assert result.primary_intent == INTENT_PAST_REFERENCE
    assert result.should_call_engine("RetrievalEngine.retrieve")


# =============================================================================
# MEMORY ENGINE ↔ RETRIEVAL ENGINE ROUTING (21-30)
# =============================================================================

def test_21_store_then_query_pattern(intent_gate):
    """Store intent differs from query intent."""
    store_result = intent_gate.classify("I like Ferrari")
    query_result = intent_gate.classify("What car do I like?")

    assert store_result.primary_intent == INTENT_PERSONAL_STATE
    assert query_result.primary_intent == INTENT_PAST_REFERENCE

    # Store should write, query should read
    assert store_result.should_call_engine("MemoryEngine.ingest_event")
    assert query_result.should_call_engine("RetrievalEngine.retrieve")


def test_22_personal_state_optional_retrieval(intent_gate):
    """PERSONAL_STATE has optional retrieval."""
    result = intent_gate.classify("I like Ferrari")
    assert result.primary_intent == INTENT_PERSONAL_STATE
    assert "RetrievalEngine.retrieve" in result.engines_optional


def test_23_my_name_vs_whats_my_name(intent_gate):
    """'My name is X' vs 'What's my name?' have different intents."""
    statement = intent_gate.classify("My name is Sam")
    question = intent_gate.classify("What's my name?")

    assert statement.primary_intent == INTENT_PERSONAL_STATE
    assert question.primary_intent == INTENT_PAST_REFERENCE


def test_24_i_live_vs_where_do_i_live(intent_gate):
    """'I live in X' vs 'Where do I live?' have different intents."""
    statement = intent_gate.classify("I live in New York")
    question = intent_gate.classify("Where do I live?")

    assert statement.primary_intent == INTENT_PERSONAL_STATE
    assert question.primary_intent == INTENT_PAST_REFERENCE


def test_25_my_birthday_vs_when_birthday(intent_gate):
    """'My birthday is X' vs 'When is my birthday?' differ."""
    statement = intent_gate.classify("My birthday is March 15th")
    question = intent_gate.classify("When is my birthday?")

    assert statement.primary_intent == INTENT_PERSONAL_STATE
    assert question.primary_intent == INTENT_PAST_REFERENCE


def test_26_memory_write_has_high_confidence(intent_gate):
    """Memory write intents have high confidence."""
    result = intent_gate.classify("I like pizza")
    assert result.confidence >= 0.8


def test_27_retrieval_has_high_confidence(intent_gate):
    """Retrieval intents have high confidence."""
    result = intent_gate.classify("What do I like?")
    assert result.confidence >= 0.8


def test_28_statement_vs_question_disambiguation(intent_gate):
    """Statements and questions are correctly disambiguated."""
    # Statement about preference
    statement = intent_gate.classify("I prefer dark chocolate")
    # Question about preference
    question = intent_gate.classify("What chocolate do I prefer?")

    # Both should be high confidence
    assert statement.confidence >= 0.7
    assert question.confidence >= 0.7

    # But different intents
    assert statement.primary_intent != question.primary_intent


def test_29_explicit_memory_vs_implicit(intent_gate):
    """'Remember this' explicit vs implicit preference."""
    explicit = intent_gate.classify("Remember this: I love sushi")
    implicit = intent_gate.classify("I love sushi")

    assert explicit.primary_intent == INTENT_EXPLICIT_MEMORY
    assert implicit.primary_intent == INTENT_PERSONAL_STATE


def test_30_both_require_memory_different_reasons(intent_gate):
    """Both explicit and implicit memory need memory engine."""
    explicit = intent_gate.classify("Remember this: I love sushi")
    implicit = intent_gate.classify("I love sushi")

    # Both should call memory engine
    assert explicit.should_call_engine("MemoryEngine.ingest_event")
    assert implicit.should_call_engine("MemoryEngine.ingest_event")


# =============================================================================
# TEMPORAL ENGINE → MEMORY ENGINE INTEGRATION (31-40)
# =============================================================================

def test_31_temporal_tags_include_day_of_week(temporal_engine, now):
    """Temporal tags have day_of_week for memory storage."""
    tags = temporal_engine.generate_temporal_tags(now)
    assert "day_of_week" in tags
    assert 0 <= tags["day_of_week"] <= 6


def test_32_temporal_tags_include_hour(temporal_engine, now):
    """Temporal tags have hour_of_day for memory storage."""
    tags = temporal_engine.generate_temporal_tags(now)
    assert "hour_of_day" in tags
    assert 0 <= tags["hour_of_day"] <= 23


def test_33_temporal_tags_include_season(temporal_engine, now):
    """Temporal tags have season for memory storage."""
    tags = temporal_engine.generate_temporal_tags(now)
    assert "season" in tags
    assert tags["season"] in ["spring", "summer", "fall", "winter"]


def test_34_temporal_tags_include_date(temporal_engine, now):
    """Temporal tags have ISO date for memory storage."""
    tags = temporal_engine.generate_temporal_tags(now)
    assert "date" in tags
    assert len(tags["date"]) == 10  # YYYY-MM-DD


def test_35_temporal_tags_include_weekend_flag(temporal_engine, now):
    """Temporal tags have is_weekend flag."""
    tags = temporal_engine.generate_temporal_tags(now)
    assert "is_weekend" in tags
    assert isinstance(tags["is_weekend"], bool)


def test_36_temporal_parse_for_retrieval_window(temporal_engine, now):
    """Temporal parse provides retrieval window."""
    result = temporal_engine.parse("What happened yesterday", now)
    assert result.retrieval_window_days is not None
    assert result.retrieval_window_days >= 1


def test_37_this_week_gives_7_day_window(temporal_engine, now):
    """'this week' gives 7 day retrieval window."""
    result = temporal_engine.parse("What happened this week", now)
    assert result.retrieval_window_days == 7


def test_38_last_month_gives_60_day_window(temporal_engine, now):
    """'last month' gives 60 day retrieval window."""
    result = temporal_engine.parse("What happened last month", now)
    assert result.retrieval_window_days == 60


def test_39_years_ago_disables_recency(temporal_engine, now):
    """'years ago' disables recency penalty."""
    result = temporal_engine.parse("What did I say years ago", now)
    assert result.disable_recency == True


def test_40_temporal_tags_for_milestone_extraction(temporal_engine, now):
    """Temporal engine extracts milestone dates."""
    date = temporal_engine.extract_milestone_date(
        "My dad passed on January 15, 2020",
        now
    )
    assert date == "2020-01-15"


# =============================================================================
# INTENT GATE → ORCHESTRATOR ROUTING (41-50)
# =============================================================================

def test_41_orchestrator_uses_intent_gate(orchestrator):
    """Orchestrator has intent gate."""
    assert orchestrator._intent_gate is not None


def test_42_orchestrator_routes_by_intent(orchestrator, now):
    """Orchestrator uses intent for routing decisions."""
    intent_result = IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.9,
        engines_required={"MemoryEngine.ingest_event"},
        engines_forbidden=set(),
    )

    result = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        intent_result,
        0.9
    )
    assert result == True


def test_43_orchestrator_respects_forbidden(orchestrator, now):
    """Orchestrator respects forbidden engines."""
    intent_result = IntentResult(
        primary_intent=INTENT_GENERAL_KNOWLEDGE,
        confidence=0.9,
        engines_required=set(),
        engines_forbidden={"MemoryEngine.ingest_event"},
    )

    result = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        intent_result,
        0.9
    )
    assert result == False


def test_44_orchestrator_context_stores_intent(now):
    """OrchestratorContext stores intent result."""
    intent_result = IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.85,
    )

    ctx = OrchestratorContext(
        user_id=1,
        user_input="I like Ferrari",
        session_id="test",
        now=now,
        intent_result=intent_result,
    )

    assert ctx.intent_result.primary_intent == INTENT_PERSONAL_STATE


def test_45_orchestrator_tracks_engines_called(now):
    """OrchestratorContext tracks engines called."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
    )
    ctx.engines_called.append("MemoryEngine.ingest_event")

    assert "MemoryEngine.ingest_event" in ctx.engines_called


def test_46_orchestrator_tracks_engines_blocked(now):
    """OrchestratorContext tracks engines blocked."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
    )
    ctx.engines_blocked.append("RetrievalEngine.retrieve")

    assert "RetrievalEngine.retrieve" in ctx.engines_blocked


def test_47_intent_confidence_affects_optional(orchestrator):
    """Optional engines depend on confidence."""
    intent_result = IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.9,
        engines_optional={"RetrievalEngine.retrieve"},
    )

    high_conf = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        intent_result,
        0.9
    )
    low_conf = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        intent_result,
        0.2
    )

    assert high_conf == True
    assert low_conf == False


def test_48_intent_gate_detection_method_tracked(intent_gate):
    """Intent gate tracks detection method."""
    result = intent_gate.classify("I like pizza")
    assert result.detection_method in ["fast_path", "semantic", "regex", "fallback"]


def test_49_fast_path_has_high_confidence(intent_gate):
    """Fast path detection has high confidence."""
    result = intent_gate.classify("I like Ferrari")
    if result.detection_method == "fast_path":
        assert result.confidence >= 0.8


def test_50_intent_routing_end_to_end(intent_gate, temporal_engine, now):
    """End-to-end: intent → temporal → routing works."""
    # 1. Classify intent
    intent = intent_gate.classify("I like Ferrari")
    assert intent.primary_intent == INTENT_PERSONAL_STATE

    # 2. Get temporal context
    tags = temporal_engine.generate_temporal_tags(now)
    assert "day_of_week" in tags

    # 3. Verify routing
    assert intent.should_call_engine("MemoryEngine.ingest_event")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
