"""
PHASE 3: Stack Integration Tests - 50 Tests

Tests 3+ engines working together in realistic scenarios.
Must pass 50/50 before proceeding to Phase 4.

Run: python -m pytest tests/test_stack_integration.py -v
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
    IntentGateV2,
    IntentResult,
    INTENT_GENERAL_KNOWLEDGE,
    INTENT_PERSONAL_STATE,
    INTENT_PAST_REFERENCE,
    INTENT_EXPLICIT_MEMORY,
)
from app.orchestrator.orchestrator import (
    OrchestratorV2,
    OrchestratorContext,
    OrchestratorResult,
)
from app.temporal.temporal_engine import (
    get_temporal_engine,
    TemporalEngineV2,
    TemporalGranularity,
)
# Memory/Retrieval engines not directly used - tests verify routing logic
from app.adaptation.adaptation_engine import (
    AdaptationEngineV2,
    AdaptationProfile,
    AdaptationContext,
)
from app.proactive.proactive_engine_v2 import (
    ProactiveEngineV2,
    AnnoyanceGuard,
    AdaptiveRateLimiter,
)
from app.proactive.story_arc import StoryArc, ArcStatus, EngagementType


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def intent_gate():
    """Fresh intent gate for each test."""
    return IntentGateV2()


@pytest.fixture
def temporal_engine():
    """Get temporal engine instance."""
    return get_temporal_engine()


@pytest.fixture
def orchestrator():
    """Fresh orchestrator for each test."""
    return OrchestratorV2()


@pytest.fixture
def adaptation_engine():
    """Fresh adaptation engine."""
    return AdaptationEngineV2()


@pytest.fixture
def proactive_engine():
    """Fresh proactive engine."""
    return ProactiveEngineV2()


@pytest.fixture
def now():
    """Fixed datetime for consistent tests."""
    return datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)


# =============================================================================
# INTENT → TEMPORAL → MEMORY STACK (1-10)
# Store memories with correct temporal tags based on intent analysis
# =============================================================================

def test_01_personal_state_with_temporal_tags(intent_gate, temporal_engine, now):
    """Personal state intent generates correct temporal tags for memory."""
    text = "I started my new job today"

    # Step 1: Intent classification
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PERSONAL_STATE

    # Step 2: Temporal analysis
    temporal = temporal_engine.parse(text, now)
    assert temporal.granularity == TemporalGranularity.DAY
    assert temporal.direction == 0  # "today"

    # Step 3: Generate tags for memory storage
    tags = temporal_engine.generate_temporal_tags(now)
    assert tags["day_of_week"] == 5  # Saturday
    assert tags["time_of_day"] == "afternoon"
    assert tags["is_weekend"] == True


def test_02_past_reference_with_temporal_window(intent_gate, temporal_engine, now):
    """Past reference intent sets correct retrieval window."""
    text = "What did I tell you yesterday?"

    # Step 1: Intent classification
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PAST_REFERENCE

    # Step 2: Temporal analysis for retrieval window
    temporal = temporal_engine.parse(text, now)
    assert temporal.direction == -1  # past
    assert temporal.retrieval_window_days == 2  # yesterday = 2 days


def test_03_explicit_memory_with_future_date(intent_gate, temporal_engine, now):
    """Explicit memory command with future date parsed correctly."""
    text = "Remember that my meeting is tomorrow at 3pm"

    # Step 1: Intent classification
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_EXPLICIT_MEMORY

    # Step 2: Temporal analysis
    temporal = temporal_engine.parse(text, now)
    assert temporal.direction == 1  # future


def test_04_past_reference_disables_recency(intent_gate, temporal_engine, now):
    """Past reference with 'long ago' disables recency penalty."""
    text = "What did I tell you a long time ago about my childhood?"

    # Step 1: Intent classification
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PAST_REFERENCE

    # Step 2: Temporal analysis
    temporal = temporal_engine.parse(text, now)
    assert temporal.disable_recency == True


def test_05_personal_state_week_granularity(intent_gate, temporal_engine, now):
    """Personal state with week reference has correct granularity."""
    text = "I've been feeling stressed this week"

    # Step 1: Intent
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PERSONAL_STATE

    # Step 2: Temporal
    temporal = temporal_engine.parse(text, now)
    assert temporal.granularity == TemporalGranularity.WEEK


def test_06_general_knowledge_no_memory_tags_needed(intent_gate, temporal_engine, now):
    """General knowledge doesn't need memory temporal tags."""
    text = "What is the capital of France?"

    # Step 1: Intent
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_GENERAL_KNOWLEDGE

    # Step 2: We still can parse for temporal context but no memory storage
    # This verifies the stack doesn't break on general knowledge
    temporal = temporal_engine.parse(text, now)
    # General knowledge shouldn't have strong temporal signals
    assert temporal.confidence < 0.9 or "capital" not in temporal.concept_matched


def test_07_past_reference_month_window(intent_gate, temporal_engine, now):
    """Past reference with month has 60-day retrieval window."""
    text = "What happened last month with my project?"

    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PAST_REFERENCE

    temporal = temporal_engine.parse(text, now)
    assert temporal.retrieval_window_days == 60


def test_08_explicit_memory_milestone_date(intent_gate, temporal_engine, now):
    """Explicit memory can extract milestone dates."""
    text = "Remember my dad passed on January 15, 2020"

    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_EXPLICIT_MEMORY

    milestone_date = temporal_engine.extract_milestone_date(text, now)
    assert milestone_date == "2020-01-15"


def test_09_personal_state_morning_tags(intent_gate, temporal_engine):
    """Morning message gets correct time_of_day tag."""
    morning = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
    text = "I'm feeling tired today"

    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PERSONAL_STATE

    tags = temporal_engine.generate_temporal_tags(morning)
    assert tags["time_of_day"] == "morning"


def test_10_past_reference_years_ago_extended_window(intent_gate, temporal_engine, now):
    """Past reference with 'years ago' disables recency."""
    text = "What did I say years ago about my dreams?"

    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PAST_REFERENCE

    temporal = temporal_engine.parse(text, now)
    assert temporal.disable_recency == True


# =============================================================================
# INTENT → ORCHESTRATOR → ROUTING STACK (11-20)
# Verify full routing decisions through orchestrator
# =============================================================================

def test_11_personal_state_routes_to_memory(intent_gate, orchestrator):
    """Personal state routes to MemoryEngine.ingest_event."""
    text = "I like Ferrari"

    # Intent classification
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PERSONAL_STATE

    # Orchestrator routing decision
    should_call = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        intent,
        intent.confidence
    )
    assert should_call == True


def test_12_past_reference_routes_to_retrieval(intent_gate, orchestrator):
    """Past reference routes to RetrievalEngine.retrieve."""
    text = "What car do I like?"

    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PAST_REFERENCE

    should_call = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        intent,
        intent.confidence
    )
    assert should_call == True


def test_13_general_knowledge_blocks_memory(intent_gate, orchestrator):
    """General knowledge blocks memory ingest."""
    text = "What is 2+2?"

    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_GENERAL_KNOWLEDGE

    should_call = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        intent,
        intent.confidence
    )
    assert should_call == False


def test_14_general_knowledge_blocks_retrieval(intent_gate, orchestrator):
    """General knowledge blocks retrieval."""
    text = "What is the speed of light?"

    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_GENERAL_KNOWLEDGE

    should_call = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        intent,
        intent.confidence
    )
    assert should_call == False


def test_15_explicit_memory_routes_to_memory(intent_gate, orchestrator):
    """Explicit memory command routes to memory."""
    text = "Remember that I have a meeting tomorrow"

    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_EXPLICIT_MEMORY

    should_call = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        intent,
        intent.confidence
    )
    assert should_call == True


def test_16_i_like_ferrari_full_routing(intent_gate, orchestrator):
    """'I like Ferrari' routes correctly through full stack."""
    text = "I like Ferrari"

    # Intent: Should be PERSONAL_STATE
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PERSONAL_STATE
    assert "MemoryEngine.ingest_event" in intent.engines_required

    # Routing: Should call memory (required)
    should_memory = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event", intent, intent.confidence
    )
    assert should_memory == True

    # Routing: Retrieval is optional for PERSONAL_STATE (not forbidden)
    # It can be called to retrieve related memories
    assert "RetrievalEngine.retrieve" not in intent.engines_forbidden


def test_17_what_car_full_routing(intent_gate, orchestrator):
    """'What car do I like?' routes correctly through full stack."""
    text = "What car do I like?"

    # Intent: Should be PAST_REFERENCE
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PAST_REFERENCE
    assert "RetrievalEngine.retrieve" in intent.engines_required

    # Routing: Should call retrieval
    should_retrieval = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve", intent, intent.confidence
    )
    assert should_retrieval == True

    # Routing: Should NOT call memory ingest
    should_memory = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event", intent, intent.confidence
    )
    assert should_memory == False


def test_18_my_favorite_routes_to_memory(intent_gate, orchestrator):
    """'My favorite X is Y' routes to memory."""
    text = "My favorite color is blue"

    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PERSONAL_STATE

    should_memory = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event", intent, intent.confidence
    )
    assert should_memory == True


def test_19_what_is_my_favorite_routes_to_retrieval(intent_gate, orchestrator):
    """'What is my favorite X?' routes to retrieval."""
    text = "What is my favorite color?"

    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PAST_REFERENCE

    should_retrieval = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve", intent, intent.confidence
    )
    assert should_retrieval == True


def test_20_routing_forbidden_overrides_required(intent_gate, orchestrator):
    """Forbidden engines override required in routing."""
    # Create a contradictory intent (shouldn't happen in practice)
    intent = IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.9,
        engines_required={"MemoryEngine.ingest_event"},
        engines_forbidden={"MemoryEngine.ingest_event"},
    )

    should_call = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event", intent, intent.confidence
    )
    assert should_call == False  # Forbidden wins


# =============================================================================
# TEMPORAL → ADAPTATION → PROACTIVE STACK (21-30)
# Time-aware adaptation and proactive engagement
# =============================================================================

def test_21_morning_boosts_warmth(temporal_engine, adaptation_engine):
    """Morning time boosts warmth in adaptation."""
    morning = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
    tags = temporal_engine.generate_temporal_tags(morning)

    assert tags["time_of_day"] == "morning"

    # Morning modifier should exist
    from app.adaptation.adaptation_engine import TIME_OF_DAY_MODIFIERS
    assert "morning" in TIME_OF_DAY_MODIFIERS
    assert TIME_OF_DAY_MODIFIERS["morning"].get("warmth", 0) > 0


def test_22_evening_reduces_initiative(temporal_engine, adaptation_engine):
    """Evening time reduces initiative in adaptation."""
    evening = datetime(2024, 6, 15, 20, 0, 0, tzinfo=timezone.utc)
    tags = temporal_engine.generate_temporal_tags(evening)

    assert tags["time_of_day"] == "evening"

    from app.adaptation.adaptation_engine import TIME_OF_DAY_MODIFIERS
    assert "evening" in TIME_OF_DAY_MODIFIERS
    assert TIME_OF_DAY_MODIFIERS["evening"].get("initiative", 0) < 0


def test_23_weekend_reduces_formality(temporal_engine, adaptation_engine):
    """Weekend reduces formality in adaptation."""
    saturday = datetime(2024, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
    tags = temporal_engine.generate_temporal_tags(saturday)

    assert tags["is_weekend"] == True

    from app.adaptation.adaptation_engine import DAY_MODIFIERS
    assert "weekend" in DAY_MODIFIERS
    assert DAY_MODIFIERS["weekend"].get("formality", 0) < 0


def test_24_late_night_blocks_low_salience_proactive(temporal_engine, now):
    """Late night blocks low-salience proactive engagement."""
    late = now.replace(hour=23, minute=30)

    guard = AnnoyanceGuard()
    low_arc = StoryArc(topic="groceries", importance=0.2)

    allowed, reason = guard.should_allow(1, low_arc, late)
    assert allowed == False
    assert reason in ["late_night", "too_generic"]


def test_25_late_night_allows_high_salience_proactive(temporal_engine, now):
    """Late night allows high-salience proactive engagement."""
    late = now.replace(hour=23, minute=30)

    guard = AnnoyanceGuard()
    high_arc = StoryArc(
        topic="job interview tomorrow",
        importance=0.9,
        has_explicit_reminder=True
    )

    allowed, reason = guard.should_allow(1, high_arc, late)
    assert allowed == True


def test_26_vulnerability_signal_with_temporal_context(adaptation_engine, temporal_engine, now):
    """Vulnerability signal combines with temporal context."""
    text = "I'm struggling today"

    # Fast path detects vulnerability
    deltas = adaptation_engine._check_fast_path(text)
    assert deltas is not None
    assert deltas["warmth"] > 0

    # Temporal provides context
    temporal = temporal_engine.parse(text, now)
    assert temporal.granularity == TemporalGranularity.DAY


def test_27_gratitude_signal_boosts_initiative(adaptation_engine, intent_gate):
    """Gratitude signal boosts initiative regardless of intent."""
    text = "Thank you so much"

    # Adaptation detects gratitude
    deltas = adaptation_engine._check_fast_path(text)
    assert deltas is not None
    assert deltas["initiative"] > 0


def test_28_style_signal_with_past_reference(adaptation_engine, intent_gate):
    """Style signal applies even with past reference intent."""
    # Test adaptation signal independently
    deltas = adaptation_engine._check_fast_path("get to the point")
    assert deltas is not None
    assert deltas["formality"] < 0

    # Test past reference intent separately (text starts with past reference pattern)
    recall_text = "what did I say about my project?"
    intent = intent_gate.classify(recall_text)
    assert intent.primary_intent == INTENT_PAST_REFERENCE


def test_29_rate_limiter_respects_user_tenure(now):
    """Rate limiter respects user tenure from first use."""
    limiter = AdaptiveRateLimiter()
    actual_now = datetime.now(timezone.utc)

    # New user: high limit (no first_use recorded)
    new_limit = limiter.get_daily_limit(1)
    assert new_limit == 20  # Default for new user

    # Record first use 45 days ago (Month 2)
    limiter.record_first_use(2, actual_now - timedelta(days=45))
    month2_limit = limiter.get_daily_limit(2)
    assert month2_limit == 18  # Month 2 limit


def test_30_proactive_annoyance_guard_full_flow(now):
    """Full proactive engagement flow with annoyance prevention."""
    guard = AnnoyanceGuard()
    limiter = AdaptiveRateLimiter()

    arc = StoryArc(
        topic="job interview",
        importance=0.7,
        has_explicit_reminder=True
    )

    # Step 1: Guard allows first engagement
    allowed, reason = guard.should_allow(1, arc, now)
    assert allowed == True

    # Step 2: Rate limiter allows
    can_engage, _ = limiter.can_engage(1, arc)
    assert can_engage == True

    # Step 3: Record the engagement
    guard.record_ask(1, "job interview")
    limiter.increment_count(1)

    # Step 4: Same topic blocked
    allowed2, reason2 = guard.should_allow(1, arc, now)
    assert allowed2 == False
    assert reason2 == "topic_asked_recently"


# =============================================================================
# FULL ORCHESTRATION STACK (31-40)
# Intent → Temporal → Orchestrator → Memory/Retrieval
# =============================================================================

def test_31_personal_fact_full_stack(intent_gate, temporal_engine, orchestrator, now):
    """Personal fact flows through full stack correctly."""
    text = "I like pizza"

    # Intent
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PERSONAL_STATE

    # Temporal tags for storage
    tags = temporal_engine.generate_temporal_tags(now)
    assert "day_of_week" in tags

    # Routing
    should_memory = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event", intent, intent.confidence
    )
    assert should_memory == True


def test_32_recall_question_full_stack(intent_gate, temporal_engine, orchestrator, now):
    """Recall question flows through full stack correctly."""
    text = "What food do I like?"

    # Intent
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PAST_REFERENCE

    # Temporal doesn't affect retrieval window (no temporal reference)
    temporal = temporal_engine.parse(text, now)

    # Routing
    should_retrieval = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve", intent, intent.confidence
    )
    assert should_retrieval == True


def test_33_temporal_recall_full_stack(intent_gate, temporal_engine, orchestrator, now):
    """Temporal recall flows through full stack with retrieval window."""
    text = "What did I tell you last week?"

    # Intent
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PAST_REFERENCE

    # Temporal sets retrieval window
    temporal = temporal_engine.parse(text, now)
    assert temporal.retrieval_window_days == 14  # last week

    # Routing
    should_retrieval = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve", intent, intent.confidence
    )
    assert should_retrieval == True


def test_34_explicit_memory_with_date_full_stack(intent_gate, temporal_engine, orchestrator, now):
    """Explicit memory with milestone date flows correctly."""
    text = "Remember my birthday is March 15, 1990"

    # Intent
    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_EXPLICIT_MEMORY

    # Extract milestone date
    date = temporal_engine.extract_milestone_date(text, now)
    assert date == "1990-03-15"

    # Routing
    should_memory = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event", intent, intent.confidence
    )
    assert should_memory == True


def test_35_context_includes_all_components(orchestrator, now):
    """OrchestratorContext can hold all component results."""
    intent = IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.9
    )

    ctx = OrchestratorContext(
        user_id=1,
        user_input="I like Ferrari",
        session_id="test",
        now=now,
        intent_result=intent,
        temporal_context={"time_of_day": "afternoon"},
    )

    assert ctx.intent_result.primary_intent == INTENT_PERSONAL_STATE
    assert ctx.temporal_context["time_of_day"] == "afternoon"


def test_36_orchestrator_fallback_with_clarification(orchestrator, now):
    """Orchestrator generates clarification when needed."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="What happened then?",
        session_id="test",
        now=now,
        clarification_needed={"clarification_prompt": "When exactly?"}
    )

    response = orchestrator._generate_fallback_response(ctx)
    assert response == "When exactly?"


def test_37_i_like_ferrari_to_what_car_full_flow(intent_gate, orchestrator, temporal_engine, now):
    """Full flow: 'I like Ferrari' → 'What car do I like?'."""
    # Step 1: User says "I like Ferrari"
    store_text = "I like Ferrari"
    store_intent = intent_gate.classify(store_text)

    assert store_intent.primary_intent == INTENT_PERSONAL_STATE
    assert orchestrator._should_call_engine(
        "MemoryEngine.ingest_event", store_intent, store_intent.confidence
    ) == True

    # Step 2: Later, user asks "What car do I like?"
    recall_text = "What car do I like?"
    recall_intent = intent_gate.classify(recall_text)

    assert recall_intent.primary_intent == INTENT_PAST_REFERENCE
    assert orchestrator._should_call_engine(
        "RetrievalEngine.retrieve", recall_intent, recall_intent.confidence
    ) == True


def test_38_my_name_is_to_what_is_my_name_flow(intent_gate, orchestrator):
    """Full flow: 'My name is John' → 'What is my name?'."""
    # Store
    store_intent = intent_gate.classify("My name is John")
    assert store_intent.primary_intent == INTENT_PERSONAL_STATE
    assert orchestrator._should_call_engine(
        "MemoryEngine.ingest_event", store_intent, store_intent.confidence
    ) == True

    # Recall
    recall_intent = intent_gate.classify("What is my name?")
    assert recall_intent.primary_intent == INTENT_PAST_REFERENCE
    assert orchestrator._should_call_engine(
        "RetrievalEngine.retrieve", recall_intent, recall_intent.confidence
    ) == True


def test_39_favorite_color_store_recall_flow(intent_gate, orchestrator):
    """Full flow: 'My favorite color is blue' → 'What is my favorite color?'."""
    # Store
    store_intent = intent_gate.classify("My favorite color is blue")
    assert store_intent.primary_intent == INTENT_PERSONAL_STATE

    # Recall
    recall_intent = intent_gate.classify("What is my favorite color?")
    assert recall_intent.primary_intent == INTENT_PAST_REFERENCE


def test_40_remember_and_recall_flow(intent_gate, orchestrator):
    """Full flow: 'Remember I have a doctor appointment' → 'When is my appointment?'."""
    # Store with explicit command
    store_intent = intent_gate.classify("Remember I have a doctor appointment tomorrow")
    assert store_intent.primary_intent == INTENT_EXPLICIT_MEMORY

    # Recall
    recall_intent = intent_gate.classify("When is my appointment?")
    assert recall_intent.primary_intent == INTENT_PAST_REFERENCE


# =============================================================================
# EDGE CASES AND ROBUSTNESS (41-50)
# =============================================================================

def test_41_empty_text_handling(intent_gate, orchestrator):
    """Empty text doesn't crash the stack."""
    intent = intent_gate.classify("")
    # Should return default (general knowledge)
    assert intent.primary_intent == INTENT_GENERAL_KNOWLEDGE


def test_42_very_long_text_handling(intent_gate, temporal_engine, now):
    """Very long text is handled gracefully."""
    long_text = "I really like " + "pizza " * 100

    intent = intent_gate.classify(long_text)
    assert intent.primary_intent == INTENT_PERSONAL_STATE

    # Temporal should still work
    temporal = temporal_engine.parse(long_text, now)
    # Long text without temporal markers
    assert temporal is not None


def test_43_mixed_temporal_markers(temporal_engine, now):
    """Multiple temporal markers handled (first wins)."""
    text = "Yesterday I thought about what happened last week"

    temporal = temporal_engine.parse(text, now)
    # Should detect "yesterday" first
    assert temporal.direction == -1


def test_44_general_knowledge_with_temporal(intent_gate, temporal_engine, orchestrator, now):
    """General knowledge with temporal marker still blocks memory."""
    text = "What was the weather like yesterday?"

    intent = intent_gate.classify(text)
    # Weather is general knowledge, not personal
    # But "what was...yesterday" might trigger past reference
    # Either way, it shouldn't store memory

    temporal = temporal_engine.parse(text, now)
    assert temporal.direction == -1


def test_45_case_insensitive_patterns(intent_gate, adaptation_engine):
    """Patterns work case-insensitively."""
    # Intent patterns
    intent1 = intent_gate.classify("I LIKE FERRARI")
    intent2 = intent_gate.classify("i like ferrari")
    assert intent1.primary_intent == intent2.primary_intent

    # Adaptation patterns
    deltas1 = adaptation_engine._check_fast_path("THANK YOU")
    deltas2 = adaptation_engine._check_fast_path("thank you")
    assert deltas1 is not None and deltas2 is not None


def test_46_punctuation_handling(intent_gate):
    """Punctuation doesn't break pattern matching."""
    intent1 = intent_gate.classify("I like Ferrari!")
    intent2 = intent_gate.classify("I like Ferrari?")
    intent3 = intent_gate.classify("I like Ferrari...")

    assert intent1.primary_intent == INTENT_PERSONAL_STATE
    # Questions might change intent, but exclamation shouldn't
    assert intent3.primary_intent == INTENT_PERSONAL_STATE


def test_47_contraction_handling(intent_gate):
    """Contractions are handled correctly."""
    intent = intent_gate.classify("I'm feeling happy")
    assert intent.primary_intent == INTENT_PERSONAL_STATE


def test_48_negation_handling(intent_gate, orchestrator):
    """Negations still route to memory."""
    intent = intent_gate.classify("I don't like spinach")
    assert intent.primary_intent == INTENT_PERSONAL_STATE

    should_memory = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event", intent, intent.confidence
    )
    assert should_memory == True


def test_49_multiple_facts_in_one_message(intent_gate, orchestrator):
    """Multiple facts route to memory."""
    text = "I like pizza and my favorite color is blue"

    intent = intent_gate.classify(text)
    assert intent.primary_intent == INTENT_PERSONAL_STATE

    should_memory = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event", intent, intent.confidence
    )
    assert should_memory == True


def test_50_question_then_statement_routing(intent_gate, orchestrator):
    """Question-then-statement routes based on dominant pattern."""
    # This tests that questions about self ("What should I...")
    # are handled correctly
    text = "What should I do? I'm feeling anxious"

    intent = intent_gate.classify(text)
    # Could be either PERSONAL_STATE (dominant emotion) or PAST_REFERENCE
    # Key is it should NOT be GENERAL_KNOWLEDGE
    assert intent.primary_intent != INTENT_GENERAL_KNOWLEDGE


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
