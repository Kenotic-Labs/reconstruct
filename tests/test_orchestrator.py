"""
PHASE 1, STEP 7: Orchestrator - 50 Isolated Tests

Tests orchestration flow, routing policy, and context management.
Must pass 50/50 before proceeding to Phase 2.

Run: python -m pytest tests/test_orchestrator.py -v
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import pytest

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestrator.orchestrator import (
    OrchestratorV2,
    get_orchestrator,
    OrchestratorContext,
    OrchestratorResult,
)
from app.orchestrator.intent_gate import (
    IntentResult,
    INTENT_GENERAL_KNOWLEDGE,
    INTENT_PERSONAL_STATE,
    INTENT_PAST_REFERENCE,
    INTENT_EXPLICIT_MEMORY,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def orchestrator():
    """Fresh orchestrator for each test."""
    return OrchestratorV2()


@pytest.fixture
def now():
    """Current datetime for consistent tests."""
    return datetime.now(timezone.utc)


@pytest.fixture
def general_knowledge_intent():
    """Intent result for general knowledge."""
    return IntentResult(
        primary_intent=INTENT_GENERAL_KNOWLEDGE,
        confidence=0.8,
        engines_required=set(),
        engines_forbidden={
            "MemoryEngine.ingest_event",
            "RetrievalEngine.retrieve",
        },
    )


@pytest.fixture
def personal_state_intent():
    """Intent result for personal state."""
    return IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.85,
        engines_required={"MemoryEngine.ingest_event"},
        engines_optional={"RetrievalEngine.retrieve"},
        engines_forbidden=set(),
    )


@pytest.fixture
def past_reference_intent():
    """Intent result for past reference."""
    return IntentResult(
        primary_intent=INTENT_PAST_REFERENCE,
        confidence=0.9,
        engines_required={"RetrievalEngine.retrieve"},
        engines_forbidden={"MemoryEngine.ingest_event"},
    )


# =============================================================================
# ORCHESTRATOR CONTEXT TESTS (1-10)
# =============================================================================

def test_01_context_stores_user_id(now):
    """OrchestratorContext stores user_id."""
    ctx = OrchestratorContext(
        user_id=42,
        user_input="hello",
        session_id="test",
        now=now,
    )
    assert ctx.user_id == 42


def test_02_context_stores_user_input(now):
    """OrchestratorContext stores user_input."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="How are you?",
        session_id="test",
        now=now,
    )
    assert ctx.user_input == "How are you?"


def test_03_context_stores_session_id(now):
    """OrchestratorContext stores session_id."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="session_123",
        now=now,
    )
    assert ctx.session_id == "session_123"


def test_04_context_stores_now(now):
    """OrchestratorContext stores now timestamp."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
    )
    assert ctx.now == now


def test_05_context_intent_result_default_none(now):
    """OrchestratorContext intent_result defaults to None."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
    )
    assert ctx.intent_result is None


def test_06_context_engines_called_default_empty(now):
    """OrchestratorContext engines_called defaults to empty list."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
    )
    assert ctx.engines_called == []


def test_07_context_engines_blocked_default_empty(now):
    """OrchestratorContext engines_blocked defaults to empty list."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
    )
    assert ctx.engines_blocked == []


def test_08_context_temporal_context_default_none(now):
    """OrchestratorContext temporal_context defaults to None."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
    )
    assert ctx.temporal_context is None


def test_09_context_memory_result_default_none(now):
    """OrchestratorContext memory_result defaults to None."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
    )
    assert ctx.memory_result is None


def test_10_context_clarification_needed_default_none(now):
    """OrchestratorContext clarification_needed defaults to None."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
    )
    assert ctx.clarification_needed is None


# =============================================================================
# ORCHESTRATOR RESULT TESTS (11-15)
# =============================================================================

def test_11_result_has_llm_output():
    """OrchestratorResult has llm_output."""
    result = OrchestratorResult(llm_output="Hello!")
    assert result.llm_output == "Hello!"


def test_12_result_log_path_default_none():
    """OrchestratorResult log_path defaults to None."""
    result = OrchestratorResult(llm_output="Hello!")
    assert result.log_path is None


def test_13_result_safety_blocked_default_false():
    """OrchestratorResult safety_blocked defaults to False."""
    result = OrchestratorResult(llm_output="Hello!")
    assert result.safety_blocked == False


def test_14_result_engines_called_default_empty():
    """OrchestratorResult engines_called defaults to empty list."""
    result = OrchestratorResult(llm_output="Hello!")
    assert result.engines_called == []


def test_15_result_intent_default_none():
    """OrchestratorResult intent defaults to None."""
    result = OrchestratorResult(llm_output="Hello!")
    assert result.intent is None


# =============================================================================
# ROUTING POLICY TESTS (16-30)
# =============================================================================

def test_16_should_call_forbidden_returns_false(orchestrator, general_knowledge_intent):
    """Forbidden engine returns False."""
    result = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        general_knowledge_intent,
        0.9
    )
    assert result == False


def test_17_should_call_required_returns_true(orchestrator, personal_state_intent):
    """Required engine returns True."""
    result = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        personal_state_intent,
        0.9
    )
    assert result == True


def test_18_should_call_optional_high_conf_returns_true(orchestrator, personal_state_intent):
    """Optional engine with high confidence returns True."""
    result = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        personal_state_intent,
        0.8
    )
    assert result == True


def test_19_should_call_optional_low_conf_returns_false(orchestrator, personal_state_intent):
    """Optional engine with low confidence returns False."""
    result = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        personal_state_intent,
        0.2
    )
    assert result == False


def test_20_should_call_unspecified_returns_false(orchestrator, general_knowledge_intent):
    """Unspecified engine returns False."""
    result = orchestrator._should_call_engine(
        "SomeOtherEngine.method",
        general_knowledge_intent,
        0.9
    )
    assert result == False


def test_21_past_reference_requires_retrieval(orchestrator, past_reference_intent):
    """PAST_SELF_REFERENCE requires retrieval."""
    result = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        past_reference_intent,
        0.9
    )
    assert result == True


def test_22_past_reference_forbids_memory(orchestrator, past_reference_intent):
    """PAST_SELF_REFERENCE forbids memory ingest."""
    result = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        past_reference_intent,
        0.9
    )
    assert result == False


def test_23_general_knowledge_forbids_retrieval(orchestrator, general_knowledge_intent):
    """GENERAL_KNOWLEDGE forbids retrieval."""
    result = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        general_knowledge_intent,
        0.9
    )
    assert result == False


def test_24_personal_state_allows_memory(orchestrator, personal_state_intent):
    """PERSONAL_STATE allows memory ingest."""
    result = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        personal_state_intent,
        0.9
    )
    assert result == True


def test_25_confidence_threshold_is_0_4(orchestrator, personal_state_intent):
    """Optional engine threshold is 0.4."""
    # At 0.4, should be True
    result_at_threshold = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        personal_state_intent,
        0.4
    )
    # At 0.39, should be False
    result_below = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        personal_state_intent,
        0.39
    )
    assert result_at_threshold == True
    assert result_below == False


def test_26_forbidden_overrides_required(orchestrator):
    """Forbidden takes precedence over required."""
    # Edge case: engine in both sets (shouldn't happen but test behavior)
    intent = IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.9,
        engines_required={"MemoryEngine.ingest_event"},
        engines_forbidden={"MemoryEngine.ingest_event"},  # Contradictory
    )
    result = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        intent,
        0.9
    )
    assert result == False  # Forbidden wins


def test_27_empty_policy_allows_nothing(orchestrator):
    """Empty policy doesn't allow engines."""
    intent = IntentResult(
        primary_intent=INTENT_GENERAL_KNOWLEDGE,
        confidence=0.9,
        engines_required=set(),
        engines_optional=set(),
        engines_forbidden=set(),
    )
    result = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        intent,
        0.9
    )
    assert result == False


def test_28_required_engine_ignores_confidence(orchestrator, past_reference_intent):
    """Required engine is called regardless of confidence."""
    result = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        past_reference_intent,
        0.1  # Very low confidence
    )
    assert result == True


def test_29_forbidden_engine_ignores_confidence(orchestrator, general_knowledge_intent):
    """Forbidden engine is blocked regardless of confidence."""
    result = orchestrator._should_call_engine(
        "MemoryEngine.ingest_event",
        general_knowledge_intent,
        1.0  # Perfect confidence
    )
    assert result == False


def test_30_optional_engine_respects_confidence(orchestrator, personal_state_intent):
    """Optional engine respects confidence threshold."""
    high_result = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        personal_state_intent,
        0.5
    )
    low_result = orchestrator._should_call_engine(
        "RetrievalEngine.retrieve",
        personal_state_intent,
        0.3
    )
    assert high_result == True
    assert low_result == False


# =============================================================================
# FALLBACK RESPONSE TESTS (31-35)
# =============================================================================

def test_31_fallback_with_clarification(orchestrator, now):
    """Fallback returns clarification prompt if needed."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
        clarification_needed={
            "clarification_prompt": "What time exactly?"
        }
    )
    response = orchestrator._generate_fallback_response(ctx)
    assert response == "What time exactly?"


def test_32_fallback_generic_message(orchestrator, now):
    """Fallback returns generic message when no context."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
    )
    response = orchestrator._generate_fallback_response(ctx)
    assert "help" in response.lower() or "more" in response.lower()


def test_33_fallback_default_clarification(orchestrator, now):
    """Fallback uses default clarification if prompt key missing."""
    ctx = OrchestratorContext(
        user_id=1,
        user_input="hello",
        session_id="test",
        now=now,
        clarification_needed={"temporal_ambiguity": True}  # Has key but no prompt
    )
    response = orchestrator._generate_fallback_response(ctx)
    assert "clarify" in response.lower()


def test_34_orchestrator_initialization(orchestrator):
    """Orchestrator initializes intent gate."""
    assert orchestrator._intent_gate is not None


def test_35_orchestrator_lazy_init(orchestrator):
    """Orchestrator uses lazy initialization."""
    # Before _ensure_initialized called
    assert orchestrator._initialized == False


# =============================================================================
# INTENT RESULT TESTS (36-42)
# =============================================================================

def test_36_intent_result_to_dict():
    """IntentResult to_dict works."""
    result = IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.85,
    )
    d = result.to_dict()
    assert d["primary_intent"] == INTENT_PERSONAL_STATE
    assert d["confidence"] == 0.85


def test_37_intent_result_should_call_engine():
    """IntentResult should_call_engine works."""
    result = IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.85,
        engines_required={"MemoryEngine.ingest_event"},
    )
    assert result.should_call_engine("MemoryEngine.ingest_event") == True


def test_38_intent_result_should_not_call_forbidden():
    """IntentResult should_call_engine returns False for forbidden."""
    result = IntentResult(
        primary_intent=INTENT_GENERAL_KNOWLEDGE,
        confidence=0.85,
        engines_forbidden={"MemoryEngine.ingest_event"},
    )
    assert result.should_call_engine("MemoryEngine.ingest_event") == False


def test_39_intent_result_ambiguity_flag():
    """IntentResult tracks ambiguity."""
    result = IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.5,
        ambiguity=True,
    )
    assert result.ambiguity == True


def test_40_intent_result_detection_method():
    """IntentResult tracks detection method."""
    result = IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.85,
        detection_method="fast_path",
    )
    assert result.detection_method == "fast_path"


def test_41_intent_result_ambiguity_flags():
    """IntentResult tracks ambiguity flags."""
    result = IntentResult(
        primary_intent=INTENT_PERSONAL_STATE,
        confidence=0.85,
        ambiguity_flags=["emotion_state", "personal_facts"],
    )
    assert "emotion_state" in result.ambiguity_flags


def test_42_intent_result_default_values():
    """IntentResult has sensible defaults."""
    result = IntentResult()
    assert result.primary_intent == INTENT_GENERAL_KNOWLEDGE
    assert result.confidence == 0.7
    assert result.ambiguity == False


# =============================================================================
# ORCHESTRATOR INSTANCE TESTS (43-50)
# =============================================================================

def test_43_get_orchestrator_singleton():
    """get_orchestrator returns singleton."""
    o1 = get_orchestrator()
    o2 = get_orchestrator()
    assert o1 is o2


def test_44_orchestrator_has_intent_gate(orchestrator):
    """Orchestrator has intent_gate."""
    assert hasattr(orchestrator, '_intent_gate')


def test_45_orchestrator_has_memory_store(orchestrator):
    """Orchestrator has memory_store attribute."""
    assert hasattr(orchestrator, '_memory_store')


def test_46_orchestrator_has_temporal_engine(orchestrator):
    """Orchestrator has temporal_engine attribute."""
    assert hasattr(orchestrator, '_temporal_engine')


def test_47_orchestrator_has_adaptation_engine(orchestrator):
    """Orchestrator has adaptation_engine attribute."""
    assert hasattr(orchestrator, '_adaptation_engine')


def test_48_orchestrator_has_proactive_engine(orchestrator):
    """Orchestrator has proactive_engine attribute."""
    assert hasattr(orchestrator, '_proactive_engine')


def test_49_orchestrator_ensure_initialized(orchestrator):
    """Orchestrator _ensure_initialized sets flag."""
    orchestrator._ensure_initialized()
    assert orchestrator._initialized == True


def test_50_orchestrator_repeated_init_safe(orchestrator):
    """Orchestrator _ensure_initialized is idempotent."""
    orchestrator._ensure_initialized()
    first_state = orchestrator._initialized
    orchestrator._ensure_initialized()
    second_state = orchestrator._initialized
    assert first_state == True
    assert second_state == True


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
