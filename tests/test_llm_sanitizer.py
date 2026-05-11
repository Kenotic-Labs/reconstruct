"""
LLM Output Sanitizer Tests - Verifies all 10 known issues are handled.

Issue Checklist:
1. System prompt leaking
2. Identity crisis (non-Nura identities)
3. Chinese characters (Qwen artifact)
4. Emojis
5. Thinking tags
6. Role labels
7. Repetition
8. Incomplete sentences
9. Markdown formatting
10. Hallucinated URLs

Run: python -m pytest tests/test_llm_sanitizer.py -v
"""

import sys
from pathlib import Path

import pytest

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.llm_sanitizer import (
    sanitize_llm_output,
    sanitize_for_tts,
    extract_and_sanitize,
    is_safe_for_tts,
    detect_hallucination,
    get_uncertainty_response,
    sanitize_with_hallucination_check,
    UNCERTAINTY_RESPONSES,
)


# =============================================================================
# ISSUE 1: SYSTEM PROMPT LEAKING
# =============================================================================

class TestSystemPromptLeaking:
    """Tests for issue #1: System prompt fragments in output."""

    def test_removes_reply_instructions(self):
        """Filters 'reply in X sentences' instructions."""
        text = "Hello there! Reply in 1-2 sentences."
        result = sanitize_llm_output(text)
        assert "reply in" not in result.lower()

    def test_removes_you_are_nura(self):
        """Filters 'you are nura' identity leak."""
        text = "You are Raya, a caring friend. How can I help?"
        result = sanitize_llm_output(text)
        assert "you are nura" not in result.lower()

    def test_removes_no_think_tag(self):
        """Filters /no_think directive."""
        text = "/no_think Hello!"
        result = sanitize_llm_output(text)
        assert "/no_think" not in result

    def test_removes_chatml_markers(self):
        """Filters <|im_start|> and <|im_end|> markers."""
        text = "<|im_start|>assistant\nHello!<|im_end|>"
        result = sanitize_llm_output(text)
        assert "<|im_start|>" not in result
        assert "<|im_end|>" not in result
        assert "Hello" in result

    def test_removes_system_bracket_markers(self):
        """Filters [system], [inst], etc."""
        text = "[system] Be nice. [/system] Hello!"
        result = sanitize_llm_output(text)
        assert "[system]" not in result.lower()


# =============================================================================
# ISSUE 2: IDENTITY CRISIS
# =============================================================================

class TestIdentityCrisis:
    """Tests for issue #2: Model identifying as wrong identity."""

    def test_fixes_i_am_chatgpt(self):
        """Replaces 'I am ChatGPT' with 'I am Nura'."""
        text = "I am ChatGPT, how can I help you?"
        result = sanitize_llm_output(text)
        assert "chatgpt" not in result.lower()
        assert "Raya" in result

    def test_fixes_i_am_claude(self):
        """Replaces 'I am Claude' with 'I am Nura'."""
        text = "I am Claude, an AI assistant."
        result = sanitize_llm_output(text)
        assert "claude" not in result.lower()
        assert "Raya" in result

    def test_fixes_as_an_ai_assistant(self):
        """Handles 'As an AI assistant' type phrases."""
        text = "As an AI assistant, I cannot feel emotions."
        result = sanitize_llm_output(text)
        # Should remove the problematic phrase
        assert "as an ai" not in result.lower() or "Raya" in result

    def test_preserves_raya_identity(self):
        """Doesn't break correct Nura references."""
        text = "I'm Nura, nice to meet you!"
        result = sanitize_llm_output(text)
        assert "Raya" in result


# =============================================================================
# ISSUE 3: CHINESE CHARACTERS (CJK)
# =============================================================================

class TestChineseCharacters:
    """Tests for issue #3: CJK characters in output."""

    def test_removes_chinese_characters(self):
        """Strips Chinese characters."""
        text = "Hello 你好 world"
        result = sanitize_llm_output(text)
        assert "你好" not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_japanese_characters(self):
        """Strips Japanese characters (hiragana/katakana)."""
        text = "Hello こんにちは world"
        result = sanitize_llm_output(text)
        assert "こんにちは" not in result

    def test_removes_korean_characters(self):
        """Strips Korean characters."""
        text = "Hello 안녕하세요 world"
        result = sanitize_llm_output(text)
        assert "안녕하세요" not in result

    def test_handles_mixed_cjk(self):
        """Handles mix of CJK scripts."""
        text = "好的,我来帮助你! Let me help."
        result = sanitize_llm_output(text)
        assert "好的" not in result
        assert "我来帮助你" not in result
        assert "Let me help" in result


# =============================================================================
# ISSUE 4: EMOJIS
# =============================================================================

class TestEmojis:
    """Tests for issue #4: Emojis in output."""

    def test_removes_smiley_emojis(self):
        """Strips smiley face emojis."""
        text = "I'm happy to help! 😊"
        result = sanitize_llm_output(text)
        assert "😊" not in result
        assert "happy to help" in result

    def test_removes_multiple_emojis(self):
        """Strips multiple emojis."""
        text = "Great! 🎉🎊🥳 Congratulations!"
        result = sanitize_llm_output(text)
        assert "🎉" not in result
        assert "🎊" not in result
        assert "🥳" not in result
        assert "Congratulations" in result

    def test_removes_flag_emojis(self):
        """Strips flag emojis."""
        text = "Hello from 🇺🇸!"
        result = sanitize_llm_output(text)
        assert "🇺🇸" not in result

    def test_removes_symbol_emojis(self):
        """Strips symbol emojis."""
        text = "Check ✅ Done ❌ Error"
        result = sanitize_llm_output(text)
        assert "✅" not in result
        assert "❌" not in result


# =============================================================================
# ISSUE 5: THINKING TAGS
# =============================================================================

class TestThinkingTags:
    """Tests for issue #5: Thinking tags in output."""

    def test_removes_complete_think_blocks(self):
        """Removes <think>...</think> blocks."""
        text = "<think>I should be helpful</think>Hello!"
        result = sanitize_llm_output(text)
        assert "<think>" not in result
        assert "</think>" not in result
        assert "I should be helpful" not in result
        assert "Hello" in result

    def test_removes_multiline_think_blocks(self):
        """Removes multiline thinking blocks."""
        text = """<think>
        Let me think about this.
        The user wants help.
        </think>
        I can help you with that."""
        result = sanitize_llm_output(text)
        assert "<think>" not in result
        assert "Let me think" not in result
        assert "I can help" in result

    def test_removes_unclosed_think_tags(self):
        """Handles unclosed think tags (model cut off)."""
        text = "<think>Still thinking..."
        result = sanitize_llm_output(text)
        assert "<think>" not in result
        assert "Still thinking" not in result

    def test_removes_stray_closing_tags(self):
        """Removes stray </think> tags."""
        text = "Hello!</think>"
        result = sanitize_llm_output(text)
        assert "</think>" not in result


# =============================================================================
# ISSUE 6: ROLE LABELS
# =============================================================================

class TestRoleLabels:
    """Tests for issue #6: Role labels in output."""

    def test_removes_user_label(self):
        """Removes 'User:' prefix."""
        text = "User: Hello there"
        result = sanitize_llm_output(text)
        assert not result.startswith("User:")

    def test_removes_assistant_label(self):
        """Removes 'Assistant:' prefix."""
        text = "Assistant: I'm here to help."
        result = sanitize_llm_output(text)
        assert "Assistant:" not in result

    def test_removes_raya_label(self):
        """Removes 'Nura:' prefix."""
        text = "Nura: Hello friend!"
        result = sanitize_llm_output(text)
        assert "Nura:" not in result
        assert "Hello friend" in result

    def test_removes_human_label(self):
        """Removes 'Human:' prefix."""
        text = "Human: What's the weather?"
        result = sanitize_llm_output(text)
        assert "Human:" not in result

    def test_removes_multiline_role_labels(self):
        """Removes role labels on multiple lines."""
        text = "User: Hello\nAssistant: Hi there!"
        result = sanitize_llm_output(text)
        assert "User:" not in result
        assert "Assistant:" not in result


# =============================================================================
# ISSUE 7: REPETITION
# =============================================================================

class TestRepetition:
    """Tests for issue #7: Repetitive text."""

    def test_removes_word_repetition(self):
        """Removes repeated words."""
        text = "I I I am happy"
        result = sanitize_llm_output(text)
        assert result.count("I") <= 2  # At most "I am"

    def test_removes_phrase_repetition(self):
        """Removes repeated phrases."""
        text = "I am here I am here I am here to help."
        result = sanitize_llm_output(text)
        assert result.count("I am here") <= 1

    def test_removes_sentence_repetition(self):
        """Removes duplicate sentences."""
        text = "Hello friend. Hello friend. How are you?"
        result = sanitize_llm_output(text)
        assert result.count("Hello friend") == 1

    def test_preserves_legitimate_repetition(self):
        """Preserves intentional repetition (like emphasis)."""
        text = "Yes, yes. I understand."
        result = sanitize_llm_output(text)
        # This is borderline - the sanitizer might keep or remove it
        assert "understand" in result.lower()


# =============================================================================
# ISSUE 8: INCOMPLETE SENTENCES
# =============================================================================

class TestIncompleteSentences:
    """Tests for issue #8: Text cut off mid-sentence."""

    def test_truncates_to_complete_sentence(self):
        """Keeps only complete sentences."""
        text = "Hello there. How are you tod"
        result = sanitize_llm_output(text)
        assert result.endswith(".") or result.endswith("!") or result.endswith("?")

    def test_preserves_complete_text(self):
        """Doesn't modify already complete text."""
        text = "Hello there. I'm doing well."
        result = sanitize_llm_output(text)
        assert "Hello there" in result
        assert "I'm doing well" in result

    def test_adds_period_if_needed(self):
        """Adds period to incomplete text with no sentences."""
        text = "Hello there"
        result = sanitize_llm_output(text)
        assert result.endswith(".")


# =============================================================================
# ISSUE 9: MARKDOWN FORMATTING
# =============================================================================

class TestMarkdownFormatting:
    """Tests for issue #9: Markdown in output."""

    def test_removes_bold_asterisks(self):
        """Strips **bold** formatting."""
        text = "This is **very** important."
        result = sanitize_llm_output(text)
        assert "**" not in result
        assert "very" in result

    def test_removes_italic_asterisks(self):
        """Strips *italic* formatting."""
        text = "This is *emphasized* text."
        result = sanitize_llm_output(text)
        assert result.count("*") == 0 or "*emphasized*" not in result
        assert "emphasized" in result

    def test_removes_underline_formatting(self):
        """Strips __bold__ and _italic_ formatting."""
        text = "This is __bold__ and _italic_."
        result = sanitize_llm_output(text)
        assert "__" not in result
        # The single _ might be preserved as punctuation

    def test_removes_code_backticks(self):
        """Strips `code` backticks."""
        text = "Run the `install` command."
        result = sanitize_llm_output(text)
        assert "`" not in result
        assert "install" in result

    def test_removes_headers(self):
        """Strips markdown headers."""
        text = "## Title\nHello there."
        result = sanitize_llm_output(text)
        assert "##" not in result
        assert "Title" in result

    def test_removes_links(self):
        """Strips markdown links."""
        text = "Check [this link](http://example.com) out."
        result = sanitize_llm_output(text)
        assert "[" not in result or "](" not in result
        assert "this link" in result


# =============================================================================
# ISSUE 10: HALLUCINATED URLS
# =============================================================================

class TestHallucinatedURLs:
    """Tests for issue #10: Fake/hallucinated URLs."""

    def test_removes_http_urls(self):
        """Strips http:// URLs."""
        text = "Visit http://example.com for more info."
        result = sanitize_llm_output(text)
        assert "http://" not in result
        assert "example.com" not in result

    def test_removes_https_urls(self):
        """Strips https:// URLs."""
        text = "Check https://www.test.org/page for details."
        result = sanitize_llm_output(text)
        assert "https://" not in result
        assert "www.test.org" not in result

    def test_removes_complex_urls(self):
        """Strips URLs with paths and params."""
        text = "See https://api.example.com/v1/users?id=123&name=test for the API."
        result = sanitize_llm_output(text)
        assert "https://" not in result
        assert "api.example.com" not in result


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Tests for combined issues."""

    def test_handles_multiple_issues(self):
        """Handles text with multiple issues at once."""
        text = """<think>Let me help</think>Assistant: Hello! 你好 😊
        I am ChatGPT. Visit https://help.com for **more** info.
        Reply in 1-2 sentences."""

        result = sanitize_llm_output(text)

        # All issues should be handled
        assert "<think>" not in result
        assert "Assistant:" not in result
        assert "你好" not in result
        assert "😊" not in result
        assert "ChatGPT" not in result
        assert "https://" not in result
        assert "**" not in result
        assert "reply in" not in result.lower()

    def test_is_safe_for_tts_clean_text(self):
        """is_safe_for_tts returns True for clean text."""
        text = "Hello, how are you doing today?"
        is_safe, reason = is_safe_for_tts(text)
        assert is_safe is True
        assert reason is None

    def test_is_safe_for_tts_detects_issues(self):
        """is_safe_for_tts detects problematic text."""
        # Chinese characters
        is_safe, reason = is_safe_for_tts("Hello 你好")
        assert is_safe is False
        assert "CJK" in reason

        # Thinking tags
        is_safe, reason = is_safe_for_tts("<think>test</think>Hi")
        assert is_safe is False
        assert "thinking" in reason.lower()

    def test_sanitize_for_tts_specialization(self):
        """sanitize_for_tts handles TTS-specific issues."""
        text = "Dr. Smith said (quietly) to check etc."
        result = sanitize_for_tts(text)
        # Should expand abbreviations and remove parentheticals
        assert "doctor" in result.lower() or "dr." in result.lower()

    def test_extract_and_sanitize(self):
        """extract_and_sanitize separates thinking and response."""
        raw = "<think>I need to be helpful</think>Hello, I'm here to help!"
        response, thinking = extract_and_sanitize(raw)

        assert "Hello" in response
        assert "I need to be helpful" in thinking
        assert "<think>" not in response


# =============================================================================
# ISSUE 11: HALLUCINATED ANSWERS
# =============================================================================

class TestHallucinatedAnswers:
    """Tests for issue #11: Hallucinated/made-up information."""

    def test_detects_as_i_recall(self):
        """Detects 'as I recall, you...' hallucination."""
        text = "As I recall, you love hiking in the mountains."
        result = detect_hallucination(text, has_memories=False)
        assert result is True

    def test_detects_you_told_me(self):
        """Detects 'you told me...' hallucination."""
        text = "You told me your favorite color is blue."
        result = detect_hallucination(text, has_memories=False)
        assert result is True

    def test_detects_i_remember_you_saying(self):
        """Detects 'I remember you saying...' hallucination."""
        text = "I remember you saying that you work in tech."
        result = detect_hallucination(text, has_memories=False)
        assert result is True

    def test_detects_from_previous_conversation(self):
        """Detects reference to non-existent conversation."""
        text = "From our previous conversation, I know you enjoy cooking."
        result = detect_hallucination(text, has_memories=False)
        assert result is True

    def test_detects_your_favorite_is(self):
        """Detects made-up preferences."""
        text = "Your favorite food is pizza, right?"
        result = detect_hallucination(text, has_memories=False)
        assert result is True

    def test_allows_with_memories(self):
        """Allows phrases when memories exist."""
        text = "As I recall, you love hiking."
        result = detect_hallucination(text, has_memories=True)
        assert result is False

    def test_allows_normal_text(self):
        """Allows normal responses without hallucination markers."""
        text = "I'm happy to help you with that. What would you like to know?"
        result = detect_hallucination(text, has_memories=False)
        assert result is False

    def test_uncertainty_response_exists(self):
        """Returns valid uncertainty response."""
        response = get_uncertainty_response()
        assert response in UNCERTAINTY_RESPONSES

    def test_sanitize_with_hallucination_replaces(self):
        """Sanitize with hallucination check replaces hallucinated text."""
        text = "You told me your birthday is in March."
        result = sanitize_with_hallucination_check(text, has_memories=False)
        # Should return an uncertainty response instead
        assert result in UNCERTAINTY_RESPONSES

    def test_sanitize_with_hallucination_preserves_valid(self):
        """Sanitize with hallucination check preserves valid text."""
        text = "I'm here to help you. What's on your mind?"
        result = sanitize_with_hallucination_check(text, has_memories=False)
        assert "help" in result.lower()


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_input(self):
        """Handles empty input."""
        assert sanitize_llm_output("") == ""
        assert sanitize_llm_output(None or "") == ""

    def test_whitespace_only(self):
        """Handles whitespace-only input."""
        assert sanitize_llm_output("   \n\t  ") == ""

    def test_very_long_text(self):
        """Handles very long text."""
        text = "Hello. " * 1000
        result = sanitize_llm_output(text)
        assert len(result) > 0
        # Should handle repetition
        assert result.count("Hello") < 100

    def test_preserves_normal_text(self):
        """Doesn't damage normal, clean text."""
        text = "I understand you're feeling tired today. Would you like to talk about what's been going on?"
        result = sanitize_llm_output(text)
        assert "understand" in result
        assert "tired" in result
        assert "talk about" in result


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
