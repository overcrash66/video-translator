"""
Unit tests for LLM-based Translation (Llama, ALMA, HY-MT).

Tests the LLMTranslator class initialization, prompt generation,
and context-aware translation integration with the Translator class.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestLLMTranslatorInitialization:
    """Tests for LLMTranslator model type detection during initialization."""

    def test_llm_translator_init_detects_llama_model(self):
        """When initialized with Llama model ID, should set is_llama=True."""
        from src.translation.text_translator import LLMTranslator
        
        translator = LLMTranslator("meta-llama/Meta-Llama-3.1-8B-Instruct")
        
        assert translator.is_llama is True
        assert translator.is_alma is False
        assert translator.is_hymt is False

    def test_llm_translator_init_detects_alma_model(self):
        """When initialized with ALMA model ID, should set is_alma=True."""
        from src.translation.text_translator import LLMTranslator
        
        translator = LLMTranslator("haoranxu/ALMA-7B-R")
        
        assert translator.is_llama is False
        assert translator.is_alma is True


class TestLLMTranslatorPromptGeneration:
    """Tests for LLMTranslator prompt generation for different model types."""

    def test_llama_prompt_includes_context_and_instructions(self):
        """Llama prompts should include translator role and context information."""
        from src.translation.text_translator import LLMTranslator
        
        translator = LLMTranslator("meta-llama/Meta-Llama-3.1-8B-Instruct")
        translator.tokenizer = MagicMock()
        translator.tokenizer.chat_template = None  # Force manual template path
        translator.tokenizer.pad_token_id = 0
        
        prompt = translator.get_prompt("Hello", "en", "es", context_prev="Previous")
        
        assert "You are a professional video translator" in prompt
        assert "Context (Previous Line): Previous" in prompt
        assert "Translate from English to Spanish" in prompt

    def test_alma_prompt_uses_correct_format(self):
        """ALMA prompts should follow the expected format with source/target markers."""
        from src.translation.text_translator import LLMTranslator
        
        translator = LLMTranslator("haoranxu/ALMA-7B-R")
        
        prompt = translator.get_prompt("Hello", "en", "es")
        
        # ALMA format: "Translate this from {src} to {tgt}:\n{src}: {text}\n{tgt}:"
        assert "Translate this from English to Spanish" in prompt
        assert "English: Hello" in prompt
        assert "Spanish:" in prompt


class TestTranslatorContextAwareIntegration:
    """Tests for context-aware translation in the main Translator class."""

    def test_translator_passes_previous_context_to_llm(self):
        """
        When optimize=True, Translator should pass previous translation
        as context for subsequent segments.
        
        Expected sequence:
        - Segment 1: prev_text=None
        - Segment 2: prev_text="<translation of segment 1>"
        """
        from src.translation.text_translator import Translator
        
        translator = Translator()
        translator.cache = {}  # Force cache miss to trigger LLM calls
        
        with patch('src.translation.text_translator.LLMTranslator') as MockLLM:
            mock_instance = MockLLM.return_value
            mock_instance.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            mock_instance.translate_context_aware.return_value = "Translated"
            
            segments = [
                {"text": "Hello", "start": 0, "end": 1}, 
                {"text": "World", "start": 1, "end": 2}
            ]
            
            result = translator.translate_segments(
                segments, "Spanish", model="llama", optimize=True
            )
            
            # Verify results
            assert len(result) == 2
            assert result[0]['translated_text'] == "Translated"
            
            # Verify sequential context passing
            assert mock_instance.translate_context_aware.call_count == 2
            
            call_args_1 = mock_instance.translate_context_aware.call_args_list[0]
            call_args_2 = mock_instance.translate_context_aware.call_args_list[1]
            
            assert call_args_1.kwargs['prev_text'] is None  # First segment: no context
            assert call_args_2.kwargs['prev_text'] == "Translated"  # Second: uses prev
