import pytest
from unittest.mock import MagicMock, patch

def test_llm_translator_init_llama():
    from src.translation.text_translator import LLMTranslator
    # Test Llama initialization
    t = LLMTranslator("meta-llama/Meta-Llama-3.1-8B-Instruct")
    assert t.is_llama is True
    assert t.is_alma is False
    assert t.is_hymt is False

def test_llm_translator_init_alma():
    from src.translation.text_translator import LLMTranslator
    t = LLMTranslator("haoranxu/ALMA-7B-R")
    assert t.is_llama is False
    assert t.is_alma is True

def test_get_prompt_llama():
    from src.translation.text_translator import LLMTranslator
    t = LLMTranslator("meta-llama/Meta-Llama-3.1-8B-Instruct")
    # Mock tokenizer
    t.tokenizer = MagicMock()
    t.tokenizer.chat_template = None # Force manual template path
    t.tokenizer.pad_token_id = 0
    
    prompt = t.get_prompt("Hello", "en", "es", context_prev="Previous")
    assert "You are a professional video translator" in prompt
    assert "Context (Previous Line): Previous" in prompt
    # Implementation uses get_language_name() to convert codes to full names
    assert "Translate from English to Spanish" in prompt

def test_get_prompt_alma():
    from src.translation.text_translator import LLMTranslator
    t = LLMTranslator("haoranxu/ALMA-7B-R")
    # Use language CODES, not names, as the implementation calls get_language_name() to convert
    prompt = t.get_prompt("Hello", "en", "es")
    # ALMA format: "Translate this from {src_name} to {tgt_name}:\n{src_name}: {text}\n{tgt_name}:"
    assert "Translate this from English to Spanish" in prompt
    assert "English: Hello" in prompt
    assert "Spanish:" in prompt

def test_translator_integration_context():
    from src.translation.text_translator import Translator
    tr = Translator()
    
    # Mock LLMTranslator
    with patch('src.translation.text_translator.LLMTranslator') as MockLLM:
        mock_instance = MockLLM.return_value
        mock_instance.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        mock_instance.translate_context_aware.return_value = "Translated"
        
        segments = [{"text": "Hello", "start": 0, "end": 1}, {"text": "World", "start": 1, "end": 2}]
        
        # Optimize = True triggers context aware loop
        res = tr.translate_segments(segments, "Spanish", model="llama", optimize=True)
        
        assert len(res) == 2
        assert res[0]['translated_text'] == "Translated"
        
        # Verify sequential calls
        # 1st call: prev_text=None
        # 2nd call: prev_text="Translated"
        assert mock_instance.translate_context_aware.call_count == 2
        
        # Check call args
        args1 = mock_instance.translate_context_aware.call_args_list[0]
        args2 = mock_instance.translate_context_aware.call_args_list[1]
        
        assert args1.kwargs['prev_text'] is None
        assert args2.kwargs['prev_text'] == "Translated"
