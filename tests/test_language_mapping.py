
import pytest
from unittest.mock import MagicMock
from src.utils.languages import LANGUAGE_CODE_MAP, get_language_name
from src.translation.text_translator import LLMTranslator

def test_get_language_name_all_codes():
    """Verify reverse lookup works for all defined codes."""
    for name, code in LANGUAGE_CODE_MAP.items():
        resolved_name = get_language_name(code)
        # Note: multiple names might map to same code (Chinese -> zh), so we check if the resolved name maps back to the code
        assert LANGUAGE_CODE_MAP[resolved_name] == code
        # Also check it is not returning 'English' fallback unless code is 'en'
        if code != 'en':
            assert resolved_name != "English"

def test_llm_prompt_uses_names():
    """Verify LLM prompts use full names, not codes."""
    translator = LLMTranslator("dummy-model")
    # Mock tokenizer to avoid loading model
    translator.tokenizer = MagicMock()
    translator.tokenizer.chat_template = None 
    
    # Test Llama
    translator.is_llama = True
    translator.is_alma = False
    
    prompt = translator.get_prompt("Hello", "en", "fr")
    assert "Translate from English to French" in prompt
    assert "Translate from en to fr" not in prompt

    # Test ALMA
    translator.is_llama = False
    translator.is_alma = True
    
    prompt = translator.get_prompt("Hello", "en", "fr")
    assert "Translate this from English to French" in prompt
    assert "English: Hello" in prompt
    assert "French:" in prompt

    # Test HY-MT
    translator.is_alma = False
    translator.is_hymt = True # Defaults to else branch
    
    prompt = translator.get_prompt("Hello", "en", "fr")
    assert "Translate from English to French" in prompt

def test_all_languages_in_prompt():
    """Ensures every single language can be correctly formatted into a prompt."""
    translator = LLMTranslator("dummy-model")
    translator.tokenizer = MagicMock()
    translator.tokenizer.chat_template = None
    translator.is_alma = True 
    
    for name, code in LANGUAGE_CODE_MAP.items():
        # Test as source
        prompt = translator.get_prompt("Text", code, "en")
        lang_name = get_language_name(code)
        assert lang_name in prompt
        
        # Test as target
        prompt = translator.get_prompt("Text", "en", code)
        lang_name = get_language_name(code)
        assert lang_name in prompt
