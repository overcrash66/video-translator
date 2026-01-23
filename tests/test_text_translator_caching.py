import pytest
from unittest.mock import MagicMock, patch
from src.translation.text_translator import Translator
from src.utils import config
import json

@pytest.fixture
def clean_cache(tmp_path):
    # Redirect cache file to tmp
    orig = config.TEMP_DIR
    config.TEMP_DIR = tmp_path
    yield tmp_path
    config.TEMP_DIR = orig

def test_translation_caching(clean_cache):
    translator = Translator()
    translator.cache_file = clean_cache / "test_cache.json"
    translator.translator_cache = {} # Reset google objects
    
    segments = [{'text': 'Hello'}]
    
    # Mock Google Translator
    with patch('src.translation.text_translator.GoogleTranslator') as MockGT:
        instance = MockGT.return_value
        instance.translate.return_value = "Hola"
        
        # First Run - Should call API
        res1 = translator.translate_segments(segments, "es", model="google")
        assert res1[0]['translated_text'] == "Hola"
        assert instance.translate.call_count == 1
        
        # Check integrity of cache in memory
        key = "en|es|Hello|google"
        assert key in translator.cache
        
        # Second Run - Should HIT cache and NOT call API
        instance.translate.reset_mock()
        res2 = translator.translate_segments(segments, "es", model="google")
        assert res2[0]['translated_text'] == "Hola"
        assert instance.translate.call_count == 0

def test_cache_persistence(clean_cache):
    translator = Translator()
    cache_file = clean_cache / "cache.json"
    translator.cache_file = cache_file
    
    # Pre-populate cache file
    data = {"en|es|Test|google": "Prueba"}
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f)
        
    # Reload
    translator = Translator()
    translator.cache_file = cache_file
    translator.cache = translator._load_cache() # Manually reload from correct path
    
    segments = [{'text': 'Test'}]
    
    with patch('src.translation.text_translator.GoogleTranslator') as MockGT:
        instance = MockGT.return_value
        
        # Should hit cache
        res = translator.translate_segments(segments, "es", model="google")
        assert res[0]['translated_text'] == "Prueba"
        assert instance.translate.call_count == 0
