"""
Unit tests for TTS Engine validation functions.
Tests the new _sanitize_text and _validate_audio_file methods.
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestTTSValidation:
    """Tests for TTS validation helper methods."""
    
    def test_sanitize_text_empty(self):
        """Test that empty text returns None."""
        from src.synthesis.tts import TTSEngine
        engine = TTSEngine()
        
        assert engine._sanitize_text("") is None
        assert engine._sanitize_text(None) is None
        assert engine._sanitize_text("   ") is None
    
    def test_sanitize_text_punctuation_only(self):
        """Test that punctuation-only text returns None."""
        from src.synthesis.tts import TTSEngine
        engine = TTSEngine()
        
        assert engine._sanitize_text("...") is None
        assert engine._sanitize_text("!@#$%") is None
        assert engine._sanitize_text("---") is None
    
    def test_sanitize_text_valid(self):
        """Test that valid text is returned stripped."""
        from src.synthesis.tts import TTSEngine
        engine = TTSEngine()
        
        assert engine._sanitize_text("Hello world") == "Hello world"
        assert engine._sanitize_text("  Hello  ") == "Hello"
        assert engine._sanitize_text("Test123") == "Test123"
    
    def test_sanitize_text_cjk(self):
        """Test that CJK characters are accepted."""
        from src.synthesis.tts import TTSEngine
        engine = TTSEngine()
        
        # Chinese
        assert engine._sanitize_text("你好世界") == "你好世界"
        # Japanese
        assert engine._sanitize_text("こんにちは") == "こんにちは"
        # Mixed
        assert engine._sanitize_text("Hello 你好") == "Hello 你好"
    
    def test_validate_audio_file_missing(self):
        """Test validation of non-existent file."""
        from src.synthesis.tts import TTSEngine
        engine = TTSEngine()
        
        assert engine._validate_audio_file("/non/existent/file.wav") is False
    
    def test_validate_audio_file_empty(self):
        """Test validation of empty file."""
        from src.synthesis.tts import TTSEngine
        engine = TTSEngine()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            assert engine._validate_audio_file(temp_path) is False
        finally:
            os.unlink(temp_path)
    
    def test_validate_audio_file_too_small(self):
        """Test validation of too-small file."""
        from src.synthesis.tts import TTSEngine
        engine = TTSEngine()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"x" * 50)  # Less than 100 bytes
            temp_path = f.name
        
        try:
            assert engine._validate_audio_file(temp_path, min_size=100) is False
        finally:
            os.unlink(temp_path)
    
    def test_validate_audio_file_valid(self):
        """Test validation of valid-sized file."""
        from src.synthesis.tts import TTSEngine
        engine = TTSEngine()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"x" * 500)  # More than 100 bytes
            temp_path = f.name
        
        try:
            assert engine._validate_audio_file(temp_path) is True
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
