import pytest
from unittest.mock import MagicMock, patch
from src.synthesis.tts import TTSEngine
from pathlib import Path

@pytest.fixture
def tts_engine():
    # Patch the F5TTSWrapper in the backend module where it is actually used
    with patch('src.synthesis.backends.f5_tts.F5TTSWrapper'):
        engine = TTSEngine()
        # Mock validation to always pass
        engine.validate_reference = MagicMock(return_value=True)
        engine._check_reference_audio = MagicMock(return_value=True)
        engine._sanitize_text = MagicMock(side_effect=lambda t: t)
        
        # Mock backends
        engine.backends["f5"] = MagicMock()
        engine.backends["xtts"] = MagicMock()
        engine.backends["edge"] = MagicMock()
        
        return engine

def test_cross_lingual_cloning_allowed_f5(tts_engine):
    """
    Verify that F5-TTS works when source_lang != target_lang
    """
    # Act
    tts_engine.generate_audio(
        text="Bonjour",
        speaker_wav_path="ref.wav",
        language="fr",
        source_lang="en",
        model="f5",
        output_path="out.wav"
    )
    
    # Assert
    # Should use F5 backend
    tts_engine.backends['f5'].generate.assert_called_once()
    # Should NOT fallback to Edge
    tts_engine.backends['edge'].generate.assert_not_called()

def test_cross_lingual_cloning_allowed_xtts(tts_engine):
    """
    Verify that XTTS works when source_lang != target_lang
    """
    tts_engine.generate_audio(
        text="Hola",
        speaker_wav_path="ref.wav",
        language="es",
        source_lang="en",
        model="xtts",
        output_path="out.wav"
    )
    
    tts_engine.backends['xtts'].generate.assert_called_once()

def test_fallback_still_occurs_on_invalid_ref(tts_engine):
    """
    Verify fallback logic still works if reference is INVALID
    """
    # Setup - Reference Invalid
    tts_engine.validate_reference = MagicMock(return_value=False)
    
    # Act
    tts_engine.generate_audio(
        text="Test",
        speaker_wav_path="bad_ref.wav",
        language="fr",
        source_lang="en",
        model="f5",
        output_path="out.wav"
    )
    
    # Assert
    # Should NOT call F5 backend
    tts_engine.backends['f5'].generate.assert_not_called()
    # Should Fallback to Edge
    tts_engine.backends['edge'].generate.assert_called()
        
def test_cross_lingual_cloning_allowed_f5_english(tts_engine):
    """
    Verify that F5-TTS works when language is English
    """
    tts_engine.generate_audio(
        text="Hello",
        speaker_wav_path="ref.wav",
        language="en",
        source_lang="fr", 
        model="f5",
        output_path="out.wav"
    )
    tts_engine.backends['f5'].generate.assert_called_once()
