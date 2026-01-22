import pytest
from unittest.mock import MagicMock, patch
from src.synthesis.tts import TTSEngine
from pathlib import Path

@pytest.fixture
def tts_engine():
    with patch('src.synthesis.tts.F5TTSWrapper'):
        engine = TTSEngine()
        # Mock validation to always pass so we can test the language logic
        engine._check_reference_audio = MagicMock(return_value=True)
        # Mock sanitize to pass
        engine._sanitize_text = MagicMock(side_effect=lambda t: t)
        
        # Mock internal generators
        engine._generate_f5 = MagicMock(return_value="f5_out.wav")
        engine._generate_xtts = MagicMock(return_value="xtts_out.wav")
        # Mock generate_audio recursion for fallback (we want to ensure this is NOT called)
        # But we create a spy or check call args if it calls itself?
        # Actually generate_audio calls itself recursively for fallback.
        # We can check if _generate_f5 was called.
        return engine

def test_cross_lingual_cloning_allowed_f5(tts_engine):
    """
    Verify that F5-TTS works when source_lang != target_lang
    """
    # Act
    tts_engine.generate_audio(
        text="Bonjour",
        speaker_wav_path="ref.wav",
        language="fr",      # Target
        source_lang="en",   # Source
        model="f5",
        output_path="out.wav"
    )
    
    # Assert
    # Should call _generate_f5
    tts_engine._generate_f5.assert_called_once()
    # Validate reference check should have been called
    tts_engine._check_reference_audio.assert_called_once()

def test_cross_lingual_cloning_allowed_xtts(tts_engine):
    """
    Verify that XTTS works when source_lang != target_lang
    """
    tts_engine.generate_audio(
        text="Hola",
        speaker_wav_path="ref.wav",
        language="es",      # Target
        source_lang="en",   # Source
        model="xtts",
        output_path="out.wav"
    )
    
    tts_engine._generate_xtts.assert_called_once()

def test_fallback_still_occurs_on_invalid_ref(tts_engine):
    """
    Verify fallback logic still works if reference is INVALID
    """
    # Setup - Reference Invalid
    tts_engine._check_reference_audio.return_value = False
    
    with patch.object(tts_engine, 'generate_audio', side_effect=tts_engine.generate_audio) as spy:
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
        # Should NOT call _generate_f5 because reference validation failed
        tts_engine._generate_f5.assert_not_called()
        
        # Should have verified reference
        tts_engine._check_reference_audio.assert_called()
