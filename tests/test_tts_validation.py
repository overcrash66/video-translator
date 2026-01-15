
import pytest
from unittest.mock import MagicMock, patch
from src.synthesis.tts import TTSEngine

@pytest.fixture
def tts_engine():
    return TTSEngine()

@patch('soundfile.info')
@patch('soundfile.read')
@patch('pathlib.Path.exists', return_value=True)
def test_validation_variable_duration(mock_exists, mock_read, mock_info, tts_engine):
    """
    Test that _check_reference_audio respects the min_duration parameter.
    """
    # Setup mock audio info with 1.5s duration
    mock_info.return_value = MagicMock(duration=1.5, samplerate=22050)
    
    # Setup valid signal (not silent)
    import numpy as np
    mock_read.return_value = (np.random.normal(0, 0.1, 1000), 22050)
    
    # Case 1: Default (2.0s) -> Should Fail for 1.5s audio
    valid = tts_engine._check_reference_audio("dummy.wav")
    assert not valid, "Should fail for 1.5s audio with default 2.0s limit"
    
    # Case 2: Custom (1.0s) -> Should Pass for 1.5s audio
    valid = tts_engine._check_reference_audio("dummy.wav", min_duration=1.0)
    assert valid, "Should pass for 1.5s audio with 1.0s limit"

@patch('src.synthesis.tts.TTSEngine._check_reference_audio')
@patch('src.synthesis.tts.TTSEngine._generate_f5')
@patch('src.synthesis.tts.TTSEngine._generate_xtts')
def test_generate_call_validation(mock_xtts, mock_f5, mock_check, tts_engine):
    """
    Test that generate_audio calls validation with correct duration for different models.
    """
    mock_check.return_value = True
    
    # Test F5 -> Should use min_duration=1.0
    tts_engine.generate_audio("text", "ref.wav", model="f5")
    mock_check.assert_called_with("ref.wav", min_duration=1.0)
    
    # Test XTTS -> Should use min_duration=2.0
    tts_engine.generate_audio("text", "ref.wav", model="xtts")
    mock_check.assert_called_with("ref.wav", min_duration=2.0)
