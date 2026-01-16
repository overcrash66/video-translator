
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import soundfile as sf
from pathlib import Path
from src.audio.diarization import Diarizer

# Mock soundfile.read to return dummy audio
def mock_sf_read(file, **kwargs):
    # Return 10 seconds of noise at 16kHz
    sr = 16000
    duration = 10
    # Use random noise to pass RMS checks (> 0.01)
    audio = np.random.uniform(-0.5, 0.5, size=int(sr * duration))
    return audio, sr

# Mock soundfile.write to do nothing
def mock_sf_write(file, data, samplerate, **kwargs):
    pass

@pytest.fixture
def diarizer():
    return Diarizer()

@patch('soundfile.read', side_effect=mock_sf_read)
@patch('soundfile.write', side_effect=mock_sf_write)
def test_extract_profile_constraints(mock_write, mock_read, diarizer, tmp_path):
    """
    Test that relaxed constraints allow extracting profiles from shorter segments.
    """
    # Create a dummy audio file path
    audio_path = tmp_path / "test_audio.wav"
    audio_path.touch()
    
    # Define segments that WOULD fail under old constraints (e.g. 0.6s)
    # Old constraints: Segment >= 1.0s, Total >= 3.0s
    # New constraints: Segment >= 0.5s, Total >= 1.0s
    
    # 3 segments of 0.6s each = 1.8s total. 
    # Should FAIL with old code (skipped segments).
    # Should PASS with new code.
    segments = [
        {'start': 0.0, 'end': 0.6, 'speaker': 'SPEAKER_00'},
        {'start': 1.0, 'end': 1.6, 'speaker': 'SPEAKER_00'},
        {'start': 2.0, 'end': 2.6, 'speaker': 'SPEAKER_00'}
    ]
    
    output_dir = tmp_path / "profiles"
    
    # Run extraction
    profiles = diarizer.extract_speaker_profiles(audio_path, segments, output_dir)
    
    # Assertions
    # With new constraints, we EXPECT a profile to be created
    if "SPEAKER_00" in profiles:
        print("Success: Profile created for shorter segments.")
    else:
        pytest.fail("Profile was NOT created. Constraints might be too strict.")

    # Verify write was called
    if "SPEAKER_00" in profiles:
        assert mock_write.called
