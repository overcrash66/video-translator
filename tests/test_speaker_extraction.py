import os
import shutil
import unittest
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path
from src.audio.diarization import Diarizer
from src.utils import config

@pytest.mark.requires_real_audio
class TestSpeakerExtraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("temp_test_extraction")
        self.test_dir.mkdir(exist_ok=True)
        self.diarizer = Diarizer()
        
        # Create a dummy audio file (mono, 16kHz, 5 seconds)
        self.sr = 16000
        self.duration = 5.0
        self.audio_path = self.test_dir / "test_audio.wav"
        
        # Generate synthetic audio:
        # 0-2s: Tone A (Speaker 1)
        # 2-3s: Silence
        # 3-5s: Tone B (Speaker 2)
        t = np.linspace(0, self.duration, int(self.sr * self.duration))
        self.audio = np.concatenate([
            np.sin(2 * np.pi * 440 * t[:int(2*self.sr)]), # 440Hz
            np.zeros(int(1*self.sr)),
            np.sin(2 * np.pi * 880 * t[:int(2*self.sr)])  # 880Hz
        ])
        
        sf.write(str(self.audio_path), self.audio, self.sr)
        
        # Dummy segments based on the synthetic audio
        self.segments = [
            {'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'},
            {'start': 3.0, 'end': 5.0, 'speaker': 'SPEAKER_01'}
        ]

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_extract_speaker_profiles(self):
        output_dir = self.test_dir / "profiles"
        profiles = self.diarizer.extract_speaker_profiles(
            self.audio_path, 
            self.segments, 
            output_dir
        )
        
        # Check if profiles were returned
        self.assertIn("SPEAKER_00", profiles)
        self.assertIn("SPEAKER_01", profiles)
        
        # Check if files exist
        p1 = Path(profiles["SPEAKER_00"])
        p2 = Path(profiles["SPEAKER_01"])
        
        self.assertTrue(p1.exists())
        self.assertTrue(p2.exists())
        
        # Check file contents (simple duration check)
        audio1, sr1 = sf.read(str(p1))
        audio2, sr2 = sf.read(str(p2))
        
        # Should be roughly 2 seconds each
        self.assertAlmostEqual(len(audio1)/sr1, 2.0, delta=0.5)
        self.assertAlmostEqual(len(audio2)/sr2, 2.0, delta=0.5)

if __name__ == '__main__':
    unittest.main()
