import os
import shutil
import unittest
import numpy as np
import soundfile as sf
from pathlib import Path
from src.audio.diarization import Diarizer
from src.utils import config

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
        
        # Create a dedicated mock for soundfile to ensure total control
        mock_sf = unittest.mock.MagicMock()
        # Return random noise to pass RMS check (> 0.01)
        noise = np.random.uniform(-0.5, 0.5, size=(int(self.sr * self.duration), 1)).astype(np.float32)
        mock_sf.read.return_value = (noise, self.sr)
        
        # Patch soundfile in sys.modules so the local import in extract_speaker_profiles gets our mock
        with unittest.mock.patch.dict("sys.modules", {"soundfile": mock_sf}):
             
             profiles = self.diarizer.extract_speaker_profiles(
                self.audio_path, 
                self.segments, 
                output_dir
             )
        
        # Check if profiles were returned
        self.assertIn("SPEAKER_00", profiles)
        self.assertIn("SPEAKER_01", profiles)
        
        # Check if files exist (mock sf.write might be called)
        # Note: Since we mocked soundfile, sf.write is a mock. It won't write to disk.
        # So we cannot check p1.exists().
        # We must check if sf.write was CALLED.
        
        mock_sf.write.assert_called()
        self.assertEqual(mock_sf.write.call_count, 2)
        
        # We manually populate profiles dict in the method, so 'profiles' dict keys exist.
        # But values point to paths. Paths won't exist.
        
        # Verify content logic by checking call args
        # Call args: (file_path, data, samplerate)
        args_list = mock_sf.write.call_args_list
        # We expect 2 calls.
        self.assertEqual(len(args_list), 2)
        
        # Check that we are writing to the output dir
        path0 = str(args_list[0][0][0])
        path1 = str(args_list[1][0][0])
        self.assertTrue("profiles" in path0)
        self.assertTrue("profiles" in path1)

if __name__ == '__main__':
    unittest.main()
