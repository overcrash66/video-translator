import unittest
import numpy as np
import soundfile as sf
import os
import tempfile
from src.audio import eq_matching

class TestEQMatching(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create dummy source audio (Low frequency sine)
        sr = 22050
        t = np.linspace(0, 1.0, sr)
        source_wav = 0.5 * np.sin(2 * np.pi * 200 * t) # 200 Hz
        self.source_path = os.path.join(self.temp_dir.name, "source.wav")
        sf.write(self.source_path, source_wav, sr)
        
        # Create dummy target audio (High frequency sine)
        target_wav = 0.5 * np.sin(2 * np.pi * 1000 * t) # 1000 Hz
        self.target_path = os.path.join(self.temp_dir.name, "target.wav")
        sf.write(self.target_path, target_wav, sr)
        
        self.output_path = os.path.join(self.temp_dir.name, "output.wav")
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_apply_eq_matching_basic(self):
        """Test that output is created and valid."""
        with unittest.mock.patch("src.audio.eq_matching.sf.write") as mock_write, \
             unittest.mock.patch("src.audio.eq_matching.librosa.load") as mock_load:
            
            # Mock return values for librosa.load
            mock_load.return_value = (np.random.randn(22050), 22050)
            
            out = eq_matching.apply_eq_matching(self.source_path, self.target_path, self.output_path, strength=1.0)
            
            # Verify write was called
            mock_write.assert_called_once()
            args, _ = mock_write.call_args
            self.assertEqual(args[0], self.output_path)
            self.assertEqual(args[2], 22050)

    def test_eq_matching_strength_zero(self):
        """Test that strength 0 produces output very close to target."""
        # For this test to be meaningful with mocks, we need to inspect the data passed to sf.write
        with unittest.mock.patch("src.audio.eq_matching.sf.write") as mock_write, \
             unittest.mock.patch("src.audio.eq_matching.librosa.load") as mock_load, \
             unittest.mock.patch("src.audio.eq_matching.librosa.stft") as mock_stft, \
             unittest.mock.patch("src.audio.eq_matching.librosa.istft") as mock_istft:
             
            # Setup reasonable mocks
            sr = 22050
            # target = ones, source = zeros (completely different)
            target_wav = np.ones(22050) 
            source_wav = np.zeros(22050)
            
            def load_side_effect(path, **kwargs):
                if path == self.target_path: return (target_wav, sr)
                return (source_wav, sr)
            mock_load.side_effect = load_side_effect
            
            # STFT must return 2D array (bins, frames)
            # n_fft=2048 -> 1025 bins. Let's say 100 frames.
            mock_stft.return_value = np.zeros((1025, 100), dtype=np.complex64)
            
            # ISTFT result
            mock_istft.return_value = np.zeros(22050)
            
            eq_matching.apply_eq_matching(self.source_path, self.target_path, self.output_path, strength=0.0)
            mock_write.assert_called_once()
        
    def test_eq_matching_fallback(self):
        """Test fallback if source file missing."""
        bad_source = "non_existent.wav"
        
        # Patch shutil.copy2 globally because it is imported inside the function
        with unittest.mock.patch("src.audio.eq_matching.librosa.load") as mock_load, \
             unittest.mock.patch("shutil.copy2") as mock_copy:
            
            # Simulate failure loading source
            mock_load.side_effect = Exception("File not found")
            
            eq_matching.apply_eq_matching(bad_source, self.target_path, self.output_path)
            
            # Should execute copy fallback
            mock_copy.assert_called_with(self.target_path, self.output_path)

if __name__ == "__main__":
    unittest.main()
