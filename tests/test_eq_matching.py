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
        out = eq_matching.apply_eq_matching(self.source_path, self.target_path, self.output_path, strength=1.0)
        
        self.assertTrue(os.path.exists(out))
        y, sr = sf.read(out)
        self.assertEqual(sr, 22050)
        # Assert duration is roughly same (within samples)
        self.assertEqual(len(y), 22050)
        
    def test_eq_matching_strength_zero(self):
        """Test that strength 0 produces output very close to target."""
        eq_matching.apply_eq_matching(self.source_path, self.target_path, self.output_path, strength=0.0)
        
        y_out, _ = sf.read(self.output_path)
        y_target, _ = sf.read(self.target_path)
        
        # Should be almost identical (ISTFT reconstruction error exists but small)
        self.assertTrue(np.allclose(y_out, y_target, atol=1e-4))
        
    def test_eq_matching_fallback(self):
        """Test fallback if source file missing."""
        bad_source = "non_existent.wav"
        eq_matching.apply_eq_matching(bad_source, self.target_path, self.output_path)
        
        # Should execute copy fallback
        self.assertTrue(os.path.exists(self.output_path))
        y_out, _ = sf.read(self.output_path)
        y_target, _ = sf.read(self.target_path)
        self.assertTrue(np.allclose(y_out, y_target))

if __name__ == "__main__":
    unittest.main()
