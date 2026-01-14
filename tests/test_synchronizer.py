"""
Unit tests for AudioSynchronizer improvements.
Tests cross-fade window generation and time-stretching.
"""

import pytest
import numpy as np


class TestCrossfadeWindow:
    """Tests for crossfade window generation."""
    
    def test_linear_fade(self):
        """Test linear crossfade window."""
        from src.processing.synchronization import generate_crossfade_window
        
        window = generate_crossfade_window(100, 'linear')
        
        assert len(window) == 100
        assert window[0] == 0.0
        assert window[-1] == 1.0
        assert np.allclose(window[50], 0.5, atol=0.02)
    
    def test_cosine_fade(self):
        """Test cosine (S-curve) crossfade window."""
        from src.processing.synchronization import generate_crossfade_window
        
        window = generate_crossfade_window(100, 'cosine')
        
        assert len(window) == 100
        assert np.isclose(window[0], 0.0, atol=0.01)
        assert np.isclose(window[-1], 1.0, atol=0.01)
        # Cosine is smoother - midpoint should be 0.5
        assert np.isclose(window[50], 0.5, atol=0.01)
    
    def test_exponential_fade(self):
        """Test exponential crossfade window."""
        from src.processing.synchronization import generate_crossfade_window
        
        window = generate_crossfade_window(100, 'exponential')
        
        assert len(window) == 100
        assert window[0] == 0.0
        assert window[-1] == 1.0
        # Exponential rises faster at start
        assert window[25] > 0.4  # Should be sqrt(0.25) = 0.5
    
    def test_zero_length_window(self):
        """Test that zero-length window returns empty array."""
        from src.processing.synchronization import generate_crossfade_window
        
        window = generate_crossfade_window(0, 'linear')
        assert len(window) == 0
    
    def test_negative_length_window(self):
        """Test that negative length returns empty array."""
        from src.processing.synchronization import generate_crossfade_window
        
        window = generate_crossfade_window(-10, 'linear')
        assert len(window) == 0


class TestAudioSynchronizer:
    """Tests for AudioSynchronizer methods."""
    
    def test_sync_segment_close_duration(self):
        """Test that segments with similar duration are just copied."""
        import tempfile
        import soundfile as sf
        from src.processing.synchronization import AudioSynchronizer
        
        sync = AudioSynchronizer()
        
        # Create a test audio file (1 second at 16kHz)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio = np.random.randn(16000).astype(np.float32) * 0.1
            sf.write(f.name, audio, 16000)
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        
        try:
            # Target duration close to actual (within 5%)
            result = sync.sync_segment(input_path, target_duration=1.02, output_path=output_path)
            assert result is not None
        finally:
            import os
            os.unlink(input_path)
            try:
                os.unlink(output_path)
            except:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
