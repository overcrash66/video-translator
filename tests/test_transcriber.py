"""
Unit tests for Transcriber improvements.
Tests VAD preprocessing, model size mapping, and segment cleaning.
"""

import pytest
import numpy as np


class TestModelSizeMapping:
    """Tests for Whisper model size mapping."""
    
    def test_turbo_model_mapping(self):
        """Test that Turbo model name maps correctly."""
        from src.audio.transcription import MODEL_SIZE_MAP
        
        # UI-friendly name should map to internal name
        assert MODEL_SIZE_MAP.get("Large v3 Turbo (Fast)") == "large-v3-turbo"
        assert MODEL_SIZE_MAP.get("Large v3") == "large-v3"
        assert MODEL_SIZE_MAP.get("Medium") == "medium"
        assert MODEL_SIZE_MAP.get("Base") == "base"
    
    def test_direct_model_names(self):
        """Test that direct model names also work."""
        from src.audio.transcription import MODEL_SIZE_MAP
        
        assert MODEL_SIZE_MAP.get("large-v3-turbo") == "large-v3-turbo"
        assert MODEL_SIZE_MAP.get("large-v3") == "large-v3"
        assert MODEL_SIZE_MAP.get("medium") == "medium"


class TestSileroVAD:
    """Tests for Silero VAD functionality."""
    
    def test_vad_initialization(self):
        """Test that SileroVAD initializes correctly."""
        from src.audio.transcription import SileroVAD
        
        vad = SileroVAD()
        assert vad.model is None
        assert vad._loaded is False
    
    def test_vad_detect_speech_no_model(self):
        """Test that VAD returns None when model not available."""
        from src.audio.transcription import SileroVAD
        
        vad = SileroVAD()
        vad._loaded = False  # Force model unavailable
        
        result = vad.detect_speech("nonexistent.wav")
        assert result is None


class TestTranscriberHelpers:
    """Tests for Transcriber helper methods."""
    
    def test_segment_overlaps_speech_true(self):
        """Test segment overlap detection - overlapping case."""
        from src.audio.transcription import Transcriber
        
        t = Transcriber()
        vad_segments = [
            {'start': 0.0, 'end': 2.0},
            {'start': 5.0, 'end': 7.0}
        ]
        
        # Segment overlaps with first VAD region
        assert t._segment_overlaps_speech(1.0, 1.5, vad_segments) is True
        
        # Segment overlaps with second VAD region
        assert t._segment_overlaps_speech(5.5, 6.5, vad_segments) is True
    
    def test_segment_overlaps_speech_false(self):
        """Test segment overlap detection - non-overlapping case."""
        from src.audio.transcription import Transcriber
        
        t = Transcriber()
        vad_segments = [
            {'start': 0.0, 'end': 2.0},
            {'start': 5.0, 'end': 7.0}
        ]
        
        # Segment between VAD regions (no overlap)
        assert t._segment_overlaps_speech(3.0, 4.0, vad_segments) is False
        
        # Segment after all VAD regions
        assert t._segment_overlaps_speech(8.0, 9.0, vad_segments) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
