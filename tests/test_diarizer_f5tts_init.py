"""
Unit tests for Diarizer and F5-TTS initialization.

Tests the lazy loading and initialization behavior of:
- Diarizer: Models should be None until explicitly loaded
- F5TTSWrapper: Pipeline should be None until load() is called
- TTSEngine: F5 backend dispatch should work correctly
"""

import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock imports for heavy libs if not present
sys.modules['nemo.collections.asr'] = MagicMock()
sys.modules['nemo.collections.asr.models'] = MagicMock()
sys.modules['f5_tts.api'] = MagicMock()


class TestDiarizerInitialization:
    """Tests for Diarizer class initialization."""

    def test_diarizer_init_sets_models_to_none(self):
        """Diarizer should initialize with all models set to None (lazy loading)."""
        from src.audio.diarization import Diarizer
        
        diarizer = Diarizer()
        
        assert diarizer.embedding_model is None
        assert diarizer.nemo_model is None


class TestF5TTSWrapperInitialization:
    """Tests for F5-TTS wrapper initialization."""

    def test_f5_wrapper_init_defers_model_loading(self):
        """F5TTSWrapper should initialize with pipeline=None until explicitly loaded."""
        from src.synthesis.f5_tts import F5TTSWrapper
        
        wrapper = F5TTSWrapper()
        
        assert wrapper.pipeline is None
        assert wrapper.model_loaded is False


class TestTTSEngineBackendDispatch:
    """Tests for TTSEngine backend routing."""

    def test_tts_engine_dispatches_to_f5_backend(self):
        """When model='f5', TTSEngine should route generate_audio call to F5 backend."""
        from src.synthesis.tts import TTSEngine
        
        engine = TTSEngine()
        engine.backends["f5"] = MagicMock()
        
        engine.generate_audio("test text", "output.wav", model="f5", force_cloning=True)
        
        engine.backends["f5"].generate.assert_called_once()
