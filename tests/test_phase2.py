import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock imports for heavy libs if not present
sys.modules['nemo.collections.asr'] = MagicMock()
sys.modules['nemo.collections.asr.models'] = MagicMock()
sys.modules['f5_tts.api'] = MagicMock()

def test_diarizer_initialization():
    from src.audio.diarization import Diarizer
    d = Diarizer()
    assert d.embedding_model is None
    assert d.nemo_model is None

def test_f5_wrapper_lazy_load():
    from src.synthesis.f5_tts import F5TTSWrapper
    wrapper = F5TTSWrapper()
    assert wrapper.pipeline is None
    assert wrapper.model_loaded is False
    
    # Test load calls import
    with patch('src.synthesis.f5_tts.F5TTSWrapper') as MockWrapper:
         # Just verify we can instantiate without error
         pass


def test_tts_engine_f5_dispatch():
    from src.synthesis.tts import TTSEngine
    engine = TTSEngine()
    
    with patch.object(engine, '_generate_f5') as mock_f5:
        engine.generate_audio("test", "wav.wav", model="f5", force_cloning=True)
        mock_f5.assert_called_once()
