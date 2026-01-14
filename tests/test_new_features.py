
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock backend modules to avoid import errors
sys.modules["faster_whisper"] = MagicMock()
sys.modules["pyannote.audio"] = MagicMock()
sys.modules["TTS.api"] = MagicMock()
sys.modules["edge_tts"] = MagicMock()
sys.modules["torchaudio"] = MagicMock()
sys.modules["soundfile"] = MagicMock()
sys.modules["pydub"] = MagicMock()
sys.modules["src.synthesis.f5_tts"] = MagicMock()

# Need to reload if already imported (unlikely in fresh run but good practice in notebook, here script is fresh)


from src.audio.transcription import Transcriber
from src.audio.diarization import Diarizer
from src.synthesis.tts import TTSEngine

class TestTranscriberFeatures:
    @patch("src.audio.transcription.WhisperModel")
    def test_transcribe_beam_size(self, mock_whisper_cls):
        """Test that beam_size is passed correctly to model.transcribe."""
        mock_model = MagicMock()
        mock_whisper_cls.return_value = mock_model
        
        # Setup mock return
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.text = "test"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([mock_seg], mock_info)
        
        transcriber = Transcriber()
        transcriber.load_model("base")
        
        # Call with specific beam size
        transcriber.transcribe("dummy.wav", beam_size=3, use_vad=False)
        
        # Verify call args
        args, kwargs = mock_model.transcribe.call_args
        assert kwargs["beam_size"] == 3
        
        # Test default
        transcriber.transcribe("dummy.wav", use_vad=False)
        args, kwargs = mock_model.transcribe.call_args
        assert kwargs["beam_size"] == 5

class TestDiarizerFeatures:
    def test_run_pyannote_community(self):
        """Test PyAnnote pipeline loading and execution."""
        # Setup mock on the module directly
        mock_pipeline_cls = MagicMock()
        sys.modules["pyannote.audio"].Pipeline = mock_pipeline_cls
        
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        
        # Mock pipeline output
        mock_turn = MagicMock()
        mock_turn.start = 0.0
        mock_turn.end = 1.0
        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = [(mock_turn, None, "SPEAKER_00")]
        mock_pipeline.return_value = mock_annotation
        
        diarizer = Diarizer()
        
        # Test direct method
        segments = diarizer._run_pyannote("dummy.wav")
        
        assert len(segments) == 1
        assert segments[0]["speaker"] == "SPEAKER_00"
        mock_pipeline_cls.from_pretrained.assert_called()
        
    @patch("src.audio.diarization.Diarizer._run_pyannote")
    def test_diarize_backend_dispatch(self, mock_run_pyannote):
        """Test that 'pyannote_community' backend triggers the correct method."""
        diarizer = Diarizer()
        diarizer.diarize("dummy.wav", backend="pyannote_community")
        mock_run_pyannote.assert_called_once()

class TestTTSFeatures:
    def test_xtts_params(self):
        """Test that guidance_scale and emotion are passed to XTTS."""
        # Setup mock on the module directly
        mock_tts_cls = MagicMock()
        sys.modules["TTS.api"].TTS = mock_tts_cls
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model  # Important: .to() is called and return value assigned
        mock_tts_cls.return_value = mock_model
        
        engine = TTSEngine()
        
        # Call generate_audio with XTTS and params
        engine.generate_audio(
            text="Hello", 
            speaker_wav_path="ref.wav", 
            model="xtts", 
            guidance_scale=1.5, 
            emotion="happy",
            output_path="out.wav"
        )
        
        # Verify tts_to_file calls
        # We need to verify that kwargs were passed
        args, kwargs = mock_model.tts_to_file.call_args
        assert kwargs["guidance_scale"] == 1.5
        assert kwargs["emotion"] == "happy"
