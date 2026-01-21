"""
Unit tests for the LipSyncer wrapper (src.processing.lipsync).
"""
import pytest
from unittest.mock import MagicMock, patch
from src.processing.lipsync import LipSyncer

class TestLipSyncer:
    
    @patch('src.processing.lipsync.Wav2LipSyncer')
    def test_init_creates_engine(self, mock_wav2lip_cls):
        """Test initialization creates Wav2Lip engine."""
        mock_engine = MagicMock()
        mock_wav2lip_cls.return_value = mock_engine
        
        syncer = LipSyncer()
        assert syncer.engine == mock_engine
        
    @patch('src.processing.lipsync.Wav2LipSyncer')
    def test_load_model_delegates(self, mock_wav2lip_cls):
        """Test load_model calls engine load_model."""
        mock_engine = MagicMock()
        mock_wav2lip_cls.return_value = mock_engine
        
        syncer = LipSyncer()
        syncer.load_model()
        mock_engine.load_model.assert_called_once()
        
    @patch('src.processing.lipsync.Wav2LipSyncer')
    def test_sync_lips_delegates(self, mock_wav2lip_cls):
        """Test sync_lips calls engine sync_lips with correct args."""
        mock_engine = MagicMock()
        mock_wav2lip_cls.return_value = mock_engine
        
        syncer = LipSyncer()
        video_path = "video.mp4"
        audio_path = "audio.wav"
        output_path = "out.mp4"
        
        syncer.sync_lips(video_path, audio_path, output_path)
        mock_engine.sync_lips.assert_called_with(video_path, audio_path, output_path, enhance_face=False)
