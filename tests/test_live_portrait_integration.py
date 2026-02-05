import pytest
from unittest.mock import MagicMock, patch
from src.processing.lipsync import LipSyncer
from src.processing.live_portrait import LivePortraitSyncer

class TestLivePortraitIntegration:
    
    def test_syncer_initialization(self):
        """Test that LipSyncer can initialize with LivePortrait."""
        syncer = LipSyncer()
        assert "live_portrait" in syncer.engines
        assert isinstance(syncer.engines["live_portrait"], LivePortraitSyncer)
        
    @patch('src.processing.live_portrait.LivePortraitSyncer.load_models')
    def test_load_model_switching(self, mock_load):
        """Test that switching models updates the active engine."""
        syncer = LipSyncer()
        
        # Switch to LivePortrait
        syncer.load_model("live_portrait")
        assert syncer.current_engine_name == "live_portrait"
        assert syncer.engine == syncer.engines["live_portrait"]
        mock_load.assert_called_once()
        
    @pytest.mark.requires_models
    @patch('src.processing.live_portrait.LivePortraitSyncer.sync_lips')
    def test_sync_lips_delegation(self, mock_sync):
        """Test that sync_lips delegates to LivePortrait engine when selected."""
        syncer = LipSyncer()
        syncer.load_model("live_portrait")
        
        mock_sync.return_value = "output.mp4"
        
        result = syncer.sync_lips("input.mp4", "audio.wav", "output.mp4", model_name="live_portrait")
        
        assert result == "output.mp4"
        mock_sync.assert_called_with("input.mp4", "audio.wav", "output.mp4", enhance_face=False)
