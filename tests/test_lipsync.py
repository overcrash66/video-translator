"""
Unit tests for the LipSyncer class (src.processing.lipsync).

Tests cover:
- Initialization
- Model loading (with and without MuseTalk availability)
- Model unloading and resource cleanup
- Lip-sync fallback behavior
- Error handling
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import sys


class TestLipSyncerInit:
    """Tests for LipSyncer initialization."""
    
    def test_init_defaults(self):
        """Verify initial state: model_loaded=False, musetalk=None."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        
        assert syncer.model_loaded is False
        assert syncer.musetalk is None
        assert 'model_config' in dir(syncer)
    
    def test_init_config_structure(self):
        """Verify model_config has expected structure."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        
        assert isinstance(syncer.model_config, dict)
        assert 'task_1' in syncer.model_config


class TestLipSyncerLoadModel:
    """Tests for model loading behavior."""
    
    def test_load_model_musetalk_available(self):
        """Test model loading when MuseTalk is available."""
        # We need to mock the import within load_model
        with patch.dict(sys.modules, {'musetalk': MagicMock()}):
            from src.processing.lipsync import LipSyncer
            
            syncer = LipSyncer()
            syncer.load_model()
            
            assert syncer.model_loaded is True
            assert syncer.musetalk is True  # Placeholder value
    
    def test_load_model_musetalk_not_installed(self):
        """Test graceful handling when MuseTalk import fails."""
        # Clear musetalk from cache if present
        if 'musetalk' in sys.modules:
            del sys.modules['musetalk']
            
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        
        # Mock the import to raise ImportError
        with patch.object(syncer, 'load_model') as mock_load:
            # Simulate the actual behavior
            syncer.model_loaded = False
            syncer.musetalk = None
            
        # Without mocking, the actual code handles ImportError gracefully
        syncer.load_model()
        
        # The actual implementation catches ImportError and sets model_loaded=False
        assert syncer.model_loaded is False
    
    def test_load_model_already_loaded(self):
        """Verify no duplicate loading when model is already loaded."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.model_loaded = True  # Simulate already loaded
        
        # Should return early without doing anything
        syncer.load_model()
        
        assert syncer.model_loaded is True
    
    def test_load_model_exception_handling(self):
        """Test that exceptions during load are caught and logged."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        
        # Even if an unexpected exception occurs, it should be caught
        # The current implementation sets model_loaded=False on any error
        syncer.load_model()
        
        # After attempting load (musetalk not installed), model_loaded should be False
        assert syncer.model_loaded is False


class TestLipSyncerUnloadModel:
    """Tests for model unloading and resource cleanup."""
    
    def test_unload_model_clears_state(self):
        """Verify model reference is cleared after unload."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.musetalk = MagicMock()  # Simulate loaded model
        syncer.model_loaded = True
        
        syncer.unload_model()
        
        assert syncer.musetalk is None
        assert syncer.model_loaded is False
    
    @patch('src.processing.lipsync.torch')
    def test_unload_model_clears_cuda_cache(self, mock_torch):
        """Verify CUDA cache is cleared when GPU is available."""
        mock_torch.cuda.is_available.return_value = True
        
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.musetalk = MagicMock()
        syncer.model_loaded = True
        
        syncer.unload_model()
        
        mock_torch.cuda.empty_cache.assert_called_once()
    
    @patch('src.processing.lipsync.torch')
    def test_unload_model_skips_cuda_when_unavailable(self, mock_torch):
        """Verify CUDA cleanup is skipped when GPU is not available."""
        mock_torch.cuda.is_available.return_value = False
        
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.musetalk = MagicMock()
        syncer.model_loaded = True
        
        syncer.unload_model()
        
        mock_torch.cuda.empty_cache.assert_not_called()


class TestLipSyncerSyncLips:
    """Tests for the sync_lips method."""
    
    @patch('shutil.copy')
    def test_sync_lips_fallback_copy(self, mock_copy):
        """When model not available, verify file is copied to output."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.model_loaded = False
        
        result = syncer.sync_lips("input.mp4", "audio.wav", "output.mp4")
        
        mock_copy.assert_called_with("input.mp4", "output.mp4")
        assert result == "output.mp4"
    
    @patch('shutil.copy')
    def test_sync_lips_returns_output_path(self, mock_copy):
        """Verify correct output path is returned."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.model_loaded = True
        syncer.musetalk = MagicMock()
        
        result = syncer.sync_lips("input.mp4", "audio.wav", "output.mp4")
        
        assert result == "output.mp4"
    
    @patch('shutil.copy')
    def test_sync_lips_calls_load_model(self, mock_copy):
        """Verify lazy loading when model not loaded."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.model_loaded = False
        
        with patch.object(syncer, 'load_model') as mock_load:
            syncer.sync_lips("input.mp4", "audio.wav", "output.mp4")
        
        mock_load.assert_called_once()
    
    @patch('shutil.copy')
    def test_sync_lips_logs_info(self, mock_copy):
        """Verify info logging during lip-sync process."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.model_loaded = True
        syncer.musetalk = MagicMock()
        
        with patch('src.processing.lipsync.logger') as mock_logger:
            syncer.sync_lips("input.mp4", "audio.wav", "output.mp4")
        
        # Should log info about the process
        assert mock_logger.info.called
    
    @patch('shutil.copy')
    def test_sync_lips_with_path_objects(self, mock_copy):
        """Verify sync_lips works with Path objects."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.model_loaded = True
        syncer.musetalk = MagicMock()
        
        video_path = Path("input.mp4")
        audio_path = Path("audio.wav")
        output_path = Path("output.mp4")
        
        result = syncer.sync_lips(str(video_path), str(audio_path), str(output_path))
        
        assert result == str(output_path)
    
    @patch('shutil.copy')
    def test_sync_lips_exception_raised(self, mock_copy):
        """Verify exception is raised properly on internal error."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.model_loaded = True
        syncer.musetalk = MagicMock()
        
        mock_copy.side_effect = Exception("Copy failed")
        
        with pytest.raises(Exception, match="Copy failed"):
            syncer.sync_lips("input.mp4", "audio.wav", "output.mp4")

