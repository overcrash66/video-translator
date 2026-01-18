"""
Unit tests for the LipSyncer class (src.processing.lipsync).

Tests cover:
- Initialization
- Model loading (checking paths and imports)
- Model unloading and resource cleanup
- Lip-sync orchestration (mocking actual inference)
- Error handling
"""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import sys
import os

class TestLipSyncerInit:
    """Tests for LipSyncer initialization."""
    
    def test_init_defaults(self):
        """Verify initial state."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        
        assert syncer.model_loaded is False
        assert syncer.vae is None
        assert syncer.unet is None
        assert syncer.whisper is None
        assert syncer.face_parser is None
        assert syncer.batch_size == 8
        assert syncer.version == "v15"

class TestLipSyncerLoadModel:
    """Tests for model loading behavior."""
    
    @patch('src.processing.lipsync.Path.exists')
    def test_check_models_exist_failure(self, mock_exists):
        """Test missing model files detection."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        mock_exists.return_value = False
        
        assert syncer._check_models_exist() is False
    
    @patch('src.processing.lipsync.Path.exists')
    def test_check_models_exist_success(self, mock_exists):
        """Test successful model files detection."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        mock_exists.return_value = True
        
        assert syncer._check_models_exist() is True

    @patch('src.processing.lipsync.LipSyncer._check_models_exist')
    @patch('src.processing.lipsync.os.chdir')
    def test_load_model_success(self, mock_chdir, mock_check_exists):
        """Test successful model loading."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        mock_check_exists.return_value = True
        
        with patch.dict(sys.modules, {
            'musetalk.utils.utils': MagicMock(),
            'musetalk.utils.preprocessing': MagicMock(),
            'musetalk.utils.blending': MagicMock(),
            'musetalk.utils.face_parsing': MagicMock(),
            'musetalk.utils.audio_processor': MagicMock(),
            'transformers': MagicMock(),
        }):
             sys.modules['musetalk.utils.utils'].load_all_model.return_value = (MagicMock(), MagicMock(), MagicMock())
             
             syncer.load_model()
             assert syncer.model_loaded is True
             assert syncer.device is not None

    @patch('src.processing.lipsync.LipSyncer._check_models_exist')
    def test_load_model_missing_files(self, mock_check_exists):
        """Test load_model returns False if files missing."""
        from src.processing.lipsync import LipSyncer
        syncer = LipSyncer()
        mock_check_exists.return_value = False
        
        assert syncer.load_model() is False
        assert syncer.model_loaded is False


class TestLipSyncerUnloadModel:
    """Tests for model unloading."""
    
    @patch('src.processing.lipsync.torch')
    @patch('src.processing.lipsync.gc')
    def test_unload_model(self, mock_gc, mock_torch):
        """Verify resources are freed."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.model_loaded = True
        syncer.vae = MagicMock()
        syncer.unet = MagicMock()
        
        syncer.unload_model()
        
        assert syncer.model_loaded is False
        assert syncer.vae is None
        assert syncer.unet is None
        mock_gc.collect.assert_called()
        if mock_torch.cuda.is_available():
            mock_torch.cuda.empty_cache.assert_called()

class TestLipSyncerSyncLips:
    """Tests for the sync_lips orchestration."""
    
    @patch('src.processing.lipsync.LipSyncer.load_model')
    @patch('shutil.copy')
    def test_sync_lips_fallback_if_load_fails(self, mock_copy, mock_load):
        """Test fallback to copy if model fails to load."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        mock_load.return_value = False # Load failed
        
        result = syncer.sync_lips("in.mp4", "aud.wav", "out.mp4")
        
        assert result == "out.mp4"
        mock_copy.assert_called_with("in.mp4", "out.mp4")

    @patch('src.processing.lipsync.LipSyncer.load_model')
    @patch('src.processing.lipsync.os.system')
    @patch('src.processing.lipsync.glob.glob')
    @patch('src.processing.lipsync.shutil.rmtree')
    @patch('src.processing.lipsync.cv2')
    def test_sync_lips_success_flow(self, mock_cv2, mock_rmtree, mock_glob, mock_system, mock_load):
        """
        Test the happy path of lip sync.
        We need to mock a LOT of internal calls since they come from `self._imports`.
        """
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.model_loaded = True
        syncer.device = "cpu"
        syncer.audio_processor = MagicMock()
        syncer.unet = MagicMock()
        syncer.vae = MagicMock()
        syncer.pe = MagicMock()
        syncer.whisper = MagicMock()
        
        # Create a Mock Frame that behaves like a numpy array (has shape)
        frame_mock = MagicMock()
        frame_mock.shape = (256, 256, 3)
        frame_mock.__getitem__.return_value = frame_mock
        
        # Mock cv2.resize to return the same frame mock
        mock_cv2.resize.return_value = frame_mock
        
        # Populate _imports with mocks so the method can call them
        syncer._imports = {
            'get_file_type': MagicMock(return_value="video"),
            'get_video_fps': MagicMock(return_value=25),
            'datagen': MagicMock(return_value=[(MagicMock(), MagicMock())]), 
            'get_landmark_and_bbox': MagicMock(return_value=([ (0,0,100,100) ], [ frame_mock ])),
            'read_imgs': MagicMock(),
            'coord_placeholder': (0,0,0,0),
            'get_image': MagicMock(return_value=frame_mock), 
        }
        
        mock_glob.return_value = ["frame1.png"]
        syncer.audio_processor.get_audio_feature.return_value = (MagicMock(), 100)
        syncer.audio_processor.get_whisper_chunk.return_value = [MagicMock()]
        
        syncer.vae.get_latents_for_unet.return_value = MagicMock()
        syncer.vae.decode_latents.return_value = [MagicMock()] 
        
        syncer.unet.model.return_value.sample = MagicMock()
        
        with patch('pathlib.Path.exists', return_value=True): 
             result = syncer.sync_lips("video.mp4", "audio.wav", "output.mp4")
             
        assert result == str(Path("output.mp4"))
        assert mock_system.call_count >= 2 

    @patch('shutil.copy')
    def test_sync_lips_exception_handling(self, mock_copy):
        """Test that exceptions trigger fallback copy."""
        from src.processing.lipsync import LipSyncer
        
        syncer = LipSyncer()
        syncer.model_loaded = True
        
        # Force an error by having empty Imports
        syncer._imports = {} 
        
        # Should catch KeyError/Exception and fallback
        result = syncer.sync_lips("video.mp4", "audio.wav", "output.mp4")
        
        mock_copy.assert_called()
        assert result == "output.mp4"
