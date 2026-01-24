import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Mock insightface/onnxruntime if not present
try:
    import insightface
except ImportError:
    insightface = None
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

from src.processing.live_portrait import LivePortraitSyncer

class TestLivePortraitSyncer:
    
    @pytest.fixture
    def syncer(self):
        return LivePortraitSyncer()

    def test_init(self, syncer):
        """Test initialization of LivePortraitSyncer."""
        assert syncer.model_dir == Path("models/live_portrait_onnx")
        assert syncer.appearance_extractor is None
        # Check specific filename with subdir
        assert "liveportrait_onnx/appearance_feature_extractor.onnx" in syncer.onnx_files.values()

    @patch("src.processing.live_portrait.hf_hub_download")
    def test_download_models(self, mock_download, syncer):
        """Test model downloading logic."""
        syncer.download_models()
        
        # Check specific file request from warmshao repo
        calls = [args[1]['filename'] for args in mock_download.call_args_list]
        repos = [args[1]['repo_id'] for args in mock_download.call_args_list]
        
        # Should now be requesting files with subdir prefix OR checking logic 
        # The key is what `hf_hub_download` receives. 
        # Our code passes 'liveportrait_onnx/appearance...'.
        assert "liveportrait_onnx/appearance_feature_extractor.onnx" in calls
        assert "warmshao/FasterLivePortrait" in repos

    @patch("src.processing.live_portrait.insightface.app.FaceAnalysis")
    @patch("src.processing.live_portrait.ort.InferenceSession")
    def test_load_models_mock(self, mock_ort, mock_face_analysis, syncer):
        """Test model loading with mocked dependencies."""
        
        # Setup mocks
        mock_session = MagicMock()
        mock_ort.return_value = mock_session
        
        with patch("src.processing.live_portrait.LivePortraitSyncer.download_models") as mock_down:
            with patch("pathlib.Path.exists") as mock_exists:
                # Force exists=True for model paths to bypass check
                mock_exists.return_value = True
                
                syncer.load_models()
                
                mock_down.assert_called_once()
                
                # Verify InsightFace app was initialized
                mock_face_analysis.assert_called_once()
                
                # Verify ORT sessions were created (3 main models: app, mot, warping_spade)
                # Landmark might not load explicitly if not used, or if used inside face analysis?
                # Code loads 3 explicitly.
                assert mock_ort.call_count >= 3

    @patch("src.processing.live_portrait.LivePortraitSyncer._run_inference")
    @patch("src.processing.live_portrait.LivePortraitSyncer._detect")
    @patch("src.processing.live_portrait.LivePortraitSyncer._align_crop")
    @patch("src.processing.live_portrait.LivePortraitSyncer._paste_back")
    @patch("src.processing.live_portrait.cv2.VideoCapture")
    @patch("src.processing.live_portrait.cv2.VideoWriter")
    def test_animate_video_loop(self, mock_writer, mock_cap, mock_paste, mock_align, mock_detect, mock_inf, syncer):
        """Test the animation loop logic (mocked)."""
        # Mock VideoCapture to return different instances for source and driving
        cap_src = MagicMock()
        cap_src.read.side_effect = [(True, "frame1"), (True, "frame2"), (False, None)]
        cap_src.get.return_value = 100
        
        cap_drv = MagicMock()
        cap_drv.read.side_effect = [(True, "frame1"), (True, "frame2"), (False, None)]
        cap_drv.get.return_value = 100
        
        mock_cap.side_effect = [cap_src, cap_drv]
        
        # Mock detection/crop
        mock_detect.return_value = MagicMock() # found face
        mock_align.return_value = ("crop", "M")
        mock_inf.return_value = "out_img"
        mock_paste.return_value = "final_frame"
        
        with patch("src.processing.live_portrait.os.remove"): # suppress remove
             syncer._animate_video("src.mp4", "drv.mp4", "out.mp4")
        
        # Should have processed 2 frames
        assert mock_inf.call_count == 2
        assert mock_writer.return_value.write.call_count == 2
