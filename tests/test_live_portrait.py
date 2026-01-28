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
    @patch("src.processing.live_portrait.Path.exists")
    def test_download_models(self, mock_exists, mock_download, syncer):
        """Test model downloading logic."""
        # Force exists() to return False so download triggers
        mock_exists.return_value = False
        
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
        frame_mock = MagicMock()
        frame_mock.copy.return_value = frame_mock
        frame_mock.shape = (100, 100, 3) # fake shape needed for logic?
        
        cap_src.read.side_effect = [(True, frame_mock), (True, frame_mock), (False, None)]
        cap_src.get.return_value = 100
        
        cap_drv = MagicMock()
        cap_drv.read.side_effect = [(True, frame_mock), (True, frame_mock), (False, None)]
        cap_drv.get.return_value = 100
        
        mock_cap.side_effect = [cap_src, cap_drv]
        
        # Mock detection/crop
        mock_detect.return_value = MagicMock() # found face
        mock_align.return_value = ("crop", "M")
        mock_inf.return_value = "out_img"
        mock_paste.return_value = "final_frame"
        
        # Prepare patches for ffmpeg and os operations
        with patch("src.processing.live_portrait.subprocess.run") as mock_run:
            with patch("src.processing.live_portrait.os.remove") as mock_remove:
                with patch("src.processing.live_portrait.os.path.exists", return_value=True):
                    
                    syncer._animate_video("src.mp4", "drv.mp4", "out.mp4")
                    
                    # Verify ffmpeg called
                    mock_run.assert_called_once()
        
        # Should have processed 2 frames
        assert mock_inf.call_count == 2
        assert mock_writer.return_value.write.call_count == 2

    def test_stitching_models_in_config(self, syncer):
        """Test that stitching and lip retargeting models are configured."""
        assert "stitching" in syncer.onnx_files
        assert "stitching_lip" in syncer.onnx_files
        assert "liveportrait_onnx/stitching.onnx" == syncer.onnx_files["stitching"]
        assert "liveportrait_onnx/stitching_lip.onnx" == syncer.onnx_files["stitching_lip"]

    def test_simple_lip_transfer(self, syncer):
        """Test the fallback simple lip transfer function."""
        import numpy as np
        
        # Create mock keypoints (1, 21, 3)
        kp_source = np.zeros((1, 21, 3), dtype=np.float32)
        kp_driving = np.ones((1, 21, 3), dtype=np.float32) * 0.5
        
        result = syncer._simple_lip_transfer(kp_source, kp_driving)
        
        # Result should be same shape
        assert result.shape == (1, 21, 3)
        
        # Lip indices (17, 18, 19, 20) should have changed
        # Non-lip indices should remain zero
        for i in range(17):
            np.testing.assert_array_equal(result[0, i], kp_source[0, i])
        
        # Lip indices should have some motion applied
        for i in [17, 18, 19, 20]:
            assert not np.array_equal(result[0, i], kp_source[0, i])

    def test_apply_lip_retargeting_fallback(self, syncer):
        """Test that _apply_lip_retargeting falls back when module not loaded."""
        import numpy as np
        
        # Ensure lip_retargeting is None (not loaded)
        syncer.lip_retargeting = None
        
        kp_source = np.zeros((1, 21, 3), dtype=np.float32)
        kp_driving = np.ones((1, 21, 3), dtype=np.float32)
        
        # Should not raise, should fall back to simple transfer
        result = syncer._apply_lip_retargeting(kp_source, kp_driving)
        
        assert result.shape == (1, 21, 3)

    def test_apply_stitching_fallback(self, syncer):
        """Test that _apply_stitching handles errors gracefully."""
        import numpy as np
        
        syncer.stitching_module = None  # Not loaded
        
        kp_source = np.zeros((1, 21, 3), dtype=np.float32)
        kp_driving = np.ones((1, 21, 3), dtype=np.float32)
        
        # Should not raise, should return driving unchanged
        result = syncer._apply_stitching(kp_source, kp_driving)
        np.testing.assert_array_equal(result, kp_driving)

    def test_apply_stitching_with_extra_outputs(self, syncer):
        """Test that _apply_stitching handles output shape of 65 (keypoints + ratios)."""
        import numpy as np
        
        # Mock stitching module to return 65 elements
        syncer.stitching_module = MagicMock()
        # Mock run method to return list of numpy arrays, first item having 65 elements
        # Shape must be (1, 65)
        mock_output = np.zeros((1, 65), dtype=np.float32)
        syncer.stitching_module.run.return_value = [mock_output]
        
        kp_source = np.zeros((1, 21, 3), dtype=np.float32)
        kp_driving = np.ones((1, 21, 3), dtype=np.float32)
        
        # Should not raise exception and should produce correct shape
        result = syncer._apply_stitching(kp_source, kp_driving)
        
        # Result should be correctly reshaped to (1, 21, 3)
        assert result.shape == (1, 21, 3)
