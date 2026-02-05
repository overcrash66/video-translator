"""
Unit tests for Wav2Lip CUDA error fallback to CPU.

Tests the automatic fallback mechanism that switches Wav2Lip from GPU to CPU
when a CUDA error (e.g., "no kernel image is available") occurs during
face detection. This ensures the pipeline completes successfully even on
incompatible GPU architectures.
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.processing.wav2lip import Wav2LipSyncer


@pytest.fixture
def cuda_env_backup():
    """Backup and restore CUDA_VISIBLE_DEVICES environment variable."""
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    yield
    if original is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = original


class TestWav2LipCudaFallback:
    """Tests for CUDA error handling and CPU fallback in Wav2Lip."""

    @pytest.mark.requires_models  # Requires CUDA fallback behavior in implementation
    @patch('src.processing.wav2lip.config')
    @patch('src.processing.wav2lip.Wav2Lip')
    @patch('src.processing.wav2lip.face_alignment.FaceAlignment')
    @patch('torch.load')
    @patch('torch.cuda.is_available', return_value=True)
    def test_cuda_error_triggers_cpu_fallback_and_hides_gpu(
        self, 
        mock_cuda_avail, 
        mock_torch_load, 
        mock_face_alignment_cls, 
        mock_wav2lip_cls, 
        mock_config,
        cuda_env_backup
    ):
        """
        When a CUDA error occurs in face detection, the syncer should:
        1. Set CUDA_VISIBLE_DEVICES="" to hide GPU
        2. Set fallback_active = True
        3. Switch device to CPU
        4. Successfully retry face detection
        """
        # Create mock face detector instances
        # instance1: Raises CUDA error (simulates incompatible GPU)
        instance1 = MagicMock()
        instance1.get_landmarks.side_effect = RuntimeError(
            "CUDA error: no kernel image is available"
        )
        
        # instance2: Works on CPU (fallback success)
        instance2 = MagicMock()
        instance2.get_landmarks.return_value = [[[10, 10], [20, 10], [20, 20], [10, 20]]]
        
        # FaceAlignment returns instance2 when called after fallback
        mock_face_alignment_cls.return_value = instance2
        
        # Create syncer and inject the failing detector
        syncer = Wav2LipSyncer()
        syncer.model = MagicMock()
        syncer.detector = instance1
        
        # Prepare test data
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frames = [dummy_frame]
        
        # Execute: detect_faces should catch error, switch to CPU, and retry
        with patch.dict(os.environ, {}, clear=False):
            results = syncer.detect_faces(frames)
            
            # Verify: GPU hidden
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "", \
                "Should set CUDA_VISIBLE_DEVICES to empty string"
            
            # Verify: Fallback mode active
            assert syncer.fallback_active is True, \
                "Fallback should be active"
            
            # Verify: Device switched to CPU
            assert str(syncer.device) == "cpu", \
                "Device should be CPU"
            
            # Verify: Detection succeeded with fallback detector
            assert results[0] is not None, \
                "Should have obtained face detection results"
