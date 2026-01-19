
import unittest
from unittest.mock import MagicMock, patch
import os
import torch
import sys

# Adjust path to include src if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.wav2lip import Wav2LipSyncer

class TestWav2LipHangFix(unittest.TestCase):
    def setUp(self):
        self.original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    def tearDown(self):
        if self.original_cuda_visible_devices is None:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.original_cuda_visible_devices

    @patch('src.processing.wav2lip.config')
    @patch('src.processing.wav2lip.Wav2Lip')
    @patch('src.processing.wav2lip.face_alignment.FaceAlignment')
    @patch('torch.load')
    @patch('torch.cuda.is_available', return_value=True)
    def test_cuda_error_fallback_hides_gpu(self, mock_cuda_avail, mock_torch_load, mock_face_alignment_cls, mock_wav2lip_cls, mock_config):
        # Setup mocks
        mock_detector_instance = MagicMock()
        mock_face_alignment_cls.return_value = mock_detector_instance
        
        # Mock exceptions: First call works (init), second call (get_landmarks) raises CUDA error
        # Actually, the error happens in detect_faces -> get_landmarks
        
        # We need to simulate the loop in detect_faces
        # First call to get_landmarks raises exception
        mock_detector_instance.get_landmarks.side_effect = RuntimeError("CUDA error: no kernel image is available")
        
        syncer = Wav2LipSyncer()
        
        # Mock load_model to avoid real loading
        syncer.model = MagicMock()
        syncer.detector = mock_detector_instance
        
        # Create a dummy frame
        import numpy as np
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frames = [dummy_frame]
        
        # Run detect_faces
        # It should catch the error, switch to CPU, and set CUDA_VISIBLE_DEVICES=""
        
        # We need to patch os.environ to verify the set
        with patch.dict(os.environ, {}, clear=False) as mock_env:
            # We also need to mock the re-init of FaceAlignment to stop the infinite loop/recurson in the test
            # The code does: del self.detector; ...; self.detector = face_alignment.FaceAlignment(..., device='cpu')
            # If we don't change side_effect, the NEW detector will also raise error if get_landmarks is called again on 'continue'.
            # BUT the fallback logic does 'continue' which re-calls get_landmarks.
            # So the side_effect should be an iterator: [Error, Success]
            
            mock_detector_instance.get_landmarks.side_effect = [
                RuntimeError("CUDA error: no kernel image is available"),
                [[[10, 10], [50, 50]]] # Success on retry (landmarks)
            ]
            
            # Since the code deletes the detector and creates a NEW one, the new one comes from mock_face_alignment_cls().
            # ensure the new instance also shares the side effect or has a working one.
            # The easiest way is to have the class return the SAME mock instance (singleton-ish for test) 
            # or a second instance that works.
            
            # Let's make the class return a sequence of instances
            instance1 = MagicMock()
            instance1.get_landmarks.side_effect = RuntimeError("CUDA error: no kernel image is available")
            
            instance2 = MagicMock()
            instance2.get_landmarks.return_value = [[[10, 10], [20, 10], [20, 20], [10, 20]]] # Dummy landmarks
            
            mock_face_alignment_cls.side_effect = [instance1, instance2] # Init 1 (load_model), Init 2 (fallback)
            
            # BUT wait, load_model is NOT called in test setup manually above, we manually set syncer.detector = mock_detector_instance.
            # The code calls `del self.detector`.
            # Then `self.detector = face_alignment.FaceAlignment(...)`
            
            # So initially:
            syncer.detector = instance1
            
            # We also need to mock load_model logic if we want, but we just set .detector
            # But the code inside `except` calls `face_alignment.FaceAlignment(...)` which uses the class mock.
            # So the class mock should return instance2 on next call.
            mock_face_alignment_cls.side_effect = None
            mock_face_alignment_cls.return_value = instance2
            
            # Reset instance2 side effect just in case
            instance2.get_landmarks.side_effect = None
            instance2.get_landmarks.return_value = [[[10, 10], [20, 10], [20, 20], [10, 20]]]
            
            
            # Run
            results = syncer.detect_faces(frames)
            
            # Verification
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "", "Should have set CUDA_VISIBLE_DEVICES to empty string")
            self.assertTrue(syncer.fallback_active, "Fallback should be active")
            self.assertEqual(str(syncer.device), "cpu", "Device should be CPU")
            
            # Verify results were obtained (from instance2)
            self.assertIsNotNone(results[0])
            
if __name__ == "__main__":
    unittest.main()
