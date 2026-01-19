import logging
import unittest
from unittest.mock import MagicMock, patch
import torch

# Configure logging to show up in test output
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("src.processing.wav2lip")

from src.processing.wav2lip import Wav2LipSyncer

class TestLoopSafety(unittest.TestCase):
    def setUp(self):
        self.syncer = Wav2LipSyncer()
        self.syncer.detector = MagicMock()
        import numpy as np
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_cuda_error_retry_once_then_skip(self):
        """
        Simulate persistent CUDA error. 
        Expectation: 
        1. Frame 0 fails (CUDA error) -> Activates Fallback -> Retries Frame 0.
        2. Frame 0 fails again (CUDA error) -> Detects Fallback Active -> Skips Frame 0.
        3. Frame 1 processed.
        Loop should terminate.
        """
        frames = [self.frame, self.frame] # 2 frames
        
        # Side effect: raise Exception with "CUDA" text
        self.syncer.detector.get_landmarks.side_effect = Exception("CUDA error: no kernel image")
        
        with patch('src.processing.wav2lip.face_alignment.FaceAlignment') as MockFA:
            # Mock the CPU detector re-initialization
            cpu_detector = MagicMock()
            # If the CPU detector ALSO fails with CUDA error (worst case)
            cpu_detector.get_landmarks.side_effect = Exception("CUDA error: persistent")
            MockFA.return_value = cpu_detector
            
            results = self.syncer.detect_faces(frames)
            
            # Assertions
            self.assertEqual(len(results), 2)
            self.assertIsNone(results[0]) # Skipped
            self.assertIsNone(results[1]) # Skipped
            
            # Check fallback happened
            self.assertTrue(self.syncer.fallback_active)
            logger.info("Test finished: Infinite loop avoided.")

    def test_cuda_error_then_success(self):
        """
        Simulate CUDA error on GPU, then success on CPU.
        """
        frames = [self.frame]
        
        # First call (GPU) raises Error
        # Second call (CPU) returns value
        # But we need to handle the fact that 'detector' is replaced.
        
        # Initial detector (mocked in setup)
        self.syncer.detector.get_landmarks.side_effect = Exception("CUDA error: init")
        
        with patch('src.processing.wav2lip.face_alignment.FaceAlignment') as MockFA:
            cpu_detector = MagicMock()
            cpu_detector.get_landmarks.return_value = [[(0,0), (10,10)]] # Fake landmarks
            MockFA.return_value = cpu_detector
            
            results = self.syncer.detect_faces(frames)
            
            self.assertEqual(len(results), 1)
            self.assertIsNotNone(results[0])
            self.assertTrue(self.syncer.fallback_active)

if __name__ == "__main__":
    unittest.main()
