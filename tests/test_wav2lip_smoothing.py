
import unittest
import numpy as np
from src.processing.wav2lip import Wav2LipSyncer

class TestWav2LipSmoothing(unittest.TestCase):
    def test_get_smooth_box_basic(self):
        syncer = Wav2LipSyncer()
        
        # Create a sequence of boxes: constant
        boxes = [[10, 10, 50, 50]] * 10
        smoothed = syncer.get_smooth_box(boxes, window_size=5)
        
        self.assertEqual(len(smoothed), 10)
        np.testing.assert_array_equal(smoothed[0], [10, 10, 50, 50])
        
    def test_get_smooth_box_jitter(self):
        syncer = Wav2LipSyncer()
        
        # Jittery box: 10, 12, 10, 8, 10 -> Avg 10
        boxes = [
            [10, 10, 50, 50],
            [12, 10, 50, 50],
            [10, 10, 50, 50],
            [8, 10, 50, 50],
            [10, 10, 50, 50]
        ]
        smoothed = syncer.get_smooth_box(boxes, window_size=5)
        
        # Center frame (idx 2) should be avg of all 5
        # 10+12+10+8+10 = 50 / 5 = 10
        self.assertEqual(smoothed[2], [10, 10, 50, 50])
        
    def test_get_smooth_box_none_filling(self):
        syncer = Wav2LipSyncer()
        
        # [Box, None, Box] -> None should be filled with previous
        boxes = [
            [10, 10, 50, 50],
            None,
            [20, 20, 60, 60]
        ]
        smoothed = syncer.get_smooth_box(boxes, window_size=1) # Window 1 = no averaging, just fill check
        
        self.assertIsNotNone(smoothed[1])
        self.assertEqual(smoothed[1], [10, 10, 50, 50]) # Forward filled
        
    def test_get_smooth_box_all_none(self):
        syncer = Wav2LipSyncer()
        boxes = [None, None]
        smoothed = syncer.get_smooth_box(boxes)
        self.assertEqual(smoothed, boxes) # Should remain None if nothing to use

if __name__ == "__main__":
    unittest.main()
