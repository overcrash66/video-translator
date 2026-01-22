import unittest
from src.audio.transcription import Transcriber

class TestSegmentMerging(unittest.TestCase):
    
    def setUp(self):
        self.transcriber = Transcriber()
        # Prevent loading actual models in tests
        self.transcriber.model = "DUMMY"
        
    def test_merge_short_segments_basic(self):
        """Test merging two short segments with small gap."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "words": []},
            {"start": 1.2, "end": 2.2, "text": "world", "words": []}
        ]
        
        # Duration 1.0 < 2.0, Gap 0.2 < 0.5 -> Should merge
        merged = self.transcriber.merge_short_segments(segments, min_duration=2.0, max_gap=0.5)
        
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['start'], 0.0)
        self.assertEqual(merged[0]['end'], 2.2)
        self.assertEqual(merged[0]['text'], "Hello world")
        
    def test_no_merge_long_gap(self):
        """Test NOT merging if gap is too large."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "words": []},
            {"start": 2.0, "end": 3.0, "text": "world", "words": []}
        ]
        
        # Gap 1.0 > 0.5 -> Should NOT merge
        merged = self.transcriber.merge_short_segments(segments, min_duration=2.0, max_gap=0.5)
        
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]['text'], "Hello")
        self.assertEqual(merged[1]['text'], "world")
        
    def test_no_merge_long_segment(self):
        """Test NOT merging if first segment is long enough."""
        segments = [
            {"start": 0.0, "end": 3.0, "text": "This is a long sentence.", "words": []},
            {"start": 3.1, "end": 4.1, "text": "Next.", "words": []}
        ]
        
        # Duration 3.0 > 2.0 -> Should NOT merge
        merged = self.transcriber.merge_short_segments(segments, min_duration=2.0, max_gap=0.5)
        
        self.assertEqual(len(merged), 2)
        
    def test_chain_merge(self):
        """Test merging a chain of short segments."""
        segments = [
            {"start": 0.0, "end": 0.5, "text": "A", "words": []},
            {"start": 0.6, "end": 1.1, "text": "B", "words": []},
            {"start": 1.2, "end": 1.7, "text": "C", "words": []},
            {"start": 5.0, "end": 6.0, "text": "D", "words": []} # Far away
        ]
        
        # A (0.5s) + B (0.5s) + C (0.5s) -> Combined
        # Gap between C(1.7) and D(5.0) is 3.3s -> No merge
        
        merged = self.transcriber.merge_short_segments(segments, min_duration=5.0, max_gap=0.5)
        
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]['text'], "A B C")
        self.assertEqual(merged[1]['text'], "D")

    def test_empty_segments(self):
        self.assertEqual(self.transcriber.merge_short_segments([]), [])

if __name__ == "__main__":
    unittest.main()
