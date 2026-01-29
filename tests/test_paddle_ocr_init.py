"""
Unit tests for PaddleOCR initialization and memory optimizations.

Verifies that:
- PaddleOCR is initialized with correct arguments (no deprecated 'show_log')
- Memory optimization methods work correctly (frame resizing, box scaling)
- LRU cache sizes are bounded
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.translation.visual_translator import (
    VisualTranslator, 
    MAX_OCR_WIDTH, 
    GC_INTERVAL_FRAMES
)


class TestPaddleOCRInitialization:
    """Tests for PaddleOCR initialization in VisualTranslator."""

    @patch('src.translation.visual_translator.PADDLE_AVAILABLE', True)
    @patch('src.translation.visual_translator.PaddleOCR')
    def test_paddle_ocr_init_excludes_deprecated_show_log_arg(self, mock_paddle_cls):
        """
        PaddleOCR should NOT be initialized with 'show_log' argument
        which is deprecated and causes crashes in newer versions.
        """
        translator = VisualTranslator()
        
        translator.load_model(source_lang='en', ocr_engine="PaddleOCR")
        
        mock_paddle_cls.assert_called_once()
        _, kwargs = mock_paddle_cls.call_args
        
        assert 'show_log' not in kwargs, \
            "PaddleOCR initialized with deprecated 'show_log' argument"

    @patch('src.translation.visual_translator.PADDLE_AVAILABLE', True)
    @patch('src.translation.visual_translator.PaddleOCR')
    def test_paddle_ocr_init_includes_required_args(self, mock_paddle_cls):
        """PaddleOCR should be initialized with required configuration arguments."""
        translator = VisualTranslator()
        
        translator.load_model(source_lang='en', ocr_engine="PaddleOCR")
        
        _, kwargs = mock_paddle_cls.call_args
        
        assert kwargs.get('use_angle_cls') is True
        assert kwargs.get('lang') == 'en'


class TestMemoryOptimizations:
    """Tests for memory optimization features."""

    def test_max_ocr_width_constant_is_reasonable(self):
        """MAX_OCR_WIDTH should be set to a reasonable value for VRAM savings."""
        assert MAX_OCR_WIDTH > 0
        assert MAX_OCR_WIDTH <= 1920  # Should not exceed 1080p width
        assert MAX_OCR_WIDTH == 1280  # Current expected value

    def test_gc_interval_frames_constant_is_reasonable(self):
        """GC_INTERVAL_FRAMES should be frequent enough to prevent memory buildup."""
        assert GC_INTERVAL_FRAMES > 0
        assert GC_INTERVAL_FRAMES <= 200  # Should be reasonably frequent
        assert GC_INTERVAL_FRAMES == 100  # Current expected value

    def test_resize_for_ocr_small_frame_unchanged(self):
        """Frames smaller than MAX_OCR_WIDTH should not be resized."""
        translator = VisualTranslator()
        
        # Create a frame smaller than MAX_OCR_WIDTH
        small_frame = np.zeros((720, 1000, 3), dtype=np.uint8)
        
        resized, scale = translator._resize_for_ocr(small_frame)
        
        assert scale == 1.0
        assert resized.shape == small_frame.shape

    def test_resize_for_ocr_large_frame_resized(self):
        """Frames larger than MAX_OCR_WIDTH should be resized."""
        translator = VisualTranslator()
        
        # Create a 4K frame (3840x2160)
        large_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        
        resized, scale = translator._resize_for_ocr(large_frame)
        
        assert scale < 1.0
        assert resized.shape[1] == MAX_OCR_WIDTH
        # Height should be proportionally scaled
        expected_height = int(2160 * scale)
        assert resized.shape[0] == expected_height

    def test_scale_boxes_identity_scale(self):
        """Boxes with scale=1.0 should be returned unchanged."""
        translator = VisualTranslator()
        
        boxes = [
            [[0, 0], [100, 0], [100, 50], [0, 50]],
            [[200, 100], [300, 100], [300, 150], [200, 150]]
        ]
        
        scaled = translator._scale_boxes(boxes, 1.0)
        
        assert scaled == boxes

    def test_scale_boxes_rescales_coordinates(self):
        """Boxes should be scaled back to original coordinates."""
        translator = VisualTranslator()
        
        # Simulate boxes detected on a frame resized by 0.5
        boxes = [
            [[50, 25], [100, 25], [100, 50], [50, 50]]
        ]
        scale = 0.5  # Frame was resized to half
        
        scaled = translator._scale_boxes(boxes, scale)
        
        # Coordinates should be doubled (inverse of 0.5 scale)
        expected = [
            [[100.0, 50.0], [200.0, 50.0], [200.0, 100.0], [100.0, 100.0]]
        ]
        
        assert len(scaled) == len(expected)
        for i, box in enumerate(scaled):
            for j, pt in enumerate(box):
                assert pt[0] == pytest.approx(expected[i][j][0])
                assert pt[1] == pytest.approx(expected[i][j][1])

    def test_lru_cache_sizes_are_bounded(self):
        """LRU caches should have bounded sizes to prevent RAM growth."""
        translator = VisualTranslator()
        
        # Check cache info for both cached methods
        cached_translate_info = translator._cached_translate.cache_info()
        translate_text_info = translator._translate_text.cache_info()
        
        # Verify maxsize is bounded (not unlimited)
        assert cached_translate_info.maxsize is not None
        assert translate_text_info.maxsize is not None
        
        # Verify reasonable bounds
        assert cached_translate_info.maxsize <= 500
        assert translate_text_info.maxsize <= 200


class TestVisualTranslatorScaleFactorTracking:
    """Tests for scale factor tracking in VisualTranslator."""

    def test_ocr_scale_factor_initialized(self):
        """VisualTranslator should initialize _ocr_scale_factor."""
        translator = VisualTranslator()
        
        assert hasattr(translator, '_ocr_scale_factor')
        assert translator._ocr_scale_factor == 1.0
