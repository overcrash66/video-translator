"""
Unit tests for PaddleOCR initialization.

Verifies that PaddleOCR is initialized with correct arguments,
particularly ensuring deprecated arguments like 'show_log' are not used.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.translation.visual_translator import VisualTranslator


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
        assert kwargs.get('enable_mkldnn') is False
        assert kwargs.get('lang') == 'en'
