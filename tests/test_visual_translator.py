"""
Unit tests for the VisualTranslator class (src.translation.visual_translator).

Tests cover:
- Initialization (finding fonts)
- Model loading/unloading
- translation filtering (language detection)
- Text rendering (PIL)
- Video processing with interval
"""

import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
from pathlib import Path

# Mock external dependencies for integration tests
@pytest.fixture
def mock_pil():
    with patch('src.translation.visual_translator.Image') as mock_img, \
         patch('src.translation.visual_translator.ImageDraw') as mock_draw, \
         patch('src.translation.visual_translator.ImageFont') as mock_font:
        yield mock_img, mock_draw, mock_font

@pytest.fixture
def translator():
    from src.translation.visual_translator import VisualTranslator
    val = VisualTranslator()
    # Mock finding font to avoid system dependency in tests
    val.font_path = "arial.ttf"
    return val

class TestVisualTranslatorInit:
    
    def test_init_defaults(self, translator):
        """Verify initial state."""
        assert translator.ocr_model is None
        assert translator.model_loaded is False
        assert translator.font_path is not None

class TestVisualTranslatorLanguageDetection:

    def test_detect_language_success(self, translator):
        with patch('src.translation.visual_translator.LANGDETECT_AVAILABLE', True), \
             patch('src.translation.visual_translator.detect') as mock_detect:
            
            mock_detect.return_value = 'fr'
            lang = translator._detect_language("Bonjour le monde")
            assert lang == 'fr'

    def test_detect_language_too_short(self, translator):
        """Short text returns unknown."""
        lang = translator._detect_language("Hi")
        assert lang == "unknown"

    def test_translate_text_skips_same_language(self, translator):
        """Should skip translation if detected language == target language."""
        
        # Mock detection to say text is ALREADY French
        with patch.object(translator, '_detect_language', return_value='fr'):
            # Try to translate TO French
            result = translator._translate_text("Bonjour", target_lang='fr')
            # Should return original
            assert result == "Bonjour"
            
    def test_translate_text_skips_different_source(self, translator):
        """Should skip translation if detected != source (and source is enforced)."""
        # User says Source is EN.
        # Detector says text is AR.
        # Should SKIP (return original text).
        with patch.object(translator, '_detect_language', return_value='ar'):
            result = translator._translate_text("Some Arabic Text", target_lang='fr', source_lang='en')
            assert result == "Some Arabic Text"

    def test_translate_text_proceeds_if_source_match(self, translator):
        """Proceeds if detected == source."""
        with patch.object(translator, '_detect_language', return_value='en'), \
             patch.object(translator, '_cached_translate', return_value='Bonjour'):
             
             result = translator._translate_text("Hello", target_lang='fr', source_lang='en')
             assert result == "Bonjour"

class TestVisualTranslatorProcess:
    
    @patch('src.translation.visual_translator.PADDLE_AVAILABLE', True)
    @patch('src.translation.visual_translator.cv2')
    @patch('src.translation.visual_translator.PaddleOCR')
    def test_translate_video_text_interval(self, mock_ocr_cls, mock_cv2, translator, mock_pil):
        """Verify OCR runs only at specified intervals."""
        # Setup mock video
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_COUNT: 60
        }.get(x, 0)
        
        # 60 frames
         # side_effect generator
        frames = [True] * 60 + [False]
        mock_cap.read.side_effect = [(True, np.zeros((100,100,3), dtype=np.uint8)) for _ in range(60)] + [(False, None)]
        
        mock_writer = MagicMock()
        mock_cv2.VideoWriter.return_value = mock_writer
        
        # Mock OCR instance
        mock_ocr = MagicMock()
        mock_ocr_cls.return_value = mock_ocr
        # Return some text
        mock_ocr.ocr.return_value = [[([[10,10],[20,10],[20,20],[10,20]], ("Hello", 0.9))]]
        
        # Run with interval 1.0s (every 30 frames)
        translator.load_model()
        
        # Mock translation to always return changed text
        with patch.object(translator, '_translate_text', return_value="Bonjour"):
            translator.translate_video_text("in.mp4", "out.mp4", ocr_interval_sec=1.0)
        
        # OCR should be called twice (Frame 0 and Frame 30) for 60 frames (0..59)
        assert mock_ocr.ocr.call_count == 2
        
        # Writer should handle 60 frames
        assert mock_writer.write.call_count == 60

    @patch('src.translation.visual_translator.PADDLE_AVAILABLE', True)
    @patch('src.translation.visual_translator.cv2')
    @patch('src.translation.visual_translator.PaddleOCR')
    def test_translate_video_inpaint_persistence(self, mock_ocr_cls, mock_cv2, translator, mock_pil):
        """Verify that inpainting mask persists between OCR calls."""
         # Setup mock video - 10 frames
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            mock_cv2.CAP_PROP_FPS: 30.0,
        }.get(x, 0)
        
        mock_cap.read.side_effect = [(True, np.zeros((100,100,3), dtype=np.uint8)) for _ in range(10)] + [(False, None)]
        
        mock_writer = MagicMock()
        mock_cv2.VideoWriter.return_value = mock_writer
        
        mock_ocr = MagicMock()
        mock_ocr_cls.return_value = mock_ocr
        # Return text ONLY on first call
        mock_ocr.ocr.return_value = [[([[10,10],[20,10],[20,20],[10,20]], ("Hello", 0.9))]]
        
        # Interval 10s (300 frames) -> OCR runs only on frame 0
        translator.load_model()
        
        with patch.object(translator, '_inpaint_text_regions') as mock_inpaint:
            # Mock overlay too to avoid PIL issues here if not fully mocked
            with patch.object(translator, '_overlay_translated_text_pil') as mock_overlay:
                 # Mock translation to ensure it triggers inpaint
                with patch.object(translator, '_translate_text', return_value="Bonjour"):
                    translator.translate_video_text("in.mp4", "out.mp4", ocr_interval_sec=10.0)
                
                # Verify OCR called once
                assert mock_ocr.ocr.call_count == 1
                
                # Verify inpaint called 10 times (persisted)
                assert mock_inpaint.call_count == 10
