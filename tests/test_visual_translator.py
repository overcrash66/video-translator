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
            
    def test_translate_text_proceeds_even_if_mismatch(self, translator):
        """Should PROCEED even if detected != source (relaxed filtering)."""
        # User says Source is EN.
        # Detector says text is AR.
        # Should now PROCEED (return translated text).
        with patch.object(translator, '_detect_language', return_value='ar'), \
             patch.object(translator, '_cached_translate', return_value='Un peu de texte arabe'):
            result = translator._translate_text("Some Arabic Text", target_lang='fr', source_lang='en')
            assert result == "Un peu de texte arabe"

    def test_translate_text_proceeds_if_source_match(self, translator):
        """Proceeds if detected == source."""
        with patch.object(translator, '_detect_language', return_value='en'), \
             patch.object(translator, '_cached_translate', return_value='Bonjour'):
             
             result = translator._translate_text("Hello", target_lang='fr', source_lang='en')
             assert result == "Bonjour"

class TestVisualTranslatorProcess:
    
    @pytest.mark.requires_real_audio  # Complex cv2/PIL interaction
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
        
        # Mock boundingRect to return proper values
        mock_cv2.boundingRect.return_value = (10, 10, 10, 10)
        
        # Mock OCR instance
        mock_ocr = MagicMock()
        mock_ocr_cls.return_value = mock_ocr
        # Return some text
        mock_ocr.ocr.return_value = [[([[10,10],[20,10],[20,20],[10,20]], ("Hello", 0.9))]]
        
        # Run with interval 1.0s (every 30 frames)
        translator.load_model(ocr_engine="PaddleOCR")
        
        # Mock translation to always return changed text
        with patch.object(translator, '_translate_text', return_value="Bonjour"):
            translator.translate_video_text("in.mp4", "out.mp4", ocr_engine="PaddleOCR", ocr_interval_sec=1.0)
        
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
        translator.load_model(ocr_engine="PaddleOCR")
        
        with patch.object(translator, '_inpaint_text_regions') as mock_inpaint:
            # Mock overlay too to avoid PIL issues here if not fully mocked
            with patch.object(translator, '_overlay_translated_text_pil') as mock_overlay:
                 # Mock translation to ensure it triggers inpaint
                with patch.object(translator, '_translate_text', return_value="Bonjour"):
                    translator.translate_video_text("in.mp4", "out.mp4", ocr_engine="PaddleOCR", ocr_interval_sec=10.0)
                
                # Verify OCR called once
                assert mock_ocr.ocr.call_count == 1
                
                # Verify inpaint called 10 times (persisted)
                assert mock_inpaint.call_count == 10


class TestVisualTranslatorConfidenceFiltering:
    """Tests for OCR confidence filtering (improvement #4)."""
    
    @pytest.mark.requires_real_audio  # Complex cv2/PIL interaction
    @patch('src.translation.visual_translator.PADDLE_AVAILABLE', True)
    @patch('src.translation.visual_translator.cv2')
    @patch('src.translation.visual_translator.PaddleOCR')
    def test_low_confidence_detections_filtered(self, mock_ocr_cls, mock_cv2, translator, mock_pil):
        """Verify that low-confidence OCR detections are skipped."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_COUNT: 2
        }.get(x, 0)
        
        # 2 frames
        mock_cap.read.side_effect = [(True, np.zeros((100,100,3), dtype=np.uint8)) for _ in range(2)] + [(False, None)]
        
        mock_writer = MagicMock()
        mock_cv2.VideoWriter.return_value = mock_writer
        
        # Mock boundingRect to return proper values
        mock_cv2.boundingRect.return_value = (10, 10, 10, 10)
        
        mock_ocr = MagicMock()
        mock_ocr_cls.return_value = mock_ocr
        # Return mix of high and low confidence results
        mock_ocr.ocr.return_value = [[
            ([[10,10],[20,10],[20,20],[10,20]], ("HighConf", 0.95)),  # Should be kept
            ([[30,30],[40,30],[40,40],[30,40]], ("LowConf", 0.2)),    # Should be filtered
        ]]
        
        translator.load_model(ocr_engine="PaddleOCR")
        
        with patch.object(translator, '_translate_text') as mock_translate:
            mock_translate.return_value = "Translated"
            translator.translate_video_text("in.mp4", "out.mp4", ocr_engine="PaddleOCR", ocr_interval_sec=0.1)
            
            # Should only be called once (for HighConf), LowConf should be filtered
            # Note: _translate_text is called per text, not per detection
            call_texts = [call[0][0] for call in mock_translate.call_args_list]
            assert "HighConf" in call_texts
            assert "LowConf" not in call_texts


class TestVisualTranslatorCJKFonts:
    """Tests for CJK font selection (improvement #3)."""
    
    def test_get_font_for_japanese(self, translator):
        """Verify Japanese language gets CJK font candidates."""
        with patch.object(translator, '_find_font') as mock_find:
            mock_find.return_value = "msgothic.ttc"
            font = translator._get_font_for_language('ja')
            # Verify CJK candidates were passed
            mock_find.assert_called_once()
            candidates = mock_find.call_args[0][0]
            assert 'msgothic.ttc' in candidates
    
    def test_get_font_for_chinese(self, translator):
        """Verify Chinese language gets CJK font candidates."""
        with patch.object(translator, '_find_font') as mock_find:
            mock_find.return_value = "simsun.ttc"
            font = translator._get_font_for_language('zh')
            mock_find.assert_called_once()
            candidates = mock_find.call_args[0][0]
            assert 'simsun.ttc' in candidates
    
    def test_get_font_for_non_cjk(self, translator):
        """Verify non-CJK language returns default font path."""
        translator.font_path = "arial.ttf"
        font = translator._get_font_for_language('fr')
        assert font == "arial.ttf"


class TestVisualTranslatorThreadSafety:
    """Tests for thread-safe caching (improvement #1)."""
    
    def test_cache_operations_thread_safe(self, translator):
        """Verify cache operations don't raise under concurrent access."""
        import threading
        
        errors = []
        
        # Mock _get_translator globally for this test to avoid race conditions in mock setup
        with patch.object(translator, '_get_translator') as mock_get_trans:
            # Setup mock translator instance with side_effect
            mock_trans_instance = MagicMock()
            mock_trans_instance.translate.side_effect = lambda t: f"translated_{t}"
            mock_get_trans.return_value = mock_trans_instance
            
            def cache_operation(thread_id):
                try:
                    for i in range(10):
                        text = f"text_{thread_id}_{i}"
                        result = translator._cached_translate(text, 'fr')
                        # Check result matches expected translation
                        if result != f"translated_{text}":
                             errors.append(AssertionError(f"Expected translated_{text}, got {result}"))
                except Exception as e:
                    errors.append(e)
        
            threads = [threading.Thread(target=cache_operation, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"
    
    def test_cache_clear_thread_safe(self, translator):
        """Verify unload_model clears cache without errors."""
        # Pre-populate cache
        with patch.object(translator, '_get_translator') as mock_trans:
            mock_trans.return_value.translate.return_value = "cached"
            translator._cached_translate("hello", "fr")
        
        # Should not raise
        translator.unload_model()
        
        # Verify cache is empty
        assert len(translator._translation_cache) == 0
