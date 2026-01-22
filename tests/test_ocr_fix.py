import pytest
from unittest.mock import MagicMock, patch
from src.translation.visual_translator import VisualTranslator

@patch('src.translation.visual_translator.PADDLE_AVAILABLE', True)
@patch('src.translation.visual_translator.PaddleOCR')
def test_paddle_ocr_init_arguments(mock_paddle_cls):
    """
    Verify PaddleOCR is initialized WITHOUT 'show_log' argument which causes crash.
    """
    translator = VisualTranslator()
    
    # Act
    translator.load_model(source_lang='en', ocr_engine="PaddleOCR")
    
    # Assert
    mock_paddle_cls.assert_called_once()
    
    # Check call arguments
    args, kwargs = mock_paddle_cls.call_args
    
    # Ensure 'show_log' is NOT in kwargs
    assert 'show_log' not in kwargs, "PaddleOCR initialized with deprecated 'show_log' argument"
    
    # Ensure other required args ARE present
    assert kwargs.get('use_angle_cls') is True
    assert kwargs.get('enable_mkldnn') is False
    assert kwargs.get('lang') == 'en'
