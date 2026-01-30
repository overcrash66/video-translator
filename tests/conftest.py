import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Pre-import to avoid DLL conflicts on Windows
try:
    import ctranslate2
    print("DEBUG: ctranslate2 pre-imported in tests/conftest.py")
except ImportError:
    pass

@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory for test outputs."""
    return tmp_path

@pytest.fixture
def mock_video_path(temp_dir):
    """Creates a minimal test video file."""
    video_file = temp_dir / "test.mp4"
    video_file.touch()
    return video_file

@pytest.fixture
def mock_components():
    """Common mock components for VideoTranslator tests."""
    return {
        'separator': MagicMock(),
        'transcriber': MagicMock(),
        'translator': MagicMock(),
        'tts_engine': MagicMock(),
        'synchronizer': MagicMock(),
        'processor': MagicMock(),
        'diarizer': MagicMock(),
        'lipsyncer': MagicMock(),
        'visual_translator': MagicMock(),
        'voice_enhancer': MagicMock()
    }
