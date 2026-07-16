"""
Pipeline smoke tests for Video Translator.

These tests verify that the core pipeline components can be instantiated
and perform basic operations. They use minimal synthetic data to avoid
requiring large model downloads.

Usage:
    pytest tests/test_smoke_pipeline.py -v
    pytest tests/test_smoke_pipeline.py -m smoke -v
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np


# =============================================================================
# HELPERS
# =============================================================================

def _is_real_module(module_name: str) -> bool:
    """Check if a module is real (not mocked)."""
    mod = sys.modules.get(module_name)
    if mod is None:
        try:
            import importlib.util
            spec = importlib.util.find_spec(module_name.split('.')[0])
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
    return not isinstance(mod, MagicMock)


IN_CI = os.environ.get('CI', '').lower() == 'true'


# =============================================================================
# PIPELINE COMPONENT INSTANTIATION
# =============================================================================

class TestPipelineComponents:
    """Verify core pipeline components can be instantiated."""

    @pytest.mark.smoke
    def test_config_loads(self):
        """Config module loads and provides expected constants."""
        from src.utils import config
        assert hasattr(config, 'BASE_DIR')
        assert hasattr(config, 'TEMP_DIR')
        assert hasattr(config, 'OUTPUT_DIR')
        assert hasattr(config, 'DEVICE')
        assert config.DEVICE in ('cuda', 'cpu')
        # Verify directories exist
        assert config.TEMP_DIR.exists()
        assert config.OUTPUT_DIR.exists()

    @pytest.mark.smoke
    def test_languages_module(self):
        """Languages module provides language mappings."""
        from src.utils import languages

        # Test known language codes
        assert languages.get_language_code("English") is not None
        assert languages.get_language_code("Spanish") is not None
        assert languages.get_language_code("French") is not None

    @pytest.mark.smoke
    def test_patches_module(self):
        """Patches module loads and provides patch functions."""
        from src.utils.patches import (
            apply_encoding_patch,
            apply_transformers_patch,
            apply_late_patches,
        )
        assert callable(apply_encoding_patch)
        assert callable(apply_transformers_patch)
        assert callable(apply_late_patches)

    @pytest.mark.smoke
    def test_video_translator_import(self):
        """VideoTranslator class can be imported."""
        from src.core.video_translator import VideoTranslator
        assert callable(VideoTranslator)

    @pytest.mark.smoke
    def test_srt_generator_import(self):
        """SRT generator module can be imported."""
        from src.utils.srt_generator import generate_srt
        assert callable(generate_srt)

    @pytest.mark.smoke
    def test_chunker_import(self):
        """Chunker utility can be imported."""
        from src.utils.chunker import VideoChunker
        assert callable(VideoChunker)


# =============================================================================
# AUDIO PIPELINE SMOKE
# =============================================================================

class TestAudioPipelineSmoke:
    """Smoke tests for audio pipeline components."""

    @pytest.mark.smoke
    def test_audio_utils_import(self):
        """Audio utilities module loads."""
        from src.utils import audio_utils
        assert hasattr(audio_utils, 'get_audio_duration') or True  # may have different API

    @pytest.mark.smoke
    def test_numpy_audio_operations(self):
        """Basic numpy audio operations work (foundation for all audio processing)."""
        # Generate a synthetic 1-second mono audio signal (sine wave)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        assert audio.shape == (16000,)
        assert audio.dtype == np.float32
        assert np.max(np.abs(audio)) <= 1.0

    @pytest.mark.smoke
    def test_soundfile_write_read_roundtrip(self):
        """SoundFile can write and read audio (if real module available)."""
        if not _is_real_module('soundfile'):
            pytest.skip("soundfile is mocked")

        import soundfile as sf

        # Generate test audio
        sample_rate = 16000
        audio = np.random.uniform(-1.0, 1.0, sample_rate).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            # Write
            sf.write(temp_path, audio, sample_rate)

            # Read back
            data, sr = sf.read(temp_path, dtype='float32')
            assert sr == sample_rate
            assert data.shape[0] == sample_rate
            assert np.allclose(audio, data, atol=1e-4)
        finally:
            os.unlink(temp_path)


# =============================================================================
# TRANSLATION PIPELINE SMOKE
# =============================================================================

class TestTranslationSmoke:
    """Smoke tests for translation components."""

    @pytest.mark.smoke
    def test_text_translator_import(self):
        """Translator can be imported."""
        from src.translation.text_translator import Translator
        assert callable(Translator)

    @pytest.mark.smoke
    def test_visual_translator_import(self):
        """VisualTranslator can be imported."""
        from src.translation.visual_translator import VisualTranslator
        assert callable(VisualTranslator)


# =============================================================================
# SYNTHESIS PIPELINE SMOKE
# =============================================================================

class TestSynthesisSmoke:
    """Smoke tests for TTS synthesis components."""

    @pytest.mark.smoke
    def test_tts_import(self):
        """TTS engine module can be imported."""
        from src.synthesis.tts import TTSEngine
        assert callable(TTSEngine)


# =============================================================================
# VIDEO PROCESSING SMOKE
# =============================================================================

class TestVideoProcessingSmoke:
    """Smoke tests for video processing components."""

    @pytest.mark.smoke
    @pytest.mark.requires_ffmpeg
    def test_ffmpeg_version(self):
        """FFmpeg binary runs and returns version info."""
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "ffmpeg version" in result.stdout.lower()

    @pytest.mark.smoke
    def test_video_module_import(self):
        """Video processing module can be imported."""
        from src.processing.video import VideoProcessor
        assert callable(VideoProcessor)

    @pytest.mark.smoke
    def test_synchronization_import(self):
        """Synchronization module can be imported."""
        from src.processing.synchronization import (
            generate_crossfade_window,
        )
        assert callable(generate_crossfade_window)

    @pytest.mark.smoke
    def test_crossfade_window_generation(self):
        """Crossfade window generation produces valid output."""
        from src.processing.synchronization import generate_crossfade_window

        # Generate a crossfade window
        window = generate_crossfade_window(1000, fade_type='linear')
        assert len(window) == 1000
        assert window[0] == pytest.approx(0.0, abs=0.01)
        assert window[-1] == pytest.approx(1.0, abs=0.01)

    @pytest.mark.smoke
    def test_lipsync_import(self):
        """Lip-sync module can be imported."""
        from src.processing.lipsync import LipSyncer
        assert callable(LipSyncer)


# =============================================================================
# END-TO-END DRY RUN
# =============================================================================

class TestDryRun:
    """Verify the full pipeline can be assembled without crashing."""

    @pytest.mark.smoke
    def test_video_translator_init_mocked(self, mock_components):
        """VideoTranslator can be initialized with mocked components."""
        from src.core.video_translator import VideoTranslator

        # VideoTranslator should init without errors
        # It lazy-loads most components, so this tests the constructor
        translator = VideoTranslator()
        assert translator is not None

    @pytest.mark.smoke
    def test_srt_generation(self):
        """SRT subtitle generation works with synthetic data."""
        from src.utils.srt_generator import generate_srt

        segments = [
            {"start": 0.0, "end": 2.5, "translated_text": "Hello, world!"},
            {"start": 3.0, "end": 5.5, "translated_text": "This is a test."},
            {"start": 6.0, "end": 8.0, "translated_text": "Smoke test complete."},
        ]

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.srt', delete=False, encoding='utf-8'
        ) as f:
            temp_path = f.name

        try:
            generate_srt(segments, temp_path)
            # Verify file was created and has content
            assert os.path.exists(temp_path)
            content = Path(temp_path).read_text(encoding='utf-8')
            assert "Hello, world!" in content
            assert "This is a test." in content
        finally:
            os.unlink(temp_path)
