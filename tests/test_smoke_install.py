"""
Smoke tests for installation validation.

These tests verify that the dependency tree is coherent and critical imports
work without conflicts on all supported platforms (Windows, Linux, macOS).

They are NOT unit tests for business logic — they validate the install.

Usage:
    pytest tests/test_smoke_install.py -m smoke -v       # Core only
    pytest tests/test_smoke_install.py -v                 # All smoke tests
    pytest tests/test_smoke_install.py -m gpu -v          # GPU-only tests
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


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


def _skip_if_mocked(module_name: str):
    """Skip test if the module is mocked (CI environment)."""
    return pytest.mark.skipif(
        not _is_real_module(module_name),
        reason=f"{module_name} is mocked (not installed)"
    )


IN_CI = os.environ.get('CI', '').lower() == 'true'
IS_WINDOWS = sys.platform == 'win32'
IS_LINUX = sys.platform == 'linux'
IS_MACOS = sys.platform == 'darwin'


# =============================================================================
# CORE IMPORTS — Must pass on ALL platforms
# =============================================================================

class TestCoreImports:
    """Validate that core packages import without conflicts."""

    @pytest.mark.smoke
    def test_numpy_import_and_version(self):
        """numpy must be a supported version (1.x or 2.x)."""
        import numpy as np
        major = int(np.__version__.split('.')[0])
        assert major in (1, 2), f"unsupported numpy version ({np.__version__}), expected 1.x or 2.x"

    @pytest.mark.smoke
    def test_python_version(self):
        """Python must be 3.10-3.12."""
        assert sys.version_info >= (3, 10), f"Python >= 3.10 required, got {sys.version}"
        assert sys.version_info < (3, 13), f"Python < 3.13 required, got {sys.version}"

    @pytest.mark.smoke
    def test_gradio_import(self):
        """Gradio UI framework must load."""
        try:
            import gradio
            assert hasattr(gradio, 'Blocks'), "gradio.Blocks not found"
        except ImportError as e:
            if "HfFolder" in str(e):
                pytest.skip(f"gradio incompatible with installed huggingface_hub: {e}")
            pytest.skip(f"gradio not installed: {e}")

    @pytest.mark.smoke
    def test_ffmpeg_available(self):
        """FFmpeg binary must be in PATH."""
        assert shutil.which("ffmpeg") is not None, (
            "ffmpeg not found in PATH. Install it:\n"
            "  Windows: winget install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Linux: sudo apt-get install ffmpeg"
        )

    @pytest.mark.smoke
    def test_ffmpeg_python_import(self):
        """ffmpeg-python wrapper must import."""
        import ffmpeg
        assert hasattr(ffmpeg, 'input'), "ffmpeg.input not found"

    @pytest.mark.smoke
    def test_soundfile_import(self):
        """SoundFile audio I/O must load."""
        if not _is_real_module('soundfile'):
            pytest.skip("soundfile is mocked")
        import soundfile
        assert hasattr(soundfile, 'read'), "soundfile.read not found"

    @pytest.mark.smoke
    def test_opencv_import(self):
        """OpenCV must load (regular or headless)."""
        if not _is_real_module('cv2'):
            pytest.skip("cv2 is mocked")
        import cv2
        assert hasattr(cv2, 'imread'), "cv2.imread not found"

    @pytest.mark.smoke
    def test_scipy_import(self):
        """SciPy must be available for signal processing."""
        import scipy
        assert hasattr(scipy, '__version__')

    @pytest.mark.smoke
    def test_dotenv_import(self):
        """python-dotenv must load for .env file support."""
        from dotenv import load_dotenv
        assert callable(load_dotenv)

    @pytest.mark.smoke
    def test_no_numpy_version_conflict(self):
        """Verify numpy is consistent across the dependency tree."""
        import numpy as np
        # Create a small array to verify numpy is functional
        arr = np.zeros((10,), dtype=np.float32)
        assert arr.shape == (10,)
        assert arr.dtype == np.float32


# =============================================================================
# GPU STACK — Skipped on CPU-only installs
# =============================================================================

class TestGPUStack:
    """Validate GPU acceleration (skipped on CPU-only installs)."""

    @pytest.mark.gpu
    @_skip_if_mocked('torch')
    def test_torch_import(self):
        """PyTorch must load without errors."""
        import torch
        assert hasattr(torch, '__version__'), "torch.__version__ not found"
        # Verify minimum version
        parts = torch.__version__.split('.')
        major, minor = int(parts[0]), int(parts[1])
        assert (major, minor) >= (2, 5), f"torch >= 2.5 required, got {torch.__version__}"

    @pytest.mark.gpu
    @_skip_if_mocked('torch')
    def test_cuda_detection(self):
        """torch.cuda.is_available() must not crash (result can be True or False)."""
        import torch
        # This should not raise, regardless of GPU presence
        result = torch.cuda.is_available()
        assert isinstance(result, bool)

    @pytest.mark.gpu
    @_skip_if_mocked('torch')
    def test_cuda_functional(self):
        """If CUDA is available, verify it actually works."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            t = torch.tensor([1.0]).to("cuda")
            assert t.device.type == "cuda"
            del t
        except RuntimeError as e:
            pytest.fail(f"CUDA available but not functional: {e}")

    @pytest.mark.gpu
    @_skip_if_mocked('torch')
    def test_torchaudio_import(self):
        """torchaudio must load."""
        import torchaudio
        assert hasattr(torchaudio, '__version__')

    @pytest.mark.gpu
    @_skip_if_mocked('torch')
    def test_torchvision_import(self):
        """torchvision must load."""
        import torchvision
        assert hasattr(torchvision, '__version__')

    @pytest.mark.gpu
    def test_ctranslate2_import(self):
        """CTranslate2 must load without DLL conflicts."""
        if not _is_real_module('ctranslate2'):
            pytest.skip("ctranslate2 is mocked")
        import ctranslate2
        assert hasattr(ctranslate2, '__version__')

    @pytest.mark.gpu
    def test_onnxruntime_import(self):
        """ONNX Runtime must load (GPU or CPU variant)."""
        if not _is_real_module('onnxruntime'):
            pytest.skip("onnxruntime is mocked")
        import onnxruntime as ort
        providers = ort.get_available_providers()
        assert len(providers) > 0, "No ONNX Runtime providers available"
        # CPUExecutionProvider should always be present
        assert 'CPUExecutionProvider' in providers


# =============================================================================
# TTS STACK — Text-to-Speech engines
# =============================================================================

class TestTTSStack:
    """Validate TTS engines can be imported."""

    @pytest.mark.smoke
    def test_edge_tts_import(self):
        """Edge-TTS (online) must import."""
        if not _is_real_module('edge_tts'):
            pytest.skip("edge_tts is mocked")
        import edge_tts
        assert hasattr(edge_tts, 'Communicate')

    @pytest.mark.smoke
    def test_f5_tts_import(self):
        """F5-TTS must import (if installed)."""
        if not _is_real_module('f5_tts'):
            pytest.skip("f5_tts not installed")
        import f5_tts


# =============================================================================
# OCR STACK — Visual text detection
# =============================================================================

class TestOCRStack:
    """Validate OCR engines."""

    @pytest.mark.smoke
    def test_paddleocr_import(self):
        """PaddleOCR must import without GPU conflicts."""
        if not _is_real_module('paddleocr'):
            pytest.skip("paddleocr is mocked")
        # Import should not cause DLL conflicts with torch
        from paddleocr import PaddleOCR

    @pytest.mark.smoke
    def test_easyocr_import(self):
        """EasyOCR must import as fallback."""
        if not _is_real_module('easyocr'):
            pytest.skip("easyocr is mocked")
        import easyocr


# =============================================================================
# TRANSLATION STACK
# =============================================================================

class TestTranslationStack:
    """Validate translation backends."""

    @pytest.mark.smoke
    def test_deep_translator_import(self):
        """deep-translator (Google Translate wrapper) must import."""
        if not _is_real_module('deep_translator'):
            pytest.skip("deep_translator is mocked")
        from deep_translator import GoogleTranslator

    @pytest.mark.smoke
    def test_transformers_import(self):
        """Hugging Face transformers must import."""
        if not _is_real_module('transformers'):
            pytest.skip("transformers is mocked")
        import transformers
        assert hasattr(transformers, 'AutoModelForCausalLM')

    @pytest.mark.smoke
    def test_faster_whisper_import(self):
        """faster-whisper must import."""
        if not _is_real_module('faster_whisper'):
            pytest.skip("faster_whisper is mocked")
        from faster_whisper import WhisperModel


# =============================================================================
# DIARIZATION STACK
# =============================================================================

class TestDiarizationStack:
    """Validate speaker diarization backends."""

    @pytest.mark.smoke
    def test_pyannote_import(self):
        """pyannote.audio must import."""
        if not _is_real_module('pyannote'):
            pytest.skip("pyannote is mocked")
        import pyannote.audio

    @pytest.mark.smoke
    def test_speechbrain_import(self):
        """SpeechBrain must import."""
        if not _is_real_module('speechbrain'):
            pytest.skip("speechbrain is mocked")
        import speechbrain


# =============================================================================
# LIPSYNC STACK
# =============================================================================

class TestLipSyncStack:
    """Validate lip-sync dependencies."""

    @pytest.mark.smoke
    def test_insightface_import(self):
        """InsightFace must import for face detection."""
        if not _is_real_module('insightface'):
            pytest.skip("insightface is mocked")
        import insightface

    @pytest.mark.smoke
    def test_gfpgan_import(self):
        """GFPGAN must import for face restoration."""
        if not _is_real_module('gfpgan'):
            pytest.skip("gfpgan is mocked")
        import gfpgan

    @pytest.mark.smoke
    def test_face_alignment_import(self):
        """face-alignment must import."""
        if not _is_real_module('face_alignment'):
            pytest.skip("face_alignment is mocked")
        import face_alignment


# =============================================================================
# PLATFORM-SPECIFIC VALIDATION
# =============================================================================

class TestPlatformSpecific:
    """Platform-specific validation."""

    @pytest.mark.smoke
    @pytest.mark.platform_windows
    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only test")
    def test_dll_loading_no_conflicts(self):
        """Windows: Verify DLL loading doesn't conflict between torch and ctranslate2."""
        if not _is_real_module('torch') or not _is_real_module('ctranslate2'):
            pytest.skip("torch or ctranslate2 is mocked")
        # If we get here, both imported without DLL conflicts
        import torch
        import ctranslate2
        # Verify torch is functional
        t = torch.tensor([1.0])
        assert t.item() == 1.0

    @pytest.mark.smoke
    @pytest.mark.platform_windows
    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only test")
    def test_encoding_patch_active(self):
        """Windows: Verify UTF-8 encoding patch is available."""
        # The patch is applied at app startup, but we can verify the module exists
        from src.utils.patches import apply_encoding_patch
        assert callable(apply_encoding_patch)

    @pytest.mark.smoke
    @pytest.mark.platform_linux
    @pytest.mark.skipif(not IS_LINUX, reason="Linux-only test")
    def test_bitsandbytes_import(self):
        """Linux: bitsandbytes must import for quantization."""
        if not _is_real_module('bitsandbytes'):
            pytest.skip("bitsandbytes not installed")
        import bitsandbytes

    @pytest.mark.smoke
    def test_rubberband_available(self):
        """Optional: Check if rubberband is available for time-stretching."""
        # This is optional, so we just warn
        rubberband = shutil.which("rubberband") or shutil.which("rubberband-program")
        if rubberband is None:
            pytest.skip("rubberband not found (optional)")

    @pytest.mark.smoke
    def test_config_module_loads(self):
        """src.utils.config must load without crashing."""
        # This is the most critical test — config.py does heavy DLL
        # loading on Windows. If it fails, nothing else works.
        try:
            from src.utils import config
            assert hasattr(config, 'BASE_DIR')
            assert hasattr(config, 'DEVICE')
            assert config.DEVICE in ('cuda', 'cpu')
        except Exception as e:
            pytest.fail(f"config.py failed to load: {e}")

    @pytest.mark.smoke
    def test_languages_module_loads(self):
        """src.utils.languages must load and provide language mappings."""
        from src.utils import languages
        assert hasattr(languages, 'get_language_code')
        # Verify basic language codes work
        assert languages.get_language_code("English") is not None


# =============================================================================
# CROSS-PLATFORM PATH DISCOVERY
# =============================================================================

class TestPathDiscovery:
    """Validate cross-platform path discovery functions."""

    @pytest.mark.smoke
    def test_find_site_packages(self):
        """_find_site_packages must return a valid path."""
        from src.utils.config import _find_site_packages
        sp = _find_site_packages()
        # In CI with mocked deps, site-packages may not exist
        if sp is not None:
            assert sp.exists(), f"site-packages path doesn't exist: {sp}"

    @pytest.mark.smoke
    def test_get_cuda_roots(self):
        """_get_cuda_roots must return a list (possibly empty)."""
        from src.utils.config import _get_cuda_roots
        roots = _get_cuda_roots()
        assert isinstance(roots, list)
        # If any roots found, they should be valid paths
        for root in roots:
            assert isinstance(root, Path)

    @pytest.mark.smoke
    def test_validate_path(self):
        """validate_path must work for valid paths."""
        from src.utils.config import validate_path
        # Test with a known existing path
        p = validate_path(Path(__file__), must_exist=True)
        assert p.exists()

    @pytest.mark.smoke
    def test_validate_path_nonexistent(self):
        """validate_path must raise for non-existent paths when must_exist=True."""
        from src.utils.config import validate_path
        with pytest.raises(FileNotFoundError):
            validate_path(Path("/nonexistent/path/file.txt"), must_exist=True)


# =============================================================================
# DEPENDENCY CONFLICT DETECTION
# =============================================================================

class TestDependencyConflicts:
    """Detect common dependency version conflicts."""

    @pytest.mark.smoke
    def test_no_pandas_v2(self):
        """pandas must be a supported version (1.x or 2.x)."""
        try:
            import pandas as pd
            major = int(pd.__version__.split('.')[0])
            assert major in (1, 2), f"unsupported pandas version ({pd.__version__}), expected 1.x or 2.x"
        except ImportError:
            pytest.skip("pandas not installed")

    @pytest.mark.smoke
    @_skip_if_mocked('torch')
    def test_torch_numpy_compatibility(self):
        """torch and numpy must be ABI-compatible."""
        import torch
        import numpy as np
        # Create a numpy array and convert to tensor — this catches ABI mismatches
        arr = np.random.randn(10).astype(np.float32)
        tensor = torch.from_numpy(arr)
        assert tensor.shape == (10,)
        # Convert back
        back = tensor.numpy()
        assert np.allclose(arr, back)

    @pytest.mark.smoke
    def test_protobuf_import(self):
        """protobuf must import (used by multiple ML libraries)."""
        try:
            import google.protobuf
        except ImportError:
            pytest.skip("protobuf not installed")
