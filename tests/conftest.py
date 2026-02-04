"""
Pytest configuration with import-time mocking for heavy dependencies.

This module conditionally mocks GPU-dependent and large modules ONLY when they
are not installed. This allows tests to use real packages locally while still
working in CI environments where heavy packages aren't installed.
"""
# =============================================================================
# IMPORT-TIME MOCKING - Must be FIRST before any other imports
# =============================================================================
import sys
from unittest.mock import MagicMock
from pathlib import Path


def _is_module_available(module_name: str) -> bool:
    """Check if a module can be imported without actually importing it."""
    if module_name in sys.modules:
        # Already imported (might be a mock or real)
        return not isinstance(sys.modules[module_name], MagicMock)
    
    try:
        import importlib.util
        spec = importlib.util.find_spec(module_name.split('.')[0])
        return spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


class SmartMock(MagicMock):
    """
    Mock that handles numpy/torch array-like attributes gracefully.
    
    This is needed because many tests access .ndim, .shape, or convert
    mocked tensors to numpy arrays. Also handles common module attributes
    like __version__ that MagicMock normally blocks.
    """
    
    # Handle dunder attributes that MagicMock normally blocks
    __version__ = "0.0.0-mock"
    __path__ = []
    __file__ = "mock"
    
    @property
    def ndim(self):
        return 2
    
    @property
    def shape(self):
        return (1, 16000)
    
    def __array__(self, dtype=None):
        import numpy as np
        return np.zeros((1, 16000), dtype=dtype or np.float32)
    
    def cuda(self, *args, **kwargs):
        return self
    
    def cpu(self):
        return self
    
    def to(self, *args, **kwargs):
        return self
    
    def float(self):
        return self
    
    def half(self):
        return self
    
    def eval(self):
        return self
    
    def train(self, mode=True):
        return self


# =============================================================================
# CONDITIONAL MOCKING - Only mock modules that aren't installed
# =============================================================================

# Heavy modules that cause import failures in CI if not installed
# These are organized by category for clarity
MOCK_MODULES = [
    # PyTorch ecosystem - typically not in CI
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.nn.utils', 
    'torch.nn.utils.rnn', 'torch.cuda', 'torch.cuda.amp',
    'torch.utils', 'torch.utils.data', 'torch.optim', 'torch.fft', 
    'torch.amp', 'torch.backends', 'torch.backends.cudnn',
    'torchvision', 'torchvision.transforms', 'torchvision.models',
    'torchaudio', 'torchaudio.transforms', 'torchaudio.functional',
    
    # ML frameworks
    'ctranslate2', 'faster_whisper',
    
    # Speaker diarization
    'pyannote', 'pyannote.audio', 'pyannote.audio.pipelines',
    'pyannote.core', 'speechbrain', 'speechbrain.pretrained', 
    'speechbrain.inference', 'speechbrain.inference.speaker',
    'diart',
    
    # Audio separation
    'demucs', 'demucs.apply', 'demucs.pretrained',
    
    # TTS engines
    'TTS', 'TTS.api', 'TTS.tts', 'TTS.tts.configs',
    'edge_tts', 'piper', 'piper.voice',
    
    # OCR
    'paddleocr', 'paddlepaddle', 'paddle', 'easyocr',
    
    # ONNX and inference
    'onnxruntime', 'onnxruntime_gpu', 'onnx',
    
    # Face processing
    'insightface', 'gfpgan', 'basicsr', 'basicsr.utils',
    'kornia', 'face_alignment',
    
    # Transformers ecosystem
    'transformers', 'diffusers', 'accelerate',
    'huggingface_hub', 'safetensors', 'einops',
    'bitsandbytes', 'sentencepiece',
    
    # Audio processing - only mock if not available
    'pyworld', 'voicefixer',
]

# Modules that should use real implementation if available
# These are commonly used in tests and work well when installed
PREFER_REAL_MODULES = [
    'soundfile', 'librosa', 'librosa.core', 'librosa.feature',
    'cv2', 'ffmpeg',
]


def _setup_mock_module(mod_name: str) -> None:
    """Set up a SmartMock for a module."""
    if mod_name not in sys.modules:
        sys.modules[mod_name] = SmartMock()


# First, mock all heavy modules that aren't installed
for mod in MOCK_MODULES:
    if not _is_module_available(mod):
        _setup_mock_module(mod)

# For preferred-real modules, only mock if not available
for mod in PREFER_REAL_MODULES:
    if not _is_module_available(mod):
        _setup_mock_module(mod)


# =============================================================================
# CONFIGURE TORCH MOCK (if mocked)
# =============================================================================
torch_mock = sys.modules.get('torch')
if torch_mock and isinstance(torch_mock, MagicMock):
    import numpy as np
    
    # Link submodules to parent mock
    torch_cuda_mock = sys.modules.get('torch.cuda')
    if torch_cuda_mock:
        torch_mock.cuda = torch_cuda_mock
        # Mock CUDA detection to return False (CPU-only mode for tests)
        torch_cuda_mock.is_available = MagicMock(return_value=False)
        torch_cuda_mock.device_count = MagicMock(return_value=0)
    
    # torch.nn linkage
    torch_nn_mock = sys.modules.get('torch.nn')
    if torch_nn_mock:
        torch_mock.nn = torch_nn_mock
        torch_nn_functional = sys.modules.get('torch.nn.functional')
        if torch_nn_functional:
            torch_nn_mock.functional = torch_nn_functional
    
    # Common torch functions/types
    torch_mock.tensor = lambda x, *args, **kwargs: SmartMock()
    torch_mock.zeros = lambda *args, **kwargs: SmartMock()
    torch_mock.ones = lambda *args, **kwargs: SmartMock()
    torch_mock.randn = lambda *args, **kwargs: SmartMock()
    torch_mock.from_numpy = lambda x: SmartMock()
    torch_mock.float32 = 'float32'
    torch_mock.float16 = 'float16'
    torch_mock.int64 = 'int64'
    torch_mock.int32 = 'int32'
    torch_mock.device = lambda x: x
    torch_mock.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    torch_mock.inference_mode = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))


# =============================================================================
# STANDARD PYTEST FIXTURES
# =============================================================================
import pytest


# Pre-import to avoid DLL conflicts on Windows (if available)
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
