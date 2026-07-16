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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import importlib.machinery
        mod_name = getattr(self, '_mock_name', None) or "mock"
        self.__spec__ = importlib.machinery.ModuleSpec(mod_name, loader=None)
    
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
    
    # Utilities
    'tqdm', 'tqdm.auto',
    'cachetools',
    
    # Translation
    'deep_translator', 'deep_translator.exceptions',
    'langdetect',
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
    
    # Link submodules to parent mock - always ensure cuda mock exists
    torch_cuda_mock = sys.modules.get('torch.cuda')
    if not torch_cuda_mock or not isinstance(torch_cuda_mock, MagicMock):
        torch_cuda_mock = MagicMock()
        sys.modules['torch.cuda'] = torch_cuda_mock
    torch_mock.cuda = torch_cuda_mock
    # Mock CUDA detection to return False (CPU-only mode for tests)
    torch_cuda_mock.is_available = MagicMock(return_value=False)
    torch_cuda_mock.device_count = MagicMock(return_value=0)
    
    # torch.nn linkage - always ensure nn mock exists
    torch_nn_mock = sys.modules.get('torch.nn')
    if not torch_nn_mock or not isinstance(torch_nn_mock, MagicMock):
        torch_nn_mock = MagicMock()
        sys.modules['torch.nn'] = torch_nn_mock
    torch_mock.nn = torch_nn_mock
    torch_nn_functional = sys.modules.get('torch.nn.functional')
    if not torch_nn_functional or not isinstance(torch_nn_functional, MagicMock):
        torch_nn_functional = MagicMock()
        sys.modules['torch.nn.functional'] = torch_nn_functional
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
# CONFIGURE SOUNDFILE/LIBROSA MOCKS (if mocked)
# =============================================================================
import numpy as np

sf_mock = sys.modules.get('soundfile')
if sf_mock and isinstance(sf_mock, MagicMock):
    # soundfile.read reads actual WAV files written by our mock sf.write
    def mock_sf_read(path, *args, **kwargs):
        import struct as _struct
        try:
            with open(str(path), 'rb') as f:
                # Parse WAV header
                riff = f.read(4)
                if riff != b'RIFF':
                    return np.ones((24000, 2), dtype=np.float32) * 0.5, 24000
                f.read(4)  # file size
                wave = f.read(4)
                if wave != b'WAVE':
                    return np.ones((24000, 2), dtype=np.float32) * 0.5, 24000
                # Find fmt chunk
                while True:
                    chunk_id = f.read(4)
                    chunk_size = _struct.unpack('<I', f.read(4))[0]
                    if chunk_id == b'fmt ':
                        fmt_data = f.read(chunk_size)
                        audio_fmt, channels, samplerate = _struct.unpack('<HHI', fmt_data[:8])[:3]
                        break
                    f.read(chunk_size)
                # Find data chunk
                while True:
                    chunk_id = f.read(4)
                    chunk_size = _struct.unpack('<I', f.read(4))[0]
                    if chunk_id == b'data':
                        raw = f.read(chunk_size)
                        break
                    f.read(chunk_size)
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
                if channels > 1:
                    data = data.reshape(-1, channels)
                return data, samplerate
        except Exception:
            return np.ones((24000, 2), dtype=np.float32) * 0.5, 24000
    sf_mock.read = mock_sf_read

    # soundfile.write actually writes a WAV file so downstream tools (ffmpeg) work
    def mock_sf_write(path, data, samplerate, **kwargs):
        import struct as _struct
        data = np.asarray(data, dtype=np.float32)
        # soundfile expects (frames, channels) format
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        frames = int(data.shape[0])
        channels = int(data.shape[1])
        try:
            samplerate = min(int(samplerate), 65535)
        except (TypeError, ValueError):
            samplerate = 24000
        with open(str(path), 'wb') as f:
            f.write(b'RIFF')
            data_size = frames * channels * 2
            f.write(_struct.pack('<I', 36 + data_size))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write(_struct.pack('<IHHIIHH', 16, 1, channels, samplerate,
                                samplerate * channels * 2, channels * 2, 16))
            f.write(b'data')
            f.write(_struct.pack('<I', data_size))
            int_data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
            f.write(int_data.tobytes())
    sf_mock.write = mock_sf_write

    # soundfile.info returns an object with numeric attributes
    class _MockSoundFileInfo:
        samplerate = 24000
        channels = 2
        duration = 5.0
        frames = 120000
        format = "WAV"
        subtype = "PCM_16"
    sf_mock.info = MagicMock(return_value=_MockSoundFileInfo())

librosa_mock = sys.modules.get('librosa')
if librosa_mock and isinstance(librosa_mock, MagicMock):
    # librosa.load returns (data, sample_rate)
    def mock_librosa_load(path, sr=None, *args, **kwargs):
        return np.zeros((16000,), dtype=np.float32), sr or 16000
    librosa_mock.load = mock_librosa_load

    # librosa.effects.time_stretch returns input unchanged when mocked
    librosa_effects_mock = sys.modules.get('librosa.effects')
    if librosa_effects_mock and isinstance(librosa_effects_mock, MagicMock):
        librosa_effects_mock.time_stretch = lambda y, rate=1.0: y
    elif isinstance(librosa_mock.effects, MagicMock):
        librosa_mock.effects.time_stretch = lambda y, rate=1.0: y


# =============================================================================
# CONFIGURE CV2 MOCK (if mocked)
# =============================================================================
cv2_mock = sys.modules.get('cv2')
if cv2_mock and isinstance(cv2_mock, MagicMock):
    # cv2.cvtColor should return a proper numpy array
    def mock_cvtColor(img, *args, **kwargs):
        if hasattr(img, '__array__'):
            return np.asarray(img)
        return np.zeros((100, 100, 3), dtype=np.uint8)
    cv2_mock.cvtColor = mock_cvtColor
    cv2_mock.boundingRect = lambda pts: (10, 10, 50, 50)
    cv2_mock.COLOR_BGR2RGB = 4
    cv2_mock.COLOR_RGB2BGR = 4
    cv2_mock.INPAINT_TELEA = 0


# =============================================================================
# STANDARD PYTEST FIXTURES
# =============================================================================
import pytest


# Pre-import to avoid DLL conflicts on Windows (if available)
try:
    from src.utils import config
    import ctranslate2
    print("DEBUG: ctranslate2 pre-imported in tests/conftest.py")
except (ImportError, ValueError, AttributeError):
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


# =============================================================================
# CI DETECTION AND SKIP MARKERS
# =============================================================================
import os

# Detect if running in CI environment
IN_CI = os.environ.get('CI', '').lower() == 'true' or os.environ.get('GITHUB_ACTIONS', '').lower() == 'true'


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_real_audio: mark test as requiring real audio files (skip in CI)"
    )
    config.addinivalue_line(
        "markers", "requires_ffmpeg: mark test as requiring ffmpeg binary (skip in CI)"
    )
    config.addinivalue_line(
        "markers", "requires_models: mark test as requiring ML models (skip in CI)"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked with certain markers when in CI."""
    if not IN_CI:
        return  # Don't skip anything when running locally
    
    skip_markers = {
        'requires_real_audio': pytest.mark.skip(reason="Requires real audio files (CI)"),
        'requires_ffmpeg': pytest.mark.skip(reason="Requires ffmpeg (CI)"),
        'requires_models': pytest.mark.skip(reason="Requires ML models (CI)"),
    }
    
    for item in items:
        for marker_name, skip_mark in skip_markers.items():
            if marker_name in [m.name for m in item.iter_markers()]:
                item.add_marker(skip_mark)
