import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

# Force UNIT_TEST mode for all tests
os.environ["UNIT_TEST"] = "true"

import pytest

# Mock heavy dependencies
modules_to_mock = [
    "torch", "torchaudio", "torchvision",
    "ctranslate2", "faster_whisper", "soundfile", "ffmpeg", "cv2", "subprocess",
    "librosa", "scipy", "pydub",
    "PIL", "moviepy", "gradio", "paddle", "paddleocr",
    "onnx", "onnxruntime",
    "skimage", "matplotlib", "sklearn", "pandas", "insightface",
    "huggingface_hub", "transformers", "sentencepiece", "tokenizers", "openai", "whisper",
    "speechbrain", "hyperpyyaml", "TTS", "langdetect", "easyocr",
    "face_alignment", "gfpgan", "basicsr", "facexlib", "kornia", "numba", "resampy", "typeguard"
]

class MockException(Exception):
    pass

class SmartMock(MagicMock):
    """A MagicMock that returns sensible defaults for common ML/Image/Audio calls."""
    
    def __getattr__(self, name):
        if any(x in name for x in ["Error", "Exception", "Warning"]):
            return MockException
        return super().__getattr__(name)

    def __array__(self, *args, **kwargs):
        shape = getattr(self, "shape", (100, 100, 3))
        if not isinstance(shape, tuple): shape = (100, 100, 3)
        return np.zeros(shape, dtype=np.uint8)
        
    @property
    def __array_interface__(self):
        arr = self.__array__()
        return arr.__array_interface__

    def __call__(self, *args, **kwargs):
        name = self._mock_name or ""
        
        if name in ["input", "filter", "overwrite_output", "output"]:
            m = getattr(self, name)
            if name == "output" and args:
                out = args[0] if isinstance(args[0], (str, Path)) else (args[1] if len(args) > 1 and isinstance(args[1], (str, Path)) else None)
                if out:
                    m._last_output = out
                    # Propagate to root
                    root = self
                    while hasattr(root, "_mock_parent") and root._mock_parent: root = root._mock_parent
                    root._last_output = out
            return m

        if "getbbox" in name or "textbbox" in name: return (0, 0, 40, 40)
        if "boundingRect" in name: return (0, 0, 50, 50)
        if "threshold" in name: return (0.0, args[0]) if args and isinstance(args[0], np.ndarray) else (0.0, np.zeros((100, 100), dtype=np.uint8))
        if any(x in name for x in ["cvtColor", "resize", "fromarray", "imread"]): return args[0] if args and isinstance(args[0], np.ndarray) else np.zeros((100, 100, 3), dtype=np.uint8)
        
        if name == "open" and any(x in str(self._mock_parent) for x in ["PIL", "Image"]):
            m = SmartMock(name="Image")
            m.size = (100, 100); m.ndim = 3; m.shape = (100, 100, 3)
            return m

        if name == "stft": return np.zeros((1025, 100))
        if name == "istft": return np.zeros(22050)
        
        if name in ["load", "read"]:
            full_name = []
            curr = self
            while curr:
                if hasattr(curr, "_mock_name") and curr._mock_name: full_name.append(curr._mock_name)
                curr = getattr(curr, "_mock_parent", None)
            lib = ".".join(reversed(full_name))
            
            duration_sec = 1
            fname = str(args[0]) if args else ""
            if any(x in fname.lower() for x in ["extraction", "video", "chunk", "profile"]):
                duration_sec = 10 if "profile" not in fname.lower() else 2
            
            samples = 22050 * duration_sec
            if any(x in fname.lower() for x in ["source", "target", "dummy", "profile", "output", "result"]):
                t = np.linspace(0, duration_sec, samples)
                data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            else:
                data = np.random.uniform(-0.1, 0.1, samples).astype(np.float32)
            
            if any(x in lib for x in ["sf", "librosa", "soundfile"]): return (data, 22050)
            if "wavfile" in lib: return (22050, data)
            return (data, 22050)

        if name == "probe":
            # Robust probe mock
            if args and isinstance(args[0], list) and any("ffprobe" in str(x) for x in args[0]):
                # Assume it's an ffprobe command
                if "-show_entries" in args[0] and "format=duration" in args[0]:
                    return {'format': {'duration': 10.0}}
                if "-show_entries" in args[0] and "stream=width,height,duration" in args[0]:
                    return {'streams': [{'codec_type': 'video', 'width': 640, 'height': 360, 'duration': 10.0}]}
            return {'format': {'duration': 10.0}, 'streams': [{'codec_type': 'video', 'width': 640, 'height': 360, 'duration': 10.0}]}
            
        if name == "run":
            from src.utils import config as app_config
            for i in range(5):
                try:
                    (app_config.TEMP_DIR / f"test_video_chunk_{i:03d}.mp4").write_bytes(b"dummy")
                    (app_config.TEMP_DIR / f"test_audio_chunk_{i:03d}.wav").write_bytes(b"dummy")
                except: pass
            
            root = self
            while hasattr(root, "_mock_parent") and root._mock_parent: root = root._mock_parent
            out = getattr(root, "_last_output", None)
            
            if out and isinstance(out, (str, Path)) and "%" not in str(out):
                try:
                    p = Path(str(out)); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"dummy_output")
                except: pass
            
            # Subprocess return logic
            all_str_args = str(args) + str(kwargs)
            is_ffprobe = "ffprobe" in all_str_args.lower()
            
            comp_proc = MagicMock()
            val = b"10.0" if is_ffprobe else b""
            if kwargs.get("text") or kwargs.get("universal_newlines"):
                val = val.decode()
            
            comp_proc.stdout = val
            comp_proc.stderr = val if not is_ffprobe else b"" # Just in case
            comp_proc.returncode = 0
            return comp_proc

        if name == "write":
            if args:
                try:
                    p = Path(str(args[0]))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b"dummy")
                except: pass
            return None

        res = super().__call__(*args, **kwargs)
        if isinstance(res, SmartMock) and args and isinstance(args[0], np.ndarray):
             res.shape = args[0].shape
             if len(res.shape) >= 2:
                  res.size = (res.shape[1], res.shape[0])
        return res

def _configure_mock(m, name):
    m.__name__ = name
    m.__file__ = f"mocked_{name}.py"
    m.__path__ = []
    
    # Pre-set some things to be sure
    if "torch" in name:
        m.cuda.is_available.return_value = False
        m.device.return_value.__str__.return_value = "cpu"
        m.randn.return_value.shape = (1, 1, 100)

class MockFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in modules_to_mock:
            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, MockLoader())
        return None

class MockLoader:
    def create_module(self, spec):
        m = SmartMock(name=spec.name)
        m.__spec__ = spec
        _configure_mock(m, spec.name)
        return m
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, MockFinder())

for name in modules_to_mock:
    m = SmartMock(name=name)
    _configure_mock(m, name)
    sys.modules[name] = m

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def mock_video_path(temp_dir):
    video_file = temp_dir / "test.mp4"
    video_file.touch()
    return video_file

@pytest.fixture
def mock_components():
    return {k: MagicMock() for k in [
        'separator', 'transcriber', 'translator', 'tts_engine', 
        'synchronizer', 'processor', 'diarizer', 'lipsyncer', 
        'visual_translator', 'voice_enhancer'
    ]}
