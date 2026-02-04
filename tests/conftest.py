import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock
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
    "huggingface_hub",
    "skimage", "matplotlib", "sklearn", "pandas", "insightface",
    "transformers", "sentencepiece", "tokenizers", "openai", "whisper",
    "hyperpyyaml", "TTS", "langdetect", "easyocr",
    "face_alignment", "gfpgan", "basicsr", "facexlib", "kornia", "numba", "resampy", "typeguard"
]

class MockException(Exception):
    pass

class SmartMock(MagicMock):
    """A MagicMock that returns sensible defaults for common ML/Image/Audio calls."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mock_new_child = SmartMock
        
        # Initialize default shape directly in __dict__ to avoid recursion
        name = (getattr(self, "_mock_name", "") or "").lower()
        if any(x in name for x in ["torch", "audio", "wav", "mel", "sf", "librosa"]):
            self.__dict__["_shape"] = (2, 22050)
        else:
            self.__dict__["_shape"] = (100, 100, 3)

    @property
    def shape(self): return self.__dict__.get("_shape", (100, 100, 3))
    @shape.setter
    def shape(self, v): self.__dict__["_shape"] = v

    @property
    def ndim(self): return len(self.shape)
    @property
    def size(self): return np.prod(self.shape)
    @property
    def dtype(self): return np.float32

    def __getattr__(self, name):
        # Mandatory bypass for MagicMock internals and private attrs
        if name.startswith("_"):
             return super().__getattr__(name) if name.startswith("_mock_") else object.__getattribute__(self, name)
        
        if name == "__version__": return "2.1.0"
        
        if name == "ndim": return self.ndim
        if name == "size": return self.size
        # Specific check for torch.Tensor.ndim which assertive tests use
        if any(x in name.lower() for x in ["error", "exception", "warning"]):
            return MockException
        return super().__getattr__(name)

    # Use __getattribute__ ONLY for ndim to avoid shadowing
    def __getattribute__(self, name):
        if name in ["ndim", "size"]:
             s = object.__getattribute__(self, "__dict__").get("_shape", (100, 100, 3))
             # Handle integer case (if shape was overwritten with a number)
             if isinstance(s, int): return 1 if name == "ndim" else s
             return len(s) if name == "ndim" else np.prod(s)
        return super().__getattribute__(name)

    def __bool__(self): return True
    def __int__(self): return 1
    def __index__(self): return 1
    def __float__(self): return 1.0

    def __array__(self, *args, **kwargs):
        return np.zeros(self.shape, dtype=np.uint8)
        
    @property
    def __array_interface__(self):
        arr = self.__array__()
        return arr.__array_interface__

    def __call__(self, *args, **kwargs):
        if hasattr(self, "forward") and callable(self.forward) and not isinstance(self.forward, MagicMock):
            return self.forward(*args, **kwargs)

        name = (getattr(self, "_mock_name", "") or "").lower()
        
        if name == "get":
            if args:
                prop = args[0]
                if prop == 5: return 30.0 # FPS
                if prop == 7: return 100 # FRAME_COUNT
                if prop == 3: return 640 # WIDTH
                if prop == 4: return 360 # HEIGHT
            return 1

        if name in ["transpose", "unsqueeze", "squeeze", "mean", "view", "reshape", "zeros", "randn", "from_numpy"]:
             res = SmartMock(name=name, _mock_parent=self)
             old_shape = self.shape
             
             if name in ["zeros", "randn"] and args:
                  if isinstance(args[0], tuple): res.shape = args[0]
                  elif all(isinstance(x, int) for x in args): res.shape = tuple(args)
                  else: res.shape = old_shape
             elif name == "transpose" and args:
                  if isinstance(args[0], tuple):
                       axes = args[0]
                       try: res.shape = tuple(old_shape[i] for i in axes)
                       except: res.shape = old_shape
                  elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
                       s = list(old_shape)
                       a1, a2 = args[0], args[1]
                       if a1 < len(s) and a2 < len(s): s[a1], s[a2] = s[a2], s[a1]
                       res.shape = tuple(s)
                  else: res.shape = (old_shape[1], old_shape[0]) if len(old_shape) == 2 else old_shape
             elif name == "unsqueeze" and args:
                  axis = args[0] if isinstance(args[0], int) else 0
                  s = list(old_shape); s.insert(axis, 1); res.shape = tuple(s)
             elif name in ["squeeze", "mean"]: res.shape = old_shape[1:] if old_shape and old_shape[0] == 1 else (old_shape if name == "squeeze" else (1,) + old_shape[1:] if len(old_shape) > 1 else (1,))
             elif name in ["randn", "from_numpy"] and args and isinstance(args[0], np.ndarray):
                  res.shape = args[0].shape
             else: res.shape = old_shape
             return res

        if name == "resize" and args:
            if len(args) > 1 and isinstance(args[1], tuple):
                w, h = args[1]
                m = SmartMock(name="resized", _mock_parent=self)
                m.shape = (h, w, 3)
                return m

        if name in ["stft", "melspectrogram", "wav2mel", "abs", "log10"]:
             # Standard mel shape
             return np.zeros((100, 80), dtype=np.float32)

        # Check name OR parent name for audio context
        parent_name = str(getattr(self, "_mock_parent", "")).lower()
        # Be more aggressive for audio read
        is_audio_read = (
            any(x == name for x in ["read", "load", "load_audio"]) and
            not any(x in parent_name for x in ["cv2", "image", "video", "capture"])
        )
        
        if is_audio_read or ("soundfile" in parent_name and "read" in name):
             # Default audio mock
             return (np.zeros((22050, 2), dtype=np.float32), 22050)

        if name == "__version__":
            return "2.1.0"

        if name == "run":
             comp_proc = MagicMock()
             is_text = kwargs.get("text", False) or kwargs.get("universal_newlines", False) or kwargs.get("encoding")
             
             # Handle ffprobe
             if args and "ffprobe" in str(args[0]):
                 # Fake 10s duration for test checks
                 fake_probe = '{"format": {"duration": "10.0"}}'
                 comp_proc.stdout = fake_probe if is_text else fake_probe.encode()
                 comp_proc.stderr = "" if is_text else b""
                 comp_proc.returncode = 0
                 return comp_proc

             if args and "ffmpeg" in str(args[0]):
                  try:
                       out = args[0][-1]
                       Path(out).touch()
                  except: pass
             
             if is_text:
                 comp_proc.stdout = ""
                 comp_proc.stderr = ""
             else:
                 comp_proc.stdout = b""
                 comp_proc.stderr = b""
                 
             comp_proc.returncode = 0
             return comp_proc

        # Fix for Resample vs resample conflict
        # If it's a class init (Resample), args are usually ints. If functional (resample), arg 0 is array.
        if "resample" in name:
             if args and isinstance(args[0], (np.ndarray, SmartMock)):
                  # Functional: resample(waveform, ...)
                  return args[0]
             elif name == "resample" and not args:
                  # Maybe attribute access?
                  pass
             else:
                  # Class init: Resample(sr, target_sr) -> return a SmartMock (the transform object)
                  # Do NOT return array here. Fall through to infinite mock creation.
                  pass
        
        elif any(x in name for x in ["cvtColor", "fromarray", "imread"]): 
             return args[0] if args and isinstance(args[0], np.ndarray) else np.zeros((100, 100, 3), dtype=np.uint8)
        
        res = super().__call__(*args, **kwargs)
        if isinstance(res, SmartMock) and args:
             arg = args[0]
             if isinstance(arg, (np.ndarray, SmartMock)):
                  if name not in ["transpose", "unsqueeze", "squeeze", "mean", "view", "reshape", "zeros", "randn", "from_numpy", "resize"]:
                       res.shape = getattr(arg, "shape", res.shape)
        return res

def _configure_mock(m, name):
    m.__name__ = name
    m.__file__ = f"mocked_{name}.py"
    m.__path__ = []
    
    if any(x in name.lower() for x in ["torch", "audio", "wav", "mel", "sf", "librosa"]):
        m.shape = (2, 22050)
    else:
        m.shape = (100, 100, 3)

    if "torch" in name:
        m.cuda.is_available.return_value = False
        m.device.return_value.__str__.return_value = "cpu"
        
        # Use a safe class for Module to avoid metaclass conflicts
        class SafeModule:
             def __init__(self, *args, **kwargs): pass
             def __call__(self, *args, **kwargs): return SmartMock()
             def to(self, device): return self
             def eval(self): return self
             def load_state_dict(self, *args): pass
             def parameters(self): return []
             
        m.nn.Module = SafeModule
        m.Tensor = SmartMock
        m.__version__ = "2.1.0"
    
    if "cv2" in name:
        m.error = MockException

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

def inject_mocks():
    root_mocks = {}
    for name in modules_to_mock:
        parts = name.split(".")
        root = parts[0]
        if root not in root_mocks:
            m = SmartMock(name=root)
            _configure_mock(m, root)
            root_mocks[root] = m
            sys.modules[root] = m
        
        curr = root_mocks[root]
        for i in range(1, len(parts)):
            sub = parts[i]
            full_sub_name = ".".join(parts[:i+1])
            if not hasattr(curr, sub):
                sm = SmartMock(name=sub)
                sm._mock_parent = curr
                setattr(curr, sub, sm)
            curr = getattr(curr, sub)
            sys.modules[full_sub_name] = curr

inject_mocks()
sys.meta_path.insert(0, MockFinder())

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
