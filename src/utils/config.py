import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from unittest.mock import MagicMock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv()

# --- CRASH DEBUGGING LOGGER ---
DEBUG_LOG_PATH = BASE_DIR / "debug_crash.log"
def debug_log(msg):
    """Writes immediate debug log to file for tracking hard crashes."""
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            import datetime
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {msg}\n")
            f.flush()
    except:
        pass

debug_log("--- SESSION START ---")
debug_log(f"Python: {sys.version}")
debug_log(f"Platform: {sys.platform}")


# Directories
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")


# Windows DLL loading fix for CUDA conflicts between torch and ctranslate2
def setup_zlib_dll():
    """Adds src/lib (zlibwapi) to DLL path for CTranslate2."""
    if sys.platform != "win32" or os.getenv("UNIT_TEST") == "true": return
    
    _src_lib = BASE_DIR / "src" / "lib"
    if _src_lib.exists():
        try:
            os.add_dll_directory(str(_src_lib))
            print(f"[Config] Added DLL directory: {_src_lib}")
        except Exception as e:
            print(f"[Config] Failed to add src/lib: {e}")
    else:
        print(f"[Config] src/lib not found!")

def setup_nvidia_dlls():
    """
    Adds torch/lib paths for dependencies.
    """
    if sys.platform != "win32" or os.getenv("UNIT_TEST") == "true": return
    
    _site_packages = BASE_DIR / "venv" / "Lib" / "site-packages"
    
    # Add torch/lib (Critical: contains matching cuDNN/cuBLAS/zlibwapi)
    _torch_lib = _site_packages / "torch" / "lib"
    
    if _torch_lib.exists():
        try:
            os.add_dll_directory(str(_torch_lib))
            print(f"[Config] Added DLL directory: {_torch_lib}")
        except Exception as e:
            print(f"[Config] Failed to add torch/lib: {e}")
    else:
        print(f"[Config] torch/lib not found at expected path: {_torch_lib}")

    # Fallback to system CUDA only if needed.
    preferred_cuda_roots = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"),
    ]
    
    env_cuda = os.environ.get("CUDA_PATH")
    if env_cuda:
        p = Path(env_cuda)
        if "v11.8" not in str(p) and "v11" not in str(p):
             if p not in preferred_cuda_roots:
                  preferred_cuda_roots.append(p)
    
    for cuda_root in preferred_cuda_roots:
        bin_path = cuda_root / "bin"
        if bin_path.exists():
            try:
                os.add_dll_directory(str(bin_path))
                print(f"[Config] Added System CUDA: {bin_path}")
            except Exception as e:
                 pass

# PHASE 1: Setup DLLs
if os.getenv("UNIT_TEST") != "true":
    setup_zlib_dll()
    setup_nvidia_dlls()

# PHASE 2: Load CTranslate2
try:
    import ctranslate2 
except Exception as e:
    if os.getenv("UNIT_TEST") == "true":
        if 'ctranslate2' not in sys.modules or sys.modules['ctranslate2'] is None:
            sys.modules['ctranslate2'] = MagicMock()
        import ctranslate2
    else:
        raise

# PHASE 3: Shim
def setup_cuda_dlls():
    # Already done at module level
    pass

try:
    import torch
except Exception as e:
    if os.getenv("UNIT_TEST") == "true":
         if 'torch' not in sys.modules or sys.modules['torch'] is None:
             sys.modules['torch'] = MagicMock()
         import torch
    else:
         raise
# Device configuration
def get_device() -> str:
    """
    Determines the best available device.
    Checks if CUDA is available AND functional (compatible with installed PyTorch).
    """
    if os.getenv("USE_CUDA", "true").lower() != "true":
        return "cpu"
        
    if not torch.cuda.is_available():
        return "cpu"
        
    try:
        # Functional test: Attempt to create a tensor on the device
        # This catches "no kernel image is available" errors for unsupported architectures
        t = torch.tensor([1.0]).to("cuda")
        del t
        return "cuda"
    except Exception as e:
        print(f"WARNING: CUDA is available but failed functioning test (likely architecture mismatch): {e}")
        print("Falling back to CPU.")
        return "cpu"

DEVICE = get_device()
# Device logic usually generic, but specific libraries check nicely.

# Model Configurations
WHISPER_MODEL_SIZE = "large-v3"

# -----------------------------
# Algorithm Parameters
# -----------------------------

# TTS Validation
TTS_MIN_TEXT_LENGTH = 2
TTS_MIN_AUDIO_SIZE = 100  # bytes
REFERENCE_RMS_THRESHOLD = 0.02
REFERENCE_VAR_THRESHOLD = 1e-5
XTTS_MIN_DURATION = 2.0  # seconds
F5_MIN_DURATION = 1.0    # seconds

# Segmentation Merging
MERGE_MIN_DURATION = 2.0 # seconds
MERGE_MAX_GAP = 0.5      # seconds

# Fallbacks
DUMMY_AUDIO_DURATION_MAX = 5.0 # seconds

from src.utils import languages

def get_language_code(name: str) -> str:
    return languages.get_language_code(name)

# -----------------------------
# Processing Parameters (Refactored from Magic Numbers)
# -----------------------------
DEMUCS_CHUNK_SECONDS = 10
DEMUCS_OVERLAP_SECONDS = 1

WAV2LIP_BOX_SMOOTH_WINDOW = 5

# Chunking (seconds)
CHUNK_DURATION = int(os.getenv("CHUNK_DURATION", "300"))  # 5 minutes default
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "2"))  # 2 second overlap

OCR_INTERVAL_DEFAULT = 1.0

def validate_path(path: str | Path, must_exist: bool = False, allowed_dirs: list[Path] | None = None) -> Path:
    """
    Validates and resolves a path, checking existence and containment.
    
    :param path: The input path string or Path object.
    :param must_exist: If True, raises FileNotFoundError if path doesn't exist.
    :param allowed_dirs: Optional list of directories the path must be inside.
    :return: Resolved absolute Path object.
    :raises FileNotFoundError: If must_exist is True and path is missing.
    :raises ValueError: If path is outside allowed_dirs.
    """
    try:
        resolved = Path(path).resolve()
    except Exception as e:
        raise ValueError(f"Invalid path format: {path}") from e

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    
    if allowed_dirs:
        # Check if resolved path starts with any allowed directory
        is_allowed = any(str(resolved).startswith(str(d.resolve())) for d in allowed_dirs)
        if not is_allowed:
             raise ValueError(f"Path outside allowed directories: {resolved}")
             
    return resolved
