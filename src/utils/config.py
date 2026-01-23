import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Directories
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
DIARIZATION_MODEL_PATH = os.getenv("DIARIZATION_MODEL_PATH") # Optional: Local path to model folder/yaml
import torch
# Device configuration
def get_device():
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

def get_language_code(name):
    return languages.get_language_code(name)

# -----------------------------
# Processing Parameters (Refactored from Magic Numbers)
# -----------------------------
DEMUCS_CHUNK_SECONDS = 10
DEMUCS_OVERLAP_SECONDS = 1

WAV2LIP_BOX_SMOOTH_WINDOW = 5

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
