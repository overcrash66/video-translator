import sys
import os
from pathlib import Path
import pytest
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.audio.diarization import Diarizer
from src.utils import config

def test_load_speechbrain_no_symlink():
    """Test that SpeechBrain model loads without WinError 1314."""
    
    print("\nInitializing Diarizer...")
    diarizer = Diarizer()
    
    print("Loading SpeechBrain model...")
    # This calls the method we patched
    try:
        diarizer._load_speechbrain()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    
    assert diarizer.embedding_model is not None
    print("SpeechBrain model loaded successfully.")

    # Verify that the model files are in the expected directory
    model_dir = config.BASE_DIR / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"
    assert model_dir.exists()
    assert (model_dir / "hyperparams.yaml").exists()
    print(f"Verified model files exist in {model_dir}")
    
if __name__ == "__main__":
    test_load_speechbrain_no_symlink()
