import sys
import os
from pathlib import Path
import pytest
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.audio.diarization import Diarizer
from src.utils import config

import sys
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.audio.diarization import Diarizer
from src.utils import config

def test_load_speechbrain_no_symlink(tmp_path):
    """Test that SpeechBrain model loads without WinError 1314 using mocks."""
    
    # Create fake model directory structure in tmp_path
    fake_model_dir = tmp_path / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"
    fake_model_dir.mkdir(parents=True)
    (fake_model_dir / "hyperparams.yaml").touch()
    (fake_model_dir / "label_encoder.txt").touch()
    
    # Mock config.BASE_DIR to use tmp_path
    with patch("src.utils.config.BASE_DIR", tmp_path), \
         patch("huggingface_hub.snapshot_download") as mock_download, \
         patch("speechbrain.inference.speaker.EncoderClassifier.from_hparams") as mock_classifier, \
         patch("shutil.copy2") as mock_copy:
         
        # Setup mock return for download
        mock_download.return_value = str(fake_model_dir)
        
        # Setup mock classifier
        mock_model = MagicMock()
        mock_classifier.return_value = mock_model
        
        print("\nInitializing Diarizer...")
        diarizer = Diarizer()
        
        print("Loading SpeechBrain model...")
        diarizer._load_speechbrain()
        
        # Verify download called with correct args
        mock_download.assert_called_once()
        args, kwargs = mock_download.call_args
        assert kwargs["local_dir_use_symlinks"] is False
        
        # Verify model loaded
        assert diarizer.embedding_model is not None
        mock_classifier.assert_called_once()
        
        print("SpeechBrain model loaded successfully (mocked).")

if __name__ == "__main__":
    pytest.main([__file__])
