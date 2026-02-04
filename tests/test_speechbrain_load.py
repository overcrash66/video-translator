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
    
    # Create a MagicMock for SpeechBrain
    mock_sb = MagicMock()
    # Configure it to mimic the structure Diarizer expects
    # sb.inference.speaker.EncoderClassifier.from_hparams
    # When we mock sys.modules['speechbrain'], imports of submodules might verify hierarchy
    # Simplest: mock sys.modules['speechbrain.inference.speaker'] specifically
    
    mock_speaker_module = MagicMock()
    mock_classifier_cls = MagicMock()
    mock_speaker_module.EncoderClassifier = mock_classifier_cls
    
    # Mock return of from_hparams
    mock_model_instance = MagicMock()
    mock_classifier_cls.from_hparams.return_value = mock_model_instance
    
    with patch.dict(sys.modules, {
            "speechbrain": MagicMock(),
            "speechbrain.inference": MagicMock(),
            "speechbrain.inference.speaker": mock_speaker_module
         }), \
         patch("src.utils.config.BASE_DIR", tmp_path), \
         patch("huggingface_hub.snapshot_download") as mock_download, \
         patch("shutil.copy2") as mock_copy:
         
        # Setup mock return for download
        mock_download.return_value = str(fake_model_dir)
        
        print("\nInitializing Diarizer...")
        diarizer = Diarizer()
        
        print("Loading SpeechBrain model...")
        diarizer._load_speechbrain()
        
        # Verify download called with correct args
        mock_download.assert_called_once()
        args, kwargs = mock_download.call_args
        assert kwargs["local_dir_use_symlinks"] is False
        
        # Verify model loaded
        assert diarizer.embedding_model is mock_model_instance
        mock_classifier_cls.from_hparams.assert_called_once()
        
        print("SpeechBrain model loaded successfully (mocked).")
        
if __name__ == "__main__":
    pytest.main([__file__])
