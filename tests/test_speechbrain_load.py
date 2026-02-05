import sys
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch


def test_load_speechbrain_no_symlink(tmp_path):
    """Test that SpeechBrain model loads without WinError 1314 using mocks."""
    
    # Must import after conftest has set up mocks
    from src.audio.diarization import Diarizer
    
    # Create fake model directory structure in tmp_path
    fake_model_dir = tmp_path / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"
    fake_model_dir.mkdir(parents=True)
    (fake_model_dir / "hyperparams.yaml").touch()
    (fake_model_dir / "label_encoder.txt").touch()
    # Create other expected files
    for fname in ["embedding_model.ckpt", "mean_var_norm_emb.ckpt", "classifier.ckpt", "custom.py"]:
        (fake_model_dir / fname).touch()
    
    # Mock config.BASE_DIR to use tmp_path
    # Patch where the module is USED (src.audio.diarization imports from these)
    with patch("src.utils.config.BASE_DIR", tmp_path), \
         patch("src.audio.diarization.shutil.copy2") as mock_copy:
        
        # Mock the imports that happen inside _load_speechbrain
        mock_classifier_cls = MagicMock()
        mock_model = MagicMock()
        mock_classifier_cls.from_hparams.return_value = mock_model
        
        mock_snapshot = MagicMock(return_value=str(fake_model_dir))
        
        # Mock fetching module for the Windows patch
        mock_fetching = MagicMock()
        mock_fetching.fetch = MagicMock()
        
        with patch.dict('sys.modules', {
            'speechbrain': MagicMock(),
            'speechbrain.inference': MagicMock(),
            'speechbrain.inference.speaker': MagicMock(EncoderClassifier=mock_classifier_cls),
            'speechbrain.utils': MagicMock(),
            'speechbrain.utils.fetching': mock_fetching,
            'huggingface_hub': MagicMock(snapshot_download=mock_snapshot),
        }):
            # Mock platform.system to return Windows to trigger the patch
            with patch("platform.system", return_value="Windows"):
                print("\nInitializing Diarizer...")
                diarizer = Diarizer()
                
                print("Loading SpeechBrain model...")
                diarizer._load_speechbrain()
                
                # Verify download called with correct args
                mock_snapshot.assert_called_once()
                args, kwargs = mock_snapshot.call_args
                assert kwargs.get("local_dir_use_symlinks") is False
                
                # Verify model loaded
                assert diarizer.embedding_model is not None
                mock_classifier_cls.from_hparams.assert_called_once()
                
                print("SpeechBrain model loaded successfully (mocked).")

if __name__ == "__main__":
    pytest.main([__file__])

