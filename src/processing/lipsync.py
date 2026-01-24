"""
Lip-Sync Integration Wrapper
"""
import logging
from src.processing.wav2lip import Wav2LipSyncer
from src.processing.live_portrait import LivePortraitSyncer

logger = logging.getLogger(__name__)

class LipSyncer:
    """
    Wrapper for Lip Syncing that supports multiple engines (Wav2Lip, LivePortrait).
    """
    
    def __init__(self):
        self.engines = {
            "wav2lip": Wav2LipSyncer(),
            "live_portrait": LivePortraitSyncer()
        }
        self.current_engine_name = "wav2lip"
        self.engine = self.engines["wav2lip"]
        
    def load_model(self, model_name: str = "wav2lip") -> bool:
        """Loads the specified lip-sync model."""
        if model_name not in self.engines:
            logger.warning(f"Unknown lip-sync model '{model_name}'. Defaulting to wav2lip.")
            model_name = "wav2lip"
            
        self.current_engine_name = model_name
        self.engine = self.engines[model_name]
        
        if hasattr(self.engine, 'load_model'):
            self.engine.load_model()
        elif hasattr(self.engine, 'load_models'):
            self.engine.load_models()
        return True

    def unload_model(self) -> None:
        """Unloads the current engine."""
        if hasattr(self.engine, 'unload_model'):
            self.engine.unload_model()
        elif hasattr(self.engine, 'unload_models'):
            self.engine.unload_models()

    def sync_lips(self, video_path: str, audio_path: str, output_path: str, model_name: str = "wav2lip", enhance_face: bool = False) -> str:
        """
        Synchronizes lips in video_path to match audio_path using the selected engine.
        
        Args:
            video_path: Path to the input video file.
            audio_path: Path to the audio file to sync lips to.
            output_path: Path where the output video will be saved.
            model_name: Name of the lip sync engine to use ('wav2lip' or 'live_portrait').
            enhance_face: If True, uses GFPGAN for face enhancement (Wav2Lip only).
            
        Returns:
            Path to the output video file.
        """
        # Ensure correct engine is loaded
        if self.current_engine_name != model_name:
            self.load_model(model_name)
            
        return self.engine.sync_lips(video_path, audio_path, output_path, enhance_face=enhance_face)
