"""
Wav2Lip Lip-Sync Integration
"""
import logging
from src.processing.wav2lip import Wav2LipSyncer

logger = logging.getLogger(__name__)

class LipSyncer:
    """
    Wrapper for Lip Syncing using Wav2Lip.
    """
    
    def __init__(self):
        self.engine = Wav2LipSyncer()
        
    def load_model(self):
        """Pass-through to Wav2Lip load_model"""
        self.engine.load_model()
        return True

    def unload_model(self):
        """Pass-through to Wav2Lip unload_model"""
        # Wav2Lip doesn't have explicit unload yet, but we can add if needed.
        # For now, just a placeholder.
        pass

    def sync_lips(self, video_path: str, audio_path: str, output_path: str, model_name: str = "wav2lip", enhance_face: bool = False) -> str:
        """
        Synchronizes lips in video_path to match audio_path using Wav2Lip.
        """
        return self.engine.sync_lips(video_path, audio_path, output_path, enhance_face=enhance_face)
