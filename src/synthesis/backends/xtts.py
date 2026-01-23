import logging
import torch
from pathlib import Path
from src.utils import config
from src.synthesis.backends.base import TTSBackend

logger = logging.getLogger(__name__)

class XttsBackend(TTSBackend):
    def __init__(self, device):
        self.device = device
        self.model = None

    def load_model(self):
        if self.model:
            return
        
        logger.info("Loading XTTS-v2 model...")
        try:
            from TTS.api import TTS
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            logger.info("XTTS-v2 model loaded.")
        except Exception as e:
            if "CUDA" in str(e) and self.device == "cuda":
                logger.warning(f"CUDA Error loading XTTS: {e}")
                logger.warning("Switching to CPU fallback for XTTS...")
                self.device = "cpu"
                if self.model:
                     del self.model
                     self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Retry
                self.load_model()
                return

            logger.error(f"Failed to load XTTS model: {e}")
            raise

    def unload_model(self):
        if self.model:
            logger.info("Unloading XTTS model...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate(self, text, output_path, language="en", speaker_wav=None, **kwargs):
        """
        Generates audio using XTTS.
        Requires speaker_wav.
        """
        if not speaker_wav:
             raise ValueError("XTTS requires a speaker_wav reference audio.")

        # Check unsupported params
        guidance_scale = kwargs.get("guidance_scale")
        emotion = kwargs.get("emotion")
        if guidance_scale is not None or emotion:
             logger.warning("CFG (guidance_scale) and emotion are NOT supported by XTTS-v2. Ignoring.")
             
        try:
            self.load_model()
            logger.info(f"Generating XTTS audio for: '{text[:20]}...'")
            
            if not self.model:
                 raise RuntimeError("XTTS model failed to load.")

            self.model.tts_to_file(
                text=text, 
                speaker_wav=str(speaker_wav), 
                language=language, 
                file_path=str(output_path)
            )
            return output_path
        except Exception as e:
            logger.error(f"XTTS generation failed: {e}")
            raise
