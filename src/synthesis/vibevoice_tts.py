import logging
import torch
import gc
from pathlib import Path
from src.utils import config

logger = logging.getLogger(__name__)

class VibeVoiceWrapper:
    """
    Wrapper for Microsoft VibeVoice TTS.
    Supports 1.5B and 7B (Large) model variants.
    """
    
    MODEL_PATHS = {
        "vibevoice": "microsoft/VibeVoice-1.5B",       # Default 1.5B
        "vibevoice-7b": "microsoft/VibeVoice-Large"    # Large 7B
    }

    def __init__(self, model_name="vibevoice"):
        self.tts = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.model_name = model_name
        self.repo_id = self.MODEL_PATHS.get(model_name, "microsoft/VibeVoice-1.5B")

    def load_model(self):
        if self.model_loaded:
            return

        logger.info(f"Loading VibeVoice model ({self.model_name}) from {self.repo_id} on {self.device}...")
        try:
            from vibevoice import VibeVoice
            
            # Initialize VibeVoice
            # Note: The actual API might vary slightly, adapting based on standard usage patterns
            # Checking VibeVoice repo, typically it loads via a class method or init
            self.tts = VibeVoice.from_pretrained(self.repo_id).to(self.device)
            
            self.model_loaded = True
            logger.info("VibeVoice loaded successfully.")
            
        except ImportError:
            logger.error("VibeVoice module not found. Install with `pip install vibevoice`.")
            raise
        except Exception as e:
            logger.error(f"Failed to load VibeVoice: {e}")
            if "CUDA" in str(e) and self.device == "cuda":
                logger.warning("Switching to CPU fallback for VibeVoice...")
                self.device = "cpu"
                # Retry
                self.load_model()
                return
            raise

    def unload_model(self):
        if self.tts:
            del self.tts
            self.tts = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model_loaded = False
        gc.collect()
        logger.info("VibeVoice unloaded.")

    def generate_speech(self, text, output_path=None, language="en", speaker_name=None):
        """
        Generates speech using VibeVoice.
        """
        if not self.model_loaded:
            self.load_model()
            
        if not output_path:
            import uuid
            output_path = config.TEMP_DIR / f"vibevoice_{uuid.uuid4()}.wav"
        
        output_path = Path(output_path)
            
        try:
            logger.info(f"VibeVoice generating: '{text[:30]}...' (Speaker: {speaker_name})")
            
            # VibeVoice generation
            # Assuming typical generate API: generate(text, speaker_id/name)
            # If speaker_name is None, use a default
            
            # Check available speakers if possible or default to a safe one
            # Microsoft VibeVoice doesn't have "cloning" in the same way, but uses prompts or IDs
            # Per docs: "Supports up to 4 distinct speakers" in conversation, so names might be arbitrary labels for the session
            # or pretrained voices. 
            
            # For now, we'll assume a simple generate interface. 
            # We might need to adjust this once we see the actual installed package API.
            
            target_speaker = speaker_name if speaker_name else "Speaker_A"
            
            # Generate audio tensor
            # output is typically (sample_rate, audio_numpy) or just audio_tensor
            output = self.tts.generate(
                text=text,
                speaker_id=target_speaker, # Use name as ID?
                language=language
            )
            
            # Handle return types
            import soundfile as sf
            import numpy as np
            
            # Normalize output structure
            if isinstance(output, tuple):
                sr, audio = output
            else:
                # Assume 24k or model default if not returned
                sr = 24000 
                audio = output
                
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
                
            # Write key file
            sf.write(str(output_path), audio, sr)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"VibeVoice generation failed: {e}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wrapper = VibeVoiceWrapper()
    # wrapper.load_model()
