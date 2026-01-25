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
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.model_name = model_name
        self.repo_id = self.MODEL_PATHS.get(model_name, "microsoft/VibeVoice-1.5B")

    def load_model(self):
        if self.model_loaded:
            return

        logger.info(f"Loading VibeVoice model ({self.model_name}) from {self.repo_id} on {self.device}...")
        try:
            # VibeVoice uses HuggingFace-style imports from submodules
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            
            # Load processor and model
            self.processor = VibeVoiceProcessor.from_pretrained(self.repo_id)
            self.tts = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.repo_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.model_loaded = True
            logger.info("VibeVoice loaded successfully.")
            
        except ImportError as e:
            logger.error(f"VibeVoice module not found: {e}. Install with `pip install vibevoice`.")
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
        
        if self.processor:
            del self.processor
            self.processor = None
            
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
            
            # VibeVoice uses script-style format: "Speaker 1: text"
            # If no speaker_name provided, default to "Speaker 1"
            target_speaker = speaker_name if speaker_name else "Speaker 1"
            
            # Format text as VibeVoice script (single speaker format)
            script_text = f"{target_speaker}: {text}"
            
            # Process input through the processor
            inputs = self.processor(
                text=script_text,
                return_tensors="pt",
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate audio
            with torch.no_grad():
                output = self.tts.generate(**inputs)
            
            # Handle return types
            import soundfile as sf
            import numpy as np
            
            # Extract audio from output
            # VibeVoice returns VibeVoiceGenerationOutput with speech_outputs
            if hasattr(output, 'speech_outputs') and output.speech_outputs:
                audio = output.speech_outputs[0]
            else:
                audio = output
                
            # VibeVoice uses 24kHz sample rate
            sr = 24000 
                
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            
            # Flatten if needed
            if audio.ndim > 1:
                audio = audio.squeeze()
                
            # Write output file
            sf.write(str(output_path), audio, sr)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"VibeVoice generation failed: {e}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wrapper = VibeVoiceWrapper()
    # wrapper.load_model()
