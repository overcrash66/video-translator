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
        "vibevoice": "microsoft/VibeVoice-1.5B",       # Default 1.5B (Official/Mirror)
        "vibevoice-7b": "rsxdalv/VibeVoice-Large"      # Community Mirror (Official 7B removed)
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
            
            # [Fix] Do NOT use device_map - it triggers accelerate's meta tensor initialization
            # Instead, load model normally then manually move to device
            self.tts = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.repo_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=False, # FORCE FULL LOAD (Avoid meta tensors)
                device_map=None,         # DISABLE ACCELERATE MAP
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
            
            # VibeVoice works best with explicit "Speaker X" format and numeric IDs
            # The processor regex strictly requires "Speaker \d+"
            target_speaker = "Speaker 1"
            
            # Format text as VibeVoice script (single speaker format)
            script_text = f"{target_speaker}: {text}"
            
            # Process input through the processor
            inputs = self.processor(
                text=script_text,
                return_tensors="pt",
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # [Fix] Populate generation_config if missing
            if getattr(self.tts, "generation_config", None) is None:
                from transformers import GenerationConfig
                self.tts.generation_config = GenerationConfig()
            
            # Determine BOS token
            bos_id = None
            tokenizer = getattr(self.processor, "tokenizer", None)
            
            if tokenizer is not None:
                if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
                    bos_id = tokenizer.bos_token_id
                elif hasattr(tokenizer, "eos_token_id"):
                    bos_id = tokenizer.eos_token_id
            
            if bos_id is None:
                bos_id = 1 # Fallback
                
            # Set in generation_config
            # Access generation_config safely
            gen_config = getattr(self.tts, "generation_config", None)
            if gen_config is not None:
                 if getattr(gen_config, "bos_token_id", None) is None:
                    gen_config.bos_token_id = bos_id
            else:
                 logger.warning("VibeVoice generation_config is None even after attempted creation.")

            # Set in model config
            # Access config safely
            model_config = getattr(self.tts, "config", None)
            if model_config is not None:
                if getattr(model_config, "bos_token_id", None) is None:
                    model_config.bos_token_id = bos_id

            # Generate audio
            with torch.no_grad():
                # Explicitly pass tokenizer as required by VibeVoice library logic
                # It uses tokenizer.bos_token_id internally even if generation_config is set
                output = self.tts.generate(**inputs, tokenizer=tokenizer)
            
            # [Fix] Handle case where model returns None (Portable App silent failure)
            if output is None:
                diagnostics = {
                    "device": self.device,
                    "input_shape": inputs['input_ids'].shape if 'input_ids' in inputs else "N/A",
                    "model_config": getattr(self.tts, 'config', "Missing"),
                    "generation_config": getattr(self.tts, 'generation_config', "Missing")
                }
                logger.error(f"VibeVoice generation returned None. Portable Diagnostics: {diagnostics}")
                raise RuntimeError("VibeVoice model returned None. This usually indicates a missing dependency or corrupted model in the portable environment.")

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
                
            # Save audio
            sf.write(str(output_path), audio.astype(np.float32), sr)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"VibeVoice generation failed: {e}")
            raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wrapper = VibeVoiceWrapper()
    # wrapper.load_model()
