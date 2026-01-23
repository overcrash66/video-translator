import logging
from pathlib import Path
from src.synthesis.backends.base import TTSBackend
from src.synthesis.f5_tts import F5TTSWrapper

logger = logging.getLogger(__name__)

class F5TTSBackend(TTSBackend):
    def __init__(self):
        self.wrapper = None

    def load_model(self):
        if not self.wrapper:
            self.wrapper = F5TTSWrapper()
        self.wrapper.load_model()

    def unload_model(self):
        if self.wrapper:
            self.wrapper.unload_model()

    def generate(self, text, output_path, language="en", speaker_wav=None, **kwargs):
        """
        Generates audio using F5-TTS.
        kwargs: cfg_strength
        """
        if not self.wrapper:
            self.wrapper = F5TTSWrapper()
            
        cfg_strength = kwargs.get("cfg_strength", 2.0)
        # Handle guidance_scale which is alias for cfg
        if kwargs.get("guidance_scale"):
             cfg_strength = float(kwargs.get("guidance_scale"))

        logger.info(f"F5-TTS generating with cfg={cfg_strength}")
        
        try:
             # F5TTSWrapper handles its own loading if needed, but explicit is better?
             # The wrapper implementation checks model_loaded flag.
             return self.wrapper.generate_voice_clone(
                 text, 
                 speaker_wav, 
                 ref_text="", 
                 output_path=output_path, 
                 cfg_strength=cfg_strength,
                 language=language
             )
        except Exception as e:
             logger.error(f"F5 generation failed: {e}")
             raise
