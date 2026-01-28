import logging
from pathlib import Path
from src.synthesis.backends.base import TTSBackend
from src.synthesis.vibevoice_tts import VibeVoiceWrapper

logger = logging.getLogger(__name__)

class VibeVoiceBackend(TTSBackend):
    def __init__(self, model_version="vibevoice"):
        """
        Initialize backend for a specific VibeVoice model version.
        model_version: "vibevoice" (1.5B) or "vibevoice-7b" (Large)
        """
        self.wrapper = None
        self.model_version = model_version

    def load_model(self):
        if not self.wrapper:
            self.wrapper = VibeVoiceWrapper(model_name=self.model_version)
        self.wrapper.load_model()

    def unload_model(self):
        if self.wrapper:
            self.wrapper.unload_model()

    def generate(self, text, output_path, language="en", speaker_wav=None, **kwargs):
        """
        Generates audio using VibeVoice.
        """
        if not self.wrapper:
            self.wrapper = VibeVoiceWrapper(model_name=self.model_version)
            
        # VibeVoice uses speaker names. 
        # If 'speaker_id' is passed in kwargs (from UI voice selector), use it.
        # Otherwise, if we have a speaker profile flow, we might map it, but VibeVoice isn't a cloner in the same way.
        # It's a multi-speaker generator.
        
        target_speaker = kwargs.get('speaker_id')
        if not target_speaker:
            # Default fallback
            target_speaker = "Alice" # Default name commonly used in demos
            
        return self.wrapper.generate_speech(
            text=text, 
            output_path=output_path, 
            language=language,
            speaker_name=target_speaker
        )
