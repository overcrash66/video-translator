import asyncio
import edge_tts
import config
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self):
        self.device = config.DEVICE
        # Mapping from language code to Edge-TTS Voice
        self.voice_map = {
            "en": "en-US-AriaNeural",
            "es": "es-ES-ElviraNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural",
            "it": "it-IT-ElsaNeural",
            "pt": "pt-BR-FranciscaNeural",
            "pl": "pl-PL-ZofiaNeural",
            "tr": "tr-TR-EmelNeural",
            "ru": "ru-RU-SvetlanaNeural",
            "nl": "nl-NL-ColetteNeural",
            "cs": "cs-CZ-VlastaNeural",
            "ar": "ar-SA-ZariyahNeural",
            "zh-cn": "zh-CN-XiaoxiaoNeural",
            "ja": "ja-JP-NanamiNeural",
            "ko": "ko-KR-SunHiNeural",
            "hi": "hi-IN-SwaraNeural"
        }

    def load_model(self):
        # Edge-TTS is API based (or rather, no local model load needed in same sense)
        pass

    def generate_audio(self, text, speaker_wav_path, language="en", output_path=None):
        """
        Generates audio using Edge-TTS. 
        speaker_wav_path is ignored as Edge-TTS doesn't support zero-shot cloning in this library.
        """
        if not output_path:
            output_path = config.TEMP_DIR / "tts_output.wav"
        
        output_path = str(output_path)
        
        # Select voice
        voice = self.voice_map.get(language, "en-US-AriaNeural")
        
        logger.info(f"Generating TTS for lang='{language}' using voice='{voice}'...")
        
        try:
            # Async wrapper check
            async def _gen():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(output_path)
            
            asyncio.run(_gen())
            
            return output_path
        except Exception as e:
            logger.error(f"Edge-TTS Generation failed: {e}")
            return self._generate_dummy_audio(text, output_path)

    def _generate_dummy_audio(self, text, output_path):
        import soundfile as sf
        import numpy as np
        # Generate 1 sec of beeps
        sr = 24000
        # Smart length estimate? ~3 words per sec
        word_count = len(text.split()) if text else 1
        duration = max(1.0, word_count * 0.4) 
        
        t = np.linspace(0, duration, int(sr * duration))
        wav = np.sin(2 * 3.14159 * 440 * t) # Mono
        output_path = str(output_path) if output_path else str(config.TEMP_DIR / "dummy.wav")
        # Soundfile write: file, data, samplerate
        sf.write(output_path, wav, sr)
        return output_path

if __name__ == "__main__":
    eng = TTSEngine()
    eng.generate_audio("Hello world, this is a test.", None, "en", "test_edge.mp3")
