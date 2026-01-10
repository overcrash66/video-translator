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
        self.xtts_model = None
        
        # Mapping for Piper (language code -> model name)
        # We use a default 'high' quality voice for each language if available
        self.piper_map = {
            "en": "en_US-lessac-high",
            "es": "es_ES-sharvard-medium",
            "fr": "fr_FR-siwis-medium",
            "de": "de_DE-thorsten-medium",
            "it": "it_IT-riccardo-x_low", # Limited options in public index, this is just a placeholder logic
            # For robustness, we will default to english if specific lang model not found, or use a generic one.
            # Real implementation would query the piper face or json index.
        }


    def load_model(self):
        # Edge-TTS is API based (or rather, no local model load needed in same sense)
        pass

    def _load_xtts(self):
        if self.xtts_model:
            return
        
        logger.info("Loading XTTS-v2 model...")
        try:
            from TTS.api import TTS
            self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            logger.info("XTTS-v2 model loaded.")
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            raise

    def generate_audio(self, text, speaker_wav_path, language="en", output_path=None, model="edge"):
        """
        Generates audio using Edge-TTS, Piper, or XTTS.
        model: "edge", "piper", or "xtts"
        """
        if not output_path:
            output_path = config.TEMP_DIR / "tts_output.wav"
            
        output_path = str(output_path)
            
        if model == "piper":
             return self._generate_piper(text, language, output_path)
        elif model == "xtts":
             return self._generate_xtts(text, language, speaker_wav_path, output_path)

        # Default Edge-TTS logic
        
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

    def _generate_piper(self, text, language, output_path):
        """
        Generates audio using Piper TTS binary (subprocess).
        Checks for binary and model, downloads if needed.
        """
        try:
            # 1. Resolve Voice Model
            model_name = self.piper_map.get(language, "en_US-lessac-high")
            model_dir = config.TEMP_DIR / "piper_models"
            model_dir.mkdir(exist_ok=True)
            
            onnx_path = model_dir / f"{model_name}.onnx"
            conf_path = model_dir / f"{model_name}.onnx.json"
            
            # 2. Download Model if missing
            if not onnx_path.exists():
                logger.info(f"Downloading Piper model: {model_name}...")
                self._download_piper_model(model_name, model_dir)
            
            # 3. Check for Piper Binary
            piper_bin_dir = config.TEMP_DIR / "piper_bin"
            piper_exe = piper_bin_dir / "piper" / "piper.exe"
            
            if not piper_exe.exists():
                logger.info("Piper binary not found. Downloading...")
                self._download_piper_binary(piper_bin_dir)
                
            if not piper_exe.exists():
                raise RuntimeError("Piper binary missing after download attempt.")

            logger.info(f"Generating Piper TTS (Binary): '{text[:20]}...' ({model_name})")
            
            # 4. Execute Piper
            # Command: echo text | piper.exe --model model.onnx --output_file out.wav
            
            import subprocess
            
            cmd = [
                str(piper_exe),
                "--model", str(onnx_path),
                "--output_file", str(output_path)
            ]
            
            # We assume single speaker for now, or default. 
            # If multi-speaker, we'd add --speaker_id.
            
            process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True, # text mode for stdin (echo)
                encoding='utf-8' # Ensure utf-8
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                raise RuntimeError(f"Piper binary failed (code {process.returncode}): {stderr}")
            
            if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
                 raise RuntimeError(f"Piper binary produced empty file: {stderr}")

            logger.info(f"Saved Piper WAV to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Piper generation failed: {e}")
            logger.info("Falling back to Edge-TTS...")
            return self.generate_audio(text, None, language, output_path, model="edge")

    def _download_piper_model(self, model_name, dest_dir):
        """
        Downloads .onnx and .json from Hugging Face (rhasspy/piper-voices)
        """
        import requests
        
        # Build URL (using standard structure for rhasspy/piper-voices v1.0.0)
        # Structure: https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/[lang]/[region]/[voice]/[quality]/[voice].onnx
        # But we only have the model name e.g. en_US-lessac-high
        # Parsing: lang_region, voice, quality
        try:
            parts = model_name.split("-") # ['en_US', 'lessac', 'high']
            lang_region = parts[0]
            voice = parts[1]
            quality = parts[2]
            
            # Extract lang code (e.g. "en" from "en_US")
            lang_code = lang_region.split("_")[0]
            
            # Correct URL Structure: v1.0.0/[lang_code]/[lang_region]/[voice]/[quality]/[model_name]
            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{lang_code}/{lang_region}/{voice}/{quality}/{model_name}"
            
            for ext in [".onnx", ".onnx.json"]:
                url = base_url + ext
                logger.info(f"Downloading {url}...")
                r = requests.get(url)
                r.raise_for_status()
                with open(dest_dir / (model_name + ext), "wb") as f:
                    f.write(r.content)
                    
        except Exception as e:
             raise RuntimeError(f"Could not download model {model_name}: {e}")

    def _download_piper_binary(self, dest_dir):
        """
        Downloads Piper Windows binary.
        """
        import requests
        import zipfile
        import io
        
        url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
        logger.info(f"Downloading Piper binary from {url}...")
        
        try:
            r = requests.get(url)
            r.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(dest_dir)
                
            logger.info("Piper binary downloaded and extracted.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to download Piper binary: {e}")

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
