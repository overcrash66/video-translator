import asyncio
import edge_tts
import config
import logging
from pathlib import Path
import torchaudio
import soundfile as sf
import torch

# MONKEYPATCH: Torchaudio 2.9+ broken backend API fix for Windows
# Forces soundfile backend for load() to bypass TorchCodec requirements
def _custom_load_patch(filepath, **kwargs):
    # Fallback to soundfile directly
    # FORCE float32 to match PyTorch model weights (fixes "expected Double found Float" error)
    data, samplerate = sf.read(filepath, dtype='float32')
    # Convert to standard (channels, frames) format
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0) # (1, frames)
    else:
        tensor = tensor.transpose(0, 1) # (frames, channels)
    return tensor, samplerate

# Apply the patch forcibly
torchaudio.load = _custom_load_patch
try:
    if hasattr(torchaudio, "set_audio_backend"):
         torchaudio.set_audio_backend("soundfile")
except:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self):
        self.device = config.DEVICE
        # Mapping from language code to Edge-TTS Voices (Gender-aware, multiple voices per gender)
        # Each gender has a LIST of voices to support multiple speakers
        self.voice_map = {
            "en": {
                "Female": ["en-US-AriaNeural", "en-US-JennyNeural", "en-GB-SoniaNeural"],
                "Male": ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-GB-RyanNeural"]
            },
            "es": {
                "Female": ["es-ES-ElviraNeural", "es-MX-DaliaNeural"],
                "Male": ["es-ES-AlvaroNeural", "es-MX-JorgeNeural"]
            },
            "fr": {
                "Female": ["fr-FR-DeniseNeural", "fr-CA-SylvieNeural"],
                "Male": ["fr-FR-HenriNeural", "fr-CA-JeanNeural"]
            },
            "de": {
                "Female": ["de-DE-KatjaNeural", "de-AT-IngridNeural"],
                "Male": ["de-DE-ConradNeural", "de-AT-JonasNeural"]
            },
            "it": {
                "Female": ["it-IT-ElsaNeural", "it-IT-IsabellaNeural"],
                "Male": ["it-IT-DiegoNeural", "it-IT-GiuseppeNeural"]
            },
            "pt": {
                "Female": ["pt-BR-FranciscaNeural", "pt-PT-RaquelNeural"],
                "Male": ["pt-BR-AntonioNeural", "pt-PT-DuarteNeural"]
            },
            "pl": {
                "Female": ["pl-PL-ZofiaNeural", "pl-PL-AgnieszkaNeural"],
                "Male": ["pl-PL-MarekNeural"]
            },
            "tr": {
                "Female": ["tr-TR-EmelNeural"],
                "Male": ["tr-TR-AhmetNeural"]
            },
            "ru": {
                "Female": ["ru-RU-SvetlanaNeural", "ru-RU-DariyaNeural"],
                "Male": ["ru-RU-DmitryNeural"]
            },
            "nl": {
                "Female": ["nl-NL-ColetteNeural", "nl-NL-FennaNeural"],
                "Male": ["nl-NL-MaartenNeural"]
            },
            "cs": {
                "Female": ["cs-CZ-VlastaNeural"],
                "Male": ["cs-CZ-AntoninNeural"]
            },
            "ar": {
                "Female": ["ar-SA-ZariyahNeural", "ar-EG-SalmaNeural"],
                "Male": ["ar-SA-HamedNeural", "ar-EG-ShakirNeural"]
            },
            "zh-cn": {
                "Female": ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-XiaochenNeural"],
                "Male": ["zh-CN-YunxiNeural", "zh-CN-YunjianNeural", "zh-CN-YunyeNeural"]
            },
            "ja": {
                "Female": ["ja-JP-NanamiNeural", "ja-JP-AoiNeural"],
                "Male": ["ja-JP-KeitaNeural", "ja-JP-DaichiNeural"]
            },
            "ko": {
                "Female": ["ko-KR-SunHiNeural", "ko-KR-JiMinNeural"],
                "Male": ["ko-KR-InJoonNeural", "ko-KR-BongJinNeural"]
            },
            "hi": {
                "Female": ["hi-IN-SwaraNeural"],
                "Male": ["hi-IN-MadhurNeural"]
            }
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

    def _sanitize_text(self, text):
        """
        Sanitizes text for TTS to avoid empty/problematic inputs.
        Returns None if text is invalid/empty.
        """
        if not text:
            return None
        
        # Strip whitespace
        text = text.strip()
        if not text:
            return None
        
        # Minimum length check (very short strings often fail)
        if len(text) < 2:
            logger.warning(f"Text too short for TTS: '{text}'")
            return None
        
        # Check if text contains any speakable characters
        # (avoid punctuation-only strings that Edge-TTS can't handle)
        import re
        # Match letters (any language), numbers, or CJK characters
        if not re.search(r'[a-zA-Z0-9\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0400-\u04ff\u0600-\u06ff\uac00-\ud7af]', text):
            logger.warning(f"Skipping TTS for non-speakable text: '{text[:50]}'")
            return None
        
        # Remove or replace problematic characters that cause Edge-TTS issues
        # (some invisible unicode characters, control chars, etc.)
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Control characters
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        return text.strip() if text.strip() else None
    
    def _validate_audio_file(self, file_path, min_size=100):
        """
        Validates that an audio file exists and has minimum size.
        Returns True if valid, False otherwise.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Audio file does not exist: {file_path}")
                return False
            
            size = path.stat().st_size
            if size < min_size:
                logger.warning(f"Audio file too small ({size} bytes): {file_path}")
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Failed to validate audio file {file_path}: {e}")
            return False

    def generate_audio(self, text, speaker_wav_path, language="en", output_path=None, model="edge", gender="Female", speaker_id=None):
        """
        Generates audio using Edge-TTS, Piper, or XTTS.
        model: "edge", "piper", or "xtts"
        gender: "Male" or "Female" (used for default/edge mapping)
        speaker_id: Speaker identifier (e.g., "SPEAKER_00") used to select unique voice
        """
        if not output_path:
            output_path = config.TEMP_DIR / "tts_output.wav"
            
        output_path = str(output_path)
        
        # Sanitize text before processing
        sanitized_text = self._sanitize_text(text)
        if not sanitized_text:
            logger.warning(f"Skipping TTS generation for empty/invalid text")
            return self._generate_dummy_audio(text or "silence", output_path)
            
        if model == "piper":
             return self._generate_piper(sanitized_text, language, output_path)
        elif model == "xtts":
             return self._generate_xtts(sanitized_text, language, speaker_wav_path, output_path)

        # Default Edge-TTS logic
        
        # Get voice list for language and gender
        opts = self.voice_map.get(language, self.voice_map.get("en", {}))
        voice_list = opts.get(gender, opts.get("Female", ["en-US-AriaNeural"]))
        
        # Handle legacy single-voice format (backward compatibility)
        if isinstance(voice_list, str):
            voice_list = [voice_list]
        
        # Select voice based on speaker_id
        voice_index = 0
        if speaker_id:
            try:
                # Extract speaker number from ID like "SPEAKER_00", "SPEAKER_01", etc.
                speaker_num = int(speaker_id.split("_")[-1])
                voice_index = speaker_num % len(voice_list)  # Cycle through available voices
            except (ValueError, IndexError):
                pass
        
        voice = voice_list[voice_index]
        
        logger.info(f"Generating TTS: lang='{language}', gender='{gender}', speaker='{speaker_id}' -> voice='{voice}'")
        
        # Retry logic with exponential backoff for transient Edge-TTS failures
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Async wrapper
                async def _gen():
                    communicate = edge_tts.Communicate(sanitized_text, voice)
                    await communicate.save(output_path)
                
                asyncio.run(_gen())
                
                # Validate the generated file
                if self._validate_audio_file(output_path):
                    return output_path
                else:
                    raise RuntimeError("Edge-TTS produced invalid/empty file")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Edge-TTS attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    import time
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying Edge-TTS in {wait_time}s...")
                    time.sleep(wait_time)
                    
                    # Try alternate voice on retry
                    if len(voice_list) > 1:
                        voice_index = (voice_index + 1) % len(voice_list)
                        voice = voice_list[voice_index]
                        logger.info(f"Switching to alternate voice: {voice}")
        
        # All retries failed - generate dummy audio as fallback
        logger.error(f"Edge-TTS failed after {max_retries} attempts: {last_error}")
        logger.info(f"Generating placeholder audio for text: '{sanitized_text[:50]}...'")
        return self._generate_dummy_audio(sanitized_text, output_path)

    def _generate_xtts(self, text, language, speaker_wav, output_path):
        try:
            self._load_xtts()
            logger.info(f"Generating XTTS audio for: '{text[:20]}...'")
            
            # XTTS language codes might differ slightly, but usually ISO 2-letter
            # TTS api handles file saving
            if not self.xtts_model:
                 raise RuntimeError("XTTS model failed to load.")

            self.xtts_model.tts_to_file(
                text=text, 
                speaker_wav=str(speaker_wav), 
                language=language, 
                file_path=str(output_path)
            )
            
            logger.info(f"Saved XTTS output to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"XTTS generation failed: {e}")
            # Fallback
            return self.generate_audio(text, speaker_wav, language, output_path, model="edge")

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
        """
        Generates a short silence/tone as fallback when TTS fails.
        IMPORTANT: Writes to the ORIGINAL path to maintain caller expectations.
        Converts to WAV internally but saves to whatever path is requested.
        """
        import soundfile as sf
        import numpy as np
        
        sr = 24000
        # Smart length estimate based on text: ~3 words per sec
        word_count = len(text.split()) if text else 1
        duration = max(0.5, min(word_count * 0.3, 5.0))  # Cap at 5 seconds
        
        # Generate silence with very subtle noise (avoids completely silent segments)
        samples = int(sr * duration)
        wav = np.random.randn(samples) * 0.001  # Very quiet noise
        
        output_path = str(output_path) if output_path else str(config.TEMP_DIR / "dummy.wav")
        
        # IMPORTANT: Keep the original path that the caller expects
        # soundfile can write to .mp3 paths as WAV data (browser/ffmpeg will handle it)
        # Or we can use a proper format based on extension
        path_obj = Path(output_path)
        
        try:
            # Try to write to the original path
            # soundfile writes WAV format by default
            sf.write(output_path, wav, sr)
            logger.info(f"Generated placeholder audio: {output_path} ({duration:.1f}s)")
        except Exception as e:
            # If writing to original path fails, try with .wav extension
            logger.warning(f"Failed to write to {output_path}: {e}")
            wav_path = str(path_obj.with_suffix('.wav'))
            sf.write(wav_path, wav, sr)
            output_path = wav_path
            logger.info(f"Generated placeholder audio (fallback): {output_path} ({duration:.1f}s)")
        
        return output_path

if __name__ == "__main__":
    eng = TTSEngine()
    eng.generate_audio("Hello world, this is a test.", None, "en", "test_edge.mp3")
