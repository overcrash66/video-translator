import asyncio
from pathlib import Path
import logging
import soundfile as sf
import torch
import torchaudio

from src.utils import config
from src.utils import languages
from src.synthesis.backends.edge_tts import EdgeTTSBackend
from src.synthesis.backends.piper_tts import PiperTTSBackend
from src.synthesis.backends.xtts import XttsBackend
from src.synthesis.backends.f5_tts import F5TTSBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# [Fix] Enforce soundfile backend to avoid TorchCodec errors via monkey-patching
# Recent torchaudio versions removed set_audio_backend, so we intercept load()
# [Fix] Enforce soundfile backend to avoid TorchCodec errors via monkey-patching
# Post-2.x torchaudio on Windows is unstable with backends. We bypass it using soundfile directly.
try:
    import soundfile as sf
    import torch
    
    _original_load = torchaudio.load
    def _safe_load(filepath, **kwargs):
        # Ignore backend args, strictly use soundfile
        try:
            # sf.read returns (frames, channels) or (frames,)
            data, samplerate = sf.read(str(filepath), dtype='float32')
            
            # torchaudio.load returns (channels, time) tensor
            if data.ndim == 1:
                # Mono: (T,) -> (1, T)
                tensor = torch.from_numpy(data).unsqueeze(0)
            else:
                # Multi-channel: (T, C) -> (C, T)
                tensor = torch.from_numpy(data.T)
                
            return tensor, samplerate
        except Exception as e:
            logger.warning(f"Soundfile fallback failed calling original load: {e}")
            return _original_load(filepath, **kwargs)
        
    torchaudio.load = _safe_load
    logger.info("Monkey-patched torchaudio.load to use soundfile library directly.")
except Exception as e:
    logger.warning(f"Failed to patch torchaudio.load: {e}")

class TTSEngine:
    def __init__(self):
        self.device = config.DEVICE
        
        # Initialize Backends
        self.backends = {
            "edge": EdgeTTSBackend(languages.EDGE_TTS_VOICE_MAP),
            "piper": PiperTTSBackend(languages.PIPER_MODEL_MAP),
            "xtts": XttsBackend(self.device),
            "f5": F5TTSBackend()
        }

    def get_available_voices(self, model_name: str, language_code: str) -> list:
        """
        Returns a list of available voices for the given model and language.
        Delegates to specific knowledge where possible, though mostly static config currently.
        """
        if model_name == "edge":
            opts = languages.EDGE_TTS_VOICE_MAP.get(language_code, {})
            voices = []
            for gender in ["Female", "Male"]:
                v_list = opts.get(gender, [])
                if isinstance(v_list, str): v_list = [v_list]
                voices.extend(v_list)
            return sorted(voices)
            
        elif model_name == "piper":
            val = languages.PIPER_MODEL_MAP.get(language_code)
            return [val] if val else []
            
        elif model_name in ["xtts", "f5"]:
            return ["Cloning (Reference Audio)"]
            
        return []

    def load_model(self, model_name="all"):
        """
        Loads specific model or all reasonable defaults.
        Usually controlled by VideoTranslator calling specific backends implicitly via use.
        But exposed for pre-loading.
        """
        if model_name == "xtts":
            self.backends["xtts"].load_model()
        elif model_name == "f5":
            self.backends["f5"].load_model()
    
    def unload_model(self):
        """Unload heavy models."""
        self.backends["xtts"].unload_model()
        self.backends["f5"].unload_model()

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
        if len(text) < config.TTS_MIN_TEXT_LENGTH:
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
    
    def _validate_audio_file(self, file_path: str | Path, min_size: int = config.TTS_MIN_AUDIO_SIZE) -> bool:
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

    def validate_reference(self, wav_path: str | Path, model_name: str = "XTTS") -> bool:
        """
        Public API to check if a reference audio is valid for the target model.
        XTTS requires ~2.0s, F5-TTS can handle ~1.0s.
        """
        # Determine strictness based on model
        # [Fix] Case insensitive check
        min_dur = config.F5_MIN_DURATION if "f5" in model_name.lower() else config.XTTS_MIN_DURATION
        return self._check_reference_audio(wav_path, min_duration=min_dur)

    def _check_reference_audio(self, wav_path: str | Path, min_duration: float = 2.0) -> bool:
        """
        Checks if reference audio is suitable for cloning (has signal, reasonable duration).
        Returns True if valid, False otherwise.
        """
        if not wav_path or not Path(wav_path).exists():
            return False
            
        try:
            import soundfile as sf
            import numpy as np
            
            # Using sf.info first for fast duration check
            info = sf.info(wav_path)
            # [Fix] XTTS prone to crashing with < 2.0s audio, but F5 can handle down to 1.0s
            if info.duration < min_duration: 
                logger.warning(f"Reference audio too short ({info.duration:.2f}s < {min_duration}s). Skipping clone.")
                return False
                
            # Check for silence/signal
            # Limit read to first 10 seconds to avoid loading huge files
            data, sr = sf.read(wav_path, dtype='float32', frames=int(info.samplerate * 10))
            if len(data.shape) > 1:
                data = data.mean(axis=1) # mix to mono
                
            rms = np.sqrt(np.mean(data**2))
            if rms < config.REFERENCE_RMS_THRESHOLD: 
                logger.warning(f"Reference audio too silent (RMS={rms:.4f}). Skipping clone.")
                return False
                
            # Check for flat signal (low variance) which might be just DC offset or hum
            variance = np.var(data)
            if variance < config.REFERENCE_VAR_THRESHOLD:
                logger.warning(f"Reference audio has low variance ({variance}). Likely background noise only. Skipping clone.")
                return False
                
            return True
        except Exception as e:
            logger.warning(f"Failed to analyze reference audio: {e}")
            return False

    def generate_audio(self, 
                       text: str, 
                       speaker_wav_path: str | Path | None, 
                       language: str = "en", 
                       output_path: str | Path | None = None, 
                       model: str = "edge", 
                       gender: str = "Female", 
                       speaker_id: str | None = None, 
                       guidance_scale: float | None = None, 
                       emotion: str | None = None, 
                       force_cloning: bool = False, 
                       voice_selector=None, 
                       source_lang: str | None = None, 
                       preferred_voice: str | None = None) -> str | None:
        """
        Generates audio using the selected backend with robust fallback capability.
        Strategy:
        1. Try selected backend (e.g., XTTS, F5).
        2. If failed, fallback to Edge-TTS.
        3. If failed, fallback to Dummy Audio (silence/noise).
        
        :param text: Text to synthesize.
        :param speaker_wav_path: Path to reference audio for cloning.
        :param language: Target language code.
        :param output_path: Destination path for audio file.
        :param model: Primary model to use.
        :param gender: Gender hint for generic voices.
        :param speaker_id: Specific speaker ID for multi-speaker models.
        :param guidance_scale: CFG scale for models supporting it.
        :param emotion: Emotion prompt for supported models.
        :param force_cloning: Whether to bypass some validation checks.
        :param voice_selector: Callback to select voices.
        :param source_lang: Source language code.
        :param preferred_voice: Specific voice name to request.
        :return: Path to generated audio file or None if absolutely failed.
        """
        if not output_path:
            output_path = config.TEMP_DIR / "tts_output.wav"
        output_path = str(output_path)
        
        # Sanitize text
        sanitized_text = self._sanitize_text(text)
        if not sanitized_text:
            logger.warning(f"Skipping TTS generation for empty/invalid text")
            return self._generate_dummy_audio(text or "silence", output_path)

        # Select Backend
        backend = self.backends.get(model)
        if not backend:
            logger.warning(f"Unknown TTS model '{model}'. Falling back to 'edge'.")
            backend = self.backends["edge"]
            model = "edge"

        try:
            # Prepare kwargs
            kwargs = {
                "gender": gender,
                "speaker_id": speaker_id,
                "guidance_scale": guidance_scale,
                "emotion": emotion,
                "voice_selector": voice_selector,
                "preferred_voice": preferred_voice
            }
            
            # Additional cloning validation logic?
            # The backends (XTTS/F5) will use speaker_wav.
            # We used to have validation logic here. 
            # Reference Validation Logic:
            if model in ["xtts", "f5"]:
                should_attempt_clone = force_cloning
                if not force_cloning:
                     perform_check = True
                     # if model == "f5" and language != "en": perform_check = False 
                     
                     if perform_check:
                              if self.validate_reference(speaker_wav_path, model):
                                  should_attempt_clone = True
                              else:
                                  should_attempt_clone = False
                                  
                if not should_attempt_clone:
                    logger.warning(f"Cloning validation failed or disabled. Fallback to Edge-TTS.")
                    # Switch to edge backend immediately
                    backend = self.backends["edge"]
                    # Clean kwargs for edge? Edge ignores unused kwargs usually.
                    
            # Generate
            result = backend.generate(
                text=sanitized_text,
                output_path=output_path,
                language=language,
                speaker_wav=speaker_wav_path,
                **kwargs
            )
            
            if result:
                return result
            # If backend returns None without raising (some paths in my impl might), treat as fail
            raise RuntimeError("Backend returned None")

        except Exception as e:
            logger.error(f"TTS Configured Backend ({model}) failed: {e}")
            
            # Fallback to Edge if we weren't already using it
            if model != "edge":
                logger.info("Attempting Fallback to Edge-TTS...")
                try:
                    return self.backends["edge"].generate(
                        text=sanitized_text,
                        output_path=output_path,
                        language=language,
                        gender=gender,
                        # Edge vars
                        preferred_voice=preferred_voice,
                        speaker_id=speaker_id,
                        voice_selector=voice_selector
                    )
                except Exception as e2:
                    logger.error(f"Fallback Edge-TTS also failed: {e2}")
            
            # Ultimate Fallback
            logger.info("Generating placeholder dummy audio.")
            return self._generate_dummy_audio(sanitized_text, output_path)

    def generate_batch(self, tasks: list, model="edge") -> list:
        """
        Synthesizes a batch of text segments into audio.
        Delegates to the backend's `generate_batch` for optimized processing (e.g., parallel async requests).
        Handles cleanup, fallbacks, and retries for individual failures.
        
        :param tasks: List of dictionaries containing TTS parameters per segment.
                      Keys: text, output_path, language, speaker_wav, etc.
        :param model: The TTS model identifier (e.g., 'edge', 'xtts').
        :return: List of paths to generated audio files (or fallback dummy audio).
        """
        # 1. Select Backend
        backend = self.backends.get(model)
        if not backend:
            logger.warning(f"Unknown TTS model '{model}'. Batch fallback to 'edge'.")
            backend = self.backends["edge"]
            model = "edge"
            
        # 2. Pre-processing (Sanitize)
        valid_tasks = []
        original_indices = [] # Track alignment
        
        results = [None] * len(tasks)
        
        for i, t in enumerate(tasks):
             safe_text = self._sanitize_text(t['text'])
             if not safe_text:
                 results[i] = self._generate_dummy_audio("silence", t['output_path'])
                 continue
             
             # Create a mutable copy and update text
             t_copy = t.copy()
             t_copy['text'] = safe_text
             # Pre-populate defaults if missing?
             t_copy.setdefault('language', 'en')
             
             # Validation Logic (Cloning)
             # Logic is seemingly per-task, but backends might batch it?
             # For simplicity, if model is XTTS/F5, we might validate here.
             # But let's assume raw access for now or update later.
             
             valid_tasks.append(t_copy)
             original_indices.append(i)
             
        if not valid_tasks:
            return results
        
        # 3. Process Batch
        # We try strict batch call first
        try:
             batch_results = backend.generate_batch(valid_tasks)
        except Exception as e:
             logger.error(f"Batch generation failed: {e}. Falling back to sequential.")
             # Fallback to sequential calls via self.generate_audio (which handles fallbacks internally)
             batch_results = [None] * len(valid_tasks) # Will trigger individual loop below
        
        # 4. Handle Failures & Fallbacks
        for j, res in enumerate(batch_results):
            orig_idx = original_indices[j]
            task = valid_tasks[j]
            
            if res:
                results[orig_idx] = res
            else:
                # Individual Failure -> Fallback to sequential generate_audio
                # This ensures we get Edge fallback -> Dummy fallback
                logger.warning(f"Batch item {j} failed/missing. Retrying individually.")
                try:
                    # Map task keys to generate_audio args
                    # generate_audio(text, speaker_wav_path, language, output_path, model...)
                    # We might need to map keys carefully
                    
                    # Construct args
                    # We pass the Original sanitized text, not dummy
                    results[orig_idx] = self.generate_audio(
                        text=task['text'],
                        speaker_wav_path=task.get('speaker_wav'),
                        language=task.get('language', 'en'),
                        output_path=task.get('output_path'),
                        model=model,
                        gender=task.get('gender', 'Female'),
                        speaker_id=task.get('speaker_id'),
                        guidance_scale=task.get('guidance_scale'),
                        emotion=task.get('emotion'),
                        force_cloning=task.get('force_cloning', False),
                        voice_selector=task.get('voice_selector'),
                        source_lang=task.get('source_lang'),
                        preferred_voice=task.get('preferred_voice')
                    )
                except Exception as e:
                     logger.error(f"Retry failed for item {j}: {e}")
                     results[orig_idx] = self._generate_dummy_audio("error", task['output_path'])

        return results

    def _download_piper_model(self, model_name, dest_dir):
        # Deprecated: Logic moved to PiperBackend
        pass

    def _download_piper_binary(self, dest_dir):
        # Deprecated: Logic moved to PiperBackend
        pass

    def _generate_dummy_audio(self, text, output_path):
        """
        Generates a short silence/tone as fallback when TTS fails.
        """
        import soundfile as sf
        import numpy as np
        
        sr = 24000
        # Smart length estimate
        word_count = len(text.split()) if text else 1
        duration = max(0.5, min(word_count * 0.3, config.DUMMY_AUDIO_DURATION_MAX))
        
        # Generate silence with very subtle noise
        samples = int(sr * duration)
        wav = np.random.randn(samples) * 0.001 
        
        output_path = str(output_path) if output_path else str(config.TEMP_DIR / "dummy.wav")
        path_obj = Path(output_path)
        
        try:
             sf.write(output_path, wav, sr)
             logger.info(f"Generated placeholder audio: {output_path} ({duration:.1f}s)")
        except Exception as e:
             logger.warning(f"Failed to write to {output_path}: {e}")
             wav_path = str(path_obj.with_suffix('.wav'))
             sf.write(wav_path, wav, sr)
             output_path = wav_path
             logger.info(f"Generated placeholder audio (fallback): {output_path} ({duration:.1f}s)")
        
        return output_path

if __name__ == "__main__":
    eng = TTSEngine()
    eng.generate_audio("Hello world, this is a test.", None, "en", "test_edge.mp3")
