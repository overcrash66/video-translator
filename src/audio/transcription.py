from faster_whisper import WhisperModel
import torch
import torchaudio
import numpy as np
from src.utils import config
from src.utils import audio_utils
import soundfile as sf
import logging
import sys # Added for subprocess executable path
import os
import subprocess
import json
from pathlib import Path

# logging.basicConfig(level=logging.INFO) # Centralized in app.py
logger = logging.getLogger(__name__)

# Model size mapping for UI display names to actual model names
MODEL_SIZE_MAP = {
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "medium": "medium",
    "base": "base",
    "small": "small",
    # UI-friendly names
    "Large v3": "large-v3",
    "Large v3 Turbo (Fast)": "large-v3-turbo",
    "Medium": "medium",
    "Base": "base",
    "Small": "small",
}


class SileroVAD:
    """
    Silero Voice Activity Detection wrapper.
    Used to filter non-speech regions before transcription to reduce hallucinations.
    """
    
    def __init__(self):
        self.model = None
        self.get_speech_timestamps = None
        self._loaded = False
    
    def load(self):
        """Load Silero VAD model."""
        if self._loaded:
            return
        
        try:
            logger.info("Loading Silero VAD model...")
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self.model = model
            self.get_speech_timestamps = utils[0]  # get_speech_timestamps function
            self._loaded = True
            logger.info("Silero VAD model loaded.")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}. Proceeding without VAD.")
            self._loaded = False
    
    def detect_speech(self, audio_path: str, min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 1000) -> list:
        """
        Detect speech segments in audio file.
        
        Args:
            audio_path: Path to audio file
            min_speech_duration_ms: Minimum speech duration to consider
            
        Returns:
            List of dicts with 'start' and 'end' times in seconds
        """
        if not self._loaded:
            self.load()
        
        if not self._loaded or self.model is None:
            logger.warning("VAD not available, returning full audio as speech.")
            return None  # Signal to process entire audio
        
        try:
            # Load audio at 16kHz (required by Silero)
            # Use utility for safety
            waveform, sr = audio_utils.load_audio(audio_path, target_sr=16000, mono=True)
            
            # Squeeze to 1D [Time] as Silero expects
            waveform = waveform.squeeze()
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                waveform,
                self.model,
                sampling_rate=sr,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=400,  # Buffer to avoid clipping
                threshold=0.6,
            )
            
            # Convert to seconds
            segments = []
            for ts in speech_timestamps:
                segments.append({
                    'start': ts['start'] / sr,
                    'end': ts['end'] / sr
                })
            
            logger.info(f"VAD detected {len(segments)} speech segments.")
            return segments
            
        except Exception as e:
            logger.warning(f"VAD processing failed: {e}. Processing full audio.")
            return None


class Transcriber:
    """
    Audio transcription using Faster-Whisper with VAD preprocessing.
    
    Supports models:
    - large-v3: Best accuracy, slower
    - large-v3-turbo: Fast with comparable accuracy (recommended)
    - medium, base, small: Progressively faster but less accurate
    """
    
    def __init__(self):
        self.model_size = config.WHISPER_MODEL_SIZE
        self.device = config.DEVICE
        self._original_device = config.DEVICE  # Track original for reset after fallback
        self.model = None
        self.vad = SileroVAD()
        self.use_vad = False  # VAD disabled by default (user can enable via UI)
        self.min_word_confidence = 0.0  # Disabled - keep all transcribed words
        self._retry_count = 0  # Track retries to prevent infinite loops
        self.max_retries = 1  # Max retries (1 = try once more on CPU)

    def load_model(self, size=None):
        # In subprocess mode, we don't load the model here.
        # Just update config if needed.
        target_size = MODEL_SIZE_MAP.get(size, size) if size else self.model_size
        self.model_size = target_size
        config.debug_log(f"Transcriber: Configured for {self.model_size} (Subprocess Mode)")

    def unload_model(self):
        # Nothing to unload in main process
        pass

    def transcribe(self, audio_path, language=None, model_size=None, use_vad=None, beam_size=5, min_silence_duration_ms=1000):
        """
        Transcribes audio using a subprocess to isolate CTranslate2/Whisper.
        """
        
        self.load_model(model_size) # Updates self.model_size
        
        logger.info(f"Transcribing {audio_path} with language='{language}'...")
        
        # Determine VAD usage
        should_use_vad = use_vad if use_vad is not None else self.use_vad
        vad_segments = None
        if should_use_vad:
            logger.info(f"Running VAD preprocessing (Min Silence: {min_silence_duration_ms}ms)...")
            vad_segments = self.vad.detect_speech(str(audio_path), min_silence_duration_ms=min_silence_duration_ms)
            if vad_segments:
                 logger.info(f"VAD: {len(vad_segments)} speech regions detected.")

        # Prepare Subprocess Command
        worker_script = Path(__file__).parent / "transcription_worker.py"
        python_exe = sys.executable
        
        compute_type = "float16" if self.device == "cuda" else "int8"
        
        cmd = [
            python_exe, str(worker_script),
            "--audio_path", str(audio_path),
            "--model_size", str(self.model_size),
            "--language", str(language if language else "auto"),
            "--compute_type", compute_type,
            "--device", self.device
        ]
        
        config.debug_log(f"Transcriber: Launching subprocess: {' '.join(cmd)}")
        
        try:
            # Run worker
            # Pass environment to ensure DLLs are found (though worker sets them up too)
            env = os.environ.copy()
            
            # [Fix] Sanitize PYTHONHASHSEED which causes fatal startup errors if invalid
            if "PYTHONHASHSEED" in env:
                seed_val = env["PYTHONHASHSEED"]
                logger.info(f"Subprocess Env: Found PYTHONHASHSEED='{seed_val}'")
                
                # Check validity: must be "random" or integer [0; 4294967295]
                is_valid = seed_val == "random" or (seed_val.isdigit() and 0 <= int(seed_val) <= 4294967295)
                
                if not is_valid:
                    logger.warning(f"⚠️ Removing invalid PYTHONHASHSEED='{seed_val}' from subprocess environment to prevent crash.")
                    del env["PYTHONHASHSEED"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                env=env,
                check=True
            )
            
            # Parse JSON
            raw_output = result.stdout
            
            # Extract JSON block
            import re
            match = re.search(r"<<<<JSON>>>>\s*(.*?)\s*<<<<ENDJSON>>>>", raw_output, re.DOTALL)
            if not match:
                logger.error(f"Worker Output: {raw_output}")
                raise RuntimeError("Could not find JSON block in worker output.")
                
            json_str = match.group(1)
            data = json.loads(json_str)
            
            # Reconstruct objects (dict -> slightly compatible structure)
            # Actually we just need list of dicts for the rest of pipeline
            raw_segments = data["segments"]
            detected_language = data.get("language", "en")
            
            logger.info(f"Subprocess returned {len(raw_segments)} segments. (Lang: {detected_language})")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Transcription Subprocess Failed! Exit code: {e.returncode}")
            logger.error(f"Stderr: {e.stderr}")
            
            # [Fix] Check if stdout actually contains valid output despite non-zero exit
            # This can happen when CUDA cleanup crashes after successful transcription
            if e.stdout:
                import re
                match = re.search(r"<<<<JSON>>>>\s*(.*?)\s*<<<<ENDJSON>>>>", e.stdout, re.DOTALL)
                if match:
                    logger.warning("Worker crashed after successful transcription. Attempting to recover output...")
                    try:
                        json_str = match.group(1)
                        data = json.loads(json_str)
                        raw_segments = data["segments"]
                        detected_language = data.get("language", "en")
                        logger.info(f"Recovered {len(raw_segments)} segments from crashed worker")
                        # Fall through to post-processing below (need to restructure)
                        # For now, skip to post-processing by setting variables
                    except (json.JSONDecodeError, KeyError) as parse_err:
                        logger.error(f"Failed to parse recovered output: {parse_err}")
                        raw_segments = None
                    
                    if raw_segments:
                        # Jump to post-processing (below the except blocks)
                        pass  # Will be handled by the variable being set
                    else:
                        pass  # Fall through to normal error handling
            else:
                raw_segments = None
            
            config.debug_log(f"Transcriber Crash: {e.stderr}")
            
            # If we recovered valid output, skip the retry logic and continue to post-processing
            if raw_segments is not None:
                logger.info("Continuing with recovered segments despite worker crash...")
                # detected_language should already be set from recovery
            else:
                # [Fix] Fallback to CPU if CUDA crashes (Access Violation / DLL issue)
                # But limit retries to prevent infinite loops
                if self.device == "cuda" and self._retry_count < self.max_retries:
                    self._retry_count += 1
                    logger.warning(f"Worker crashed on CUDA. Retrying with CPU fallback... (Attempt {self._retry_count}/{self.max_retries})")
                    config.debug_log("Switching to CPU fallback for Transcription worker...")
                    
                    # Update settings for CPU
                    self.device = "cpu"
                    
                    try:
                        # Recursive retry
                        result = self.transcribe(audio_path, language, model_size, use_vad, beam_size, min_silence_duration_ms)
                        # Reset for next transcription call (so we try GPU again next time)
                        self.device = self._original_device
                        self._retry_count = 0
                        return result
                    except Exception as retry_error:
                        # Reset state even on failure
                        self.device = self._original_device
                        self._retry_count = 0
                        raise
                
                # Reset retry count for next call
                self._retry_count = 0
                raise RuntimeError(f"Transcription failed after {self.max_retries} retries: {e.stderr}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse worker output: {result.stdout}")
            config.debug_log(f"Transcriber JSON Error: {result.stdout}")
            raise RuntimeError("Transcription worker returned invalid JSON")
            
        
        # Post-Processing (VAD Filtering & Cleanup)
        # Re-use existing logic
        segments = []
        for seq_dict in raw_segments:
            # Convert dict back to object-like if needed, or just use dict
            # Existing logic below expects objects 'seg.start' OR dicts?
            # Looking at original code: 'seg.start', 'seg.text'. It used namedtuples from faster_whisper.
            # But wait, original code converted 'seg' to dict:
            # segments.append({ "start": seg.start ... })
            # So I can just adapt here.
            
            start = seq_dict['start']
            end = seq_dict['end']
            text = seq_dict['text']
            words = seq_dict['words'] # List of dicts
            
            # Filter by VAD regions if available
            if vad_segments:
                if not self._segment_overlaps_speech(start, end, vad_segments):
                    logger.debug(f"Skipping segment outside VAD regions: '{text[:20]}...'")
                    continue
            
            # Word-level confidence filtering (Mocking object structure for words if needed)
            # Simplification: If min_word_confidence > 0, we need 'words' objects.
            # My worker returns dicts.
            # Existing logic: confidence = getattr(w, 'probability', 1.0)
            # So I need to handle dicts or objects.
            # Let's rewrite the filtering loop below to handle dicts.

            final_words = []
            if words and self.min_word_confidence > 0:
                 for w in words:
                     prob = w.get('probability', 1.0)
                     if prob >= self.min_word_confidence:
                         final_words.append(w)
                 if final_words:
                      text = ' '.join(w['word'] for w in final_words).strip()
            else:
                 final_words = words

            if not text: continue
            
            segments.append({
                "start": start,
                "end": end,
                "text": text,
                "words": final_words
            })
            
        logger.info(f"Raw segments: {len(segments)}. Running cleanup...")
        segments = self._clean_segments(segments)
        logger.info(f"Cleanup complete. Found {len(segments)} unique segments.")
        return segments, detected_language
    
    def _get_audio_duration(self, audio_path) -> float:
        """Get audio file duration in seconds."""
        try:
            info = sf.info(str(audio_path))
            return info.duration
        except Exception:
            return 0.0
    
    def _segment_overlaps_speech(self, start: float, end: float, vad_segments: list) -> bool:
        """Check if a transcription segment overlaps with any VAD speech region."""
        for vad in vad_segments:
            # Check for overlap
            if start < vad['end'] and end > vad['start']:
                return True
        return False

    def _clean_segments(self, segments):
        """
        Removes duplicates, merges adjacent identical segments, filters empty/noise.
        """
        if not segments:
            return []

        cleaned = []
        
        # Track phrase counts to detect loops (3+ repeats)
        phrase_counts = {}
        
        for i, seg in enumerate(segments):
            text = seg["text"].strip()
            
            # 1. Filter empty or very short noise (< 0.5s if empty text)
            if not text:
                continue
                
            # Loop detection: Count normalized text occurrences
            # Check for substantial text (not just "Ah", "Oh")
            if len(text) > 3:
                norm_text = text.lower()
                phrase_counts[norm_text] = phrase_counts.get(norm_text, 0) + 1
                
                # If this specific phrase has appeared many times, likely a hallucination loop
                # especially in TV shows (lyrics, banners, etc)
                if phrase_counts[norm_text] > 4:
                     logger.warning(f"Dropping likely hallucination loop segment: '{text}' (Count: {phrase_counts[norm_text]})")
                     continue
            
            # 2. Match with previous
            if cleaned:
                prev = cleaned[-1]
                
                # Check for exact duplicate text (hallucination loop)
                is_duplicate_text = text.lower() == prev["text"].lower().strip()
                
                # Calculate gap
                gap = seg["start"] - prev["end"]
                
                if is_duplicate_text:
                    # Only merge if they are close in time (e.g. < 5 seconds gap)
                    # If the gap is huge (e.g. minutes), it's likely a legitimate reprise or separate hallucination block
                    # But merging 30 mins into one segment is bad.
                    # Whisper hallucinations usually happen consecutively with small overlap or small gap.
                    if gap < 5.0:
                        logger.info(f"Merging duplicate segment '{text}' ({prev['end']}->{seg['end']})")
                        prev["end"] = seg["end"] 
                        continue
                    else:
                        logger.info(f"Duplicate text '{text}' found but gap {gap:.2f}s is too large. Keeping separate.")

            cleaned.append(seg)
            
        return cleaned

    def merge_short_segments(self, segments: list, min_duration: float = 2.0, max_gap: float = 0.5) -> list:
        """
        Merges short segments into longer, coherent sentences.
        
        Args:
            segments: List of segments {start, end, text, words}
            min_duration: Threshold - merge if segment duration < this (seconds)
            max_gap: Maximum gap to next segment to allow merging (seconds)
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
            
        merged = []
        current_seg = segments[0].copy()
        
        for next_seg in segments[1:]:
            # Calculate properties
            current_duration = current_seg['end'] - current_seg['start']
            gap = next_seg['start'] - current_seg['end']
            
            # Check merge conditions:
            # 1. Current segment is short
            # 2. Gap is small (part of same flow)
            # 3. (Optional) Check punctuation? For now, we assume short segments usually need merging.
            #    Ideally we wouldn't merge if current ends in specific punctuation like '?' or '!', 
            #    but often Whisper breaks sentences mid-phrase.
            
            should_merge = (current_duration < min_duration) and (gap < max_gap)
            
            if should_merge:
                # Merge into current_seg
                current_seg['end'] = next_seg['end']
                
                # Intelligent text merging
                curr_text = current_seg['text'].strip()
                next_text = next_seg['text'].strip()
                
                # Avoid double spacing or missing spacing
                if curr_text.endswith('-') or next_text.startswith('-'):
                     # Hyphenated break
                     current_seg['text'] = curr_text + next_text
                else:
                     current_seg['text'] = f"{curr_text} {next_text}"
                     
                # Merge words list if available
                if 'words' in current_seg and 'words' in next_seg:
                    current_seg['words'] = current_seg['words'] + next_seg['words']
                    
                logger.debug(f"Merged segment: '{current_seg['text']}' (New duration: {current_seg['end'] - current_seg['start']:.2f}s)")
                
            else:
                # Push current and promote next
                merged.append(current_seg)
                current_seg = next_seg.copy()
                
        # Append the final segment
        merged.append(current_seg)
        
        return merged

if __name__ == "__main__":
    t = Transcriber()
    # Dummy test
    # t.transcribe("path/to/audio.wav") 
    pass
