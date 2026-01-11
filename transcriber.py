from faster_whisper import WhisperModel
import torch
import torchaudio
import numpy as np
import config
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
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
    
    def detect_speech(self, audio_path: str, min_speech_duration_ms: int = 250) -> list:
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
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
                sr = 16000
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.squeeze()
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                waveform,
                self.model,
                sampling_rate=sr,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=100,
                speech_pad_ms=30,
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
        self.model = None
        self.vad = SileroVAD()
        self.use_vad = True  # Enable VAD preprocessing by default
        self.min_word_confidence = 0.5  # Filter words with low confidence

    def load_model(self, size=None):
        """Load Whisper model, handling model name mapping."""
        # Map display name to actual model name
        target_size = MODEL_SIZE_MAP.get(size, size) if size else self.model_size
        
        # Check if we need to reload
        if self.model and self.model_size == target_size:
            return

        if self.model:
           self.unload_model()
           
        self.model_size = target_size
        logger.info(f"Loading Faster-Whisper model: {self.model_size} on {self.device}...")
        try:
            # compute_type="float16" for GPU, "int8" for CPU usually standard
            compute_type = "float16" if self.device == "cuda" else "int8"
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)
            logger.info("Faster-Whisper model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}")
            logger.info("Attempting to load 'base' model as fallback.")
            self.model = WhisperModel("base", device=self.device, compute_type="int8")

    def unload_model(self):
        if self.model:
            logger.info("Unloading Whisper model...")
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Whisper model unloaded.")

    def transcribe(self, audio_path, language=None, model_size=None, use_vad=None):
        """
        Transcribes the audio file and returns segments with timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code or 'auto' for auto-detection
            model_size: Whisper model size (large-v3, large-v3-turbo, medium, base)
            use_vad: Override VAD preprocessing (default: self.use_vad)
            
        Returns:
            list of dicts {start, end, text, words}
        """
        self.load_model(model_size)
        if not self.model:
            raise RuntimeError("Whisper model not loaded.")

        logger.info(f"Transcribing {audio_path} with language='{language}'...")
        
        # Determine if we should use VAD
        should_use_vad = use_vad if use_vad is not None else self.use_vad
        
        # VAD preprocessing to identify speech regions
        vad_segments = None
        if should_use_vad:
            logger.info("Running VAD preprocessing...")
            vad_segments = self.vad.detect_speech(str(audio_path))
            if vad_segments:
                total_audio_duration = self._get_audio_duration(audio_path)
                speech_duration = sum(s['end'] - s['start'] for s in vad_segments)
                logger.info(f"VAD: {len(vad_segments)} speech regions, {speech_duration:.1f}s speech / {total_audio_duration:.1f}s total")
        
        # Mapping parameters for faster-whisper
        lang_arg = language if (language and language != "auto") else None
        
        # Transcribe with Whisper
        segments_generator, info = self.model.transcribe(
            str(audio_path), 
            language=lang_arg,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt="This is a dialogue. Transcribe it accurately.",
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vad_filter=True,  # Enable Whisper's built-in VAD as backup
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        logger.info(f"Detected language '{info.language}' with probability {info.language_probability}")

        # Unroll generator
        result_segments = list(segments_generator)
        
        segments = []
        for seg in result_segments:
            # Filter by VAD regions if available
            if vad_segments:
                if not self._segment_overlaps_speech(seg.start, seg.end, vad_segments):
                    logger.debug(f"Skipping segment outside VAD regions: '{seg.text[:50]}...'")
                    continue
            
            # Word-level confidence filtering
            text = seg.text.strip()
            words = seg.words if hasattr(seg, 'words') and seg.words else []
            
            if words and self.min_word_confidence > 0:
                # Filter low-confidence words and reconstruct text
                filtered_words = []
                for w in words:
                    # Word object has: start, end, word, probability
                    confidence = getattr(w, 'probability', 1.0)
                    if confidence >= self.min_word_confidence:
                        filtered_words.append(w)
                    else:
                        logger.debug(f"Filtering low-confidence word: '{w.word}' ({confidence:.2f})")
                
                if filtered_words:
                    text = ' '.join(w.word for w in filtered_words).strip()
                    words = filtered_words
            
            if not text:
                continue
                
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "words": words
            })
            
        logger.info(f"Raw segments: {len(segments)}. Running cleanup...")
        segments = self._clean_segments(segments)
        logger.info(f"Cleanup complete. Found {len(segments)} unique segments.")
        return segments
    
    def _get_audio_duration(self, audio_path) -> float:
        """Get audio file duration in seconds."""
        try:
            info = torchaudio.info(str(audio_path))
            return info.num_frames / info.sample_rate
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

if __name__ == "__main__":
    t = Transcriber()
    # Dummy test
    # t.transcribe("path/to/audio.wav") 
    pass
