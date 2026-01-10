from faster_whisper import WhisperModel
import torch
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self):
        self.model_size = config.WHISPER_MODEL_SIZE
        self.device = config.DEVICE
        self.model = None

    def load_model(self, size=None):
        target_size = size or self.model_size
        
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

    def transcribe(self, audio_path, language=None, model_size=None):
        """
        Transcribes the audio file and returns segments with timestamps.
        Returns: list of dicts {start, end, text}
        """
        self.load_model(model_size)
        if not self.model:
            raise RuntimeError("Whisper model not loaded.")

        logger.info(f"Transcribing {audio_path} with language='{language}'...")
        
        # Mapping parameters for faster-whisper
        # beam_size=5 is standard
        # language=None means auto-detect
        
        lang_arg = language if (language and language != "auto") else None
        
        segments_generator, info = self.model.transcribe(
            str(audio_path), 
            language=lang_arg,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt="This is a dialogue. Transcribe it accurately.",
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )
        
        logger.info(f"Detected language '{info.language}' with probability {info.language_probability}")

        # Unroll generator
        # Faster-whisper segments differ slightly in structure
        # Segment(start, end, text, words=...)
        
        result_segments = list(segments_generator)
        
        segments = []
        for seg in result_segments:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": seg.words if hasattr(seg, 'words') else [] 
            })
            
        logger.info(f"Raw segments: {len(segments)}. Running cleanup...")
        segments = self._clean_segments(segments)
        logger.info(f"Cleanup complete. Found {len(segments)} unique segments.")
        return segments

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
