import logging
import os
import torch
import config
import numpy as np

logger = logging.getLogger(__name__)

class Diarizer:
    def __init__(self):
        self.pipeline = None
        # Pyannote pipeline moves to device automatically if specified, or we move it manually
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_pipeline(self):
        if self.pipeline:
             return
        
        token = os.getenv("HF_TOKEN")
        if not token:
             # Just warning, app might catch it later or user didn't enable it
             logger.warning("HF_TOKEN is missing. Diarization will likely fail if attempted.")
             return
             
        logger.info("Loading Pyannote Diarization pipeline (speaker-diarization-3.1)...")
        from pyannote.audio import Pipeline
        try:
             self.pipeline = Pipeline.from_pretrained(
                 "pyannote/speaker-diarization-3.1", 
                 use_auth_token=token
             )
             if self.pipeline:
                 self.pipeline.to(self.device)
                 logger.info(f"Diarization pipeline loaded on {self.device}.")
             else:
                 logger.error("Failed to load pipeline (returned None). Check HF_TOKEN capabilities.")
                 
        except Exception as e:
             logger.error(f"Failed to load Pyannote pipeline: {e}")
             raise

    def diarize(self, audio_path):
        """
        Returns list of segments: [{'start': float, 'end': float, 'speaker': str}, ...]
        """
        self._load_pipeline()
        if not self.pipeline:
            raise RuntimeError("Diarization pipeline not loaded (check HF_TOKEN).")
            
        logger.info(f"Diarizing {audio_path}...")
        # Run pipeline
        diarization = self.pipeline(audio_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        logger.info(f"Diarization complete. Found {len(segments)} segments across {len(diarization.labels())} speakers.")
        return segments

    def detect_genders(self, audio_path, segments):
        """
        Analyzes speaker segments to estimate gender (Male/Female) based on Pitch (F0).
        Returns dict {speaker_id: 'Male'|'Female'}
        """
        import librosa
        
        # Group segments by speaker
        speaker_segments = {}
        for seg in segments:
            sp = seg['speaker']
            if sp not in speaker_segments:
                speaker_segments[sp] = []
            speaker_segments[sp].append((seg['start'], seg['end']))
            
        genders = {}
        
        logger.info("Loading audio for gender detection...")
        # Load audio once, mono
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
        except Exception as e:
            logger.error(f"Librosa load failed: {e}")
            return {sp: "Male" for sp in speaker_segments} # Fallback
            
        logger.info("Analyzing pitch for speakers...")
        
        for sp, times in speaker_segments.items():
            # Collect audio for this speaker
            # Limit to max 20 seconds of speech to save time
            
            speaker_audio = []
            total_dur = 0
            for start, end in times:
                s_sample = int(start * sr)
                e_sample = int(end * sr)
                
                # Bounds check
                s_sample = max(0, s_sample)
                e_sample = min(len(y), e_sample)
                
                if s_sample >= e_sample: continue
                
                chunk = y[s_sample:e_sample]
                speaker_audio.append(chunk)
                total_dur += (end - start)
                if total_dur > 20.0: 
                     break
            
            if not speaker_audio:
                genders[sp] = "Male" # Default
                continue
                
            full_audio = np.concatenate(speaker_audio)
            
            # F0 estimation (PyIN is robust but slow-ish, so we limited duration)
            # fmin=65 (Low C2), fmax=300 (approx D4, covers typical speech fund. freq)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                full_audio, 
                fmin=65, 
                fmax=300,
                sr=sr
            )
            
            # Filter NaNs
            f0 = f0[~np.isnan(f0)]
            
            if len(f0) == 0:
                 genders[sp] = "Male" # Fallback
            else:
                 mean_f0 = np.mean(f0)
                 # Threshold: 
                 # Adult Male: 85-155 Hz
                 # Adult Female: 165-255 Hz
                 # Cutoff 160Hz
                 if mean_f0 > 160:
                     genders[sp] = "Female"
                 else:
                     genders[sp] = "Male"
                     
                 logger.info(f"Speaker {sp}: Mean F0 = {mean_f0:.1f} Hz -> {genders[sp]}")
            
        return genders

if __name__ == "__main__":
    # Test
    pass
