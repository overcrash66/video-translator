import logging
import torch
import torchaudio
import os
import gc
from pathlib import Path
import config

logger = logging.getLogger(__name__)

class F5TTSWrapper:
    """
    Wrapper for F5-TTS (Non-Autoregressive, Low Latency, High Fidelity).
    Supports Voice Cloning with Sway Sampling.
    """
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False

    def load_model(self):
        if self.model_loaded:
            return

        logger.info(f"Loading F5-TTS model on {self.device}...")
        try:
            # Import here to avoid hard dependency if not installed
            from f5_tts.api import F5TTS
            
            # Initialize with default settings or specific checkpoint if needed
            # F5TTS class typically handles model download/loading
            self.pipeline = F5TTS(device=self.device)
            self.model_loaded = True
            logger.info("F5-TTS loaded successfully.")
            
        except ImportError:
            logger.error("F5-TTS module not found. Install with `pip install f5-tts`.")
            raise
        except Exception as e:
            logger.error(f"Failed to load F5-TTS: {e}")
            raise

    def unload_model(self):
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model_loaded = False
        logger.info("F5-TTS unloaded.")

    def generate_voice_clone(self, text, ref_audio_path, ref_text="", output_path=None):
        """
        Generates speech using F5-TTS with voice cloning.
        Supports long-form text by segmenting and merging.
        """
        if not self.model_loaded:
            self.load_model()
            
        if not output_path:
            import uuid
            output_path = config.TEMP_DIR / f"f5_{uuid.uuid4()}.wav"
            
        try:
            import soundfile as sf
            import numpy as np
            import pysbd
            from pydub import AudioSegment
            
            # Segment text
            seg = pysbd.Segmenter(language="en", clean=False)
            sentences = seg.segment(text)
            
            logger.info(f"F5-TTS: Segmented text into {len(sentences)} parts.")
            
            audio_segments = []
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 2: continue
                
                logger.info(f"Generating segment {i+1}/{len(sentences)}: '{sentence[:20]}...'")
                
                # Infer single segment
                # F5TTS api.infer returns (wav, sr, spect)
                wav, sr, _ = self.pipeline.infer(
                    ref_file=str(ref_audio_path),
                    ref_text=ref_text,
                    gen_text=sentence,
                    file_wave=None,
                    file_spec=None,
                    seed=-1
                )
                
                # Convert to pydub AudioSegment
                if hasattr(wav, 'cpu'): wav = wav.squeeze().cpu().numpy()
                
                # Convert float32 numpy to int16 for pydub
                wav_int16 = (wav * 32767).astype(np.int16)
                
                segment = AudioSegment(
                    wav_int16.tobytes(), 
                    frame_rate=sr,
                    sample_width=2, 
                    channels=1
                )
                audio_segments.append(segment)
                
            if not audio_segments:
                raise ValueError("No audio generated from text segments.")
                
            # Stitch with cross-fade
            final_audio = audio_segments[0]
            cross_fade_ms = 50
            
            for next_seg in audio_segments[1:]:
                final_audio = final_audio.append(next_seg, crossfade=cross_fade_ms)
                
            # Export
            final_audio.export(str(output_path), format="wav")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"F5-TTS Generation failed: {e}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wrapper = F5TTSWrapper()
    # wrapper.load_model()
