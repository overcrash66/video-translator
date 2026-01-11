import ffmpeg
import logging
import os
from pathlib import Path
import torchaudio
import config
import soundfile as sf
import numpy as np
try:
    import pyrubberband as pyrb
except ImportError:
    pyrb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioSynchronizer:
    def sync_segment(self, audio_path, target_duration, output_path=None):
        """
        Stretches or squeezes the audio at audio_path to fit target_duration.
        """
        if not output_path:
            output_path = str(Path(audio_path).parent / f"synced_{Path(audio_path).name}")
        
        # Get current duration
        try:
            probe = ffmpeg.probe(audio_path)
            audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            current_duration = float(audio_info['duration'])
        except Exception as e:
            logger.error(f"Failed to probe audio {audio_path}: {e}")
            return audio_path # Return original if fail
        
        if current_duration <= 0 or target_duration <= 0:
            return audio_path

        speed_factor = current_duration / target_duration
        
        logger.info(f"Syncing {audio_path}: current={current_duration}s, target={target_duration}s, speed={speed_factor:.2f}")

        # 1. Close enough check (0.95 - 1.05) - Just copy
        if 0.95 < speed_factor < 1.05:
            import shutil
            shutil.copy(audio_path, output_path)
            return output_path

        # 2. [Safety Check] Extreme stretching (Fallback to Padding/Trimming)
        if speed_factor < 0.25 or speed_factor > 4.0:
            logger.warning(f"Speed factor {speed_factor:.2f} is extreme. Fallback to padding/trimming.")
            
            w, sr = sf.read(str(audio_path))
            target_samples = int(target_duration * sr)
            
            if w.ndim == 1:
                w = w[:, np.newaxis]
            
            current_samples = w.shape[0]
            
            if current_samples < target_samples:
                # Pad with silence
                padding = target_samples - current_samples
                new_w = np.vstack([w, np.zeros((padding, w.shape[1]))])
            else:
                # Trim
                new_w = w[:target_samples, :]
                
            sf.write(output_path, new_w, sr)
            return output_path

        # 3. [Optimization] Use pyrubberband for natural stretching
        if pyrb:
            try:
                # Load with soundfile
                y, sr = sf.read(str(audio_path))
                if y.ndim > 1:
                     if y.shape[1] > 1:
                        y = y[:, 0] # Force mono

                y_stretched = pyrb.time_stretch(y, sr, speed_factor)
                
                sf.write(output_path, y_stretched, sr)
                return output_path
            except Exception as e:
                logger.error(f"Pyrubberband failed: {e}. Falling back to FFmpeg.")

        # 4. [Fallback] FFmpeg 'atempo' filter
        atempo_filters = []
        remaining_factor = speed_factor
        
        while remaining_factor > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining_factor /= 2.0
        while remaining_factor < 0.5:
            atempo_filters.append("atempo=0.5")
            remaining_factor /= 0.5
             
        if remaining_factor != 1.0:
             atempo_filters.append(f"atempo={remaining_factor}")
        
        try:
            stream = ffmpeg.input(audio_path)
            for f in atempo_filters:
                 stream = stream.filter('atempo', f.split('=')[1])
            
            stream.output(output_path).run(overwrite_output=True, quiet=True)
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg sync failed: {e.stderr.decode() if e.stderr else str(e)}")
            return audio_path

    def merge_segments(self, segments, total_duration, output_path):
        """
        Concatenates segments into a single track, respecting start times.
        segments: list of dict { 'audio_path': str, 'start': float, 'end': float }
        This is complex because we need to insert silence.
        """
        # Better: [silence_d1][clip1][silence_d2][clip2]...
        
        inputs = []
        current_time = 0.0
        
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        target_sr = 24000 # default for XTTS
        
        # Create empty canvas
        total_samples = int(total_duration * target_sr) + 24000 # buffer
        # careful with memory if video is 1 hour.
        # For < 5 min videos, tensor is fine.
        
        try:
            import torch
            import soundfile as sf
            import numpy as np
            
            # Canvas in tensor [1, Samples]
            canvas = torch.zeros((1, total_samples))
            
            for seg in sorted_segments:
                path = seg['audio_path']
                start_sec = seg['start']
                
                # Validate segment file before processing
                try:
                    seg_path = Path(path)
                    if not seg_path.exists():
                        logger.warning(f"Skipping non-existent audio segment: {path}")
                        continue
                    
                    file_size = seg_path.stat().st_size
                    if file_size < 100:  # Minimum 100 bytes for valid audio
                        logger.warning(f"Skipping too-small audio segment ({file_size} bytes): {path}")
                        continue
                except Exception as e:
                    logger.warning(f"Error validating segment {path}: {e}")
                    continue
                
                # Load segment with soundfile
                try:
                    wav_np, sr = sf.read(str(path))
                except Exception as e:
                    logger.warning(f"Failed to read audio segment {path}: {e}")
                    continue
                
                if wav_np.size == 0:
                    logger.warning(f"Skipping empty audio segment: {path}")
                    continue
                
                # Convert to [Channels, Time]
                if wav_np.ndim == 1:
                    wav_np = wav_np[np.newaxis, :]
                else:
                    wav_np = wav_np.T
                
                wav = torch.from_numpy(wav_np).float()
                
                # Resample if needed
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    wav = resampler(wav)
                
                # Add to canvas
                start_sample = int(start_sec * target_sr)
                end_sample = start_sample + wav.shape[1]
                
                if end_sample > total_samples:
                    # expand canvas? or crop?
                    # pad
                    padding = end_sample - total_samples + 1000
                    canvas = torch.cat([canvas, torch.zeros(1, padding)], dim=1)
                    total_samples = canvas.shape[1]

                # Mix (overwrite) to avoid double-speech on overlaps
                # If segments overlap, the later one takes precedence (standard behavior)
                # We simply assign the values instead of adding them.
                canvas[:, start_sample:end_sample] = wav[:, :end_sample-start_sample]
            
            # Save
            # Convert back to [Time, Channels] for Soundfile
            canvas_np = canvas.cpu().numpy().T
            sf.write(output_path, canvas_np, target_sr)
            return output_path
            
        except Exception as e:
            logger.error(f"Merging failed in torch: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    pass
