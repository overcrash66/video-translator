import ffmpeg
import logging
import os
from pathlib import Path
import torchaudio
from src.utils import config
import soundfile as sf
import numpy as np
try:
    import pyrubberband as pyrb
    # Configure pyrubberband to use 'rubberband-program' on Windows
    import sys
    if sys.platform == 'win32':
        pyrb.pyrb.__RUBBERBAND_UTIL = 'rubberband-program'
except ImportError:
    pyrb = None

try:
    import librosa
except ImportError:
    librosa = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_crossfade_window(length: int, fade_type: str = 'linear') -> np.ndarray:
    """
    Generate a crossfade window for smooth audio transitions.
    
    Args:
        length: Number of samples for the fade
        fade_type: Type of fade curve ('linear', 'cosine', 'exponential')
        
    Returns:
        Fade-in curve as numpy array (multiply by audio for fade effect)
    """
    if length <= 0:
        return np.array([])
    
    if fade_type == 'cosine':
        # Smoother S-curve using cosine interpolation
        t = np.linspace(0, np.pi, length)
        return (1 - np.cos(t)) / 2
    elif fade_type == 'exponential':
        # Faster rise at the start, slower at the end
        t = np.linspace(0, 1, length)
        return t ** 0.5
    else:
        # Linear fade
        return np.linspace(0, 1, length)


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
                
                # Ensure float64 for pyrubberband
                y = y.astype(np.float64)

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

    def merge_segments(self, segments, total_duration, output_path, enable_time_stretch=False):
        """
        Concatenates segments into a single track, respecting start times.
        segments: list of dict { 'audio_path': str, 'start': float, 'end': float }
        enable_time_stretch: If True, stretch/compress audio to fit expected duration.
                            If False (default), place audio at original duration.
        """
        import torch
        import soundfile as sf
        import numpy as np
        import torchaudio
        
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        target_sr = 24000  # default for XTTS
        
        # We will dynamically calculate the new start time for each segment.
        # Max delay we allow to accumulate is 1.5 seconds.
        MAX_DELAY = 1.5
        MAX_SPEED = 1.35  # Max speed up
        MIN_SPEED = 0.8   # Max slow down
        
        accumulated_delay = 0.0
        prev_end_sec = 0.0
        
        # First, pre-calculate all segment start times and durations
        processed_segments = []
        
        for i, seg in enumerate(sorted_segments):
            path = seg['audio_path']
            orig_start = seg['start']
            orig_end = seg.get('end', orig_start + 1.0)
            orig_dur = orig_end - orig_start
            
            # Validate segment file
            try:
                seg_path = Path(path)
                if not seg_path.exists() or seg_path.stat().st_size < 100:
                    logger.warning(f"Skipping invalid segment file: {path}")
                    continue
            except Exception:
                continue
                
            # Read actual duration
            try:
                info = sf.info(str(path))
                actual_dur = info.duration
            except Exception:
                logger.warning(f"Skipping segment because sf.info failed: {path}")
                continue
                
            desired_start = orig_start + accumulated_delay
            
            # If desired_start is before prev_end_sec, we have an overlap.
            # We try to shift the start to prev_end_sec to avoid overlap.
            if desired_start < prev_end_sec:
                needed_shift = prev_end_sec - desired_start
                temp_delay = accumulated_delay + needed_shift
                # Clamp accumulated delay to MAX_DELAY
                new_delay = min(temp_delay, MAX_DELAY)
                new_start = orig_start + new_delay
                accumulated_delay = new_delay
            else:
                new_start = desired_start
                
            # Calculate max available duration before next segment starts
            if i < len(sorted_segments) - 1:
                next_orig_start = sorted_segments[i+1]['start']
                available_duration = (next_orig_start + accumulated_delay) - new_start
            else:
                available_duration = total_duration - new_start
                
            # If available_duration is negative or extremely small, set a minimum target duration
            available_duration = max(0.5, available_duration)
            
            # Determine stretch ratio
            final_dur = actual_dur
            clamped_ratio = 1.0
            
            if enable_time_stretch and orig_dur > 0:
                # Target is to fit in available_duration
                ratio = actual_dur / available_duration
                
                # Check if there is significant mismatch (more than 10%)
                if ratio < 0.9 or ratio > 1.1:
                    clamped_ratio = ratio
                    if ratio > MAX_SPEED:
                        clamped_ratio = MAX_SPEED
                    elif ratio < MIN_SPEED:
                        clamped_ratio = MIN_SPEED
                    
                    final_dur = actual_dur / clamped_ratio
            
            processed_segments.append({
                'path': path,
                'new_start': new_start,
                'final_dur': final_dur,
                'ratio': clamped_ratio,
                'actual_dur': actual_dur
            })
            
            prev_end_sec = new_start + final_dur

        # Determine canvas size
        max_end_time = prev_end_sec
        total_samples = int(max(total_duration, max_end_time) * target_sr) + 24000
        
        try:
            canvas = torch.zeros((1, total_samples))
            
            for seg_info in processed_segments:
                path = seg_info['path']
                new_start = seg_info['new_start']
                ratio = seg_info['ratio']
                
                try:
                    wav_np, sr = sf.read(str(path))
                except Exception as e:
                    logger.warning(f"Failed to read segment {path}: {e}")
                    continue
                    
                if wav_np.size == 0:
                    continue
                    
                # Perform time stretch if needed
                if ratio != 1.0:
                    stretched = False
                    
                    # Use pyrubberband
                    if pyrb and not stretched:
                        try:
                            if wav_np.ndim > 1:
                                wav_np = wav_np[:, 0]
                            wav_np = wav_np.astype(np.float64)
                            wav_np = pyrb.time_stretch(wav_np, sr, ratio)
                            stretched = True
                        except Exception as e:
                            logger.warning(f"Pyrubberband stretch failed: {e}")
                            
                    # Use librosa
                    if librosa and not stretched:
                        try:
                            if wav_np.ndim > 1:
                                wav_np = wav_np[:, 0]
                            wav_np = wav_np.astype(np.float32)
                            wav_np = librosa.effects.time_stretch(wav_np, rate=ratio)
                            stretched = True
                        except Exception as e:
                            logger.warning(f"Librosa stretch failed: {e}")
                            
                    # Fallback: trim/pad if stretching failed
                    if not stretched:
                        target_samples = int((seg_info['actual_dur'] / ratio) * sr)
                        current_samples = len(wav_np) if wav_np.ndim == 1 else wav_np.shape[0]
                        if current_samples > target_samples:
                            wav_np = wav_np[:target_samples]
                        elif current_samples < target_samples:
                            padding = np.zeros(target_samples - current_samples)
                            wav_np = np.concatenate([wav_np, padding])
                            
                # Convert to PyTorch tensor
                if wav_np.ndim == 1:
                    wav_np = wav_np[np.newaxis, :]
                else:
                    wav_np = wav_np.T
                    
                wav = torch.from_numpy(wav_np).float()
                
                # Resample if needed
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    wav = resampler(wav)
                    
                start_sample = int(new_start * target_sr)
                end_sample = start_sample + wav.shape[1]
                
                if end_sample > canvas.shape[1]:
                    padding = end_sample - canvas.shape[1] + 1000
                    canvas = torch.cat([canvas, torch.zeros(1, padding)], dim=1)
                    
                # Place segment by ADDING it to the canvas (mixing)
                # This ensures no speech is cut off!
                canvas[:, start_sample:end_sample] += wav[:, :end_sample-start_sample]
                
            # Save
            canvas_np = canvas.cpu().numpy().T
            
            # Soft-normalize to prevent clipping if signals added up
            max_val = np.max(np.abs(canvas_np))
            if max_val > 0.95:
                canvas_np = canvas_np * (0.95 / max_val)
                
            sf.write(output_path, canvas_np, target_sr)
            return output_path
            
        except Exception as e:
            logger.error(f"Merging failed in torch: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    pass
