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
                
                # Calculate expected duration for this segment
                expected_duration = seg.get('end', start_sec + 1.0) - start_sec
                actual_duration = len(wav_np) / sr if wav_np.ndim == 1 else wav_np.shape[0] / sr
                
                # Time-stretch if there's significant duration mismatch
                if expected_duration > 0 and actual_duration > 0:
                    ratio = actual_duration / expected_duration
                    
                    # Only stretch if there's more than 10% difference
                    if ratio < 0.9 or ratio > 1.1:
                        logger.info(f"Time-stretching segment: actual={actual_duration:.2f}s -> target={expected_duration:.2f}s (ratio={ratio:.2f})")
                        
                        stretched = False
                        
                        # Use pyrubberband for quality stretching (requires rubberband-cli)
                        if pyrb and not stretched:
                            try:
                                # Ensure mono and float64 for pyrubberband
                                if wav_np.ndim > 1:
                                    wav_np = wav_np[:, 0]
                                wav_np = wav_np.astype(np.float64)
                                
                                # Stretch factor: >1 = slow down (make longer), <1 = speed up (make shorter)
                                wav_np = pyrb.time_stretch(wav_np, sr, ratio)
                                stretched = True
                                logger.info(f"Time-stretched with pyrubberband (ratio={ratio:.2f})")
                            except Exception as e:
                                logger.warning(f"Pyrubberband stretch failed: {e}")
                        
                        # Fallback 1: Use librosa (pure Python, no CLI required)
                        if librosa and not stretched:
                            try:
                                # Ensure mono for librosa
                                if wav_np.ndim > 1:
                                    wav_np = wav_np[:, 0]
                                wav_np = wav_np.astype(np.float32)
                                
                                # librosa.effects.time_stretch uses phase vocoder
                                # rate > 1 = speed up, rate < 1 = slow down
                                wav_np = librosa.effects.time_stretch(wav_np, rate=ratio)
                                stretched = True
                                logger.info(f"Time-stretched with librosa (ratio={ratio:.2f})")
                            except Exception as e:
                                logger.warning(f"Librosa stretch failed: {e}")
                        
                        # Fallback 2: Mild trim/pad - but DON'T destroy content
                        # Only trim up to 30% to preserve most speech content
                        if not stretched:
                            target_samples = int(expected_duration * sr)
                            current_samples = len(wav_np) if wav_np.ndim == 1 else wav_np.shape[0]
                            
                            # Calculate what percentage we'd lose
                            if current_samples > target_samples:
                                loss_pct = (current_samples - target_samples) / current_samples
                                
                                if loss_pct <= 0.3:
                                    # Mild trim is acceptable (up to 30% loss)
                                    wav_np = wav_np[:target_samples]
                                    fade_len = min(int(target_samples * 0.1), 2400)
                                    if fade_len > 0:
                                        fade = np.linspace(1.0, 0.0, fade_len)
                                        wav_np[-fade_len:] *= fade
                                    logger.info(f"Mild trim: {current_samples} -> {target_samples} samples ({loss_pct*100:.0f}% loss)")
                                else:
                                    # Would lose too much - just use original and let overlap handling work
                                    logger.warning(f"Skipping trim (would lose {loss_pct*100:.0f}%). Using original audio.")
                            elif current_samples < target_samples:
                                # Pad with silence (this is fine, no content loss)
                                padding = np.zeros(target_samples - current_samples)
                                wav_np = np.concatenate([wav_np, padding])
                                logger.info(f"Padded segment: {current_samples} -> {target_samples} samples")
                
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
                
                # Add to canvas with proper overlap handling
                start_sample = int(start_sec * target_sr)
                end_sample = start_sample + wav.shape[1]
                
                if end_sample > total_samples:
                    # expand canvas
                    padding = end_sample - total_samples + 1000
                    canvas = torch.cat([canvas, torch.zeros(1, padding)], dim=1)
                    total_samples = canvas.shape[1]

                # Check if there's existing audio we would overwrite
                # Only apply cross-fade blending when there's actual overlap
                existing_region = canvas[:, start_sample:end_sample]
                has_existing_audio = existing_region.abs().max() > 0.01
                
                if has_existing_audio:
                    # There IS existing audio - apply proper crossfade blending
                    # Use 50ms fade (1200 samples at 24kHz)
                    fade_samples = min(1200, wav.shape[1] // 4)  # Max 25% of segment
                    
                    if fade_samples > 10:
                        # Create fade-in window for the incoming segment
                        fade_in = torch.from_numpy(
                            generate_crossfade_window(fade_samples, 'cosine')
                        ).float().unsqueeze(0)
                        # Create fade-out window for existing audio
                        fade_out = torch.from_numpy(
                            generate_crossfade_window(fade_samples, 'cosine')[::-1].copy()
                        ).float().unsqueeze(0)
                        
                        # Apply crossfade only in the overlap region at start
                        existing_start = canvas[:, start_sample:start_sample + fade_samples].clone()
                        
                        # Blend: fade out existing + fade in new
                        blended = existing_start * fade_out + wav[:, :fade_samples] * fade_in
                        
                        # Place blended region
                        canvas[:, start_sample:start_sample + fade_samples] = blended
                        # Place rest of segment at full volume
                        canvas[:, start_sample + fade_samples:end_sample] = wav[:, fade_samples:end_sample-start_sample]
                        
                        logger.debug(f"Crossfade applied at {start_sec:.2f}s (overlap with existing audio)")
                    else:
                        # Segment too short for cross-fade, just overwrite
                        canvas[:, start_sample:end_sample] = wav[:, :end_sample-start_sample]
                else:
                    # No existing audio - place segment at FULL volume (no fading!)
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
