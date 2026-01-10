import ffmpeg
import logging
import os
from pathlib import Path
import torchaudio
import config

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
        
        # Limit speed changes to avoid robotic artifacts or crashes
        # e.g., between 0.5x and 2.0x is usually safe-ish.
        # But if we must fit, we must fit. However, too fast = unintelligible.
        # If speed_factor > 2.0 (needs to be 2x faster), it's very fast.
        
        logger.info(f"Syncing {audio_path}: current={current_duration}s, target={target_duration}s, speed={speed_factor:.2f}")

        # [Safety Check] Extreme stretching
        if speed_factor < 0.25 or speed_factor > 4.0:
            logger.warning(f"Speed factor {speed_factor:.2f} is extreme. Fallback to padding/trimming.")
            import soundfile as sf
            import numpy as np
            
            w, sr = sf.read(str(audio_path))
            target_samples = int(target_duration * sr)
            
            if w.ndim == 1:
                w = w[:, np.newaxis]
                
            current_samples = w.shape[0]
            
            if current_samples < target_samples:
                # Pad with silence
                padding = target_samples - current_samples
                # Pad at end
                new_w = np.vstack([w, np.zeros((padding, w.shape[1]))])
            else:
                # Trim
                new_w = w[:target_samples, :]
                
            sf.write(output_path, new_w, sr)
            return output_path

        # atempo filter limitations: 0.5 to 2.0. Chain if needed.
        # If speed_factor is 1.0 (approx), just copy.
        if 0.95 < speed_factor < 1.05:
            # close enough
            import shutil
            shutil.copy(audio_path, output_path)
            return output_path

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
        
        filter_str = ",".join(atempo_filters)
        
        try:
            (
                ffmpeg
                .input(audio_path)
                .filter('atempo', remaining_factor) # simplified for single pass, need complex logic for chaining in ffmpeg-python?
                # Actually ffmpeg-python .filter calls can be chained.
                # Let's rebuild the chain correctly
            )
            
            stream = ffmpeg.input(audio_path)
            # Re-implement chaining loop on the stream object
            # Reset
            remaining_factor = speed_factor
            while remaining_factor > 2.0:
                stream = stream.filter('atempo', 2.0)
                remaining_factor /= 2.0
            while remaining_factor < 0.5:
                stream = stream.filter('atempo', 0.5)
                remaining_factor /= 0.5
            stream = stream.filter('atempo', remaining_factor)
            
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
        # We can construct a filter_complex command
        # or generate silence clips and concat them.
        
        # Simple approach: Create a full duration silent track, and overlay each segment.
        # But 'amix' or 'overlay' logic can be heavy.
        
        # Better: [silence_d1][clip1][silence_d2][clip2]...
        
        inputs = []
        current_time = 0.0
        
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        filter_chain = []
        
        # To avoid massive CLI arguments, maybe process in chunks or use a specific silence generator filter
        # For prototype, we might find pydub easier, but we stick to ffmpeg.
        
        # Let's try to build a list of file inputs.
        # But constructing silence needs `aevalsrc`.
        
        # implementation detail:
        # Create an concat file list?
        # No, gaps need explicit silence.
        
        # Refined approach:
        # Generate the full timeline using PyTorch/Torchaudio if possible (since we have logic).
        # But we want to use the processed files.
        # Let's use torchaudio to load, pad, and mix. It's more precise than ffmpeg CLI for this specific logic 
        # unless memory is an issue.
        
        target_sr = 24000 # default for XTTS
        
        # Create empty canvas
        total_samples = int(total_duration * target_sr) + 24000 # buffer
        # careful with memory if video is 1 hour.
        # If huge, ffmpeg is better.
        
        # For < 5 min videos, tensor is fine.
        # 5 min * 60 * 24000 * 4 bytes ~ 30MB. Safe.
        
        try:
            import torch
            import soundfile as sf
            import numpy as np
            
            # Canvas in tensor [1, Samples]
            canvas = torch.zeros((1, total_samples))
            
            for seg in sorted_segments:
                path = seg['audio_path']
                start_sec = seg['start']
                
                # Load segment with soundfile
                wav_np, sr = sf.read(str(path))
                
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
                    padding = end_sample - total_samples
                    canvas = torch.cat([canvas, torch.zeros(1, padding + 1000)], dim=1)
                    total_samples = canvas.shape[1]

                # Mix (add) or overwrite? Overwrite is safer for avoiding doubpling if overlap (rare in this logic)
                # But let's add to be safe for overlaps.
                # Ensure we only take first channel if input is stereo
                if wav.shape[0] > 1:
                    wav = wav[0:1, :]
                    
                canvas[:, start_sample:end_sample] += wav[:, :end_sample-start_sample]
            
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
